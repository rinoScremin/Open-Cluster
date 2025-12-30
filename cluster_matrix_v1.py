import torch
import paramiko
import os
import time
import socket
import threading
import subprocess
import ctypes
import struct
import numpy as np
import tempfile
import zmq
import atexit
import json
import mmap
import shutil
import glob
import math

# ======================================================
# SHAPE + NUMERICAL COMPARISON
# ======================================================

def check_combined_result_values(c_ref_path, combined):
    c_ref = torch.load(c_ref_path)
    if c_ref.shape != combined.shape:
        print(f"❌ Shape mismatch! Reference: {c_ref.shape}, Combined: {combined.shape}")
    else:
        print(f"✅ Shapes match: {c_ref.shape}")

        # Ensure both are Torch tensors (defensive)
        if not isinstance(c_ref, torch.Tensor):
            c_ref = torch.from_numpy(c_ref)
        if not isinstance(combined, torch.Tensor):
            combined = torch.from_numpy(combined)

        c_ref = c_ref.to(dtype=combined.dtype, device=combined.device)

        # Calculate absolute differences
        diff = torch.abs(c_ref - combined)

        # Basic statistics
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        print(f"Max absolute difference:  {max_diff:.6e}")
        print(f"Mean absolute difference: {mean_diff:.6e}")

        # Looser tolerance for shard / float accumulation
        tolerance = 0.15

        if torch.allclose(c_ref, combined, rtol=tolerance, atol=tolerance):
            print(f"✅ Results match within tolerance ({tolerance})")
        else:
            print(f"⚠️  Results differ beyond tolerance ({tolerance})")

        significant_diff = diff > tolerance
        num_different = torch.sum(significant_diff).item()
        total_elements = c_ref.numel()

        print(
            f"Elements with > {tolerance} difference: "
            f"{num_different}/{total_elements} "
            f"({(num_different / total_elements * 100):.2f}%)"
        )

# ======================================================
# VISUAL DEBUG: FIRST & SECOND HALF
# ======================================================

def print_matrix_sections(matrix, name, n_elements=100):
    flat = matrix.flatten()
    mid = flat.shape[0] // 2

    print(f"\n{name} — first {n_elements} elements of first half:")
    print(flat[:n_elements])

    print(f"\n{name} — first {n_elements} elements of second half:")
    print(flat[mid:mid + n_elements])

def convert_bin_matrix_to_pt(filename, force_2d=True):  
    """  
    Load a binary matrix saved in the format:  
    [num_dims(int), dim1(int), dim2(int), ..., data(float32)]  
      
    Args:  
        filename: path to binary file  
        force_2d: if True, flatten batch*depth*rows into 2D (rows x cols)  
          
    Returns:  
        PyTorch tensor  
    """  
    with open(filename, 'rb') as f:  
        # Read number of dimensions  
        ndim_bytes = f.read(4)  
        if len(ndim_bytes) < 4:  
            raise ValueError("File too short to read ndim")  
        ndim = struct.unpack('i', ndim_bytes)[0]  
  
        # Read dimensions  
        dims_bytes = f.read(ndim * 4)  
        if len(dims_bytes) < ndim * 4:  
            raise ValueError("File too short to read dimensions")  
        dims = list(struct.unpack('i' * ndim, dims_bytes))  
  
        # Read data  
        num_elements = np.prod(dims)  
        data_bytes = f.read(num_elements * 4)  
        if len(data_bytes) < num_elements * 4:  
            raise ValueError(f"File too short to read {num_elements} floats")  
        data_np = np.frombuffer(data_bytes, dtype=np.float32).copy()  # ensure writable  
  
    # Reshape  
    tensor_np = data_np.reshape(dims)  
  
    # Optionally flatten batch*depth*rows -> 2D for LLAMA-like 4D tensors  
    if force_2d and ndim == 4:  
        batch, depth, rows, cols = dims  
        tensor_np = tensor_np.reshape(batch * depth * rows, cols)  
  
    # Convert to PyTorch tensor  
    tensor_pt = torch.from_numpy(tensor_np).float()  
  
    # Info  
    print(f"✅ Loaded {filename}")  
    print(f"  Original dims: {dims}")  
    print(f"  Result tensor shape: {tensor_pt.shape}, size: {tensor_pt.numel()*4:,} bytes")  
    print(f"  Data range: [{tensor_pt.min().item():.6f}, {tensor_pt.max().item():.6f}]")  
  
    return tensor_pt

class cluster_matrix:
    def __init__(self, matrix_file_path,
                node_IP_list, CPU_GPU_select_list, node_percentages=[], back_end_select_list=[],
                split_matrix=False, dim=0, pair_split_with_shape=None):
        
        print("=" * 70)
        print("🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM")
        print("=" * 70)
        
        # =============== NODE CONFIGURATION VALIDATION ===============
        print("\n📋 VALIDATING NODE CONFIGURATION...")
        
        # Check consistency of the node configuration
        if not (len(node_IP_list) == len(CPU_GPU_select_list) == len(back_end_select_list) == len(node_percentages)):
            print("❌ NODE CONFIGURATION ERROR: Lengths do not match!")
            print(f"   - node_IP_list: {len(node_IP_list)} nodes")
            print(f"   - CPU_GPU_select_list: {len(CPU_GPU_select_list)} selections")
            print(f"   - back_end_select_list: {len(back_end_select_list)} backends")
            print(f"   - node_percentages: {len(node_percentages)} percentages")
            raise ValueError("Node configuration error: All lists must have the same length")

        # Check that percentages sum to (roughly) 1.0
        total_percent = sum(node_percentages)
        if abs(total_percent - 1.0) > 1e-6:
            print(f"❌ PERCENTAGE ERROR: Node percentages sum to {total_percent:.6f}, should be 1.0")
            raise ValueError(f"Node configuration error: percentages do not sum to 1.0 (sum={total_percent})")

        print(f"✅ Node configuration validated: {len(node_IP_list)} nodes configured")
        print(f"✅ Percentage distribution validated: {total_percent:.6f}")

        # =============== NETWORK AND PORT CONFIGURATION ===============
        print("\n🌐 CONFIGURING NETWORK SETTINGS...")
        
        # Get head node IP addresses from environment variables
        self.IP = os.environ.get('HEAD_NODE_IP', '192.168.2.100')
        self.wifi_IP = os.environ.get('HEAD_NODE_IP_WIFI', '192.168.50.113')
        
        print(f"   Head Node Ethernet IP: {self.IP}")
        print(f"   Head Node WiFi IP: {self.wifi_IP}")
        
        # ZeroMQ ports for llama communication
        self.llama_head_node_PULL_port = os.environ.get("HEAD_NODE_PULL_PORT_C", "7779")
        self.llama_head_node_PUSH_port = os.environ.get("HEAD_NODE_PUSH_PORT_C", "7780")
        self.llama_worker_node_PULL_port = os.environ.get("WORKER_NODE_PULL_PORT_C", "5557")
        self.llama_worker_node_PUSH_port = os.environ.get("WORKER_NODE_PUSH_PORT_C", "5558")
        
        print(f"   Head Node Ports: PULL={self.llama_head_node_PULL_port}, PUSH={self.llama_head_node_PUSH_port}")
        print(f"   Worker Node Ports: PULL={self.llama_worker_node_PULL_port}, PUSH={self.llama_worker_node_PUSH_port}")
        
        # Python frontend ACK / cluster barrier port
        self.python_front_end_cluster_port = os.environ.get("PYTHON_FRONT_END_CLUSTER_PORT", "7790")
        print(f"   Cluster Barrier Port: {self.python_front_end_cluster_port}")

        # =============== FOLDER PATH CONFIGURATION ===============
        print("\n📁 CONFIGURING STORAGE PATHS...")
        
        # Local paths (head node)
        self.local_matrix_results_RAM_folder = os.environ.get('LOCAL_MATRIX_RESULTS_RAM_FOLDER', '/dev/shm/matrix_results/')
        self.local_DISK_folder = os.environ.get('LOCAL_DISK_FOLDER', 'matrix_shards/')
        self.local_RAM_folder = os.environ.get('LOCAL_RAM_FOLDER', '/dev/shm/matrix_shards/')
        self.local_project_dir = os.environ.get('LOCAL_PROJECT_DIR', '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/')
        
        print(f"   Local Paths:")
        print(f"     - RAM Results: {self.local_matrix_results_RAM_folder}")
        print(f"     - Disk Folder: {self.local_DISK_folder}")
        print(f"     - RAM Folder: {self.local_RAM_folder}")
        print(f"     - Project Dir: {self.local_project_dir}")
        
        # Remote paths (worker nodes)
        self.remote_DISK_folder = os.environ.get('REMOTE_DISK_FOLDER', 'matrix_shards/')
        self.remote_RAM_folder = os.environ.get('REMOTE_RAM_FOLDER', '/dev/shm/matrix_shards/')
        self.remote_matrix_results_RAM_folder = os.environ.get('REMOTE_MATRIX_RESULTS_RAM_FOLDER', '/dev/shm/matrix_results/')
        self.remote_project_dir = os.environ.get('REMOTE_PROJECT_DIR', '/home/rino/Desktop/Open_Cluster_AI_Station_beta/')
        
        print(f"   Remote Paths:")
        print(f"     - Disk Folder: {self.remote_DISK_folder}")
        print(f"     - RAM Folder: {self.remote_RAM_folder}")
        print(f"     - RAM Results: {self.remote_matrix_results_RAM_folder}")
        print(f"     - Project Dir: {self.remote_project_dir}")
        
        # =============== INSTANCE VARIABLE INITIALIZATION ===============
        print("\n📊 INITIALIZING INSTANCE VARIABLES...")
        
        self.matrix_file_path = matrix_file_path
        self.node_IP_list = node_IP_list
        wifi_env = os.environ.get("WORKER_WIFI_IPS", "")
        if wifi_env:
            self.IP_list_wifi = [ip.strip() for ip in wifi_env.split(",") if ip.strip()]
        else:
            self.IP_list_wifi = ['192.168.3.13', '192.168.3.243', '192.168.3.165', '192.168.3.94']  # WiFi testing IPs
        self.node_percentages = node_percentages
        self.dim = dim
        self.transpose = False
        self.CPU_GPU_select_list = CPU_GPU_select_list  # True for GPU, False for CPU
        self.back_end_select_list = back_end_select_list  # 'torch', 'llama', 'opencl'
        self.split_matrix = split_matrix
        self.OG_matrix_shape = []
        self.pair_split_with_shape = pair_split_with_shape
        self.grid_rows = 0
        self.grid_cols = 0

        # Extract matrix name from file path
        matrix_file_path_split = matrix_file_path.split('/')
        self.matrix_name = matrix_file_path_split[len(matrix_file_path_split)-1].split('.pt')[0]
        print(f"   Matrix Name: {self.matrix_name}")
        print(f"   Split Matrix: {split_matrix}")
        print(f"   Dimension: {dim}")
        
        # If no backend specified, default to 'torch' for all nodes
        if self.back_end_select_list == []:
            print("   No backend specified, defaulting to 'torch' for all nodes")
            for CPU_GPU_select in self.CPU_GPU_select_list:
                self.back_end_select_list.append('torch')
        
        self.node_matrices = []
        self.matrix_file_paths_list = []  # List for storing matrix file paths
        
        # =============== CREATE LOCAL DIRECTORIES ===============
        print("\n📂 CREATING LOCAL DIRECTORIES...")
        
        directories_created = []
        if not os.path.exists(self.local_DISK_folder):
            os.makedirs(self.local_DISK_folder)
            directories_created.append(self.local_DISK_folder)
        if not os.path.exists(self.local_RAM_folder):
            os.makedirs(self.local_RAM_folder)
            directories_created.append(self.local_RAM_folder)
        if not os.path.exists(self.local_matrix_results_RAM_folder):
            os.makedirs(self.local_matrix_results_RAM_folder)
            directories_created.append(self.local_matrix_results_RAM_folder)
        
        if directories_created:
            print(f"✅ Created directories: {', '.join(directories_created)}")
        else:
            print("✅ All required directories already exist")

        # =============== ZEROMMQ SOCKET SETUP ===============
        print("\n🔌 SETTING UP ZEROMQ CONNECTIONS...")
        
        # Initialize ZeroMQ context
        self.zmq_context = zmq.Context()
        self.llama_socket_pool = {}  # For llama communication - ports 5557/5558
        self.llama_socket_pool_wifi = {}  # Placeholder to avoid cleanup errors
        self.timeout = 5000  # 5 second timeout
        
        # Create PUSH sockets for ALL remote nodes
        unique_IP_list = list(dict.fromkeys(node_IP_list))
        print(f"   Connecting to {len(unique_IP_list)} unique nodes...")
        
        for node_ip in unique_IP_list:
            if node_ip != self.IP:  # Remote nodes only
                # Llama socket (port 5557 for computation)
                try:
                    llama_socket = self.zmq_context.socket(zmq.PUSH)
                    llama_socket.connect(f"tcp://{node_ip}:{self.llama_worker_node_PULL_port}")
                    self.llama_socket_pool[node_ip] = llama_socket
                    print(f"   ✅ Connected to worker node {node_ip}:{self.llama_worker_node_PULL_port}")
                except Exception as e:
                    print(f"   ❌ Failed to connect to {node_ip}: {e}")
        
        # Optional WiFi sockets (parallel transfer)
        for idx, node_ip in enumerate(unique_IP_list):
            if idx < len(self.IP_list_wifi):
                wifi_ip = self.IP_list_wifi[idx]
                try:
                    wifi_socket = self.zmq_context.socket(zmq.PUSH)
                    wifi_socket.connect(f"tcp://{wifi_ip}:{self.llama_worker_node_PULL_port}")
                    self.llama_socket_pool_wifi[wifi_ip] = wifi_socket
                    print(f"   ✅ Connected to worker WiFi {wifi_ip}:{self.llama_worker_node_PULL_port}")
                except Exception as e:
                    print(f"   ❌ Failed to connect WiFi {wifi_ip}: {e}")

        # Connect to local head node as well
        try:
            llama_socket = self.zmq_context.socket(zmq.PUSH)
            llama_socket.connect(f"tcp://{self.IP}:{self.llama_head_node_PULL_port}")
            self.llama_socket_pool[self.IP] = llama_socket
            print(f"   ✅ Connected to head node (self) {self.IP}:{self.llama_head_node_PULL_port}")
        except Exception as e:
            print(f"   ❌ Failed to connect to head node: {e}")
        
        print(f"   Total sockets in pool: {len(self.llama_socket_pool)}")

        # =============== CLUSTER BARRIER/ACK RECEIVER SETUP ===============
        print("\n🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...")
        
        # Initialize ack receiver socket (singleton pattern)
        if not hasattr(cluster_matrix, '_ack_receiver_socket'):
            try:
                cluster_matrix._ack_receiver_socket = self.zmq_context.socket(zmq.PULL)
                cluster_matrix._ack_receiver_socket.bind(f"tcp://0.0.0.0:{self.python_front_end_cluster_port}")
                print(f"✅ Python frontend ACK receiver bound to port {self.python_front_end_cluster_port}")
            except Exception as e:
                print(f"❌ Failed to bind ACK receiver: {e}")
                raise
        else:
            print(f"✅ ACK receiver already exists on port {self.python_front_end_cluster_port}")
        
        # Reference it in the instance
        self.ack_receiver_socket = cluster_matrix._ack_receiver_socket

        # =============== CREATE REMOTE DIRECTORIES ===============
        print("\n📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...")
        
        command = f'mkdir -p {self.remote_DISK_folder} {self.remote_RAM_folder} {self.remote_matrix_results_RAM_folder}'
        print(f"   Sending command: {command}")
        
        for node_ip, socket in self.llama_socket_pool.items():
            try:
                socket.send_multipart([command.encode('utf-8')])
                print(f"   ✅ Directory creation command sent to {node_ip}")
            except Exception as e:
                print(f"   ❌ Failed to send command to {node_ip}: {e}")

    
        # =============== MATRIX DISTRIBUTION LOGIC ===============
        '''
        
        print("\n" + "=" * 70)
        print("🧮 MATRIX DISTRIBUTION PHASE")
        print("=" * 70)
        
        # Initialize with a default value
        matrix_shards_found = True  # Default to True
        matrix_exists = os.path.exists(matrix_file_path)
        
        print(f"   Matrix file exists: {matrix_exists}")
        print(f"   Split matrix mode: {split_matrix}")
        
        # Decision tree for matrix handling
        if matrix_exists and split_matrix:
            # Case 1: New matrix, needs splitting and distribution
            print("\n📝 CASE 1: NEW MATRIX - CONVERT, DISTRIBUTE, AND LOAD")
            print("   Processing steps:")
            print("   1. Convert to cluster matrix shards")
            print("   2. Distribute shards to nodes")
            print("   3. Load distributed shards")
            self.convert_to_cluster_matrix_shards()
            self.save_distribute_matrix_shards_bin()
            
        elif not matrix_exists and split_matrix:
            # Case 2: Matrix doesn't exist but split mode requested
            print("\n🔍 CASE 2: LOADING EXISTING DISTRIBUTED MATRIX SHARDS")
            print("   Attempting to load pre-existing shards...")
            matrix_shards_found = self.load_cluster_matrix_shards()
            
        elif matrix_exists and not split_matrix:
            # Case 3: Matrix exists, distribute as whole (no splitting)
            print("\n📦 CASE 3: DISTRIBUTING FULL MATRIX (NO SPLITTING)")
            print("   Processing steps:")
            print("   1. Save full matrix in binary format")
            print("   2. Distribute to all nodes")
            self.save_distribute_full_matrix_bin()
            
        elif not matrix_exists and not split_matrix:
            # Case 4: Load existing full matrix distribution
            print("\n🔍 CASE 4: LOADING EXISTING FULL MATRIX DISTRIBUTION")
            print("   Attempting to load pre-distributed full matrix...")
            matrix_shards_found = self.load_cluster_matrix()

        # =============== FINAL VALIDATION ===============
        print("\n" + "=" * 70)
        print("🏁 INITIALIZATION COMPLETE")
        print("=" * 70)
        
        if not matrix_shards_found:
            print("❌ ERROR: MATRIX SHARDS/FILES NOT FOUND!")
            print("   Possible causes:")
            print("   1. Matrix file path is incorrect")
            print("   2. Distributed shards were not properly created")
            print("   3. Network issues preventing file access")
        else:
            print("✅ Cluster matrix initialization successful!")
            print(f"   - Total nodes configured: {len(node_IP_list)}")
            print(f"   - Matrix handling mode: {'Split' if split_matrix else 'Full'}")
            print(f"   - Backends: {self.back_end_select_list}")
            print(f"   - CPU/GPU selections: {self.CPU_GPU_select_list}")
        
        '''
        
    def wait_for_acks(self, expected_count, expected_msg="ACK"):
        """
        Wait for ACKs from all expected nodes on the Python front end cluster port.
        """
        acks = 0
        while acks < expected_count:
            try:
                msg = self.ack_receiver_socket.recv_string(flags=zmq.NOBLOCK)
                if msg == expected_msg:
                    acks += 1
                    print(f"✅ Received {expected_msg} {acks}/{expected_count}")
            except zmq.Again:
                # No message yet, sleep briefly to avoid 100% CPU
                time.sleep(0.01)
        print("✅ All ACKs received!")
        return acks

    def zmq_send_command(self, worker_ip, command, timeout=5):
        """Send command using persistent connection pool"""
        if worker_ip in self.llama_socket_pool:
            socket_eth = self.llama_socket_pool[worker_ip]
            try:
                # MUST send bytes, NOT str.
                socket_eth.send(command.encode('utf-8'))
                return True
            except Exception as e:
                print(f"❌ Error sending command to {worker_ip}: {e}")
                return False
        else:
            print(f"❌ No socket found for worker {worker_ip}")
            return False
 
    def zmq_send_file(self, worker_ip, local_file_path):
        if worker_ip in self.llama_socket_pool:
            socket_eth = self.llama_socket_pool[worker_ip]
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
            
            # Use os.path.basename to get filename
            filename_only = os.path.basename(local_file_path)
            
            socket_eth.send_multipart([
                filename_only.encode(),
                file_data
            ])
            print(f"📤 Sent file {filename_only} to {worker_ip}")
        
    def cleanup(self):
        for socket in self.llama_socket_pool.values():
            socket.close()
        for socket in self.llama_socket_pool_wifi.values():
            socket.close()
        self.zmq_context.term()

    def convert_to_cluster_matrix_grid(self):

        full_matrix = torch.load(self.matrix_file_path)
        self.OG_matrix_shape = list(full_matrix.shape)

        # -------------------------------------------------
        # BOTH A (dim=2) AND B (dim=3) SPLIT ON ROWS (dim 0)
        # KEEP INNER DIMENSION = 20000
        # -------------------------------------------------
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 (A) or 3 (B)")

        split_dim = 0  # ✅ ALWAYS ROW SPLIT
        total_rows = full_matrix.size(0)

        shard_sizes = []
        remaining = total_rows

        for i, pct in enumerate(self.node_percentages):
            if i == len(self.node_percentages) - 1:
                shard = remaining
            else:
                shard = int(math.floor(total_rows * pct))
                remaining -= shard
            shard_sizes.append(shard)

        shards = torch.split(full_matrix, shard_sizes, dim=split_dim)

        self.node_matrices = list(shards)

        # Linear distribution (NOT a 2D grid)
        self.grid_rows = len(self.node_matrices)
        self.grid_cols = 1

        # Debug print (keep this)
        print(len(self.node_matrices))
        for s in self.node_matrices:
            print(s.shape)

        return self.node_matrices

    def convert_to_cluster_matrix_shards(self):
        """
        Split the original matrix into shards according to node percentages.
        Stores the shards in self.node_matrices.
        """
        if self.matrix_file_path is None:
            raise ValueError("Matrix file path not set.")

        # Load full matrix
        full_matrix = torch.load(self.matrix_file_path)
        total_rows = full_matrix.size(self.dim)  # typically dim=0
        self.node_matrices = []

        # Convert percentages to row counts
        if hasattr(self, 'node_percentages') and self.node_percentages:
            total_percentage = sum(self.node_percentages)
            if abs(total_percentage - 1.0) > 1e-6:
                raise ValueError(f"Node percentages must sum to 1. Got {total_percentage}")
            rows_per_node = [int(total_rows * p) for p in self.node_percentages]
            # Adjust for rounding error
            diff = total_rows - sum(rows_per_node)
            if diff != 0:
                rows_per_node[-1] += diff
        else:
            # Default: even split among nodes
            num_nodes = len(self.node_IP_list)
            base_rows = total_rows // num_nodes
            rows_per_node = [base_rows] * num_nodes
            rows_per_node[-1] += total_rows - sum(rows_per_node)

        # Slice the full matrix into shards
        start_idx = 0
        for node_idx, row_count in enumerate(rows_per_node):
            end_idx = start_idx + row_count
            shard = full_matrix.narrow(self.dim, start_idx, row_count).clone()
            self.node_matrices.append(shard)
            start_idx = end_idx

        print(f"✅ Created {len(self.node_matrices)} shards according to node percentages")
        for i, shard in enumerate(self.node_matrices):
            print(f"  Node {i}: shard shape {shard.shape}")

        return self.node_matrices

    def merged_matrix(self, matrix_shards, start_index, end_index):
        """
        Merge row shards from start_index (inclusive) to end_index (exclusive).
        Returns the concatenated tensor along dim=0.
        """
        if start_index < 0 or end_index <= start_index:
            raise ValueError("Invalid start_index/end_index")
        end_index = min(end_index, len(matrix_shards))
        pieces = [matrix_shards[i] for i in range(start_index, end_index)]
        if not pieces:
            raise ValueError("No shards to merge")
        return torch.cat(pieces, dim=self.dim)

    def save_distribute_matrix_shards_bin(self):
        """Save matrix shards as binary files and distribute to appropriate nodes."""
        
        # Get list of unique node IPs for ACK tracking
        unique_node_IP_list = list(set(self.node_IP_list))
        print(f"Starting distribution of {len(self.node_IP_list)} shards to {len(unique_node_IP_list)} unique nodes")
        
        # Process each shard
        for shard_index, node_IP in enumerate(self.node_IP_list):
            print(f"Processing shard {shard_index} for node {node_IP}")
            
            # Create filename for this shard
            save_name = self.matrix_name.split('.pt')[0] + '_shard_' + str(shard_index)
            
            # Handle shard for HEAD NODE (local storage)
            if node_IP == self.IP:
                save_name += '.bin'
                save_file_path_DISK = os.path.join(self.local_DISK_folder, save_name)
                save_file_path_RAM = os.path.join(self.local_RAM_folder, save_name)
                
                print(f"  Head node: Saving to DISK={save_file_path_DISK}")
                print(f"  Head node: Saving to RAM={save_file_path_RAM}")
                
                # Save tensor to binary file in both locations
                self.save_matrix_binary(self.node_matrices[shard_index].float(), save_file_path_DISK)
                self.save_matrix_binary(self.node_matrices[shard_index].float(), save_file_path_RAM)
                
                # Store RAM path for later access
                self.matrix_file_paths_list.append(save_file_path_RAM)
                print(f"  Added RAM path to file list")
                    
            # Handle shard for REMOTE NODE
            elif node_IP != self.IP:
                save_name += '.bin'
                
                print(f"  Remote node {node_IP}: Beginning distribution")
                
                # Step 1: Save shard locally first
                save_file_path_DISK = os.path.join(self.local_DISK_folder, save_name)
                print(f"  Step 1: Saving locally to {save_file_path_DISK}")
                self.save_matrix_binary(self.node_matrices[shard_index].float(), save_file_path_DISK)
                
                # Step 2: Send file to remote node via ZeroMQ
                print(f"  Step 2: Sending file to remote node {node_IP}")
                self.zmq_send_file(node_IP, save_file_path_DISK)
                
                # Step 3: Tell remote node to copy from RAM to DISK
                remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, save_name)
                remote_disk_dir_full = os.path.join(self.remote_project_dir, self.remote_DISK_folder)
                remote_save_file_path_DISK = os.path.join(remote_disk_dir_full, save_name)
                mkdir_cmd = f"mkdir -p {remote_disk_dir_full} {self.remote_RAM_folder} {self.remote_matrix_results_RAM_folder}"
                self.zmq_send_command(node_IP, mkdir_cmd)
                copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                print(f"  Step 3: Sending copy command to remote")
                self.zmq_send_command(node_IP, copy_command)
                
                # Step 4: Store remote RAM path (not local)
                self.matrix_file_paths_list.append(remote_save_file_path_RAM)
                print(f"  Added remote RAM path to file list: {remote_save_file_path_RAM}")
        
        # Wait for ACK signals from remote nodes (excluding head node)
        print(f"Waiting for ACKs from {len(unique_node_IP_list)-1} remote nodes...")
        self.wait_for_acks(len(unique_node_IP_list)-1)
        
        print(f"Distribution complete: {len(self.matrix_file_paths_list)} shards saved and distributed")
        return self.matrix_file_paths_list

    def save_distribute_full_matrix_bin(self):
        """
        Save a FULL matrix (no splitting) as binary and distribute to all nodes.
        """
        # Create filename: replace .pt with .bin
        save_name = self.matrix_name.split('.pt')[0] + '.bin'
        print(f"Preparing full matrix: {save_name}")
        
        # Define local file paths
        save_file_path_DISK = os.path.join(self.local_DISK_folder, save_name)
        local_save_file_path_RAM = os.path.join(self.local_RAM_folder, save_name)
        print(f"Local paths - DISK: {save_file_path_DISK}, RAM: {local_save_file_path_RAM}")
        
        # Load the full matrix from PyTorch file
        print(f"Loading matrix from: {self.matrix_file_path}")
        full_matrix = torch.load(self.matrix_file_path)
        print(f"Matrix loaded - Shape: {full_matrix.shape}")
        
        # Save to binary format locally
        print("Saving to local storage...")
        self.save_matrix_binary(full_matrix.float(), save_file_path_DISK)
        self.save_matrix_binary(full_matrix.float(), local_save_file_path_RAM)

        # Define remote paths (absolute disk path)
        remote_disk_dir_full = os.path.join(self.remote_project_dir, self.remote_DISK_folder)
        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, save_name)
        remote_save_file_path_DISK = os.path.join(remote_disk_dir_full, save_name)
        print(f"Remote paths - RAM: {remote_save_file_path_RAM}, DISK: {remote_save_file_path_DISK}")
        
        # Track file paths for each node
        for node_ip in self.node_IP_list:
            if node_ip == self.IP:
                # Head node uses local RAM path
                self.matrix_file_paths_list.append(local_save_file_path_RAM)
            else:
                # Remote nodes use remote RAM path
                self.matrix_file_paths_list.append(remote_save_file_path_RAM)
        
        # Get UNIQUE IPs (no duplicates)
        unique_node_IP_list = list(set(self.node_IP_list))
        unique_remote_count = len([ip for ip in unique_node_IP_list if ip != self.IP])
        
        print(f"Distributing to {unique_remote_count} remote node(s)...")
        
        # Send file to each unique remote node
        for node_ip in unique_node_IP_list:
            if node_ip != self.IP:  # Skip local node
                print(f"Sending to {node_ip}")

                # Ensure dirs exist on remote
                mkdir_cmd = f"mkdir -p {remote_disk_dir_full} {self.remote_RAM_folder} {self.remote_matrix_results_RAM_folder}"
                self.zmq_send_command(node_ip, mkdir_cmd)

                # Step 1: Send the file to remote node's RAM
                self.zmq_send_file(node_ip, save_file_path_DISK)
                
                # Step 2: Tell remote node to copy from RAM to DISK for persistence
                copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                self.zmq_send_command(node_ip, copy_command)
        
        # Wait for acknowledgments from remote nodes
        self.wait_for_acks(len(unique_node_IP_list)-1)
        
        print(f"Full matrix distribution completed")
        print(f"Total file paths tracked: {len(self.matrix_file_paths_list)}")
        return 0

    def save_distribute_matrix_grid_bin(self):
        """
        Distribute broadcast shards for distributed GEMM.

        Matrix A (dim=2):
            - Row shards A_row{i} → broadcast across grid columns

        Matrix B (dim=3):
            - Row shards B_row{j} → broadcast across grid rows
            - Transpose happens at compute time (B_row.T)

        Populates self.matrix_file_paths_list with RAM paths in correct order
        for cluster_shard_operation.
        """

        if not hasattr(self, 'node_matrices') or not self.node_matrices:
            raise ValueError("Call convert_to_cluster_matrix_grid first")

        if not self.node_IP_list:
            raise ValueError("node_IP_list empty")

        print(f"📐 Grid: {self.grid_rows} × {self.grid_cols}")

        self.matrix_file_paths_list = []
        os.makedirs(self.local_DISK_folder, exist_ok=True)
        os.makedirs(self.local_RAM_folder, exist_ok=True)

        ip_saved_files = {}
        matrix_base = self.matrix_name.split('.pt')[0]

        # ---------------------------
        # MATRIX A — ROW SHARDS
        # ---------------------------
        if self.dim == 2:
            print("\n📤 Distributing Matrix A row shards")
            for grid_row in range(self.grid_rows):
                shard = self.node_matrices[grid_row]
                filename = f"{matrix_base}_A_row{grid_row+1}.bin"

                for grid_col in range(self.grid_cols):
                    node_idx = grid_row * self.grid_cols + grid_col
                    if node_idx >= len(self.node_IP_list):
                        continue

                    node_IP = self.node_IP_list[node_idx]

                    if node_IP not in ip_saved_files:
                        ip_saved_files[node_IP] = set()

                    if filename in ip_saved_files[node_IP]:
                        continue

                    print(f"   → Node {node_IP} gets {filename} {tuple(shard.shape)}")

                    save_disk = os.path.join(self.local_DISK_folder, filename)
                    self.save_matrix_binary(shard.float(), save_disk)

                    if node_IP == self.IP:
                        save_ram = os.path.join(self.local_RAM_folder, filename)
                        self.save_matrix_binary(shard.float(), save_ram)
                        self.matrix_file_paths_list.append(save_ram)
                    else:
                        self.zmq_send_file(node_IP, save_disk)
                        remote_save_ram = os.path.join(self.remote_RAM_folder, filename)
                        self.matrix_file_paths_list.append(remote_save_ram)

                    ip_saved_files[node_IP].add(filename)

        # ---------------------------
        # MATRIX B — ROW SHARDS
        # ---------------------------
        else:
            print("\n📤 Distributing Matrix B row shards")
            for grid_col in range(self.grid_cols):
                shard = self.node_matrices[grid_col]
                filename = f"{matrix_base}_B_row{grid_col+1}.bin"

                for grid_row in range(self.grid_rows):
                    node_idx = grid_row * self.grid_cols + grid_col
                    if node_idx >= len(self.node_IP_list):
                        continue

                    node_IP = self.node_IP_list[node_idx]

                    if node_IP not in ip_saved_files:
                        ip_saved_files[node_IP] = set()

                    if filename in ip_saved_files[node_IP]:
                        continue

                    print(f"   → Node {node_IP} gets {filename} {tuple(shard.shape)}")

                    save_disk = os.path.join(self.local_DISK_folder, filename)
                    self.save_matrix_binary(shard.float(), save_disk)

                    if node_IP == self.IP:
                        save_ram = os.path.join(self.local_RAM_folder, filename)
                        self.save_matrix_binary(shard.float(), save_ram)
                        self.matrix_file_paths_list.append(save_ram)
                    else:
                        self.zmq_send_file(node_IP, save_disk)
                        remote_save_ram = os.path.join(self.remote_RAM_folder, filename)
                        self.matrix_file_paths_list.append(remote_save_ram)

                    ip_saved_files[node_IP].add(filename)

        # ---------------------------
        # ACK SYNC
        # ---------------------------
        remote_nodes = list(set(self.node_IP_list) - {self.IP})
        print(f"\n⏳ Waiting for ACKs from {len(remote_nodes)} nodes")
        self.wait_for_acks(len(remote_nodes), 'ACK')

        self.blocks_distributed = True

        print("\n✅ Distribution complete")
        print(f"Total shards: {len(self.matrix_file_paths_list)}")
        return self.matrix_file_paths_list

    def save_matrix_binary(self, matrix, filename):
        """
        Save a PyTorch tensor or numpy array as a binary file.
        
        Binary format:
        [num_dims, dim1, dim2, ..., data]
        Always saves as 4D format (batch, channel, height, width) for consistency.
        Data is always saved as float32.
        
        Args:
            matrix: PyTorch tensor or numpy array to save
            filename: Path where the binary file will be saved
        """
        print(f"Saving matrix to binary file: {filename}")
        
        # ===== CONVERT INPUT TO NUMPY ARRAY =====
        # Handle both PyTorch tensors and numpy arrays
        print("  Converting input to numpy array...")
        if isinstance(matrix, torch.Tensor):
            # Convert PyTorch tensor to numpy
            print(f"    Input is PyTorch tensor: shape={matrix.shape}, dtype={matrix.dtype}, device={matrix.device}")
            matrix_float = matrix.float().cpu()
            matrix_np = matrix_float.detach().numpy()
            print(f"    Converted to CPU float32 numpy array")
        elif isinstance(matrix, np.ndarray):
            # Already a numpy array
            print(f"    Input is numpy array: shape={matrix.shape}, dtype={matrix.dtype}")
            matrix_np = matrix.astype('float32')
            print(f"    Cast to float32")
        else:
            # Unsupported type
            error_msg = f"Unsupported matrix type: {type(matrix)}. Expected torch.Tensor or np.ndarray"
            print(f"  ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Ensure float32 dtype
        matrix_np = matrix_np.astype('float32')
        print(f"  Final numpy array: shape={matrix_np.shape}, dtype={matrix_np.dtype}")
        
        # ===== CONVERT TO 4D FORMAT =====
        # Always convert to 4D format for consistency (batch, channel, height, width)
        original_shape = matrix_np.shape
        print(f"  Converting to 4D format...")
        
        if len(matrix_np.shape) == 2:
            # 2D matrix -> reshape to (1, 1, height, width)
            new_shape = (1, 1, matrix_np.shape[0], matrix_np.shape[1])
            matrix_np = matrix_np.reshape(new_shape)
            print(f"    2D {original_shape} -> 4D {new_shape}")
        elif len(matrix_np.shape) == 3:
            # 3D matrix -> reshape to (1, channels, height, width)
            new_shape = (1, matrix_np.shape[0], matrix_np.shape[1], matrix_np.shape[2])
            matrix_np = matrix_np.reshape(new_shape)
            print(f"    3D {original_shape} -> 4D {new_shape}")
        elif len(matrix_np.shape) == 4:
            # Already 4D
            print(f"    Already 4D format: {matrix_np.shape}")
        else:
            # Unsupported dimensionality
            error_msg = f"Unsupported number of dimensions: {len(matrix_np.shape)}"
            print(f"  ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # ===== WRITE BINARY FILE =====
        print(f"  Writing binary file...")
        try:
            with open(filename, 'wb') as f:
                # Write number of dimensions (always 4 for consistency)
                ndim = 4
                f.write(struct.pack('i', ndim))
                print(f"    Wrote ndim: {ndim}")
                
                # Write all 4 dimensions
                for i, dim in enumerate(matrix_np.shape):
                    f.write(struct.pack('i', dim))
                    if i == 0:
                        print(f"    Dimensions: {dim}", end="")
                    else:
                        print(f" × {dim}", end="")
                print()  # New line after dimensions
                
                # Write the actual data
                f.write(matrix_np.tobytes())
                print(f"    Wrote {matrix_np.size:,} float32 elements")
                
        except Exception as e:
            print(f"  ERROR writing file {filename}: {e}")
            raise
        
        # ===== VERIFY FILE SIZE =====
        try:
            file_size = os.path.getsize(filename)
            # Calculate expected size: 4 bytes for ndim + 4×4 bytes for dimensions + data size
            expected_size = 4 + ndim * 4 + matrix_np.size * 4
            print(f"  File saved successfully")
            print(f"  File size: {file_size:,} bytes")
            print(f"  Expected size: {expected_size:,} bytes")
            
            # Verify file size matches expected
            if file_size == expected_size:
                print(f"  ✓ File size verification passed")
            else:
                print(f"  ⚠️  File size mismatch: got {file_size:,}, expected {expected_size:,}")
                
            # Calculate memory usage
            file_size_mb = file_size / (1024 * 1024)
            print(f"  Memory usage: {file_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"  ERROR getting file size: {e}")
        
        print(f"  Save completed: {filename}")
        return file_size

    def load_cluster_matrix(self):
        """
        Load a full matrix (not split) from disk and distribute to all nodes.
        """
        try:
            # Create filename for the binary matrix
            base_name = self.matrix_name + '.bin'
            combined_name = self.matrix_name + '_combined.bin'
            print(f"Loading full matrix: {base_name}")
            
            base_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, base_name)
            
            if not os.path.exists(base_disk_path):
                print(
                    "Error: Base matrix binary not found. Combined outputs are write-only "
                    "and cannot be reused as inputs. Regenerate shards or rerun the operation "
                    "with send_back=False to keep a distributed input."
                )
                return False

            source_path = base_disk_path
            source_filename = base_name
            print(f"Source file: {source_path}")
            # Copy to RAM for local access
            local_ram_path = os.path.join(self.local_RAM_folder, base_name)
            print(f"Copying to local RAM...")
            subprocess.run(f'cp {source_path} {self.local_RAM_folder}', shell=True, check=True)
            
            # Get unique nodes to avoid duplicate transfers
            unique_node_IP_list = list(set(self.node_IP_list))
            
            # Define remote paths (mirror the source filename)
            remote_disk_path = self.remote_DISK_folder + source_filename
            remote_RAM_path = self.remote_RAM_folder + source_filename
            
            # Track file paths for all nodes
            for node_ip in self.node_IP_list:
                if node_ip == self.IP:
                    # Head node uses local RAM path
                    self.matrix_file_paths_list.append(local_ram_path)
                else:
                    # Remote nodes use remote RAM path
                    self.matrix_file_paths_list.append(remote_RAM_path)
            
            # Distribute to remote nodes
            print(f"Distributing to remote nodes...")
            for node_ip in unique_node_IP_list:
                if node_ip != self.IP:
                    # Send file to remote node
                    self.zmq_send_file(node_ip, source_path)
                    
                    # Send command to copy from remote disk to remote RAM
                    copy_command = f'cp {self.remote_project_dir}{remote_disk_path} {self.remote_RAM_folder}'
                    self.zmq_send_command(node_ip, copy_command)
                    
        except Exception as e:
            print(f"Error loading matrix: {e}")
            return False
        
        print(f"Matrix loaded successfully")
        return True

    def load_cluster_matrix_shards(self):
        """
        Load distributed matrix shards from storage.
        
        This method checks if matrix shards are already in RAM, and if not,
        loads them from disk to RAM for both local and remote nodes.
        The controller (head node) needs all shards in its local RAM.
        Remote nodes only need their specific assigned shards.
        """
        
        # Initialize the file paths list
        self.matrix_file_paths_list = []
        
        print(f"Loading cluster matrix shards: {self.matrix_name}")
        print(f"Number of nodes/shard locations: {len(self.node_IP_list)}")
        
        # ===== CHECK IF SHARDS ARE ALREADY IN LOCAL RAM =====
        # Check if the first shard already exists in RAM (indicator all might be there)
        check_first_local_matrix_shard_ram_path = os.path.join(
            self.local_RAM_folder, 
            f"{self.matrix_name}_shard_0.bin"
        )
        print(f"Checking for existing shards in RAM: {check_first_local_matrix_shard_ram_path}")
        
        # ===== CASE 1: SHARDS ALREADY IN LOCAL RAM =====
        if os.path.exists(check_first_local_matrix_shard_ram_path):
            print("Found existing matrix shards in local RAM")
            
            # Just add all existing RAM paths to our list
            for shard_index, node_IP in enumerate(self.node_IP_list):
                local_matrix_shard_ram_path = os.path.join(
                    self.local_RAM_folder, 
                    f"{self.matrix_name}_shard_{shard_index}.bin"
                )
                self.matrix_file_paths_list.append(local_matrix_shard_ram_path)
                print(f"  Shard {shard_index}: Using existing RAM path")
        
        # ===== CASE 2: SHARDS NOT IN RAM, NEED TO LOAD FROM DISK =====
        else:
            print("Matrix shards not found in RAM, loading from disk...")
            
            for shard_index, node_IP in enumerate(self.node_IP_list):
                print(f"\nProcessing shard {shard_index} for node {node_IP}:")
                
                # Create file names
                shard_filename = f"{self.matrix_name}_shard_{shard_index}.bin"
                local_matrix_shard_ram_path = os.path.join(self.local_RAM_folder, shard_filename)
                
                # Add RAM path to our list (controller tracks all shard paths)
                self.matrix_file_paths_list.append(local_matrix_shard_ram_path)
                print(f"  Controller tracking: {local_matrix_shard_ram_path}")
                
                # ----- CONTROLLER NODE: COPY ALL SHARDS TO LOCAL RAM -----
                # The controller needs ALL shards in its RAM for coordination
                print(f"  Controller: Copying shard {shard_index} to local RAM")
                
                # Construct source and destination paths
                local_disk_source = os.path.join(
                    self.local_project_dir, 
                    self.local_DISK_folder, 
                    shard_filename
                )
                local_ram_dest = os.path.join(self.local_RAM_folder, shard_filename)
                
                # Create copy command for local system
                local_copy_command = f'cp "{local_disk_source}" "{local_ram_dest}"'
                print(f"  Local copy command: {local_copy_command}")
                
                # Execute local copy
                try:
                    subprocess.run(local_copy_command, shell=True, check=True)
                    print(f"  ✓ Successfully copied to local RAM")
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed to copy shard {shard_index} locally: {e}")
                    raise
                
                # ----- REMOTE NODES: COPY ONLY THEIR ASSIGNED SHARD -----
                # Each remote node only needs its specific shard in its RAM
                if self.IP != node_IP:
                    print(f"  Remote node {node_IP}: Setting up its assigned shard")
                    
                    # Construct remote paths
                    remote_disk_path = os.path.join(self.remote_DISK_folder, shard_filename)
                    remote_ram_path = os.path.join(self.remote_RAM_folder, shard_filename)
                    
                    # Command to copy from remote disk to remote RAM
                    remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                    print(f"  Remote copy command: {remote_copy_command}")
                    
                    # Send command to remote node via ZeroMQ
                    print(f"  Sending command to remote node {node_IP}...")
                    self.zmq_send_command(node_IP, remote_copy_command)
                    print(f"  ✓ Command sent to remote node")
        
        # ===== LOADING COMPLETE =====
        print(f"\nMatrix shard loading complete")
        print(f"Total shard paths tracked: {len(self.matrix_file_paths_list)}")
        
        return True
 
    def cluster_shard_operation(self, cluster_matrixB, TransposeA, TransposeB, send_back_result=False, operation='mul'):
        """
        Perform a distributed matrix operation across the cluster.
        
        Args:
            cluster_matrixB: Another cluster_matrix instance for the second operand
            TransposeA: Whether to transpose matrix A
            TransposeB: Whether to transpose matrix B  
            send_back_result: Whether to combine results into single file (True) 
                            or keep distributed (False)
            operation: Operation to perform ('mul', 'add', 'sub')
        
        Returns:
            Base name of the result file(s)
        """
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING CLUSTER OPERATION")
        print(f"{'='*60}")
        print(f"Matrix A: {self.matrix_name}")
        print(f"Matrix B: {cluster_matrixB.matrix_name}")
        print(f"Operation: {operation}")
        print(f"Transpose A: {TransposeA}, Transpose B: {TransposeB}")
        print(f"Send back result: {send_back_result}")
        print(f"Number of shards: {len(self.node_IP_list)}")
        start_time = time.time()
        
        # ===== TRACK GPU USAGE PER NODE =====
        # This ensures multiple GPUs on the same node get used properly
        node_gpu_counters = {}
        
        print(f"\n📤 DISTRIBUTING OPERATIONS TO NODES")
        print(f"{'-'*40}")
        
        # Send operation commands to each node for its assigned shard
        for shard_index, (node_IP, CPU_GPU_select, back_end_select, node_matrix) in enumerate(zip(
            self.node_IP_list,  
            self.CPU_GPU_select_list,
            self.back_end_select_list, 
            self.matrix_file_paths_list
        )):
            print(f"\nProcessing shard {shard_index}:")
            
            # Initialize or get GPU counter for this node
            if node_IP not in node_gpu_counters:
                node_gpu_counters[node_IP] = 0
            current_gpu_number = node_gpu_counters[node_IP]
            
            print(f"  Node: {node_IP}")
            print(f"  Backend: {back_end_select}")
            print(f"  Use GPU: {CPU_GPU_select} (GPU #{current_gpu_number})")
            
            # Get file paths for both matrices
            matrix_a = node_matrix  # Current matrix shard
            matrix_b = cluster_matrixB.matrix_file_paths_list[shard_index]  # Other matrix shard
            print(f"  Matrix A path: {matrix_a}")
            print(f"  Matrix B path: {matrix_b}")
            
            # Convert booleans to lowercase strings for command
            use_gpu_str = str(CPU_GPU_select).lower()  # "true" or "false"
            
            # ===== TRANSPOSE LOGIC HANDLING =====
            # Handle backend-specific transpose quirks.
            # GGML (llama) uses column-major; flip TransposeB and swap operand order
            # to mirror the previously working cross-backend behavior.
            local_TransposeA = TransposeA
            local_TransposeB = TransposeB
            if back_end_select == 'llama':
                local_TransposeB = not TransposeB

            TransposeA_str = str(local_TransposeA).lower()
            TransposeB_str = str(local_TransposeB).lower()
            print(f"  Final transpose flags - A: {TransposeA_str}, B: {TransposeB_str}")
            
            # ===== PREPARE SEND_BACK FLAG =====
            # Send total_shards count instead of just true/false
            if send_back_result:
                send_back_str = len(self.node_IP_list)  # Number of shards to combine
                print(f"  Send back result: Yes ({send_back_str} shards will be combined)")
            else:
                send_back_str = "0"  # 0 means no send back
                print(f"  Send back result: No (keep distributed)")
            
            send_back_str = str(send_back_str)
            
            # ===== BUILD COMMAND FOR SPECIFIC BACKEND =====
            command = (
                f"server_command={back_end_select} "
                f"{matrix_a} "          # GGML expects B first
                f"{TransposeA_str} "
                f"{matrix_b} "          # Then A
                f"{TransposeB_str} "
                f"{use_gpu_str} "
                f"{current_gpu_number} "
                f"{send_back_str} "
                f"{operation} "
                f"2 "
                f"{shard_index}"
            )
   
            # ===== SEND COMMAND TO NODE =====
            print(f"  Sending command to node...")
            socket_eth = self.llama_socket_pool[node_IP]
            socket_eth.send_multipart([command.encode()])
            print(f"  ✅ Command sent to node {node_IP}")
            
            # Only increment GPU counter if this node is using GPU
            if CPU_GPU_select:
                node_gpu_counters[node_IP] += 1
                print(f"  Incremented GPU counter for node {node_IP} to {node_gpu_counters[node_IP]}")
        
        # ===== WAIT FOR ACKS FROM ALL NODES =====
        unique_nodes = list(set(self.node_IP_list))
        expected_acks = len(self.node_IP_list)  # one ACK per shard/operation
        print(f"\n⏳ WAITING FOR ACKS FROM NODES ({expected_acks})")
        self.wait_for_acks(expected_acks, "ACK_matrixOp_complete")
        
        # ===== OPERATION COMPLETE =====
        print(f"\n{'='*60}")
        print(f"✅ CLUSTER OPERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Operation time: {time.time() - start_time:.2f} seconds")

        # When keep-distributed, return a cluster_matrix wired to the shard outputs
        # ===== SETUP RESULT FILENAMES =====
        # Result names match the operand order we send to the server:
        # self (matrix A) first, then cluster_matrixB (matrix B)

        base_result_name=''
        if back_end_select == 'torch':
            base_result_name = f"{self.matrix_name}x{cluster_matrixB.matrix_name}"
            print(f"\n📊 Result base: {base_result_name} (send_back={send_back_result})")
        if back_end_select == 'llama':
            base_result_name = f"{cluster_matrixB.matrix_name}x{self.matrix_name}"
            print(f"\n📊 Result base: {base_result_name} (send_back={send_back_result})")



        if send_back_result:
            self.wait_for_acks(1, "ACK_combined_matrix_saved")
            path = self.local_RAM_folder + base_result_name + '_combined.bin'
            combined_matrix = convert_bin_matrix_to_pt(path)
            os.remove(path)  # Clean up any existing combined file
            return combined_matrix
        else:
            result_cluster_matrix = cluster_matrix(
                base_result_name,
                self.node_IP_list,
                self.CPU_GPU_select_list,
                self.node_percentages,
                self.back_end_select_list,
                True
            )
            return result_cluster_matrix
        return False  # Return the distributed result instance

    def cluster_grid_operation(self, cluster_matrixB, TransposeA, TransposeB, send_back_result=False, operation='mul'):
        """
        Perform a distributed matrix operation across the cluster using the full grid GEMM pattern.
        Each row shard of Matrix A multiplies with each column shard of Matrix B.

        Args:
            cluster_matrixB: Another cluster_matrix instance for the second operand
            TransposeA: Whether to transpose matrix A
            TransposeB: Whether to transpose matrix B  
            send_back_result: Whether to combine results into single file (True) 
                            or keep distributed (False)
            operation: Operation to perform ('mul', 'add', 'sub')

        Returns:
            Base name of the result file(s)
        """
        print(f"\n{'='*60}")
        print(f"🚀 STARTING CLUSTER GRID OPERATION")
        print(f"{'='*60}")
        print(f"Matrix A: {self.matrix_name}")
        print(f"Matrix B: {cluster_matrixB.matrix_name}")
        print(f"Operation: {operation}")
        print(f"Transpose A: {TransposeA}, Transpose B: {TransposeB}")
        print(f"Send back result: {send_back_result}")
        print(f"A shards: {len(self.matrix_file_paths_list)}, B shards: {len(cluster_matrixB.matrix_file_paths_list)}")
        start_time = time.time()

        node_gpu_counters = {}

        total_ops = len(self.matrix_file_paths_list)  # A row shards
        print(f"\n📤 DISTRIBUTING GRID OPERATIONS TO NODES ({total_ops} ops)")

        for shard_index, (node_IP, CPU_GPU_select, back_end_select, matrix_a) in enumerate(zip(
            self.node_IP_list,
            self.CPU_GPU_select_list,
            self.back_end_select_list,
            self.matrix_file_paths_list
        )):
            # Initialize GPU counter
            if node_IP not in node_gpu_counters:
                node_gpu_counters[node_IP] = 0
            current_gpu_number = node_gpu_counters[node_IP]

            # Map A shard to correct B shard (broadcast across grid)
            b_shard_index = shard_index % len(cluster_matrixB.matrix_file_paths_list)
            matrix_b = cluster_matrixB.matrix_file_paths_list[b_shard_index]

            print(f"\nProcessing operation {shard_index}:")
            print(f"  Node: {node_IP}")
            print(f"  Backend: {back_end_select}")
            print(f"  Use GPU: {CPU_GPU_select} (GPU #{current_gpu_number})")
            print(f"  Matrix A path: {matrix_a}")
            print(f"  Matrix B path: {matrix_b}")

            # Backend-specific transpose handling
            local_TransposeA = TransposeA
            local_TransposeB = TransposeB
            if back_end_select == 'llama':
                local_TransposeB = not TransposeB

            TransposeA_str = str(local_TransposeA).lower()
            TransposeB_str = str(local_TransposeB).lower()
            use_gpu_str = str(CPU_GPU_select).lower()

            # Send-back flag
            send_back_str = str(len(self.node_IP_list)) if send_back_result else "0"
            print(f"  Final transpose flags - A: {TransposeA_str}, B: {TransposeB_str}")
            print(f"  Send back result: {send_back_result} ({send_back_str})")

            # Build command
            command = (
                f"server_command={back_end_select} "
                f"{matrix_a} "
                f"{TransposeA_str} "
                f"{matrix_b} "
                f"{TransposeB_str} "
                f"{use_gpu_str} "
                f"{current_gpu_number} "
                f"{send_back_str} "
                f"{operation} "
                f"2 "
                f"{shard_index}"
            )

            # Send command
            socket_eth = self.llama_socket_pool[node_IP]
            socket_eth.send_multipart([command.encode()])
            print(f"  ✅ Command sent to node {node_IP}")

            if CPU_GPU_select:
                node_gpu_counters[node_IP] += 1
                print(f"  Incremented GPU counter for node {node_IP} to {node_gpu_counters[node_IP]}")

        # Wait for all ACKs
        expected_acks = len(self.matrix_file_paths_list)
        print(f"\n⏳ WAITING FOR ACKS FROM NODES ({expected_acks})")
        self.wait_for_acks(expected_acks, "ACK_matrixOp_complete")

        print(f"\n{'='*60}")
        print(f"✅ CLUSTER GRID OPERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Operation time: {time.time() - start_time:.2f} seconds")

        # Prepare result cluster
        base_result_name = f"{cluster_matrixB.matrix_name}x{self.matrix_name}" if 'llama' in self.back_end_select_list else f"{self.matrix_name}x{cluster_matrixB.matrix_name}"

        if send_back_result:
            self.wait_for_acks(1, "ACK_combined_matrix_saved")
            path = self.local_RAM_folder + base_result_name + '_combined.bin'
            combined_matrix = convert_bin_matrix_to_pt(path)
            os.remove(path)
            return combined_matrix
        else:
            result_cluster_matrix = cluster_matrix(
                base_result_name,
                self.node_IP_list,
                self.CPU_GPU_select_list,
                self.node_percentages,
                self.back_end_select_list,
                True
            )
            return result_cluster_matrix

#######################################------MAIN FUNCTION - TESTING-----######################################  
if __name__ == "__main__":
    
    # ----------------- CREATE MATRICES -----------------
    '''
    A = np.random.rand(10000, 20000)
    B = np.random.rand(15000, 20000)

    A2 = np.random.rand(5000, 7000)
    B2 = np.random.rand(9000, 7000)

    A3 = np.random.rand(1500, 500)
    B3 = np.random.rand(1000, 500)

    torch.save(torch.tensor(A, dtype=torch.float32), 'model_matrixs/big_matrixA.pt')
    torch.save(torch.tensor(B, dtype=torch.float32), 'model_matrixs/big_matrixB.pt')   
    
    torch.save(torch.tensor(A2, dtype=torch.float32), 'model_matrixs/mid_matrixA.pt')
    torch.save(torch.tensor(B2, dtype=torch.float32), 'model_matrixs/mid_matrixB.pt')   

    torch.save(torch.tensor(A3, dtype=torch.float32), 'model_matrixs/small_matrixA.pt')
    torch.save(torch.tensor(B3, dtype=torch.float32), 'model_matrixs/small_matrixB.pt')
    '''  

    # ----------------- FILE PATHS -----------------
    big_test_matrix_pathA = 'model_matrixs/big_matrixA.pt'  
    big_test_matrix_pathB = 'model_matrixs/big_matrixB.pt'  

    mid_test_matrix_pathA = 'model_matrixs/mid_matrixA.pt'  
    mid_test_matrix_pathB = 'model_matrixs/mid_matrixB.pt'  

    small_test_matrix_pathA = 'model_matrixs/small_matrixA.pt'  
    small_test_matrix_pathB = 'model_matrixs/small_matrixB.pt'  

    
    # ----------------- REFERENCE RESULTS -----------------
    '''
    big_a = torch.load(big_test_matrix_pathA)
    big_b = torch.load(big_test_matrix_pathB)
    big_c_ref = big_a @ big_b.T  # A @ B.T
    torch.save(big_c_ref, 'model_matrixs/big_c_ref.pt')

    mid_a = torch.load(mid_test_matrix_pathA)
    mid_b = torch.load(mid_test_matrix_pathB)
    mid_c_ref = mid_a @ mid_b.T
    torch.save(mid_c_ref, 'model_matrixs/mid_c_ref.pt')

    small_a = torch.load(small_test_matrix_pathA)
    small_b = torch.load(small_test_matrix_pathB)
    small_c_ref = small_a @ small_b.T
    torch.save(small_c_ref, 'model_matrixs/small_c_ref.pt')
    '''
 
    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
'''
    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
    percentages = [0.4,0.4,0.1,0.1]  
    CPU_GPU_select_list = [ True, True, True, True ]  
    backend_select_list = ['llama','llama','llama','llama'] 

    big_new_matrixA = cluster_matrix(big_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0)
    big_new_matrixA.convert_to_cluster_matrix_shards()
    big_new_matrixA.save_distribute_matrix_shards_bin()
    big_new_matrixB = cluster_matrix(big_test_matrix_pathB, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, False) 
    big_new_matrixB.save_distribute_full_matrix_bin()
    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/big_c_ref.pt',big_new_matrixC)
    
    load_matrixA = cluster_matrix('big_matrixA', IP_list, CPU_GPU_select_list, 
                            percentages, backend_select_list)
    load_matrixA.load_cluster_matrix_shards()
    load_matrixB = cluster_matrix('big_matrixB', IP_list, CPU_GPU_select_list, 
                            percentages, backend_select_list) 
    load_matrixB.load_cluster_matrix()
    load_matrixD = load_matrixA.cluster_shard_operation(load_matrixB, False, True, True) 
    check_combined_result_values('model_matrixs/big_c_ref.pt',load_matrixD)

    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    mid_new_matrixA = cluster_matrix(mid_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0)
    mid_new_matrixA.convert_to_cluster_matrix_shards()
    mid_new_matrixA.save_distribute_matrix_shards_bin()
    mid_new_matrixB = cluster_matrix(mid_test_matrix_pathB, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, False) 
    mid_new_matrixB.save_distribute_full_matrix_bin()
    mid_new_matrixC = mid_new_matrixA.cluster_shard_operation(mid_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/mid_c_ref.pt',mid_new_matrixC)
    
    load_mid_matrixA = cluster_matrix('mid_matrixA', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list)
    load_mid_matrixA.load_cluster_matrix_shards()
    load_mid_matrixB = cluster_matrix('mid_matrixB', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list) 
    load_mid_matrixB.load_cluster_matrix()
    load_mid_matrixD = load_mid_matrixA.cluster_shard_operation(load_mid_matrixB, False, True, True) 
    check_combined_result_values('model_matrixs/mid_c_ref.pt',load_mid_matrixD)

    # ----------------- 5 NODE TEST -----------------
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.101','192.168.2.104']   
    percentages = [0.4,0.4,0.05,0.05,0.1]  
    CPU_GPU_select_list = [ True, True, True, True, True ]  
    backend_select_list = ['llama','llama','llama','llama', 'llama'] 

    node5_big_matrixA = cluster_matrix(big_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0)
    node5_big_matrixA.convert_to_cluster_matrix_shards()
    node5_big_matrixA.save_distribute_matrix_shards_bin()
    node5_big_matrixB = cluster_matrix(big_test_matrix_pathB, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list) 
    node5_big_matrixB.save_distribute_full_matrix_bin()
    node5_big_matrixC = node5_big_matrixA.cluster_shard_operation(node5_big_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/big_c_ref.pt',node5_big_matrixC)

    node5_big_mid_matrixA = cluster_matrix('big_matrixA', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list)    
    node5_big_mid_matrixA.load_cluster_matrix_shards()
    node5_big_mid_matrixB = cluster_matrix('big_matrixB', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list) 
    node5_big_mid_matrixB.load_cluster_matrix()
    node5_big_mid_matrixD = node5_big_mid_matrixA.cluster_shard_operation(node5_big_mid_matrixB, False, True, True) 
    check_combined_result_values('model_matrixs/big_c_ref.pt',node5_big_mid_matrixD)

    node5_mid_matrixA = cluster_matrix(mid_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0)
    node5_mid_matrixA.convert_to_cluster_matrix_shards()
    node5_mid_matrixA.save_distribute_matrix_shards_bin()
    node5_mid_matrixB = cluster_matrix(mid_test_matrix_pathB, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list) 
    node5_mid_matrixB.save_distribute_full_matrix_bin()
    node5_mid_matrixC = node5_mid_matrixA.cluster_shard_operation(node5_mid_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/mid_c_ref.pt',node5_mid_matrixC)

    node5_big_mid_matrixA = cluster_matrix('mid_matrixA', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list)    
    node5_big_mid_matrixA.load_cluster_matrix_shards()
    node5_big_mid_matrixB = cluster_matrix('mid_matrixB', IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list) 
    node5_big_mid_matrixB.load_cluster_matrix()
    node5_big_mid_matrixD = node5_big_mid_matrixA.cluster_shard_operation(node5_big_mid_matrixB, False, True, True) 
    check_combined_result_values('model_matrixs/mid_c_ref.pt',node5_big_mid_matrixD)
  
'''

############################# TESTING CLUSTER MATRIX OPERATIONS SYSTEM 2 #############################




IP_list = [
    '192.168.2.100','192.168.2.100','192.168.2.100',
    '192.168.2.101','192.168.2.101','192.168.2.104'
]

percentages = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
CPU_GPU_select_list = [True, True, True, True, True, True]
# You already have this variable defined:
backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama'] 

# Use it instead of empty list:
matrix_A = cluster_matrix(
    matrix_file_path=big_test_matrix_pathA,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,  # Use the actual variable!
    split_matrix=True,
    dim=2
)
test = matrix_A.convert_to_cluster_matrix_grid()
matrix_A.save_distribute_matrix_grid_bin()

matrix_B = cluster_matrix(
    matrix_file_path=big_test_matrix_pathB,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,  # Use the actual variable!
    split_matrix=True,
    dim=3
)
matrix_B.convert_to_cluster_matrix_grid()
matrix_B.save_distribute_matrix_grid_bin()

# With our corrected system where B is already transposed to Bᵀ:
result = matrix_A.cluster_grid_operation(matrix_B,False,True,False)

# This computes: A × Bᵀ where B is already Bᵀ from convert_to_cluster_matrix_grid()
# No additional transpose needed!

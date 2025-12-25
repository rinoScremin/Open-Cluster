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
                split_matrix=False, dim=0):
        
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

    def _clear_previous_results(self, base_result_name):
        """
        Remove stale result artifacts for a given operation base name.
        Keeps input shards intact.
        """
        patterns = [
            os.path.join(self.local_RAM_folder, f"{base_result_name}_combined.bin"),
            os.path.join(self.local_RAM_folder, f"{base_result_name}_shard_*.bin"),
            os.path.join(self.local_matrix_results_RAM_folder, f"{base_result_name}_*"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                    print(f"🧹 Removed stale result: {path}")
                except IsADirectoryError:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception as exc:
                    print(f"⚠️  Could not remove {path}: {exc}")

    def _validate_shard_counts_and_shapes(self, other, transposeA, transposeB, operation="mul"):
        """
        Ensure shard counts match and inner dimensions are compatible
        for the requested operation. Only validates shards reachable
        from this controller.
        """
        if len(self.matrix_file_paths_list) != len(other.matrix_file_paths_list):
            raise ValueError(
                f"Shard count mismatch: {len(self.matrix_file_paths_list)} (A) vs "
                f"{len(other.matrix_file_paths_list)} (B)"
            )

        for idx, (path_a, path_b) in enumerate(
            zip(self.matrix_file_paths_list, other.matrix_file_paths_list)
        ):
            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                print(
                    f"ℹ️  Skipping shape check for shard {idx}: "
                    f"{'missing A' if not os.path.exists(path_a) else 'missing B'} locally"
                )
                continue
            try:
                rows_a, cols_a = _infer_rows_cols_from_bin(path_a)
                rows_b, cols_b = _infer_rows_cols_from_bin(path_b)
                if operation in ("add", "sub"):
                    if rows_a != rows_b or cols_a != cols_b:
                        raise ValueError(
                            f"Shard {idx} shape mismatch for {operation}: A {rows_a}x{cols_a} vs B {rows_b}x{cols_b}"
                        )
                else:  # mul or other matmul-like
                    inner_a = cols_a if not transposeA else rows_a
                    inner_b = rows_b if not transposeB else cols_b
                    if inner_a != inner_b:
                        raise ValueError(
                            f"Shard {idx} inner dimension mismatch: {inner_a} (A) vs {inner_b} (B)"
                        )
            except Exception as exc:
                print(f"⚠️  Could not validate shard {idx}: {exc}")
        print("✅ Shard count validation complete")

    def wait_for_acks(self, expected_count):
        """
        Wait for ACKs from all expected nodes on the Python front end cluster port.
        """
        acks = 0
        while acks < expected_count:
            try:
                msg = self.ack_receiver_socket.recv_string(flags=zmq.NOBLOCK)
                if msg == "ACK":
                    acks += 1
                    print(f"✅ Received ACK {acks}/{expected_count}")
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

    def convert_to_cluster_matrix_shards(self, number_of_shards=100):
        print("=" * 70)
        print("🔪 CONVERTING MATRIX TO DISTRIBUTED SHARDS")
        print("=" * 70)
        
        # =============== LOAD ORIGINAL MATRIX ===============
        print("\n📦 LOADING ORIGINAL MATRIX...")
        print(f"   File path: {self.matrix_file_path}")
        
        try:
            matrix = torch.load(self.matrix_file_path)
            print(f"✅ Successfully loaded matrix")
            print(f"   - Shape: {matrix.shape}")
            print(f"   - Data type: {matrix.dtype}")
            print(f"   - Device: {matrix.device if hasattr(matrix, 'device') else 'CPU'}")
        except Exception as e:
            print(f"❌ FAILED to load matrix: {e}")
            raise
        
        # Store original matrix shape for reference
        self.OG_matrix_shape = list(matrix.shape)
        print(f"   - Original matrix shape stored: {self.OG_matrix_shape}")
        
        # =============== VALIDATE AND ADJUST SHARD COUNT ===============
        print("\n⚙️  CONFIGURING SHARD PARAMETERS...")
        print(f"   Requested shards: {number_of_shards}")
        print(f"   Split dimension (dim={self.dim}): size = {matrix.shape[self.dim]}")
        
        matrix_dim_size = matrix.shape[self.dim]
        
        # Adjust number_of_shards if matrix dimension is smaller than requested shards
        if matrix_dim_size < number_of_shards:
            print(f"   ⚠️  Matrix dimension ({matrix_dim_size}) is smaller than requested shards ({number_of_shards})")
            number_of_shards = matrix_dim_size
            print(f"   ✅ Adjusted to {number_of_shards} shards (1 shard per matrix element in dimension {self.dim})")
        else:
            print(f"   ✅ Matrix size supports {number_of_shards} shards")
        
        print(f"   Final shard count: {number_of_shards}")
        
        # =============== SPLIT MATRIX INTO INITIAL SHARDS ===============
        print("\n🔪 SPLITTING MATRIX INTO INITIAL SHARDS...")
        print(f"   Creating exact shard sizes along dim={self.dim} to cover all rows/cols")
        
        try:
            base = matrix_dim_size // number_of_shards
            remainder = matrix_dim_size % number_of_shards
            chunk_sizes = [base + 1 if i < remainder else base for i in range(number_of_shards)]
            matrix_shards = list(torch.split(matrix, chunk_sizes, dim=self.dim))
            print(f"✅ Successfully split matrix")
            print(f"   - Created {len(matrix_shards)} shards")
            print(f"   - Each shard shape: {matrix_shards[0].shape if matrix_shards else 'N/A'}")
            
            # Verify shard distribution
            shard_sizes = [shard.shape[self.dim] for shard in matrix_shards]
            print(f"   - Shard sizes along dim {self.dim}: {shard_sizes[:5]}... (first 5)")
            print(f"   - Total elements in shards: {sum(shard_sizes)} (should equal {matrix_dim_size})")
            
        except Exception as e:
            print(f"❌ FAILED to split matrix: {e}")
            raise
        
        # =============== DISTRIBUTE SHARDS TO NODES ===============
        print("\n🌐 DISTRIBUTING SHARDS TO CLUSTER NODES...")
        print(f"   Number of nodes: {len(self.node_percentages)}")
        print(f"   Node percentages: {[f'{p*100:.1f}%' for p in self.node_percentages]}")
        print(f"   Total shards available: {len(matrix_shards)}")
        
        start_merged_index = 0
        self.node_matrices = []
        total_shards = len(matrix_shards)
        
        print(f"\n   {'Node':<6} {'Shards':<8} {'Percentage':<12} {'Shape':<20} {'Start':<8} {'End':<8}")
        print("   " + "-" * 62)
        
        node_count = len(self.node_percentages)
        base_shards = total_shards // node_count if node_count > 0 else 0
        remainder = total_shards % node_count if node_count > 0 else 0
        shards_per_node = [base_shards + (1 if i < remainder else 0) for i in range(node_count)]
        
        for i, node_percentage in enumerate(self.node_percentages):
            shards_to_take = shards_per_node[i]
            end_index = start_merged_index + shards_to_take
            
            # Merge the shards for this node
            if shards_to_take > 0 and start_merged_index < end_index:
                merged_shard = self.merged_matrix(matrix_shards, start_merged_index, end_index)
                self.node_matrices.append(merged_shard)
                
                print(f"   {i:<6} {shards_to_take:<8} {node_percentage*100:>6.1f}%    {str(merged_shard.shape):<20} {start_merged_index:<8} {end_index-1:<8}")
                
                # Update starting index for next node
                start_merged_index = end_index
            else:
                empty_shape = list(matrix.shape)
                empty_shape[self.dim] = 0
                empty_tensor = torch.tensor([], dtype=matrix.dtype).reshape(empty_shape)
                self.node_matrices.append(empty_tensor)
                print(f"   {i:<6} {shards_to_take:<8} {node_percentage*100:>6.1f}%    {str(empty_tensor.shape):<20} {start_merged_index:<8} {start_merged_index-1:<8}")
        
        if start_merged_index < total_shards:
            print(f"\n   ⚠️  Remaining shards not allocated, assigning to last node")
            tail = self.merged_matrix(matrix_shards, start_merged_index, total_shards)
            self.node_matrices[-1] = torch.cat([self.node_matrices[-1], tail], dim=self.dim) if self.node_matrices else tail
            start_merged_index = total_shards
        
        print(f"\n   📊 All {total_shards} shards have been allocated")
        
        # =============== VERIFICATION ===============
        print("\n📊 DISTRIBUTION VERIFICATION:")
        
        # Calculate total elements from distributed matrices
        total_elements_after_dist = sum([node_mat.shape[self.dim] for node_mat in self.node_matrices])
        
        print(f"   Original matrix size (dim {self.dim}): {matrix_dim_size}")
        print(f"   Total after distribution (dim {self.dim}): {total_elements_after_dist}")
        
        if total_elements_after_dist == matrix_dim_size:
            print("   ✅ SUCCESS: All elements accounted for!")
        else:
            print(f"   ⚠️  WARNING: Element mismatch ({matrix_dim_size} vs {total_elements_after_dist})")
        
        # Show distribution summary
        print(f"\n   Distribution summary:")
        for i, node_mat in enumerate(self.node_matrices):
            percentage = (node_mat.shape[self.dim] / matrix_dim_size * 100) if matrix_dim_size > 0 else 0
            print(f"   Node {i}: {node_mat.shape} ({percentage:.1f}% of total)")
        
        print("\n" + "=" * 70)
        print("✅ MATRIX SHARD CONVERSION COMPLETE")
        print("=" * 70)
        
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
 
    def cluster_operation(self, cluster_matrixB, TransposeA, TransposeB, send_back_result=False, operation='mul'):
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
        
        # ===== SETUP RESULT FILENAMES =====
        # Result names match the operand order we send to the server:
        # self (matrix A) first, then cluster_matrixB (matrix B)
        base_result_name = f"{self.matrix_name}x{cluster_matrixB.matrix_name}"
        print(f"\n📊 Result base: {base_result_name} (send_back={send_back_result})")

        # ===== LOCAL SIMULATION SHORT-CIRCUIT FOR SELF-TESTS =====
        if getattr(self, "_local_simulate", False) or getattr(cluster_matrixB, "_local_simulate", False) or os.environ.get("CLUSTER_MATRIX_LOCAL_SIM", "0") == "1":
            return self._simulate_cluster_operation(
                base_result_name,
                cluster_matrixB,
                TransposeA,
                TransposeB,
                send_back_result,
                operation,
            )

        # ===== VALIDATION & CLEANUP =====
        self._validate_shard_counts_and_shapes(cluster_matrixB, TransposeA, TransposeB, operation)
        self._clear_previous_results(base_result_name)
        
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
            if back_end_select == 'llama':
                command = (
                    f"server_command={back_end_select} "
                    f"{matrix_b} "          # GGML expects B first
                    f"{TransposeB_str} "
                    f"{matrix_a} "          # Then A
                    f"{TransposeA_str} "
                    f"{use_gpu_str} "
                    f"{current_gpu_number} "
                    f"{send_back_str} "
                    f"{operation} "
                    f"2 "
                    f"{shard_index}"
                )
            elif back_end_select == 'torch':
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
            else:
                raise ValueError(f"Unsupported backend selected: {back_end_select}")
            
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
        self.wait_for_acks(expected_acks)
        
        # ===== OPERATION COMPLETE =====
        print(f"\n{'='*60}")
        print(f"✅ CLUSTER OPERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Result base name: {base_result_name}")
        print(f"Operation time: {time.time() - start_time:.2f} seconds")

        # When keep-distributed, return a cluster_matrix wired to the shard outputs
        result_cluster_matrix = cluster_matrix(
            base_result_name,
            self.node_IP_list,
            self.CPU_GPU_select_list,
            self.node_percentages,
            self.back_end_select_list,
            True
        )
        return result_cluster_matrix  # Return the distributed result instance

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

    def parallel_interface_file_transfer(self, filename, target_node_ip, split_fraction=0.5):
        """
        Transfer file using both Ethernet and WiFi interfaces in parallel.
        Sends exactly 2 parts per message (matches C++ receiver).
        
        ⚠️  EXPERIMENTAL FEATURE - STILL UNDER DEVELOPMENT ⚠️
        This feature splits file transfer across multiple network interfaces
        for potentially faster transfers, but is not fully tested.
        
        Args:
            filename: Path to file to transfer
            target_node_ip: Target node's Ethernet IP
        Args:
            split_fraction: Fraction of bytes to send over Ethernet (rest via WiFi).
        Returns:
            bool: True if transfer successful, False otherwise
        """
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return False
        
        # Get Ethernet socket for the target node
        if target_node_ip not in self.llama_socket_pool:
            print(f"Error: No Ethernet socket for {target_node_ip}")
            return False
        socket_eth = self.llama_socket_pool[target_node_ip]
        
        # Try to find corresponding WiFi socket
        socket_wifi = None
        
        # Look up WiFi IP based on Ethernet IP mapping
        unique_node_ips = list(dict.fromkeys(self.node_IP_list))
        try:
            node_index = unique_node_ips.index(target_node_ip)
            if node_index < len(self.IP_list_wifi):
                wifi_ip = self.IP_list_wifi[node_index]
                if wifi_ip in self.llama_socket_pool_wifi:
                    socket_wifi = self.llama_socket_pool_wifi[wifi_ip]
        except (ValueError, IndexError):
            print(f"Note: Could not find WiFi IP for {target_node_ip}")
        
        # Fallback to Ethernet-only if no WiFi socket
        if socket_wifi is None:
            print(f"Warning: No WiFi socket, using Ethernet only")
            return self.zmq_send_file(target_node_ip, filename)
        
        # Get file information
        file_size = os.path.getsize(filename)
        base_name = os.path.basename(filename)
        
        # Split file into two chunks based on requested ratio
        split_fraction = max(0.05, min(0.95, split_fraction))
        chunk_size = int(file_size * split_fraction)
        
        print(f"Starting parallel transfer: {filename}")
        print(f"File size: {file_size:,} bytes")
        print(f"Split size: ETH={chunk_size:,} bytes, WiFi={file_size - chunk_size:,} bytes")
        
        # Send file in 2 chunks (Ethernet and WiFi)
        with open(filename, 'rb') as f:
            # CHUNK 1: Send via Ethernet
            eth_chunk = f.read(chunk_size)
            if eth_chunk:
                eth_message = [
                    f"P_SEND_{base_name}".encode(),  # Filename with prefix
                    eth_chunk                         # Data chunk
                ]
                socket_eth.send_multipart(eth_message)
                print(f"ETH chunk sent: {len(eth_chunk):,} bytes")
            
            # CHUNK 2: Send via WiFi  
            wifi_chunk = f.read(chunk_size)
            if wifi_chunk:
                wifi_message = [
                    f"P_SEND_{base_name}".encode(),  # Same filename
                    wifi_chunk                       # Data chunk
                ]
                socket_wifi.send_multipart(wifi_message)
                print(f"WiFi chunk sent: {len(wifi_chunk):,} bytes")
            
            # Send any remaining data via Ethernet (for odd-sized files)
            remaining = f.read(chunk_size)
            if remaining:
                socket_eth.send_multipart([f"P_SEND_{base_name}".encode(), remaining])
                print(f"Extra ETH chunk sent: {len(remaining):,} bytes")
        
        # Calculate total bytes sent
        total_sent = len(eth_chunk if eth_chunk else b'') + len(wifi_chunk if wifi_chunk else b'') + len(remaining if remaining else b'')
        
        print(f"Parallel transfer complete: {total_sent:,} bytes sent")
        print(f"Note: This is an experimental feature")
        
        return True


#######################################------MAIN FUNCTION - TESTING-----######################################  
# Node configuration for the cluster
#IP_list = ['192.168.2.100', '192.168.2.100', '192.168.2.100', '192.168.2.102', '192.168.2.103']  
#percentages = [0.45, 0.35, 0.10, 0.05, 0.05]  
#CPU_GPU_select_list = [True, True, False, True, True]  
#backend_select_list = ['llama', 'llama', 'torch', 'llama', 'llama']  


#test_matrix = torch.rand((20000, 20000), device="cpu")

#torch.save(test_matrix, '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/big_matrix.pt')

# Node configuration for the cluster
IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
percentages = [0.25,0.25,0.25,0.25]  
CPU_GPU_select_list = [True, True, True, True]  
backend_select_list = ['llama','llama','llama','llama']  


# Test matrix file paths
test_matrix_path = '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/big_matrix.pt'  
test_matrix_path2 = '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/big_matrix.pt'  

test_matrix_load = 'big_matrix'
test_matrix_load2 = 'big_matrix'


print("\n" + "="*60)
print("🚀 CLUSTER MATRIX DISTRIBUTION SYSTEM TEST")
print("="*60)

# --- Create and save matrix A (split mode) ---  
print("\n📦 Creating and distributing matrix A (split=True)...")  
matrixA = cluster_matrix(test_matrix_path, IP_list, CPU_GPU_select_list, 
                        percentages, backend_select_list, True)  

# --- Create and save matrix B (no split mode) ---  
print("\n📦 Creating and distributing matrix B (split=False)...")  
matrixB = cluster_matrix(test_matrix_path2, IP_list, CPU_GPU_select_list, 
                        percentages, backend_select_list, False)  

# --- Perform cluster matrix multiplication ---  
print("\n" + "="*60)
print("🧮 PERFORMING CLUSTER MATRIX OPERATION")
print("="*60)
print("Operation: MatrixA @ MatrixB.T")
print(f"Nodes: {len(IP_list)}")
print(f"Backends: {backend_select_list}")
print(f"GPU usage: {CPU_GPU_select_list}")

cluster_start_time = time.time()
cluster_matrixC = matrixA.cluster_operation(matrixB, False, True, True)  
cluster_end_time = time.time()

print(f"\n✅ Cluster operation completed")
print(f"Result name: {cluster_matrixC.matrix_name}")
print(f"Cluster operation time: {cluster_end_time - cluster_start_time:.2f} seconds")

# ======================================================
# PYTORCH REFERENCE CALCULATIONS (SINGLE NODE)
# ======================================================
print("\n" + "="*60)
print("🔍 PYTORCH REFERENCE (SINGLE NODE)")
print("="*60)

# Load the original PyTorch matrices
a = torch.load(test_matrix_path)
b = torch.load(test_matrix_path2)

print(f"\nMatrix A shape: {a.shape}")
print(f"Matrix B shape: {b.shape}")
print(f"Matrix A sample (5x5):")
print(a[:5, :5])

# Perform reference multiplication on single node
pytorch_start_time = time.time()
c_ref = a @ b.T  # A @ B.T (TransposeB=True)
pytorch_end_time = time.time()

print(f"\nReference result shape: {c_ref.shape}")
print(f"Single-node PyTorch computation time: {pytorch_end_time - pytorch_start_time:.2f}s")
print(f"First 2500 elements of reference result:")
print(c_ref.flatten()[:2500])
print(f"\nReference result sample (5x5):")
print(c_ref[:5, :5])

# ======================================================
# LOAD CLUSTER RESULT
# ======================================================
print("\n" + "="*60)
print("📥 LOADING CLUSTER RESULT")
print("="*60)

# Load the combined result from cluster computation
#combined_result_path = "/dev/shm/matrix_shards/big_matrixxbig_matrix_combined.bin"
#print(f"Loading cluster result from: {combined_result_path}")

#combined = convert_bin_matrix_to_pt(combined_result_path)
#print(f"Cluster result shape: {combined.shape}")
#print(f"Cluster result sample (5x5):")
#print(combined[:5, :5])

# ======================================================
# PERFORMANCE COMPARISON
# ======================================================
print("\n" + "="*60)
print("🏁 PERFORMANCE COMPARISON")
print("="*60)

cluster_time = cluster_end_time - cluster_start_time
pytorch_time = pytorch_end_time - pytorch_start_time

print(f"CLUSTER OPERATION TIME:      {cluster_time:.4f} seconds")
print(f"SINGLE NODE PYTORCH TIME:    {pytorch_time:.4f} seconds")
print("-" * 60)

# Calculate speedup/difference
if pytorch_time > 0:
    speedup = pytorch_time / cluster_time
    print(f"CLUSTER vs SINGLE NODE: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

'''
# Check if shapes match
if c_ref.shape != combined.shape:
    print(f"❌ Shape mismatch! Reference: {c_ref.shape}, Combined: {combined.shape}")
else:
    print(f"✅ Shapes match: {c_ref.shape}")
    
    # Calculate differences
    diff = torch.abs(c_ref - combined)
    
    # Basic statistics
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
        
    # Looser tolerance for shard/float errors
    tolerance = 0.15
    if torch.allclose(c_ref, combined, rtol=tolerance, atol=tolerance):
        print(f"✅ Results match within tolerance ({tolerance})")
    else:
        print(f"⚠️  Results differ beyond tolerance ({tolerance})")
        
        # Count elements with significant differences
        significant_diff = diff > tolerance
        num_different = torch.sum(significant_diff).item()
        total_elements = c_ref.numel()
        print(f"Elements with > {tolerance} difference: {num_different}/{total_elements} "
            f"({(num_different/total_elements*100):.2f}%)")

# =========================
# Visual debug: first and second half
# =========================
def print_matrix_sections(matrix, name, n_elements=100):
    flat = matrix.flatten()
    mid = flat.shape[0] // 2
    print(f"\n{name} - first {n_elements} elements of first half:")
    print(flat[:n_elements])
    print(f"\n{name} - first {n_elements} elements of second half:")
    print(flat[mid:mid + n_elements])

print_matrix_sections(c_ref, "Torch reference")
# print_matrix_sections(combined, "Combined cluster result")
'''

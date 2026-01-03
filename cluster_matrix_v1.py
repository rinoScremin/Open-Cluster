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
                split_matrix=False, dim=0, matrix_labeling='' ,hierarchical_split_order=[]):
        
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
        half_node_percentages = len(node_percentages) // 2
        sys2_split_percentages = node_percentages[:half_node_percentages]
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
        self.remote_project_dir = os.environ.get('REMOTE_PROJECT_DIR', '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/')
        
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
        self.sys2_split_percentages = sys2_split_percentages
        self.matrix_labeling= matrix_labeling
        self.split_depth = len(hierarchical_split_order)
        self.hierarchical_split_order = hierarchical_split_order

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
        
        command = f'mkdir -p {self.remote_project_dir}{self.remote_DISK_folder} {self.remote_RAM_folder} {self.remote_matrix_results_RAM_folder}'
        print(f"   Sending command: {command}")
        
        for node_ip, socket in self.llama_socket_pool.items():
            try:
                socket.send_multipart([command.encode('utf-8')])
                print(f"   ✅ Directory creation command sent to {node_ip}")
            except Exception as e:
                print(f"   ❌ Failed to send command to {node_ip}: {e}")

        '''
        # =============== MATRIX DISTRIBUTION LOGIC ===============
        print("\n" + "=" * 70)
        print("🧮 MATRIX DISTRIBUTION PHASE")
        print("=" * 70)

        # Initialize with a default value
        matrix_shards_found = True  # Default to True
        matrix_exists = os.path.exists(matrix_file_path)

        # Check for shard file (this is what was working!)
        matrix_shard_file_path = self.local_project_dir + self.local_DISK_folder + self.matrix_name + '_shard_0.bin'
        matrix_shard_exists = os.path.exists(matrix_shard_file_path)

        print(f"   Matrix file exists: {matrix_exists}")
        print(f"   Split matrix mode: {split_matrix}")

        # Decision tree for matrix handling
        if matrix_exists and self.matrix_labeling == '' and split_matrix:
            print("NEW MATRIX SYSTEM 1 SPLIT")
            self.convert_to_cluster_matrix_shards()
            self.save_distribute_matrix_shards_bin()
            matrix_shards_found = True

        elif matrix_exists and self.matrix_labeling == '' and split_matrix == False:
            print("NEW MATRIX SYSTEM 1 FULL")
            self.save_distribute_full_matrix_bin()
            matrix_shards_found = True

        elif matrix_exists and self.matrix_labeling == 'a':
            print("NEW MATRIX A - SYSTEM 2 GRID")
            self.convert_to_cluster_matrix_grid()
            self.save_distribute_matrixA_grid_bin()
            matrix_shards_found = True

        elif matrix_exists and self.matrix_labeling == 'b':
            print("NEW MATRIX B - SYSTEM 2 GRID")
            self.convert_to_cluster_matrix_grid()
            self.save_distribute_matrix_shards_bin()
            matrix_shards_found = True

        # Loading existing data - USE SHARD CHECK!
        elif matrix_shard_exists and self.matrix_labeling == '' and split_matrix:
            print("LOADING EXISTING SHARDS - SYSTEM 1")
            self.load_cluster_matrix_shards()
            matrix_shards_found = True

        elif matrix_exists == False and self.matrix_labeling == '' and split_matrix == False:
            # For full matrix, check if .bin exists
            full_matrix_path = self.local_project_dir + self.local_DISK_folder + self.matrix_name + '.bin'
            if os.path.exists(full_matrix_path):
                print("LOADING EXISTING FULL MATRIX")
                self.load_cluster_matrix()
                matrix_shards_found = True
            else:
                print(f"ERROR: Full matrix file not found: {full_matrix_path}")
                matrix_shards_found = False

        elif matrix_shard_exists and self.matrix_labeling == 'a':
            print("LOADING EXISTING MATRIX A GRID")
            self.load_cluster_matrixA_grid()
            matrix_shards_found = True

        elif matrix_shard_exists and self.matrix_labeling == 'b':
            print("LOADING EXISTING MATRIX B SHARDS")
            self.load_cluster_matrix_shards()
            matrix_shards_found = True

        elif matrix_shards_found == False:
            print('FILE NOT FOUND!! Checking both matrix and shard files...')
            print(f'  Matrix path: {matrix_file_path}')
            print(f'  Shard 0 path: {matrix_shard_file_path}')

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
        
    def send_ack_confirmation(self, ack_msg="ACK"):    
        """    
        Send ACK confirmation back to C++ backend    
        """    
        try:    
            # Create a separate socket for sending confirmations    
            if not hasattr(self, 'ack_confirmation_socket'):    
                self.ack_confirmation_socket = self.zmq_context.socket(zmq.PUSH)    
                # Use self.IP for the head node IP and define confirmation port  
                confirmation_port = os.environ.get("PYTHON_ACK_CONFIRMATION_PORT", "7791")  
                self.ack_confirmation_socket.connect(f"tcp://{self.IP}:{confirmation_port}")    
            
            # Send the confirmation message    
            self.ack_confirmation_socket.send_string(ack_msg)    
            print(f"✅ Sent confirmation: {ack_msg}")    
            
        except Exception as e:    
            print(f"❌ Failed to send confirmation: {e}")

    def wait_for_acks(self, expected_count, expected_msg="ACK", time_out=120):
        """
        Wait for ACKs from all expected nodes on the Python front end cluster port.
        
        Args:
            expected_count: Number of ACKs to wait for
            expected_msg: The expected message string (default: "ACK")
            time_out: Timeout in seconds (default: 120 seconds)
        
        Returns:
            Number of ACKs actually received (may be less than expected if timeout occurs)
        """
        acks = 0
        start_time = time.time()
        
        while acks < expected_count:
            # Check if timeout has been reached
            if time.time() - start_time > time_out:
                print(f"⏰ TIMEOUT: Only received {acks}/{expected_count} ACKs after {time_out} seconds")
                return acks
                
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

    def convert_to_hierarchical_matrix_shards(self):
        """
        Apply hierarchical splits to the matrix.
        For Matrix A: starts with full matrix (10000, 20000)
        For Matrix B: starts with initial split based on node_percentages
        """
        matrix_type = 'Matrix A' if self.matrix_labeling == 'a' else 'Matrix B'
        print(f"🚀 Creating hierarchical shards for {matrix_type}")
        print(f"   Split order: {self.hierarchical_split_order}")
        print(f"   Split depth: {self.split_depth}")
        
        # ============================================================
        # INITIALIZE MATRICES
        # ============================================================
        if self.matrix_labeling == 'a':  # Matrix A
            # Load full matrix
            full_matrix = torch.load(self.matrix_file_path)
            print(f"📥 Matrix A: Loaded full matrix {full_matrix.shape}")
            current_shards = [full_matrix]
            
        else:  # Matrix B  
            # Save init values
            saved_node_IP_list = self.node_IP_list
            
            # Set new init values for initial split
            # Use the first two percentages for the initial split
            if len(self.node_percentages) >= 2:
                initial_percentages = [self.node_percentages[0], self.node_percentages[1]]
            else:
                # Default to 50/50 if not enough percentages
                initial_percentages = [0.5, 0.5]
                
            self.node_percentages = initial_percentages
            self.node_IP_list = ['0.0.0.0', '0.0.0.0']
            
            # Get initial split (depth 0) based on self.dim
            self.convert_to_cluster_matrix_shards()
            
            # Restore original values
            self.node_IP_list = saved_node_IP_list

            current_shards = self.node_matrices
            
            # Print the shard shapes
            print(f"🔄 Matrix B: Initial split (dim={self.dim})")
            for i, shard in enumerate(current_shards):
                print(f"    B[{i}]: {shard.shape}")
        
        # ============================================================
        # APPLY HIERARCHICAL SPLITS BASED ON hierarchical_split_order
        # ============================================================
        print(f"\n📊 Starting hierarchical splits with: {len(current_shards)} shard(s)")
        for i, shard in enumerate(current_shards):
            matrix_id = 'A' if self.matrix_labeling == 'a' else 'B'
            print(f"    {matrix_id}[{i}]: {shard.shape}")
        
        depth_index = 0
        
        for split_dim in self.hierarchical_split_order:
            print(f"\n🔧 Depth level {depth_index + 1}/{self.split_depth} (split along dim={split_dim})")
            
            if split_dim == 0:  # Split by rows
                print("🔄 Splitting by rows (dim=0)")
                new_shards = []
                
                for shard in current_shards:
                    # Ensure 2D tensor
                    if shard.dim() == 1:
                        shard = shard.unsqueeze(0)
                    
                    # Check if we can split rows
                    if shard.size(0) > 1:
                        # Use torch.split() for cleaner code
                        split_size = shard.size(0) // 2
                        shard1, shard2 = torch.split(shard, split_size, dim=0)
                        new_shards.extend([shard1, shard2])
                    else:
                        # Can't split further
                        new_shards.append(shard)
            
            elif split_dim == 1:  # Split by columns
                print("🔄 Splitting by columns (dim=1)")
                new_shards = []
                
                for shard in current_shards:
                    # Ensure 2D tensor
                    if shard.dim() == 1:
                        shard = shard.unsqueeze(0)
                    
                    # Check if we can split columns
                    if shard.size(1) > 1:
                        # Use torch.split() for cleaner code
                        split_size = shard.size(1) // 2
                        shard1, shard2 = torch.split(shard, split_size, dim=1)
                        new_shards.extend([shard1, shard2])
                    else:
                        # Can't split further
                        new_shards.append(shard)
            else:
                print(f"⚠️  Warning: Invalid split dimension {split_dim}, skipping")
                new_shards = current_shards
            
            current_shards = new_shards
            depth_index += 1
            
            print(f"📈 After depth {depth_index}: {len(current_shards)} shards")
            matrix_id = 'A' if self.matrix_labeling == 'a' else 'B'
            if len(current_shards) <= 10:
                for i, shard in enumerate(current_shards):
                    print(f"    {matrix_id}[{i}]: {shard.shape}")
            else:
                print(f"    (Showing first 10 of {len(current_shards)} shards)")
                for i, shard in enumerate(current_shards[:10]):
                    print(f"    {matrix_id}[{i}]: {shard.shape}")
        
        # ============================================================
        # FINAL PROCESSING
        # ============================================================
        if self.matrix_labeling == 'a':  # Matrix A
            # Duplicate for round-robin pairing with B shards
            # Important: Create copies to avoid memory sharing
            duplicated_shards = [shard.clone() for shard in current_shards]
            self.node_matrices = current_shards + duplicated_shards
        else:  # Matrix B
            self.node_matrices = current_shards

        print(f"\n✅ Hierarchical shards created for {matrix_type}!")
        print(f"   Total shards: {len(self.node_matrices)}")
        
        # Also print individual shard shapes
        print("📊 Final shard shapes:")
        matrix_id = 'A' if self.matrix_labeling == 'a' else 'B'
        for i, shard in enumerate(self.node_matrices):
            print(f"    {matrix_id}[{i}]: {shard.shape}")
        
        return self.node_matrices

    def convert_to_cluster_matrix_grid(self):
        """
        Split matrix according to System 2 pattern using self.dim.
        Matrix A (label='a'): should use dim=1 (split by columns)
        Matrix B (label='b'): should use dim=0 (split by rows)
        """
        full_matrix = torch.load(self.matrix_file_path)
        self.OG_matrix_shape = list(full_matrix.shape)
        
        split_dim = self.dim  # Always use self.dim
        
        if self.matrix_labeling == 'a':  # Matrix A
            # For System 2: Matrix A should be split into 2 equal parts
            dim_size = full_matrix.size(split_dim)
            split_size = dim_size // 2
            
            # Use torch.split to get the 2 shards
            shards = torch.split(full_matrix, split_size, dim=split_dim)
            
            # torch.split returns a tuple, convert to list
            self.node_matrices = list(shards)
            
            print(f"✅ Matrix A: {full_matrix.shape} → [{shards[0].shape}, {shards[1].shape}] (split along dim={split_dim})")

        elif self.matrix_labeling == 'b':  # Matrix B for GEMM
            # Get total size along the specified dimension
            dim_size = full_matrix.size(split_dim)
            total_nodes = len(self.node_IP_list)
            unique_B_shards = total_nodes // 2
            
            # Validate we have enough nodes
            if total_nodes < 2:
                raise ValueError(f"System 2 requires at least 2 nodes, got {total_nodes}")
            if total_nodes % 2 != 0:
                raise ValueError(f"System 2 requires even number of nodes, got {total_nodes}")
            
            # Use sys2_split_percentages if provided
            if hasattr(self, 'sys2_split_percentages') and self.sys2_split_percentages is not None:
                # Validate percentages
                if len(self.sys2_split_percentages) != unique_B_shards:
                    raise ValueError(
                        f"sys2_split_percentages must have length {unique_B_shards} "
                        f"(total_nodes//2), got {len(self.sys2_split_percentages)}"
                    )
                
                if abs(sum(self.sys2_split_percentages) - 1.0) > 0.01:
                    raise ValueError(
                        f"sys2_split_percentages must sum to 1.0, got {sum(self.sys2_split_percentages)}"
                    )
                
                print(f"✅ Matrix B: {full_matrix.shape} → splitting into {unique_B_shards} shards using percentages {self.sys2_split_percentages}")
                
                # Calculate split sizes based on percentages
                split_sizes = []
                
                for i in range(unique_B_shards - 1):
                    size = int(dim_size * self.sys2_split_percentages[i])
                    split_sizes.append(size)
                
                # Last shard gets remaining rows
                allocated = sum(split_sizes)
                last_size = dim_size - allocated
                split_sizes.append(last_size)
                
                # Validate sizes
                if sum(split_sizes) != dim_size:
                    split_sizes[-1] = dim_size - sum(split_sizes[:-1])
                
            else:
                # Default: equal split
                print(f"✅ Matrix B: {full_matrix.shape} → splitting into {unique_B_shards} equal shards")
                
                base_chunk_size = dim_size // unique_B_shards
                remainder = dim_size % unique_B_shards
                
                split_sizes = [base_chunk_size] * unique_B_shards
                for i in range(remainder):
                    split_sizes[i] += 1
            
            print(f"Split sizes for {unique_B_shards} unique shards along dim={split_dim}: {split_sizes}")
            print(f"Sum check: {sum(split_sizes)} = {dim_size} {'✓' if sum(split_sizes) == dim_size else '✗'}")
            
            # Split B into unique shards along the specified dimension
            B_unique_chunks = torch.split(full_matrix, split_sizes, dim=split_dim)
            
            # Create base repeating pattern
            base_pattern = []
            for i in range(total_nodes):
                shard_index = i % unique_B_shards
                base_pattern.append(B_unique_chunks[shard_index])
            
            self.node_matrices = base_pattern
            
            print(f"✅ Created {total_nodes} B shards (before reordering):")
            for i, chunk in enumerate(self.node_matrices):
                shard_num = i % unique_B_shards
                print(f"  Node {i} (original): gets B{shard_num} {chunk.shape}")
        
        return self.node_matrices

    def convert_to_cluster_matrix_shards(self):

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
                

                print(f'DEBUG::: {save_name}')

                # Step 2: Send file to remote node via ZeroMQ
                print(f"  Step 2: Sending file to remote node {node_IP}")
                self.zmq_send_file(node_IP, save_file_path_DISK)
                self.wait_for_acks(1,save_name)
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
        #print(f"Waiting for ACKs from {len(unique_node_IP_list)-1} remote nodes...")
        #self.wait_for_acks(len(unique_node_IP_list)-1)
        
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
                
                # Wait for acknowledgments from remote nodes
                self.wait_for_acks(1,save_name)

                # Step 2: Tell remote node to copy from RAM to DISK for persistence
                copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                self.zmq_send_command(node_ip, copy_command)
        

        
        print(f"Full matrix distribution completed")
        print(f"Total file paths tracked: {len(self.matrix_file_paths_list)}")
        return 0

    def save_distribute_matrixA_grid_bin(self):
        """
        Distribute broadcast shards for distributed GEMM.
        """

        # ---------------------------
        # MATRIX A — ROW SHARDS
        # ---------------------------
        if self.matrix_labeling == 'a':
            print("\n📤 Distributing Matrix A row shards")
            
            # ---------------------------
            # MATRIX A — SAVE LOCALLY (save both files to head node)
            # ---------------------------
            self.matrix_file_paths_list = []

            # Create disk folder path
            disk_folder_path = os.path.join(self.local_project_dir, self.local_DISK_folder)
            os.makedirs(disk_folder_path, exist_ok=True)

            # Create file paths
            matrixA1_file_path = os.path.join(self.local_RAM_folder, f'{self.matrix_name}_shard_0.bin')
            matrixA2_file_path = os.path.join(self.local_RAM_folder, f'{self.matrix_name}_shard_1.bin')

            # Save matrices locally to RAM
            self.save_matrix_binary(self.node_matrices[0], matrixA1_file_path)
            self.save_matrix_binary(self.node_matrices[1], matrixA2_file_path)

            # Copy to project directory AND disk folder
            import subprocess
            
            # Copy shard 0 to both locations
            shard0_disk_path = os.path.join(disk_folder_path, f'{self.matrix_name}_shard_0.bin')
            subprocess.run(['cp', matrixA1_file_path, self.local_project_dir], check=True)
            subprocess.run(['cp', matrixA1_file_path, shard0_disk_path], check=True)
            print(f"  Copied shard 0 to: {self.local_project_dir}/{self.matrix_name}_shard_0.bin")
            print(f"  Copied shard 0 to: {shard0_disk_path}")
            
            # Copy shard 1 to both locations
            shard1_disk_path = os.path.join(disk_folder_path, f'{self.matrix_name}_shard_1.bin')
            subprocess.run(['cp', matrixA2_file_path, self.local_project_dir], check=True)
            subprocess.run(['cp', matrixA2_file_path, shard1_disk_path], check=True)
            print(f"  Copied shard 1 to: {self.local_project_dir}/{self.matrix_name}_shard_1.bin")
            print(f"  Copied shard 1 to: {shard1_disk_path}")

            # Determine how many nodes get each shard
            total_nodes = len(self.node_IP_list)
            half_nodes = total_nodes // 2  # Integer division
            
            # Track which IPs we've already sent files to
            shard0_sent_to_ips = set()
            shard1_sent_to_ips = set()
            
            # Temporary list to store [IP, file_path] pairs
            ip_shard_pairs = []
            
            for index, node_IP in enumerate(self.node_IP_list):
                # Determine which shard this node gets
                if index < half_nodes:
                    file_path = matrixA1_file_path
                    shard_type = 0
                else:
                    file_path = matrixA2_file_path
                    shard_type = 1
                
                # Store the IP and file path pair
                ip_shard_pairs.append([node_IP, file_path])
                
                # Send files to remote nodes (skip local IP)
                if node_IP != self.IP:
                    if shard_type == 0 and node_IP not in shard0_sent_to_ips:
                        # Send the file to remote RAM
                        self.zmq_send_file(node_IP, matrixA1_file_path)
                        
                        
                        # Send command to copy from RAM to disk
                        remote_filename = os.path.basename(matrixA1_file_path)

                        
                        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, remote_filename)
                        remote_save_file_path_DISK = os.path.join(self.remote_project_dir, self.remote_DISK_folder, remote_filename)
                        copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                        self.zmq_send_command(node_IP, copy_command)
                        shard0_sent_to_ips.add(node_IP)
                        print(f'Sent shard 0 to IP: {node_IP}')
                    
                    elif shard_type == 1 and node_IP not in shard1_sent_to_ips:
                        # Send the file to remote RAM
                        self.zmq_send_file(node_IP, matrixA2_file_path)
                        # Send command to copy from RAM to disk
                        remote_filename = os.path.basename(matrixA2_file_path)
                        self.wait_for_acks(1,remote_filename)
                        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, remote_filename)
                        remote_save_file_path_DISK = os.path.join(self.remote_project_dir, self.remote_DISK_folder, remote_filename)
                        copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                        self.zmq_send_command(node_IP, copy_command)
                        
                        shard1_sent_to_ips.add(node_IP)
                        print(f'Sent shard 1 to IP: {node_IP}')
            
            # Print the IP-shard assignments for debugging
            print("\n📋 Node shard assignments:")
            for ip, path in ip_shard_pairs:
                shard_name = "shard_0" if path == matrixA1_file_path else "shard_1"
                print(f"  {ip} -> {shard_name}")
            
            # Now extract just the file paths (remove IPs) and store in matrix_file_paths_list
            self.matrix_file_paths_list = [file_path for _, file_path in ip_shard_pairs]
            
            print(f"\n✅ Final matrix_file_paths_list (paths only):")
            for i, path in enumerate(self.matrix_file_paths_list):
                shard_name = "shard_0" if path == matrixA1_file_path else "shard_1"
                print(f"  Node {i}: {shard_name}")

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
 
    def load_cluster_matrixA_grid(self):
        """
        Load Matrix A shards from disk to RAM for distributed GEMM.
        Simple version: just copy from local_DISK_folder to local_RAM_folder
        """
                
        print(f"\n📥 Loading Matrix A grid shards from disk to RAM")
        
        # Initialize the file paths list
        self.matrix_file_paths_list = []
        
        # Determine how many nodes get each shard
        total_nodes = len(self.node_IP_list)
        half_nodes = total_nodes // 2  # Integer division
        
        # Create file names for Matrix A shards
        shard0_filename = f'{self.matrix_name}_shard_0.bin'
        shard1_filename = f'{self.matrix_name}_shard_1.bin'
        
        # Define disk paths - CORRECTED: should be in local_project_dir + local_DISK_folder
        local_shard0_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, shard0_filename)
        local_shard1_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, shard1_filename)
        
        # Define RAM paths
        local_shard0_ram_path = os.path.join(self.local_RAM_folder, shard0_filename)
        local_shard1_ram_path = os.path.join(self.local_RAM_folder, shard1_filename)
        
        # Check if shards exist in disk
        print(f"Looking for shards in: {os.path.join(self.local_project_dir, self.local_DISK_folder)}")
        print(f"  Shard 0 path: {local_shard0_disk_path}")
        print(f"  Shard 1 path: {local_shard1_disk_path}")
        
        if not os.path.exists(local_shard0_disk_path):
            print(f"❌ Error: shard_0 not found at: {local_shard0_disk_path}")
            return False
        
        if not os.path.exists(local_shard1_disk_path):
            print(f"❌ Error: shard_1 not found at: {local_shard1_disk_path}")
            return False
        
        # Copy shard 0 from disk to RAM
        print(f"\n📋 Copying shard_0 from disk to RAM...")
        shard0_copy_cmd = f'cp "{local_shard0_disk_path}" "{local_shard0_ram_path}"'
        print(f"  Command: {shard0_copy_cmd}")
        subprocess.run(shard0_copy_cmd, shell=True, check=True)
        print(f"  ✅ shard_0 copied to RAM")
        
        # Copy shard 1 from disk to RAM  
        print(f"\n📋 Copying shard_1 from disk to RAM...")
        shard1_copy_cmd = f'cp "{local_shard1_disk_path}" "{local_shard1_ram_path}"'
        print(f"  Command: {shard1_copy_cmd}")
        subprocess.run(shard1_copy_cmd, shell=True, check=True)
        print(f"  ✅ shard_1 copied to RAM")
        
        # Create the distribution pattern (same as save_distribute_matrixA_grid_bin)
        print(f"\n📋 Creating distribution pattern for {total_nodes} nodes:")
        
        # Track which IPs have been processed for remote commands
        shard0_processed_ips = set()
        shard1_processed_ips = set()
        
        for index, node_IP in enumerate(self.node_IP_list):
            if index < half_nodes:
                # First half gets shard_0
                self.matrix_file_paths_list.append(local_shard0_ram_path)
                print(f"  Node {index} ({node_IP}): assigned shard_0")
                
                # Send command to remote nodes to copy their shard from disk to RAM
                if node_IP != self.IP and node_IP not in shard0_processed_ips:
                    remote_disk_path = os.path.join(self.remote_DISK_folder, shard0_filename)
                    remote_ram_path = os.path.join(self.remote_RAM_folder, shard0_filename)
                    # CORRECTED: remote_disk_path should be prefixed with remote_project_dir
                    remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                    
                    print(f"    Sending to remote {node_IP}: {remote_copy_command}")
                    self.zmq_send_command(node_IP, remote_copy_command)
                    shard0_processed_ips.add(node_IP)
                    
            else:
                # Second half gets shard_1
                self.matrix_file_paths_list.append(local_shard1_ram_path)
                print(f"  Node {index} ({node_IP}): assigned shard_1")
                
                # Send command to remote nodes to copy their shard from disk to RAM
                if node_IP != self.IP and node_IP not in shard1_processed_ips:
                    remote_disk_path = os.path.join(self.remote_DISK_folder, shard1_filename)
                    remote_ram_path = os.path.join(self.remote_RAM_folder, shard1_filename)
                    # CORRECTED: remote_disk_path should be prefixed with remote_project_dir
                    remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                    
                    print(f"    Sending to remote {node_IP}: {remote_copy_command}")
                    self.zmq_send_command(node_IP, remote_copy_command)
                    shard1_processed_ips.add(node_IP)
        
        # ===== LOADING COMPLETE =====
        print(f"\n✅ Matrix A grid loading complete")
        print(f"   Total nodes: {total_nodes}")
        print(f"   First {half_nodes} nodes: shard_0")
        print(f"   Remaining {total_nodes - half_nodes} nodes: shard_1")
        print(f"   File paths tracked: {len(self.matrix_file_paths_list)}")
        
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
            
            # Get GPU number for this node
            if node_IP not in node_gpu_counters:
                node_gpu_counters[node_IP] = 0
            
            current_gpu_number = node_gpu_counters[node_IP]
            
            # INCREMENT NOW for next operation on this node
            if CPU_GPU_select:
                node_gpu_counters[node_IP] += 1
            
            print(f"  Node: {node_IP}")
            print(f"  Backend: {back_end_select}")
            print(f"  Use GPU: {CPU_GPU_select} (GPU #{current_gpu_number})")
            print(f"  Next GPU for this node will be: #{node_gpu_counters[node_IP]}")
            
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
            send_back_str=''
            if send_back_result:
                send_back_str = len(self.node_IP_list)  # Number of shards to combine
                if (self.matrix_labeling=='a' or self.matrix_labeling=='b'):
                    send_back_str = send_back_str * -1 # make send back negative to signle system 2 combine
                print(f"  Send back result: Yes ({send_back_str} shards will be combined)")
            else:
                send_back_str = "0"  # 0 means no send back
                print(f"  Send back result: No (keep distributed)")
            
            ######################NEW CODE##################################
            """
            check if 'hierarchical_split' was used if it was append the 'hierarchical_split_order' to the 'send_back_str' string 
            in c++ server now need to parse the 'send_back_str' string by splitting by '/' to get 'number_of_shards' and now the 
            'hierarchical_split_order'


            """
            if (self.split_depth != 0):
                split_dim_string = ''
                for split_dim in self.hierarchical_split_order:
                    split_dim_string+= str(split_dim)
                send_back_str = str(send_back_str) + '/' + str(self.dim) + split_dim_string # added a  '/' as delimiter
             
            print(f'DEBUG TEST: {send_back_str}')
            ######################NEW CODE##################################
            
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
        
        # ===== WAIT FOR ACKS FROM ALL NODES =====
        unique_nodes = list(set(self.node_IP_list))
        expected_acks = len(self.node_IP_list)  # one ACK per shard/operation
        print(f"\n⏳ WAITING FOR ACKS FROM NODES ({expected_acks})")
        self.wait_for_acks(expected_acks, "ACK_matrixOp_complete")
        #self.send_ack_confirmation("ACK_can_receive")
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
            path = self.local_RAM_folder + base_result_name + '_combined.bin'
            if os.path.exists(path):
                combined_matrix = convert_bin_matrix_to_pt(path)
                os.remove(path)
            else:
                self.wait_for_acks(1, "ACK_combined_matrix_saved")
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



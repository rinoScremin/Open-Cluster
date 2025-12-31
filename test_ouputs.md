    FULL TEST OUTPUTS 

(ray-conda-env) rino@rino-Z370-HD3:~/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix$ python cluster_matrix_v1.py
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ Python frontend ACK receiver bound to port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Created 4 shards according to node percentages
  Node 0: shard shape torch.Size([4000, 20000])
  Node 1: shard shape torch.Size([4000, 20000])
  Node 2: shard shape torch.Size([1000, 20000])
  Node 3: shard shape torch.Size([1000, 20000])
Starting distribution of 4 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixA_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixA_shard_0.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: matrix_shards/big_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixA_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixA_shard_1.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: matrix_shards/big_matrixA_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixA_shard_2.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([1000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(1000, 20000), dtype=float32
  Converting to 4D format...
    2D (1000, 20000) -> 4D (1, 1, 1000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 1000 × 20000
    Wrote 20,000,000 float32 elements
  File saved successfully
  File size: 80,000,020 bytes
  Expected size: 80,000,020 bytes
  ✓ File size verification passed
  Memory usage: 76.29 MB
  Save completed: matrix_shards/big_matrixA_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixA_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
Processing shard 3 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixA_shard_3.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([1000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(1000, 20000), dtype=float32
  Converting to 4D format...
    2D (1000, 20000) -> 4D (1, 1, 1000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 1000 × 20000
    Wrote 20,000,000 float32 elements
  File saved successfully
  File size: 80,000,020 bytes
  Expected size: 80,000,020 bytes
  ✓ File size verification passed
  Memory usage: 76.29 MB
  Save completed: matrix_shards/big_matrixA_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file big_matrixA_shard_3.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
Distribution complete: 4 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Preparing full matrix: big_matrixB.bin
Local paths - DISK: matrix_shards/big_matrixB.bin, RAM: /dev/shm/matrix_shards/big_matrixB.bin
Loading matrix from: model_matrixs/big_matrixB.pt
Matrix loaded - Shape: torch.Size([15000, 20000])
Saving to local storage...
Saving matrix to binary file: matrix_shards/big_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([15000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(15000, 20000), dtype=float32
  Converting to 4D format...
    2D (15000, 20000) -> 4D (1, 1, 15000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 15000 × 20000
    Wrote 300,000,000 float32 elements
  File saved successfully
  File size: 1,200,000,020 bytes
  Expected size: 1,200,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1144.41 MB
  Save completed: matrix_shards/big_matrixB.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([15000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(15000, 20000), dtype=float32
  Converting to 4D format...
    2D (15000, 20000) -> 4D (1, 1, 15000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 15000 × 20000
    Wrote 300,000,000 float32 elements
  File saved successfully
  File size: 1,200,000,020 bytes
  Expected size: 1,200,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1144.41 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB.bin
Remote paths - RAM: /dev/shm/matrix_shards/big_matrixB.bin, DISK: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixB.bin
Distributing to 2 remote node(s)...
Sending to 192.168.2.104
📤 Sent file big_matrixB.bin to 192.168.2.104
Sending to 192.168.2.101
📤 Sent file big_matrixB.bin to 192.168.2.101
✅ Received ACK 1/2
✅ Received ACK 2/2
✅ All ACKs received!
Full matrix distribution completed
Total file paths tracked: 4

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrixA
Matrix B: big_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 4

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 8.44 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 1.189991e-02
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/150000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: big_matrixA
Number of nodes/shard locations: 4
Checking for existing shards in RAM: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 4
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading full matrix: big_matrixB.bin
Source file: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixB.bin
Copying to local RAM...
Distributing to remote nodes...
📤 Sent file big_matrixB.bin to 192.168.2.104
📤 Sent file big_matrixB.bin to 192.168.2.101
Matrix loaded successfully

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrixA
Matrix B: big_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 4

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 5.25 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 1.189991e-02
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/150000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Created 4 shards according to node percentages
  Node 0: shard shape torch.Size([2000, 7000])
  Node 1: shard shape torch.Size([2000, 7000])
  Node 2: shard shape torch.Size([500, 7000])
  Node 3: shard shape torch.Size([500, 7000])
Starting distribution of 4 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixA_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixA_shard_0.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: matrix_shards/mid_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixA_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixA_shard_1.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: matrix_shards/mid_matrixA_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixA_shard_2.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(500, 7000), dtype=float32
  Converting to 4D format...
    2D (500, 7000) -> 4D (1, 1, 500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 500 × 7000
    Wrote 3,500,000 float32 elements
  File saved successfully
  File size: 14,000,020 bytes
  Expected size: 14,000,020 bytes
  ✓ File size verification passed
  Memory usage: 13.35 MB
  Save completed: matrix_shards/mid_matrixA_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixA_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
Processing shard 3 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixA_shard_3.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(500, 7000), dtype=float32
  Converting to 4D format...
    2D (500, 7000) -> 4D (1, 1, 500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 500 × 7000
    Wrote 3,500,000 float32 elements
  File saved successfully
  File size: 14,000,020 bytes
  Expected size: 14,000,020 bytes
  ✓ File size verification passed
  Memory usage: 13.35 MB
  Save completed: matrix_shards/mid_matrixA_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file mid_matrixA_shard_3.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
Distribution complete: 4 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Preparing full matrix: mid_matrixB.bin
Local paths - DISK: matrix_shards/mid_matrixB.bin, RAM: /dev/shm/matrix_shards/mid_matrixB.bin
Loading matrix from: model_matrixs/mid_matrixB.pt
Matrix loaded - Shape: torch.Size([9000, 7000])
Saving to local storage...
Saving matrix to binary file: matrix_shards/mid_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([9000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(9000, 7000), dtype=float32
  Converting to 4D format...
    2D (9000, 7000) -> 4D (1, 1, 9000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 9000 × 7000
    Wrote 63,000,000 float32 elements
  File saved successfully
  File size: 252,000,020 bytes
  Expected size: 252,000,020 bytes
  ✓ File size verification passed
  Memory usage: 240.33 MB
  Save completed: matrix_shards/mid_matrixB.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([9000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(9000, 7000), dtype=float32
  Converting to 4D format...
    2D (9000, 7000) -> 4D (1, 1, 9000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 9000 × 7000
    Wrote 63,000,000 float32 elements
  File saved successfully
  File size: 252,000,020 bytes
  Expected size: 252,000,020 bytes
  ✓ File size verification passed
  Memory usage: 240.33 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB.bin
Remote paths - RAM: /dev/shm/matrix_shards/mid_matrixB.bin, DISK: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixB.bin
Distributing to 2 remote node(s)...
Sending to 192.168.2.104
📤 Sent file mid_matrixB.bin to 192.168.2.104
Sending to 192.168.2.101
📤 Sent file mid_matrixB.bin to 192.168.2.101
✅ Received ACK 1/2
✅ Received ACK 2/2
✅ All ACKs received!
Full matrix distribution completed
Total file paths tracked: 4

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: mid_matrixA
Matrix B: mid_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 4

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 1.08 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.287842, 1856.252319]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.260254e-02
Mean absolute difference: 5.429398e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: mid_matrixA
Number of nodes/shard locations: 4
Checking for existing shards in RAM: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 4
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 4 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading full matrix: mid_matrixB.bin
Source file: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixB.bin
Copying to local RAM...
Distributing to remote nodes...
📤 Sent file mid_matrixB.bin to 192.168.2.104
📤 Sent file mid_matrixB.bin to 192.168.2.101
Matrix loaded successfully

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: mid_matrixA
Matrix B: mid_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 4

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 0.78 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.287842, 1856.252319]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.260254e-02
Mean absolute difference: 5.429398e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Created 5 shards according to node percentages
  Node 0: shard shape torch.Size([4000, 20000])
  Node 1: shard shape torch.Size([4000, 20000])
  Node 2: shard shape torch.Size([500, 20000])
  Node 3: shard shape torch.Size([500, 20000])
  Node 4: shard shape torch.Size([1000, 20000])
Starting distribution of 5 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixA_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixA_shard_0.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: matrix_shards/big_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixA_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixA_shard_1.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: matrix_shards/big_matrixA_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4000, 20000), dtype=float32
  Converting to 4D format...
    2D (4000, 20000) -> 4D (1, 1, 4000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4000 × 20000
    Wrote 80,000,000 float32 elements
  File saved successfully
  File size: 320,000,020 bytes
  Expected size: 320,000,020 bytes
  ✓ File size verification passed
  Memory usage: 305.18 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixA_shard_2.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(500, 20000), dtype=float32
  Converting to 4D format...
    2D (500, 20000) -> 4D (1, 1, 500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 500 × 20000
    Wrote 10,000,000 float32 elements
  File saved successfully
  File size: 40,000,020 bytes
  Expected size: 40,000,020 bytes
  ✓ File size verification passed
  Memory usage: 38.15 MB
  Save completed: matrix_shards/big_matrixA_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixA_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
Processing shard 3 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixA_shard_3.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(500, 20000), dtype=float32
  Converting to 4D format...
    2D (500, 20000) -> 4D (1, 1, 500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 500 × 20000
    Wrote 10,000,000 float32 elements
  File saved successfully
  File size: 40,000,020 bytes
  Expected size: 40,000,020 bytes
  ✓ File size verification passed
  Memory usage: 38.15 MB
  Save completed: matrix_shards/big_matrixA_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixA_shard_3.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
Processing shard 4 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixA_shard_4.bin
Saving matrix to binary file: matrix_shards/big_matrixA_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([1000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(1000, 20000), dtype=float32
  Converting to 4D format...
    2D (1000, 20000) -> 4D (1, 1, 1000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 1000 × 20000
    Wrote 20,000,000 float32 elements
  File saved successfully
  File size: 80,000,020 bytes
  Expected size: 80,000,020 bytes
  ✓ File size verification passed
  Memory usage: 76.29 MB
  Save completed: matrix_shards/big_matrixA_shard_4.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file big_matrixA_shard_4.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixA_shard_4.bin
Distribution complete: 5 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Preparing full matrix: big_matrixB.bin
Local paths - DISK: matrix_shards/big_matrixB.bin, RAM: /dev/shm/matrix_shards/big_matrixB.bin
Loading matrix from: model_matrixs/big_matrixB.pt
Matrix loaded - Shape: torch.Size([15000, 20000])
Saving to local storage...
Saving matrix to binary file: matrix_shards/big_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([15000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(15000, 20000), dtype=float32
  Converting to 4D format...
    2D (15000, 20000) -> 4D (1, 1, 15000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 15000 × 20000
    Wrote 300,000,000 float32 elements
  File saved successfully
  File size: 1,200,000,020 bytes
  Expected size: 1,200,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1144.41 MB
  Save completed: matrix_shards/big_matrixB.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([15000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(15000, 20000), dtype=float32
  Converting to 4D format...
    2D (15000, 20000) -> 4D (1, 1, 15000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 15000 × 20000
    Wrote 300,000,000 float32 elements
  File saved successfully
  File size: 1,200,000,020 bytes
  Expected size: 1,200,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1144.41 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB.bin
Remote paths - RAM: /dev/shm/matrix_shards/big_matrixB.bin, DISK: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixB.bin
Distributing to 2 remote node(s)...
Sending to 192.168.2.104
📤 Sent file big_matrixB.bin to 192.168.2.104
Sending to 192.168.2.101
📤 Sent file big_matrixB.bin to 192.168.2.101
✅ Received ACK 1/2
✅ Received ACK 2/2
✅ All ACKs received!
Full matrix distribution completed
Total file paths tracked: 5

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrixA
Matrix B: big_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 5

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (5)
✅ Received ACK_matrixOp_complete 1/5
✅ Received ACK_matrixOp_complete 2/5
✅ Received ACK_matrixOp_complete 3/5
✅ Received ACK_matrixOp_complete 4/5
✅ Received ACK_matrixOp_complete 5/5
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 8.93 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 1.128532e-02
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/150000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: big_matrixA
Number of nodes/shard locations: 5
Checking for existing shards in RAM: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path
  Shard 4: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 5
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading full matrix: big_matrixB.bin
Source file: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixB.bin
Copying to local RAM...
Distributing to remote nodes...
📤 Sent file big_matrixB.bin to 192.168.2.104
📤 Sent file big_matrixB.bin to 192.168.2.101
Matrix loaded successfully

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrixA
Matrix B: big_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 5

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (5)
✅ Received ACK_matrixOp_complete 1/5
✅ Received ACK_matrixOp_complete 2/5
✅ Received ACK_matrixOp_complete 3/5
✅ Received ACK_matrixOp_complete 4/5
✅ Received ACK_matrixOp_complete 5/5
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 9.43 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 1.128532e-02
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/150000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Created 5 shards according to node percentages
  Node 0: shard shape torch.Size([2000, 7000])
  Node 1: shard shape torch.Size([2000, 7000])
  Node 2: shard shape torch.Size([250, 7000])
  Node 3: shard shape torch.Size([250, 7000])
  Node 4: shard shape torch.Size([500, 7000])
Starting distribution of 5 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixA_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixA_shard_0.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: matrix_shards/mid_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixA_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixA_shard_1.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: matrix_shards/mid_matrixA_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2000, 7000), dtype=float32
  Converting to 4D format...
    2D (2000, 7000) -> 4D (1, 1, 2000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2000 × 7000
    Wrote 14,000,000 float32 elements
  File saved successfully
  File size: 56,000,020 bytes
  Expected size: 56,000,020 bytes
  ✓ File size verification passed
  Memory usage: 53.41 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixA_shard_2.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(250, 7000), dtype=float32
  Converting to 4D format...
    2D (250, 7000) -> 4D (1, 1, 250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 250 × 7000
    Wrote 1,750,000 float32 elements
  File saved successfully
  File size: 7,000,020 bytes
  Expected size: 7,000,020 bytes
  ✓ File size verification passed
  Memory usage: 6.68 MB
  Save completed: matrix_shards/mid_matrixA_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixA_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
Processing shard 3 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixA_shard_3.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(250, 7000), dtype=float32
  Converting to 4D format...
    2D (250, 7000) -> 4D (1, 1, 250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 250 × 7000
    Wrote 1,750,000 float32 elements
  File saved successfully
  File size: 7,000,020 bytes
  Expected size: 7,000,020 bytes
  ✓ File size verification passed
  Memory usage: 6.68 MB
  Save completed: matrix_shards/mid_matrixA_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixA_shard_3.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
Processing shard 4 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixA_shard_4.bin
Saving matrix to binary file: matrix_shards/mid_matrixA_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(500, 7000), dtype=float32
  Converting to 4D format...
    2D (500, 7000) -> 4D (1, 1, 500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 500 × 7000
    Wrote 3,500,000 float32 elements
  File saved successfully
  File size: 14,000,020 bytes
  Expected size: 14,000,020 bytes
  ✓ File size verification passed
  Memory usage: 13.35 MB
  Save completed: matrix_shards/mid_matrixA_shard_4.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file mid_matrixA_shard_4.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixA_shard_4.bin
Distribution complete: 5 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Preparing full matrix: mid_matrixB.bin
Local paths - DISK: matrix_shards/mid_matrixB.bin, RAM: /dev/shm/matrix_shards/mid_matrixB.bin
Loading matrix from: model_matrixs/mid_matrixB.pt
Matrix loaded - Shape: torch.Size([9000, 7000])
Saving to local storage...
Saving matrix to binary file: matrix_shards/mid_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([9000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(9000, 7000), dtype=float32
  Converting to 4D format...
    2D (9000, 7000) -> 4D (1, 1, 9000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 9000 × 7000
    Wrote 63,000,000 float32 elements
  File saved successfully
  File size: 252,000,020 bytes
  Expected size: 252,000,020 bytes
  ✓ File size verification passed
  Memory usage: 240.33 MB
  Save completed: matrix_shards/mid_matrixB.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([9000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(9000, 7000), dtype=float32
  Converting to 4D format...
    2D (9000, 7000) -> 4D (1, 1, 9000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 9000 × 7000
    Wrote 63,000,000 float32 elements
  File saved successfully
  File size: 252,000,020 bytes
  Expected size: 252,000,020 bytes
  ✓ File size verification passed
  Memory usage: 240.33 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB.bin
Remote paths - RAM: /dev/shm/matrix_shards/mid_matrixB.bin, DISK: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixB.bin
Distributing to 2 remote node(s)...
Sending to 192.168.2.104
📤 Sent file mid_matrixB.bin to 192.168.2.104
Sending to 192.168.2.101
📤 Sent file mid_matrixB.bin to 192.168.2.101
✅ Received ACK 1/2
✅ Received ACK 2/2
✅ All ACKs received!
Full matrix distribution completed
Total file paths tracked: 5

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: mid_matrixA
Matrix B: mid_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 5

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (5)
✅ Received ACK_matrixOp_complete 1/5
✅ Received ACK_matrixOp_complete 2/5
✅ Received ACK_matrixOp_complete 3/5
✅ Received ACK_matrixOp_complete 4/5
✅ Received ACK_matrixOp_complete 5/5
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 7.38 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.287842, 1856.252319]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.260254e-02
Mean absolute difference: 5.139636e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: mid_matrixA
Number of nodes/shard locations: 5
Checking for existing shards in RAM: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path
  Shard 4: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 5
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 5 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading full matrix: mid_matrixB.bin
Source file: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixB.bin
Copying to local RAM...
Distributing to remote nodes...
📤 Sent file mid_matrixB.bin to 192.168.2.104
📤 Sent file mid_matrixB.bin to 192.168.2.101
Matrix loaded successfully

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: mid_matrixA
Matrix B: mid_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 5

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (5)
✅ Received ACK_matrixOp_complete 1/5
✅ Received ACK_matrixOp_complete 2/5
✅ Received ACK_matrixOp_complete 3/5
✅ Received ACK_matrixOp_complete 4/5
✅ Received ACK_matrixOp_complete 5/5
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 5.45 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.287842, 1856.252319]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.260254e-02
Mean absolute difference: 5.139636e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: True
   Dimension: 2

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Matrix A: torch.Size([10000, 20000]) → [torch.Size([5000, 20000]), torch.Size([5000, 20000])]

📤 Distributing Matrix A row shards
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Copied shard 0 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix//big_matrixA_shard_0.bin
  Copied shard 0 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_0.bin
  Copied shard 1 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix//big_matrixA_shard_1.bin
  Copied shard 1 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_1.bin
📤 Sent file big_matrixA_shard_1.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.101
📤 Sent file big_matrixA_shard_1.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.104

📋 Node shard assignments:
  192.168.2.100 -> shard_0
  192.168.2.100 -> shard_0
  192.168.2.100 -> shard_0
  192.168.2.101 -> shard_1
  192.168.2.101 -> shard_1
  192.168.2.104 -> shard_1

✅ Final matrix_file_paths_list (paths only):
  Node 0: shard_0
  Node 1: shard_0
  Node 2: shard_0
  Node 3: shard_1
  Node 4: shard_1
  Node 5: shard_1
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: True
   Dimension: 3

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Matrix B: torch.Size([15000, 20000]) → splitting into 3 unique shards
Split sizes for 3 unique shards: [5000, 5000, 5000]
Sum check: 15000 = 15000 ✓
✅ Created 6 B shards (repeating pattern):
  Node 0: gets B0 torch.Size([5000, 20000])
  Node 1: gets B1 torch.Size([5000, 20000])
  Node 2: gets B2 torch.Size([5000, 20000])
  Node 3: gets B0 torch.Size([5000, 20000])
  Node 4: gets B1 torch.Size([5000, 20000])
  Node 5: gets B2 torch.Size([5000, 20000])
Starting distribution of 6 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_0.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_1.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_2.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_2.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_2.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_2.bin
  Added RAM path to file list
Processing shard 3 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_3.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixB_shard_3.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixB_shard_3.bin
Processing shard 4 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_4.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_4.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixB_shard_4.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixB_shard_4.bin
Processing shard 5 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_5.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_5.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([5000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(5000, 20000), dtype=float32
  Converting to 4D format...
    2D (5000, 20000) -> 4D (1, 1, 5000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 5000 × 20000
    Wrote 100,000,000 float32 elements
  File saved successfully
  File size: 400,000,020 bytes
  Expected size: 400,000,020 bytes
  ✓ File size verification passed
  Memory usage: 381.47 MB
  Save completed: matrix_shards/big_matrixB_shard_5.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file big_matrixB_shard_5.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixB_shard_5.bin
Distribution complete: 6 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixA
   Split Matrix: True
   Dimension: 2

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100

📥 Loading Matrix A grid shards from disk to RAM
Looking for shards in: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/
  Shard 0 path: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_0.bin
  Shard 1 path: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_1.bin

📋 Copying shard_0 from disk to RAM...
  Command: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_0.bin" "/dev/shm/matrix_shards/big_matrixA_shard_0.bin"
  ✅ shard_0 copied to RAM

📋 Copying shard_1 from disk to RAM...
  Command: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_1.bin" "/dev/shm/matrix_shards/big_matrixA_shard_1.bin"
  ✅ shard_1 copied to RAM

📋 Creating distribution pattern for 6 nodes:
  Node 0 (192.168.2.100): assigned shard_0
  Node 1 (192.168.2.100): assigned shard_0
  Node 2 (192.168.2.100): assigned shard_0
  Node 3 (192.168.2.101): assigned shard_1
    Sending to remote 192.168.2.101: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_1.bin" "/dev/shm/matrix_shards/big_matrixA_shard_1.bin"
  Node 4 (192.168.2.101): assigned shard_1
  Node 5 (192.168.2.104): assigned shard_1
    Sending to remote 192.168.2.104: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_1.bin" "/dev/shm/matrix_shards/big_matrixA_shard_1.bin"

✅ Matrix A grid loading complete
   Total nodes: 6
   First 3 nodes: shard_0
   Remaining 3 nodes: shard_1
   File paths tracked: 6
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixB
   Split Matrix: True
   Dimension: 3

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: big_matrixB
Number of nodes/shard locations: 6
Checking for existing shards in RAM: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path
  Shard 4: Using existing RAM path
  Shard 5: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 6

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrixA
Matrix B: big_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 6

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_1.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #2)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_2.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 3

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_3.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 4:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_4.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 5:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_5.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (6)
✅ Received ACK_matrixOp_complete 1/6
✅ Received ACK_matrixOp_complete 2/6
✅ Received ACK_matrixOp_complete 3/6
✅ Received ACK_matrixOp_complete 4/6
✅ Received ACK_matrixOp_complete 5/6
✅ Received ACK_matrixOp_complete 6/6
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 22.78 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 6.965588e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/150000000 (0.00%)
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: True
   Dimension: 2

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Matrix A: torch.Size([5000, 7000]) → [torch.Size([2500, 7000]), torch.Size([2500, 7000])]

📤 Distributing Matrix A row shards
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2500, 7000), dtype=float32
  Converting to 4D format...
    2D (2500, 7000) -> 4D (1, 1, 2500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2500 × 7000
    Wrote 17,500,000 float32 elements
  File saved successfully
  File size: 70,000,020 bytes
  Expected size: 70,000,020 bytes
  ✓ File size verification passed
  Memory usage: 66.76 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2500, 7000), dtype=float32
  Converting to 4D format...
    2D (2500, 7000) -> 4D (1, 1, 2500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2500 × 7000
    Wrote 17,500,000 float32 elements
  File saved successfully
  File size: 70,000,020 bytes
  Expected size: 70,000,020 bytes
  ✓ File size verification passed
  Memory usage: 66.76 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Copied shard 0 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix//mid_matrixA_shard_0.bin
  Copied shard 0 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_0.bin
  Copied shard 1 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix//mid_matrixA_shard_1.bin
  Copied shard 1 to: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_1.bin
📤 Sent file mid_matrixA_shard_1.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.101
📤 Sent file mid_matrixA_shard_1.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.104

📋 Node shard assignments:
  192.168.2.100 -> shard_0
  192.168.2.100 -> shard_0
  192.168.2.100 -> shard_0
  192.168.2.101 -> shard_1
  192.168.2.101 -> shard_1
  192.168.2.104 -> shard_1

✅ Final matrix_file_paths_list (paths only):
  Node 0: shard_0
  Node 1: shard_0
  Node 2: shard_0
  Node 3: shard_1
  Node 4: shard_1
  Node 5: shard_1
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: True
   Dimension: 3

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
✅ Matrix B: torch.Size([9000, 7000]) → splitting into 3 unique shards
Split sizes for 3 unique shards: [3000, 3000, 3000]
Sum check: 9000 = 9000 ✓
✅ Created 6 B shards (repeating pattern):
  Node 0: gets B0 torch.Size([3000, 7000])
  Node 1: gets B1 torch.Size([3000, 7000])
  Node 2: gets B2 torch.Size([3000, 7000])
  Node 3: gets B0 torch.Size([3000, 7000])
  Node 4: gets B1 torch.Size([3000, 7000])
  Node 5: gets B2 torch.Size([3000, 7000])
Starting distribution of 6 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_0.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_1.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_1.bin
  Added RAM path to file list
Processing shard 2 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_2.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_2.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_2.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_2.bin
  Added RAM path to file list
Processing shard 3 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_3.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_3.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixB_shard_3.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixB_shard_3.bin
Processing shard 4 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_4.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_4.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixB_shard_4.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixB_shard_4.bin
Processing shard 5 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_5.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_5.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3000, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3000, 7000), dtype=float32
  Converting to 4D format...
    2D (3000, 7000) -> 4D (1, 1, 3000, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3000 × 7000
    Wrote 21,000,000 float32 elements
  File saved successfully
  File size: 84,000,020 bytes
  Expected size: 84,000,020 bytes
  ✓ File size verification passed
  Memory usage: 80.11 MB
  Save completed: matrix_shards/mid_matrixB_shard_5.bin
  Step 2: Sending file to remote node 192.168.2.104
📤 Sent file mid_matrixB_shard_5.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixB_shard_5.bin
Distribution complete: 6 shards saved and distributed
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixA
   Split Matrix: True
   Dimension: 2

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100

📥 Loading Matrix A grid shards from disk to RAM
Looking for shards in: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/
  Shard 0 path: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_0.bin
  Shard 1 path: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_1.bin

📋 Copying shard_0 from disk to RAM...
  Command: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_0.bin" "/dev/shm/matrix_shards/mid_matrixA_shard_0.bin"
  ✅ shard_0 copied to RAM

📋 Copying shard_1 from disk to RAM...
  Command: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_1.bin" "/dev/shm/matrix_shards/mid_matrixA_shard_1.bin"
  ✅ shard_1 copied to RAM

📋 Creating distribution pattern for 6 nodes:
  Node 0 (192.168.2.100): assigned shard_0
  Node 1 (192.168.2.100): assigned shard_0
  Node 2 (192.168.2.100): assigned shard_0
  Node 3 (192.168.2.101): assigned shard_1
    Sending to remote 192.168.2.101: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_1.bin" "/dev/shm/matrix_shards/mid_matrixA_shard_1.bin"
  Node 4 (192.168.2.101): assigned shard_1
  Node 5 (192.168.2.104): assigned shard_1
    Sending to remote 192.168.2.104: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_1.bin" "/dev/shm/matrix_shards/mid_matrixA_shard_1.bin"

✅ Matrix A grid loading complete
   Total nodes: 6
   First 3 nodes: shard_0
   Remaining 3 nodes: shard_1
   File paths tracked: 6
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 6 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: mid_matrixB
   Split Matrix: True
   Dimension: 3

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 3 unique nodes...
   ✅ Connected to worker node 192.168.2.101:5557
   ✅ Connected to worker node 192.168.2.104:5557
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to worker WiFi 192.168.3.243:5557
   ✅ Connected to worker WiFi 192.168.3.165:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 3

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.101
   ✅ Directory creation command sent to 192.168.2.104
   ✅ Directory creation command sent to 192.168.2.100
Loading cluster matrix shards: mid_matrixB
Number of nodes/shard locations: 6
Checking for existing shards in RAM: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path
  Shard 2: Using existing RAM path
  Shard 3: Using existing RAM path
  Shard 4: Using existing RAM path
  Shard 5: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 6

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: mid_matrixA
Matrix B: mid_matrixB
Operation: mul
Transpose A: False, Transpose B: True
Send back result: True
Number of shards: 6

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_1.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

Processing shard 2:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #2)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_2.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 3

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_3.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 1

Processing shard 4:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_4.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101
  Incremented GPU counter for node 192.168.2.101 to 2

Processing shard 5:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_5.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104
  Incremented GPU counter for node 192.168.2.104 to 1

⏳ WAITING FOR ACKS FROM NODES (6)
✅ Received ACK_matrixOp_complete 1/6
✅ Received ACK_matrixOp_complete 2/6
✅ Received ACK_matrixOp_complete 3/6
✅ Received ACK_matrixOp_complete 4/6
✅ Received ACK_matrixOp_complete 5/6
✅ Received ACK_matrixOp_complete 6/6
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 2.52 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.286255, 1856.249878]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.040527e-02
Mean absolute difference: 3.080465e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
(ray-conda-env) rino@rino-Z370-HD3:~/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix$ 


BELOW ARE OUTPUT FROM THE FRONT END 
======================================================================
🏁 INITIALIZATION COMPLETE
======================================================================
✅ Cluster matrix initialization successful!
   - Total nodes configured: 2
   - Matrix handling mode: Split
   - Backends: ['llama', 'llama']
   - CPU/GPU selections: [True, True]

✅ Cluster operation completed
Result name: big_matrixxbig_matrix
Cluster operation time: 8.35 seconds

============================================================
🔍 PYTORCH REFERENCE (SINGLE NODE)
============================================================

Matrix A shape: torch.Size([20000, 20000])
Matrix B shape: torch.Size([20000, 20000])
Matrix A sample (5x5):
tensor([[0.5822, 0.4205, 0.3506, 0.7201, 0.6890],
        [0.8598, 0.2469, 0.4321, 0.2997, 0.8340],
        [0.2097, 0.0890, 0.8809, 0.5067, 0.9566],
        [0.4470, 0.5123, 0.3698, 0.0967, 0.0183],
        [0.3265, 0.1357, 0.2475, 0.8788, 0.5820]])

Reference result shape: torch.Size([20000, 20000])
Single-node PyTorch computation time: 22.74s
First 2500 elements of reference result:
tensor([6642.2144, 4998.1240, 4964.3613,  ..., 5005.1006, 4977.2671,
        5025.6562])

Reference result sample (5x5):
tensor([[6642.2144, 4998.1240, 4964.3613, 4969.4556, 4994.3457],
        [4998.1240, 6641.9399, 4992.1772, 4975.1665, 5021.4575],
        [4964.3613, 4992.1772, 6628.5098, 4965.4482, 4985.1729],
        [4969.4556, 4975.1665, 4965.4482, 6643.5278, 4986.7500],
        [4994.3457, 5021.4575, 4985.1729, 4986.7500, 6677.4839]])

============================================================
📥 LOADING CLUSTER RESULT
============================================================

============================================================
🏁 PERFORMANCE COMPARISON
============================================================
CLUSTER OPERATION TIME:      8.3546 seconds
SINGLE NODE PYTORCH TIME:    22.7372 seconds
------------------------------------------------------------
CLUSTER vs SINGLE NODE: 2.72x faster
# Node configuration for the cluster
               RX-5500         RX-6400
IP_list = ['192.168.2.100','192.168.2.100']   
percentages = [0.5,0.5]  
CPU_GPU_select_list = [True, True]  
backend_select_list = ['llama','llama']  


using a RX 5500 and RX 6400 AMD GPU 



======================================================================
🏁 INITIALIZATION COMPLETE
======================================================================
✅ Cluster matrix initialization successful!
   - Total nodes configured: 2
   - Matrix handling mode: Split
   - Backends: ['llama', 'llama']
   - CPU/GPU selections: [True, True]

✅ Cluster operation completed
Result name: big_matrixxbig_matrix
Cluster operation time: 10.10 seconds

============================================================
🔍 PYTORCH REFERENCE (SINGLE NODE)
============================================================

Matrix A shape: torch.Size([20000, 20000])
Matrix B shape: torch.Size([20000, 20000])
Matrix A sample (5x5):
tensor([[0.5822, 0.4205, 0.3506, 0.7201, 0.6890],
        [0.8598, 0.2469, 0.4321, 0.2997, 0.8340],
        [0.2097, 0.0890, 0.8809, 0.5067, 0.9566],
        [0.4470, 0.5123, 0.3698, 0.0967, 0.0183],
        [0.3265, 0.1357, 0.2475, 0.8788, 0.5820]])

Reference result shape: torch.Size([20000, 20000])
Single-node PyTorch computation time: 22.29s
First 2500 elements of reference result:
tensor([6642.2144, 4998.1240, 4964.3613,  ..., 5005.1006, 4977.2671,
        5025.6562])

Reference result sample (5x5):
tensor([[6642.2144, 4998.1240, 4964.3613, 4969.4556, 4994.3457],
        [4998.1240, 6641.9399, 4992.1772, 4975.1665, 5021.4575],
        [4964.3613, 4992.1772, 6628.5098, 4965.4482, 4985.1729],
        [4969.4556, 4975.1665, 4965.4482, 6643.5278, 4986.7500],
        [4994.3457, 5021.4575, 4985.1729, 4986.7500, 6677.4839]])

============================================================
📥 LOADING CLUSTER RESULT
============================================================
Loading cluster result from: /dev/shm/matrix_shards/big_matrixxbig_matrix_combined.bin
✅ Loaded /dev/shm/matrix_shards/big_matrixxbig_matrix_combined.bin
  Original dims: [1, 1, 20000, 20000]
  Result tensor shape: torch.Size([20000, 20000]), size: 1,600,000,000 bytes
  Data range: [4816.545410, 6825.019043]
Cluster result shape: torch.Size([20000, 20000])
Cluster result sample (5x5):
tensor([[6642.2402, 4998.1519, 4964.3545, 4969.4458, 4994.3462],
        [4998.1519, 6642.0332, 4992.1909, 4975.1743, 5021.4595],
        [4964.3545, 4992.1909, 6628.5337, 4965.4272, 4985.1807],
        [4969.4458, 4975.1743, 4965.4272, 6643.5527, 4986.7388],
        [4994.3462, 5021.4595, 4985.1807, 4986.7388, 6677.5449]])

============================================================
🏁 PERFORMANCE COMPARISON
============================================================
CLUSTER OPERATION TIME:      10.0957 seconds
SINGLE NODE PYTORCH TIME:    22.2927 seconds
------------------------------------------------------------
CLUSTER vs SINGLE NODE: 2.21x faster
✅ Shapes match: torch.Size([20000, 20000])
Max absolute difference: 1.376953e-01
Mean absolute difference: 1.315311e-02
✅ Results match within tolerance (0.15)

Torch reference - first 100 elements of first half:
tensor([6642.2144, 4998.1240, 4964.3613, 4969.4556, 4994.3457, 4999.0098,
        4960.4141, 5011.5098, 4995.5801, 4955.3188, 5008.3462, 5017.0386,
        4940.8623, 5005.9199, 5012.8750, 4995.0996, 4969.3306, 4990.7939,
        5001.7642, 4986.4834, 5002.0005, 4974.2461, 4994.4048, 5015.1133,
        5014.3979, 4994.6294, 5015.9272, 4974.7793, 4987.9248, 4992.0923,
        4990.2627, 5004.7568, 5014.2886, 5019.8037, 4994.1255, 4962.1348,
        4967.3003, 5011.9282, 5008.4985, 5012.1704, 4989.1821, 4985.9629,
        4986.1226, 5001.7017, 4987.3892, 5011.2651, 5000.4028, 4974.8545,
        4967.9717, 5018.9673, 5023.8721, 4985.1094, 5024.8447, 4994.2095,
        4964.5371, 4995.6973, 5000.1904, 5016.1660, 5025.8828, 5016.1177,
        5025.7104, 5066.5820, 4977.1812, 5009.7920, 4969.0859, 4993.5454,
        5004.5093, 4996.7764, 5015.9214, 5004.4941, 5013.7920, 5001.7773,
        5008.9658, 5000.5352, 4952.4927, 4944.4858, 4995.1323, 5011.3755,
        5021.0303, 4983.2246, 5025.5337, 5009.6050, 5011.3164, 4994.6650,
        5014.9321, 5017.5444, 5021.5830, 5004.3306, 4989.3862, 4989.6895,
        5014.1343, 4953.6255, 4938.4976, 5004.1211, 4994.1587, 5023.8799,
        5004.2246, 4953.3467, 5010.6670, 4965.7173])

Torch reference - first 100 elements of second half:
tensor([5002.0723, 5016.1333, 4991.6533, 4991.3120, 5015.6411, 5033.5591,
        4996.2041, 5041.8496, 5026.8521, 4990.7983, 5042.3252, 5020.5039,
        4967.9214, 5032.2812, 5027.8701, 5033.7290, 4996.4814, 5023.4341,
        5007.7896, 4997.4297, 5000.7339, 4993.6509, 5033.0757, 5038.4414,
        5031.1611, 5016.1265, 5036.0884, 4992.3945, 5034.0869, 5006.6157,
        5007.2500, 5004.1826, 5039.1987, 5028.2642, 4999.5874, 5014.1548,
        5022.4189, 5023.9775, 5055.8071, 5004.0083, 5052.2456, 5009.9092,
        5030.5288, 5049.0449, 5029.6680, 5041.8506, 5038.9512, 4972.3301,
        4993.5986, 5049.8477, 5060.0649, 4998.5469, 5013.3677, 5012.8462,
        4997.7056, 5047.9868, 4994.0874, 5027.6543, 5051.0576, 5004.7969,
        5045.4507, 5055.8237, 5011.7920, 4996.6016, 4978.8257, 5000.4204,
        5052.4463, 5005.8726, 5028.7261, 5040.2319, 5034.6162, 5002.1138,
        5025.5190, 5023.7905, 4981.7788, 5002.6040, 5015.0122, 5069.6211,
        5040.0366, 4996.0684, 5056.1797, 5044.9092, 5055.8999, 5007.0137,
        5014.2285, 5048.7715, 5032.9038, 5015.1846, 5020.7949, 5003.4893,
        5039.9639, 4985.8799, 4984.9141, 5024.4575, 5008.4390, 5056.7822,
        5027.9995, 5002.2764, 5039.5527, 4995.6245])

above is 

   cluster_start_time = time.time()
   cluster_matrixC = matrixA.cluster_operation(matrixB, False, True, True)  
   cluster_end_time = time.time()

above is the results so you can check that they are correct 

============================================================
🏁 PERFORMANCE COMPARISON
============================================================
CLUSTER OPERATION TIME:      10.0957 seconds
SINGLE NODE PYTORCH TIME:    22.2927 seconds
------------------------------------------------------------
CLUSTER vs SINGLE NODE: 2.21x faster
✅ Shapes match: torch.Size([20000, 20000])
Max absolute difference: 1.376953e-01
Mean absolute difference: 1.315311e-02
✅ Results match within tolerance (0.15)

above show using RX 5500 and RX 6400 to run and combine the matrix in less then half the time 


FULL OUTPUT FROM FRONT END RUN! BELOW 
============================================================
🚀 CLUSTER MATRIX DISTRIBUTION SYSTEM TEST
============================================================

📦 Creating and distributing matrix A (split=True)...
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 2 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrix
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 1 unique nodes...
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 1

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ Python frontend ACK receiver bound to port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.100

======================================================================
🧮 MATRIX DISTRIBUTION PHASE
======================================================================
   Matrix file exists: True
   Split matrix mode: True

📝 CASE 1: NEW MATRIX - CONVERT, DISTRIBUTE, AND LOAD
   Processing steps:
   1. Convert to cluster matrix shards
   2. Distribute shards to nodes
   3. Load distributed shards
======================================================================
🔪 CONVERTING MATRIX TO DISTRIBUTED SHARDS
======================================================================

📦 LOADING ORIGINAL MATRIX...
   File path: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/big_matrix.pt
✅ Successfully loaded matrix
   - Shape: torch.Size([20000, 20000])
   - Data type: torch.float32
   - Device: cpu
   - Original matrix shape stored: [20000, 20000]

⚙️  CONFIGURING SHARD PARAMETERS...
   Requested shards: 100
   Split dimension (dim=0): size = 20000
   ✅ Matrix size supports 100 shards
   Final shard count: 100

🔪 SPLITTING MATRIX INTO INITIAL SHARDS...
   Creating exact shard sizes along dim=0 to cover all rows/cols
✅ Successfully split matrix
   - Created 100 shards
   - Each shard shape: torch.Size([200, 20000])
   - Shard sizes along dim 0: [200, 200, 200, 200, 200]... (first 5)
   - Total elements in shards: 20000 (should equal 20000)

🌐 DISTRIBUTING SHARDS TO CLUSTER NODES...
   Number of nodes: 2
   Node percentages: ['50.0%', '50.0%']
   Total shards available: 100

   Node   Shards   Percentage   Shape                Start    End     
   --------------------------------------------------------------
   0      50         50.0%    torch.Size([10000, 20000]) 0        49      
   1      50         50.0%    torch.Size([10000, 20000]) 50       99      

   📊 All 100 shards have been allocated

📊 DISTRIBUTION VERIFICATION:
   Original matrix size (dim 0): 20000
   Total after distribution (dim 0): 20000
   ✅ SUCCESS: All elements accounted for!

   Distribution summary:
   Node 0: torch.Size([10000, 20000]) (50.0% of total)
   Node 1: torch.Size([10000, 20000]) (50.0% of total)

======================================================================
✅ MATRIX SHARD CONVERSION COMPLETE
======================================================================
Starting distribution of 2 shards to 1 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrix_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrix_shard_0.bin
Saving matrix to binary file: matrix_shards/big_matrix_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([10000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(10000, 20000), dtype=float32
  Converting to 4D format...
    2D (10000, 20000) -> 4D (1, 1, 10000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 10000 × 20000
    Wrote 200,000,000 float32 elements
  File saved successfully
  File size: 800,000,020 bytes
  Expected size: 800,000,020 bytes
  ✓ File size verification passed
  Memory usage: 762.94 MB
  Save completed: matrix_shards/big_matrix_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrix_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([10000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(10000, 20000), dtype=float32
  Converting to 4D format...
    2D (10000, 20000) -> 4D (1, 1, 10000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 10000 × 20000
    Wrote 200,000,000 float32 elements
  File saved successfully
  File size: 800,000,020 bytes
  Expected size: 800,000,020 bytes
  ✓ File size verification passed
  Memory usage: 762.94 MB
  Save completed: /dev/shm/matrix_shards/big_matrix_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrix_shard_1.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrix_shard_1.bin
Saving matrix to binary file: matrix_shards/big_matrix_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([10000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(10000, 20000), dtype=float32
  Converting to 4D format...
    2D (10000, 20000) -> 4D (1, 1, 10000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 10000 × 20000
    Wrote 200,000,000 float32 elements
  File saved successfully
  File size: 800,000,020 bytes
  Expected size: 800,000,020 bytes
  ✓ File size verification passed
  Memory usage: 762.94 MB
  Save completed: matrix_shards/big_matrix_shard_1.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrix_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([10000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(10000, 20000), dtype=float32
  Converting to 4D format...
    2D (10000, 20000) -> 4D (1, 1, 10000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 10000 × 20000
    Wrote 200,000,000 float32 elements
  File saved successfully
  File size: 800,000,020 bytes
  Expected size: 800,000,020 bytes
  ✓ File size verification passed
  Memory usage: 762.94 MB
  Save completed: /dev/shm/matrix_shards/big_matrix_shard_1.bin
  Added RAM path to file list
Waiting for ACKs from 0 remote nodes...
✅ All ACKs received!
Distribution complete: 2 shards saved and distributed

======================================================================
🏁 INITIALIZATION COMPLETE
======================================================================
✅ Cluster matrix initialization successful!
   - Total nodes configured: 2
   - Matrix handling mode: Split
   - Backends: ['llama', 'llama']
   - CPU/GPU selections: [True, True]

📦 Creating and distributing matrix B (split=False)...
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 2 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrix
   Split Matrix: False
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 1 unique nodes...
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 1

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.100

======================================================================
🧮 MATRIX DISTRIBUTION PHASE
======================================================================
   Matrix file exists: True
   Split matrix mode: False

📦 CASE 3: DISTRIBUTING FULL MATRIX (NO SPLITTING)
   Processing steps:
   1. Save full matrix in binary format
   2. Distribute to all nodes
Preparing full matrix: big_matrix.bin
Local paths - DISK: matrix_shards/big_matrix.bin, RAM: /dev/shm/matrix_shards/big_matrix.bin
Loading matrix from: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/big_matrix.pt
Matrix loaded - Shape: torch.Size([20000, 20000])
Saving to local storage...
Saving matrix to binary file: matrix_shards/big_matrix.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([20000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(20000, 20000), dtype=float32
  Converting to 4D format...
    2D (20000, 20000) -> 4D (1, 1, 20000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 20000 × 20000
    Wrote 400,000,000 float32 elements
  File saved successfully
  File size: 1,600,000,020 bytes
  Expected size: 1,600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1525.88 MB
  Save completed: matrix_shards/big_matrix.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrix.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([20000, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(20000, 20000), dtype=float32
  Converting to 4D format...
    2D (20000, 20000) -> 4D (1, 1, 20000, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 20000 × 20000
    Wrote 400,000,000 float32 elements
  File saved successfully
  File size: 1,600,000,020 bytes
  Expected size: 1,600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 1525.88 MB
  Save completed: /dev/shm/matrix_shards/big_matrix.bin
Remote paths - RAM: /dev/shm/matrix_shards/big_matrix.bin, DISK: /home/rino/Desktop/Open_Cluster_AI_Station_beta/matrix_shards/big_matrix.bin
Distributing to 0 remote node(s)...
✅ All ACKs received!
Full matrix distribution completed
Total file paths tracked: 2

======================================================================
🏁 INITIALIZATION COMPLETE
======================================================================
✅ Cluster matrix initialization successful!
   - Total nodes configured: 2
   - Matrix handling mode: Full
   - Backends: ['llama', 'llama']
   - CPU/GPU selections: [True, True]

============================================================
🧮 PERFORMING CLUSTER MATRIX OPERATION
============================================================
Operation: MatrixA @ MatrixB.T
Nodes: 2
Backends: ['llama', 'llama']
GPU usage: [True, True]

============================================================
🚀 STARTING CLUSTER OPERATION
============================================================
Matrix A: big_matrix
Matrix B: big_matrix
Operation: mul
Transpose A: False, Transpose B: True
Send back result: False
Number of shards: 2

📊 Result base: big_matrixxbig_matrix (send_back=False)
⚠️  Could not validate shard 0: name '_infer_rows_cols_from_bin' is not defined
⚠️  Could not validate shard 1: name '_infer_rows_cols_from_bin' is not defined
✅ Shard count validation complete
🧹 Removed stale result: /dev/shm/matrix_shards/big_matrixxbig_matrix_shard_1.bin
🧹 Removed stale result: /dev/shm/matrix_shards/big_matrixxbig_matrix_shard_0.bin

📤 DISTRIBUTING OPERATIONS TO NODES
----------------------------------------

Processing shard 0:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #0)
  Matrix A path: /dev/shm/matrix_shards/big_matrix_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrix.bin
  Final transpose flags - A: false, B: false
  Send back result: No (keep distributed)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 1

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Matrix A path: /dev/shm/matrix_shards/big_matrix_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrix.bin
  Final transpose flags - A: false, B: false
  Send back result: No (keep distributed)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100
  Incremented GPU counter for node 192.168.2.100 to 2

⏳ WAITING FOR ACKS FROM NODES (2)
✅ Received ACK 1/2
✅ Received ACK 2/2
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Result base name: big_matrixxbig_matrix
Operation time: 8.35 seconds
======================================================================
🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM
======================================================================

📋 VALIDATING NODE CONFIGURATION...
✅ Node configuration validated: 2 nodes configured
✅ Percentage distribution validated: 1.000000

🌐 CONFIGURING NETWORK SETTINGS...
   Head Node Ethernet IP: 192.168.2.100
   Head Node WiFi IP: 192.168.50.113
   Head Node Ports: PULL=7779, PUSH=7780
   Worker Node Ports: PULL=5557, PUSH=5558
   Cluster Barrier Port: 7790

📁 CONFIGURING STORAGE PATHS...
   Local Paths:
     - RAM Results: /dev/shm/matrix_results/
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/
   Remote Paths:
     - Disk Folder: matrix_shards/
     - RAM Folder: /dev/shm/matrix_shards/
     - RAM Results: /dev/shm/matrix_results/
     - Project Dir: /home/rino/Desktop/Open_Cluster_AI_Station_beta/

📊 INITIALIZING INSTANCE VARIABLES...
   Matrix Name: big_matrixxbig_matrix
   Split Matrix: True
   Dimension: 0

📂 CREATING LOCAL DIRECTORIES...
✅ All required directories already exist

🔌 SETTING UP ZEROMQ CONNECTIONS...
   Connecting to 1 unique nodes...
   ✅ Connected to worker WiFi 192.168.3.13:5557
   ✅ Connected to head node (self) 192.168.2.100:7779
   Total sockets in pool: 1

🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...
✅ ACK receiver already exists on port 7790

📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...
   Sending command: mkdir -p matrix_shards/ /dev/shm/matrix_shards/ /dev/shm/matrix_results/
   ✅ Directory creation command sent to 192.168.2.100

======================================================================
🧮 MATRIX DISTRIBUTION PHASE
======================================================================
   Matrix file exists: False
   Split matrix mode: True

🔍 CASE 2: LOADING EXISTING DISTRIBUTED MATRIX SHARDS
   Attempting to load pre-existing shards...
Loading cluster matrix shards: big_matrixxbig_matrix
Number of nodes/shard locations: 2
Checking for existing shards in RAM: /dev/shm/matrix_shards/big_matrixxbig_matrix_shard_0.bin
Found existing matrix shards in local RAM
  Shard 0: Using existing RAM path
  Shard 1: Using existing RAM path

Matrix shard loading complete
Total shard paths tracked: 2

======================================================================
🏁 INITIALIZATION COMPLETE
======================================================================
✅ Cluster matrix initialization successful!
   - Total nodes configured: 2
   - Matrix handling mode: Split
   - Backends: ['llama', 'llama']
   - CPU/GPU selections: [True, True]

✅ Cluster operation completed
Result name: big_matrixxbig_matrix
Cluster operation time: 8.35 seconds

============================================================
🔍 PYTORCH REFERENCE (SINGLE NODE)
============================================================

Matrix A shape: torch.Size([20000, 20000])
Matrix B shape: torch.Size([20000, 20000])
Matrix A sample (5x5):
tensor([[0.5822, 0.4205, 0.3506, 0.7201, 0.6890],
        [0.8598, 0.2469, 0.4321, 0.2997, 0.8340],
        [0.2097, 0.0890, 0.8809, 0.5067, 0.9566],
        [0.4470, 0.5123, 0.3698, 0.0967, 0.0183],
        [0.3265, 0.1357, 0.2475, 0.8788, 0.5820]])

Reference result shape: torch.Size([20000, 20000])
Single-node PyTorch computation time: 22.74s
First 2500 elements of reference result:
tensor([6642.2144, 4998.1240, 4964.3613,  ..., 5005.1006, 4977.2671,
        5025.6562])

Reference result sample (5x5):
tensor([[6642.2144, 4998.1240, 4964.3613, 4969.4556, 4994.3457],
        [4998.1240, 6641.9399, 4992.1772, 4975.1665, 5021.4575],
        [4964.3613, 4992.1772, 6628.5098, 4965.4482, 4985.1729],
        [4969.4556, 4975.1665, 4965.4482, 6643.5278, 4986.7500],
        [4994.3457, 5021.4575, 4985.1729, 4986.7500, 6677.4839]])

============================================================
📥 LOADING CLUSTER RESULT
============================================================

============================================================
🏁 PERFORMANCE COMPARISON
============================================================
CLUSTER OPERATION TIME:      8.3546 seconds
SINGLE NODE PYTORCH TIME:    22.7372 seconds
------------------------------------------------------------
CLUSTER vs SINGLE NODE: 2.72x faster
(ray-conda-env) rino@rino-Z370-HD3:~/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix$ 

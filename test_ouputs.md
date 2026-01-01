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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 9.91 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 6.21 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 2.68 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (4 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

⏳ WAITING FOR ACKS FROM NODES (4)
✅ Received ACK_matrixOp_complete 1/4
✅ Received ACK_matrixOp_complete 2/4
✅ Received ACK_matrixOp_complete 3/4
✅ Received ACK_matrixOp_complete 4/4
✅ All ACKs received!

============================================================
✅ CLUSTER OPERATION COMPLETE
============================================================
Operation time: 1.94 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 12.36 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 9.96 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 4.55 seconds

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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_2.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_3.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 4:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_4.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (5 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 5.21 seconds

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
📤 Sent file big_matrixA_shard_0.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 0 to IP: 192.168.2.101
📤 Sent file big_matrixA_shard_1.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.104

📋 Node shard assignments:
  192.168.2.100 -> shard_0
  192.168.2.101 -> shard_0
  192.168.2.101 -> shard_0
  192.168.2.100 -> shard_1
  192.168.2.100 -> shard_1
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
✅ Matrix B: torch.Size([15000, 20000]) → splitting into 3 shards using percentages [0.5, 0.25, 0.25]
Split sizes for 3 unique shards: [7500, 3750, 3750]
Sum check: 15000 = 15000 ✓
✅ Created 6 B shards (before reordering):
  Node 0 (original): gets B0 torch.Size([7500, 20000])
  Node 1 (original): gets B1 torch.Size([3750, 20000])
  Node 2 (original): gets B2 torch.Size([3750, 20000])
  Node 3 (original): gets B0 torch.Size([7500, 20000])
  Node 4 (original): gets B1 torch.Size([3750, 20000])
  Node 5 (original): gets B2 torch.Size([3750, 20000])
Starting distribution of 6 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_0.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([7500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(7500, 20000), dtype=float32
  Converting to 4D format...
    2D (7500, 20000) -> 4D (1, 1, 7500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 7500 × 20000
    Wrote 150,000,000 float32 elements
  File saved successfully
  File size: 600,000,020 bytes
  Expected size: 600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 572.20 MB
  Save completed: matrix_shards/big_matrixB_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([7500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(7500, 20000), dtype=float32
  Converting to 4D format...
    2D (7500, 20000) -> 4D (1, 1, 7500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 7500 × 20000
    Wrote 150,000,000 float32 elements
  File saved successfully
  File size: 600,000,020 bytes
  Expected size: 600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 572.20 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_1.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3750, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3750, 20000), dtype=float32
  Converting to 4D format...
    2D (3750, 20000) -> 4D (1, 1, 3750, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3750 × 20000
    Wrote 75,000,000 float32 elements
  File saved successfully
  File size: 300,000,020 bytes
  Expected size: 300,000,020 bytes
  ✓ File size verification passed
  Memory usage: 286.10 MB
  Save completed: matrix_shards/big_matrixB_shard_1.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixB_shard_1.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixB_shard_1.bin
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_2.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3750, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3750, 20000), dtype=float32
  Converting to 4D format...
    2D (3750, 20000) -> 4D (1, 1, 3750, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3750 × 20000
    Wrote 75,000,000 float32 elements
  File saved successfully
  File size: 300,000,020 bytes
  Expected size: 300,000,020 bytes
  ✓ File size verification passed
  Memory usage: 286.10 MB
  Save completed: matrix_shards/big_matrixB_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file big_matrixB_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/big_matrixB_shard_2.bin
Processing shard 3 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_3.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_3.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([7500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(7500, 20000), dtype=float32
  Converting to 4D format...
    2D (7500, 20000) -> 4D (1, 1, 7500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 7500 × 20000
    Wrote 150,000,000 float32 elements
  File saved successfully
  File size: 600,000,020 bytes
  Expected size: 600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 572.20 MB
  Save completed: matrix_shards/big_matrixB_shard_3.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([7500, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(7500, 20000), dtype=float32
  Converting to 4D format...
    2D (7500, 20000) -> 4D (1, 1, 7500, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 7500 × 20000
    Wrote 150,000,000 float32 elements
  File saved successfully
  File size: 600,000,020 bytes
  Expected size: 600,000,020 bytes
  ✓ File size verification passed
  Memory usage: 572.20 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_3.bin
  Added RAM path to file list
Processing shard 4 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/big_matrixB_shard_4.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/big_matrixB_shard_4.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3750, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3750, 20000), dtype=float32
  Converting to 4D format...
    2D (3750, 20000) -> 4D (1, 1, 3750, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3750 × 20000
    Wrote 75,000,000 float32 elements
  File saved successfully
  File size: 300,000,020 bytes
  Expected size: 300,000,020 bytes
  ✓ File size verification passed
  Memory usage: 286.10 MB
  Save completed: matrix_shards/big_matrixB_shard_4.bin
Saving matrix to binary file: /dev/shm/matrix_shards/big_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3750, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3750, 20000), dtype=float32
  Converting to 4D format...
    2D (3750, 20000) -> 4D (1, 1, 3750, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3750 × 20000
    Wrote 75,000,000 float32 elements
  File saved successfully
  File size: 300,000,020 bytes
  Expected size: 300,000,020 bytes
  ✓ File size verification passed
  Memory usage: 286.10 MB
  Save completed: /dev/shm/matrix_shards/big_matrixB_shard_4.bin
  Added RAM path to file list
Processing shard 5 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/big_matrixB_shard_5.bin
Saving matrix to binary file: matrix_shards/big_matrixB_shard_5.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([3750, 20000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(3750, 20000), dtype=float32
  Converting to 4D format...
    2D (3750, 20000) -> 4D (1, 1, 3750, 20000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 3750 × 20000
    Wrote 75,000,000 float32 elements
  File saved successfully
  File size: 300,000,020 bytes
  Expected size: 300,000,020 bytes
  ✓ File size verification passed
  Memory usage: 286.10 MB
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
  Node 1 (192.168.2.101): assigned shard_0
    Sending to remote 192.168.2.101: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/big_matrixA_shard_0.bin" "/dev/shm/matrix_shards/big_matrixA_shard_0.bin"
  Node 2 (192.168.2.101): assigned shard_0
  Node 3 (192.168.2.100): assigned shard_1
  Node 4 (192.168.2.100): assigned shard_1
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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_0.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_1.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_2.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_3.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 4:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #2)
  Next GPU for this node will be: #3
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_4.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 5:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/big_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/big_matrixB_shard_5.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 17.71 seconds

📊 Result base: big_matrixBxbig_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/big_matrixBxbig_matrixA_combined.bin
  Original dims: [1, 1, 10000, 15000]
  Result tensor shape: torch.Size([10000, 15000]), size: 600,000,000 bytes
  Data range: [4821.820312, 5167.922852]
✅ Shapes match: torch.Size([10000, 15000])
Max absolute difference:  9.570312e-02
Mean absolute difference: 8.501702e-03
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
📤 Sent file mid_matrixA_shard_0.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 0 to IP: 192.168.2.101
📤 Sent file mid_matrixA_shard_1.bin to 192.168.2.104
✅ Received ACK 1/1
✅ All ACKs received!
Sent shard 1 to IP: 192.168.2.104

📋 Node shard assignments:
  192.168.2.100 -> shard_0
  192.168.2.101 -> shard_0
  192.168.2.101 -> shard_0
  192.168.2.100 -> shard_1
  192.168.2.100 -> shard_1
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
✅ Matrix B: torch.Size([9000, 7000]) → splitting into 3 shards using percentages [0.5, 0.25, 0.25]
Split sizes for 3 unique shards: [4500, 2250, 2250]
Sum check: 9000 = 9000 ✓
✅ Created 6 B shards (before reordering):
  Node 0 (original): gets B0 torch.Size([4500, 7000])
  Node 1 (original): gets B1 torch.Size([2250, 7000])
  Node 2 (original): gets B2 torch.Size([2250, 7000])
  Node 3 (original): gets B0 torch.Size([4500, 7000])
  Node 4 (original): gets B1 torch.Size([2250, 7000])
  Node 5 (original): gets B2 torch.Size([2250, 7000])
Starting distribution of 6 shards to 3 unique nodes
Processing shard 0 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_0.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_0.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4500, 7000), dtype=float32
  Converting to 4D format...
    2D (4500, 7000) -> 4D (1, 1, 4500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4500 × 7000
    Wrote 31,500,000 float32 elements
  File saved successfully
  File size: 126,000,020 bytes
  Expected size: 126,000,020 bytes
  ✓ File size verification passed
  Memory usage: 120.16 MB
  Save completed: matrix_shards/mid_matrixB_shard_0.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4500, 7000), dtype=float32
  Converting to 4D format...
    2D (4500, 7000) -> 4D (1, 1, 4500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4500 × 7000
    Wrote 31,500,000 float32 elements
  File saved successfully
  File size: 126,000,020 bytes
  Expected size: 126,000,020 bytes
  ✓ File size verification passed
  Memory usage: 120.16 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Added RAM path to file list
Processing shard 1 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_1.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_1.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2250, 7000), dtype=float32
  Converting to 4D format...
    2D (2250, 7000) -> 4D (1, 1, 2250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2250 × 7000
    Wrote 15,750,000 float32 elements
  File saved successfully
  File size: 63,000,020 bytes
  Expected size: 63,000,020 bytes
  ✓ File size verification passed
  Memory usage: 60.08 MB
  Save completed: matrix_shards/mid_matrixB_shard_1.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixB_shard_1.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixB_shard_1.bin
Processing shard 2 for node 192.168.2.101
  Remote node 192.168.2.101: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_2.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_2.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2250, 7000), dtype=float32
  Converting to 4D format...
    2D (2250, 7000) -> 4D (1, 1, 2250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2250 × 7000
    Wrote 15,750,000 float32 elements
  File saved successfully
  File size: 63,000,020 bytes
  Expected size: 63,000,020 bytes
  ✓ File size verification passed
  Memory usage: 60.08 MB
  Save completed: matrix_shards/mid_matrixB_shard_2.bin
  Step 2: Sending file to remote node 192.168.2.101
📤 Sent file mid_matrixB_shard_2.bin to 192.168.2.101
✅ Received ACK 1/1
✅ All ACKs received!
  Step 3: Sending copy command to remote
  Added remote RAM path to file list: /dev/shm/matrix_shards/mid_matrixB_shard_2.bin
Processing shard 3 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_3.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_3.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4500, 7000), dtype=float32
  Converting to 4D format...
    2D (4500, 7000) -> 4D (1, 1, 4500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4500 × 7000
    Wrote 31,500,000 float32 elements
  File saved successfully
  File size: 126,000,020 bytes
  Expected size: 126,000,020 bytes
  ✓ File size verification passed
  Memory usage: 120.16 MB
  Save completed: matrix_shards/mid_matrixB_shard_3.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_3.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([4500, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(4500, 7000), dtype=float32
  Converting to 4D format...
    2D (4500, 7000) -> 4D (1, 1, 4500, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 4500 × 7000
    Wrote 31,500,000 float32 elements
  File saved successfully
  File size: 126,000,020 bytes
  Expected size: 126,000,020 bytes
  ✓ File size verification passed
  Memory usage: 120.16 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_3.bin
  Added RAM path to file list
Processing shard 4 for node 192.168.2.100
  Head node: Saving to DISK=matrix_shards/mid_matrixB_shard_4.bin
  Head node: Saving to RAM=/dev/shm/matrix_shards/mid_matrixB_shard_4.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2250, 7000), dtype=float32
  Converting to 4D format...
    2D (2250, 7000) -> 4D (1, 1, 2250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2250 × 7000
    Wrote 15,750,000 float32 elements
  File saved successfully
  File size: 63,000,020 bytes
  Expected size: 63,000,020 bytes
  ✓ File size verification passed
  Memory usage: 60.08 MB
  Save completed: matrix_shards/mid_matrixB_shard_4.bin
Saving matrix to binary file: /dev/shm/matrix_shards/mid_matrixB_shard_4.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2250, 7000), dtype=float32
  Converting to 4D format...
    2D (2250, 7000) -> 4D (1, 1, 2250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2250 × 7000
    Wrote 15,750,000 float32 elements
  File saved successfully
  File size: 63,000,020 bytes
  Expected size: 63,000,020 bytes
  ✓ File size verification passed
  Memory usage: 60.08 MB
  Save completed: /dev/shm/matrix_shards/mid_matrixB_shard_4.bin
  Added RAM path to file list
Processing shard 5 for node 192.168.2.104
  Remote node 192.168.2.104: Beginning distribution
  Step 1: Saving locally to matrix_shards/mid_matrixB_shard_5.bin
Saving matrix to binary file: matrix_shards/mid_matrixB_shard_5.bin
  Converting input to numpy array...
    Input is PyTorch tensor: shape=torch.Size([2250, 7000]), dtype=torch.float32, device=cpu
    Converted to CPU float32 numpy array
  Final numpy array: shape=(2250, 7000), dtype=float32
  Converting to 4D format...
    2D (2250, 7000) -> 4D (1, 1, 2250, 7000)
  Writing binary file...
    Wrote ndim: 4
    Dimensions: 1 × 1 × 2250 × 7000
    Wrote 15,750,000 float32 elements
  File saved successfully
  File size: 63,000,020 bytes
  Expected size: 63,000,020 bytes
  ✓ File size verification passed
  Memory usage: 60.08 MB
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
  Node 1 (192.168.2.101): assigned shard_0
    Sending to remote 192.168.2.101: cp "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/mid_matrixA_shard_0.bin" "/dev/shm/matrix_shards/mid_matrixA_shard_0.bin"
  Node 2 (192.168.2.101): assigned shard_0
  Node 3 (192.168.2.100): assigned shard_1
  Node 4 (192.168.2.100): assigned shard_1
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
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_0.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 1:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_1.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 2:
  Node: 192.168.2.101
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_0.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_2.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.101

Processing shard 3:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #1)
  Next GPU for this node will be: #2
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_3.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 4:
  Node: 192.168.2.100
  Backend: llama
  Use GPU: True (GPU #2)
  Next GPU for this node will be: #3
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_4.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.100

Processing shard 5:
  Node: 192.168.2.104
  Backend: llama
  Use GPU: True (GPU #0)
  Next GPU for this node will be: #1
  Matrix A path: /dev/shm/matrix_shards/mid_matrixA_shard_1.bin
  Matrix B path: /dev/shm/matrix_shards/mid_matrixB_shard_5.bin
  Final transpose flags - A: false, B: false
  Send back result: Yes (-6 shards will be combined)
  Sending command to node...
  ✅ Command sent to node 192.168.2.104

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
Operation time: 3.83 seconds

📊 Result base: mid_matrixBxmid_matrixA (send_back=True)
✅ Received ACK_combined_matrix_saved 1/1
✅ All ACKs received!
✅ Loaded /dev/shm/matrix_shards/mid_matrixBxmid_matrixA_combined.bin
  Original dims: [1, 1, 5000, 9000]
  Result tensor shape: torch.Size([5000, 9000]), size: 180,000,000 bytes
  Data range: [1656.287842, 1856.252319]
✅ Shapes match: torch.Size([5000, 9000])
Max absolute difference:  4.260254e-02
Mean absolute difference: 3.811591e-03
✅ Results match within tolerance (0.15)
Elements with > 0.15 difference: 0/45000000 (0.00%)
(ray-conda-env) rino@rino-Z370-HD3:~/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix$ 

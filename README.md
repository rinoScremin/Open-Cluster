SYSTEM 1 IS WORKING WITH TRANFORMER GIVING THE CORRECT SHAPE AND DATA MAKING SOME CHANGES TO SYSTEM2 CODE IS NOW BACK UNDER CONSTRUCTION 

# 🚀 **The Vision: Democratizing AI Compute**

**We built this because we were tired of seeing perfectly good hardware collecting dust.** The AI revolution shouldn't require $10,000 GPUs and HPC expertise. **This system proves you can run massive models on whatever hardware you already have** - gaming PCs, old laptops, office workstations, even integrated graphics.

## 🎯 **What This Means For You:**

**If you have:**
- An old gaming PC with an AMD/NVIDIA GPU
- A MacBook with Apple Silicon  
- A Raspberry Pi cluster
- Office computers after hours
- Friends with PCs who want to contribute

**You now have:**
- A distributed supercomputer
- The ability to run models that previously needed A100s
- A system that scales by just adding more devices
- **Zero configuration complexity** - it just works

## 🔓 **Why We're Open Sourcing This:**

1. **Break the hardware monopoly** - AI shouldn't require specific NVIDIA GPUs
2. **Reduce e-waste** - Old hardware has value when it computes together
3. **Lower barriers** - Students, researchers, startups can afford distributed AI
4. **Community innovation** - Let's build the future of distributed computing together

## 🌱 **This is Version 1.0 - The Foundation**

The architecture is solid, it works, and it's **already useful**. But this is just the beginning. With community contributions, we can:
- Add more backends (ROCm, SYCL, WebGPU)
- Improve scheduling algorithms  
- Add containerization/K8s support
- Create web dashboards
- Build transformer-specific optimizations

## 🤝 **Join The Movement**

**If you believe:**
- AI should be accessible to everyone, not just big tech
- Old hardware deserves a second life
- Distributed systems should be simple to use
- Community-driven software beats closed ecosystems

**Then this project is for you.** Let's build the distributed computing platform that actually works for real people with real hardware. No PhD required, no $10k GPU needed - just the computers you already have, working together.

**This isn't just code. It's a statement: Your hardware is enough.** 🖥️💻🖥️💻=🚀


# 🔥 Cluster Matrix - Distributed Computation System

## 🎯 **What This Is**
A **hybrid distributed computing system** that turns ANY group of computers into a supercomputer. Mix CPUs, GPUs (NVIDIA/AMD/Intel), Apple Silicon - **all working together** on massive matrix operations.


        
                    ######################################## USING CLUSTER_MATRIX_V1 ########################################

IP_list = [
    "192.168.2.100",
    "192.168.2.100",
    "192.168.2.101",
    "192.168.2.104",
]

# List the IP addresses of the remote PCs to be used in the cluster.
#
# If you list the same IP more than once (e.g. "192.168.2.100" twice),
# the C++ backend will check whether the system has separate hardware
# available for each node.
#
# Example: "192.168.2.100"
# This system has 2 GPUs:
#   - AMD RX 5500
#   - AMD RX 6400
#
# In this case:
#   shard 1 → GPU #1 (RX 5500)
#   shard 2 → GPU #2 (RX 6400)
#
# If "192.168.2.100" is listed a third time and no additional GPU is
# available, the shard will fall back to CPU BLAS.
#
# NOTE:
# CPU BLAS may not support ADD operations correctly on some systems.
# If you encounter issues, CPU BLAS for add operations may need to be disabled.
#
# "192.168.2.101" is a laptop with an integrated GPU / APU.
# The next shard will run on that GPU.
# If listed again, additional shards will run on CPU BLAS
# (this laptop only has one GPU).
#
# "192.168.2.104" is an Intel(R) Core(TM) i5-6500 @ 3.20GHz system.
# This machine has no GPU and will always use CPU BLAS.

percentages = [0.35, 0.35, 0.15, 0.15]

# Define how the matrix is distributed across the cluster:
#   node 1: 35% of matrix B
#   node 2: 35% of matrix B
#   node 3: 15% of matrix B
#   node 4: 15% of matrix B

CPU_GPU_select_list = [True, True, True, True]

# Enables the backend acceleration that was compiled for each node.
# If set to False, the shard will be processed on CPU only,
# without acceleration (e.g. no BLAS).

backend_select_list = ["llama", "llama", "llama", "llama"]

# Available backends:
#
# "llama" → GGML backend (accelerated when CPU_GPU_select_list is True)
# "torch" → PyTorch backend
# "opencl" → Custom OpenCL backend
#
# You can mix and match backends depending on your hardware:
#
# backend_select_list = ["torch", "torch", "torch", "torch"]
# backend_select_list = ["llama", "torch", "llama", "torch"]
# backend_select_list = ["opencl", "torch", "llama", "opencl"]
#
# This allows you to support unusual or custom hardware.
# For example, if you build a custom accelerator using Raspberry Pis,
# you can implement an OpenCL backend for it.

post_attn_ln_w = torch.load(post_attn_ln_path, map_location="cpu")

if post_attn_ln_w.ndim != 1:
    raise ValueError(
        f"post_attention_layernorm_weight must be 1D, got {tuple(post_attn_ln_w.shape)}"
    )

if post_attn_ln_w.shape[0] != residual.shape[1]:
    raise ValueError(
        f"post_attention_layernorm_weight hidden mismatch: "
        f"weight={post_attn_ln_w.shape[0]} hidden={residual.shape[1]}"
    )

mlp_in = self.rms_norm(residual, post_attn_ln_w)  # [1, hidden]
mlp_in_col = mlp_in.t().contiguous()               # [hidden, 1]

# NOTE:
# cluster_matrix_v1.py only converts PyTorch tensors into a format
# that the cluster can use.
#
# Any "special" tensor operations (e.g. .contiguous(), transpose, reshape)
# MUST be performed in PyTorch before passing the tensor to the cluster.

mlp_in_cluster = cluster_matrix(
    matrix_file_path=mlp_in_col,  # If passing a torch.Tensor, you must provide a name
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=False,  # Matrix A (full matrix), not split
    dim=1,               # Dimension used for combining results
    auto_set_up=[1, "save"],
    matrix_name=f"layer{0}_mlp_in",
)

mlp_gate_path = f"{model_matrix_fold_dir}layers_{0}_mlp_gate_proj_weight.pt"

# Here we pass a file path instead of a torch.Tensor.
# In this case, you do NOT need to provide a matrix_name —
# cluster_matrix will automatically use the file name
# (e.g. "layers_0_mlp_gate_proj_weight").

mlp_gate_cluster = cluster_matrix(
    matrix_file_path=mlp_gate_path,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=True,   # Matrix B (sharded)
    dim=1,
    auto_set_up=[1, "save"],
)

mlp_gate_cluster = cluster_matrix(
    matrix_file_path=mlp_gate_path,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=True,   # Matrix B (sharded)
    dim=1,
    auto_set_up=[1, "load"], # after cashing the matrix shards you can just load then using the load function 
    # strongly reconmend first cashing the tensors you need for what ever you are doing then using the 'load' function when you can 
    # some times it might not be possable to be albe to pre-cash a tensor (for exsample in the case of token embeddings you would also need to use       # 'save' do to the fact you can not cash the matrix 
)

mlp_gate_cluster = cluster_matrix(
    matrix_file_path=mlp_gate_path,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=True,   # Matrix B (sharded)
    dim=1,
    auto_set_up=[1, "load"],  # After caching the matrix shards, you can load them directly
)

# Below is an example demonstrating how the `cluster_matrix` class should be used.

attn_q_proj_path = f"{self.model_matrix_fold_dir}layers_{0}_self_attn_q_proj_weight.pt"
attn_k_proj_path = f"{self.model_matrix_fold_dir}layers_{0}_self_attn_k_proj_weight.pt"
attn_v_proj_path = f"{self.model_matrix_fold_dir}layers_{0}_self_attn_v_proj_weight.pt"
attn_o_proj_path = f"{self.model_matrix_fold_dir}layers_{0}_self_attn_o_proj_weight.pt"

input_layernorm_weight_path = f"{self.model_matrix_fold_dir}layers_{0}_input_layernorm_weight.pt"
input_layernorm_weight = torch.load(input_layernorm_weight_path)

# Apply RMSNorm locally before sending to the cluster
x = self.rms_norm(input_token_embeddings, input_layernorm_weight)
x = x.unsqueeze(1)

# Matrix A (full, not sharded)
x = cluster_matrix(
    matrix_file_path=x,
    node_IP_list=self.IP_list,
    CPU_GPU_select_list=self.CPU_GPU_select_list,
    node_percentages=self.percentages,
    back_end_select_list=self.backend_select_list,
    split_matrix=False,
    dim=1,
    auto_set_up=[1, "save"],
    matrix_name="input_token_embeddings",
)

# Matrix B (sharded weights, pre-cached)
q = cluster_matrix(
    matrix_file_path=attn_q_proj_path,
    node_IP_list=self.IP_list,
    CPU_GPU_select_list=self.CPU_GPU_select_list,
    node_percentages=self.percentages,
    back_end_select_list=self.backend_select_list,
    split_matrix=True,
    dim=1,
    auto_set_up=[1, "load"],
)

k = cluster_matrix(
    matrix_file_path=attn_k_proj_path,
    node_IP_list=self.IP_list,
    CPU_GPU_select_list=self.CPU_GPU_select_list,
    node_percentages=self.percentages,
    back_end_select_list=self.backend_select_list,
    split_matrix=True,
    dim=1,
    auto_set_up=[1, "load"],
)

v = cluster_matrix(
    matrix_file_path=attn_v_proj_path,
    node_IP_list=self.IP_list,
    CPU_GPU_select_list=self.CPU_GPU_select_list,
    node_percentages=self.percentages,
    back_end_select_list=self.backend_select_list,
    split_matrix=True,
    dim=1,
    auto_set_up=[1, "load"],
)

# Perform distributed matrix multiplication
q_flat = x.cluster_shard_operation(q, True, False, True)
k_flat = x.cluster_shard_operation(k, True, False, True)
v_flat = x.cluster_shard_operation(v, True, False, True)

# The `cluster_shard_operation` method performs the distributed operation
# (e.g. matrix multiply, add, subtract).
#
# By default, the result is sent back and returned as a PyTorch tensor,
# allowing further local processing using PyTorch.
#
# Example (NOTE: not valid for this case):
# q_flat = x.cluster_shard_operation(q, True, False, True, "add")
#
# The above example would only be valid if both matrices were split and
# compatible for element-wise addition.


# ======================= MATRIX ADDITION EXAMPLE =======================

# ----------------- FILE PATHS (dim = 1 split test) -----------------
big_test_matrix_pathA_T = "model_matrixs/big_matrixA_T.pt"
big_test_matrix_pathB_T = "model_matrixs/big_matrixB_T.pt"

mid_test_matrix_pathA_T = "model_matrixs/mid_matrixA_T.pt"
mid_test_matrix_pathB_T = "model_matrixs/mid_matrixB_T.pt"

small_test_matrix_pathA_T = "model_matrixs/small_matrixA_T.pt"
small_test_matrix_pathB_T = "model_matrixs/small_matrixB_T.pt"


# Create reference result for validation
big_matrixA = torch.load(big_test_matrix_pathA_T)
big_c_ref = torch.add(big_matrixA, big_matrixA)
torch.save(big_c_ref, "model_matrixs/big_c_ref.pt")


############################# TESTING CLUSTER MATRIX OPERATIONS (SYSTEM 1) #############################

# ----------------- CLUSTER TEST (BIG MATRIX) dim = 0 split/join -----------------

IP_list = ["192.168.2.100", "192.168.2.100", "192.168.2.101", "192.168.2.104"]
percentages = [0.25, 0.25, 0.25, 0.25]
CPU_GPU_select_list = [True, True, True, False]
backend_select_list = ["llama", "llama", "llama", "llama"]

# ----------------- CLUSTER MATRICES -----------------

big_new_matrixA = cluster_matrix(
    big_test_matrix_pathA_T,
    IP_list,
    CPU_GPU_select_list,
    percentages,
    backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[1, "load"],
)

big_new_matrixB = cluster_matrix(
    big_test_matrix_pathA_T,
    IP_list,
    CPU_GPU_select_list,
    percentages,
    backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[1, "load"],
)

# Perform distributed matrix addition
big_new_matrixC = big_new_matrixA.cluster_shard_operation(
    big_new_matrixB,
    False,
    True,
    True,
    "add",
)

# For matrix addition, both Matrix A and Matrix B must be split.
# The operation is performed as:
#   matrixA_shard_i + matrixB_shard_i = matrixC_shard_i



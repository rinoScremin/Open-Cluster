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

---

# USING `cluster_matrix_v1 system #1`

This document demonstrates how to use `cluster_matrix_v1` system #1 for **distributed matrix operations** across multiple machines, GPUs, and backends, including **LLM inference workloads** (e.g. attention + MLP layers).

---

## 📡 Cluster Configuration

### Node IP List

```python
IP_list = [
    "192.168.2.100",
    "192.168.2.100",
    "192.168.2.101",
    "192.168.2.104",
]
```

#### How IP duplication works

If the **same IP is listed multiple times**, the C++ backend will attempt to bind each entry to **separate hardware** on that machine.

Example:

```
192.168.2.100
 ├─ GPU 0 → shard 1
 ├─ GPU 1 → shard 2
 └─ CPU BLAS → shard 3 (fallback if no GPU available)
```

Notes:

* If no additional GPU is available, the shard **falls back to CPU BLAS**
* CPU BLAS **may not support ADD operations correctly** on some systems
* If you encounter incorrect ADD results, disable CPU BLAS for those nodes

Hardware examples:

* `192.168.2.101` → Laptop with integrated GPU / APU

  * First shard → GPU
  * Additional shards → CPU BLAS

* `192.168.2.104` → Intel i5-6500

  * No GPU
  * Always uses CPU BLAS

---

### Matrix Split Percentages

```python
percentages = [0.35, 0.35, 0.15, 0.15]
```

Defines how **Matrix B** is distributed:

| Node   | Percentage |
| ------ | ---------- |
| Node 1 | 35%        |
| Node 2 | 35%        |
| Node 3 | 15%        |
| Node 4 | 15%        |

---

### Backend Acceleration Selection

```python
CPU_GPU_select_list = [True, True, True, True]
```

* `True` → use compiled backend acceleration
* `False` → CPU-only (no BLAS / GPU acceleration)

---

### Backend Type Selection

```python
backend_select_list = ["llama", "llama", "llama", "llama"]
```

Available backends:

| Backend  | Description                        |
| -------- | ---------------------------------- |
| `llama`  | GGML backend (CPU/GPU accelerated) |
| `torch`  | PyTorch backend                    |
| `opencl` | Custom OpenCL backend              |

You can **mix backends per node**:

```python
backend_select_list = ["llama", "torch", "opencl", "llama"]
```

This allows support for **custom or experimental hardware**, including OpenCL-based accelerators.

---

## 🧠 RMSNorm (Local Preprocessing)

```python
post_attn_ln_w = torch.load(post_attn_ln_path, map_location="cpu")

if post_attn_ln_w.ndim != 1:
    raise ValueError("LayerNorm weight must be 1D")

if post_attn_ln_w.shape[0] != residual.shape[1]:
    raise ValueError("Hidden size mismatch")

mlp_in = self.rms_norm(residual, post_attn_ln_w)
mlp_in_col = mlp_in.t().contiguous()
```

⚠️ **IMPORTANT**

`cluster_matrix_v1` does **not** perform tensor reshaping.

All operations like:

* `.contiguous()`
* `.transpose()`
* `.reshape()`

**must be done in PyTorch before sending the tensor to the cluster.**

---

## 📦 Creating Cluster Matrices

### Matrix A (Full / Not Sharded)

```python
mlp_in_cluster = cluster_matrix(
    matrix_file_path=mlp_in_col,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=False,
    dim=1,
    auto_set_up=[1, "save"],
    matrix_name="layer0_mlp_in",
)
```

---

### Matrix B (Sharded Weights)

```python
mlp_gate_cluster = cluster_matrix(
    matrix_file_path=mlp_gate_path,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=1,
    auto_set_up=[1, "load"],
)
```

💡 **Recommendation**

* Cache all large weight tensors once using `"save"`
* Reuse them later using `"load"`
* Some tensors (e.g. token embeddings) **cannot be cached** and must always use `"save"`

---

## ⚙️ Distributed Attention Example

```python
x = self.rms_norm(input_token_embeddings, input_layernorm_weight)
x = x.unsqueeze(1)

x = cluster_matrix(
    matrix_file_path=x,
    node_IP_list=IP_list,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=False,
    dim=1,
    auto_set_up=[1, "save"],
    matrix_name="input_token_embeddings",
)
```

```python
q_flat = x.cluster_shard_operation(q, True, False, True)
k_flat = x.cluster_shard_operation(k, True, False, True)
v_flat = x.cluster_shard_operation(v, True, False, True)
```

By default:

* Results are **sent back**
* Returned as **PyTorch tensors**
* Ready for further local processing

---

## ➕ Distributed Matrix Addition Example

### Cluster ADD (Sharded)

```python
big_new_matrixC = big_new_matrixA.cluster_shard_operation(
    big_new_matrixB,
    False,
    True,
    True,
    "add",
)
```

✔️ For ADD:

* **Both matrices must be split**
* Operation is performed shard-wise:

  ```
  C_i = A_i + B_i
  ```

---

## 🖥️ Single-PC / Single-GPU Mode

You can also use `cluster_matrix` **without a cluster**.

Useful if:

* You do not have CUDA
* You only have one GPU
* GPU supports Vulkan / Metal / OpenCL (via GGML)

```python
big_new_matrixA = cluster_matrix(
    matrix_file_path=big_test_matrix_pathA_T,
    node_IP_list=["192.168.2.100"],
    CPU_GPU_select_list=[True],
    node_percentages=[1],
    back_end_select_list=["llama"],
    split_matrix=False,
    dim=0,
    auto_set_up=[1, "save"],
)
```

---

## 🔄 Converting Results Back to PyTorch

After operations, results are stored as binary files:

```
/dev/shm/matrix_shards/*.bin
```

Convert back to PyTorch:

```python
big_new_matrixC.convert_bin_matrix_to_pt("path/to/output_file.bin")
```

---



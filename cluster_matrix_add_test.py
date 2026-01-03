
from cluster_matrix_v1 import cluster_matrix
import torch


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


#######################################------MAIN FUNCTION - TESTING-----######################################  
if __name__ == "__main__":
    
    # ----------------- CREATE MATRICES for dim = 0 split test -----------------
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

    # ----------------- CREATE MATRICES for dim = 1 split test -----------------
    '''
    A_T = np.random.rand(20000, 10000)
    B_T = np.random.rand(20000, 15000)

    A2_T = np.random.rand(7000, 5000)
    B2_T = np.random.rand(7000, 9000)

    A3_T = np.random.rand(500, 1500)
    B3_T = np.random.rand(500, 1000)

    torch.save(torch.tensor(A_T, dtype=torch.float32), 'model_matrixs/big_matrixA_T.pt')
    torch.save(torch.tensor(B_T, dtype=torch.float32), 'model_matrixs/big_matrixB_T.pt')   
    
    torch.save(torch.tensor(A2_T, dtype=torch.float32), 'model_matrixs/mid_matrixA_T.pt')
    torch.save(torch.tensor(B2_T, dtype=torch.float32), 'model_matrixs/mid_matrixB_T.pt')   

    torch.save(torch.tensor(A3_T, dtype=torch.float32), 'model_matrixs/small_matrixA_T.pt')
    torch.save(torch.tensor(B3_T, dtype=torch.float32), 'model_matrixs/small_matrixB_T.pt')
    '''
    
    # ----------------- FILE PATHS for dim = 0 split test-----------------
    big_test_matrix_pathA = 'model_matrixs/big_matrixA.pt'  
    big_test_matrix_pathB = 'model_matrixs/big_matrixB.pt'  

    mid_test_matrix_pathA = 'model_matrixs/mid_matrixA.pt'  
    mid_test_matrix_pathB = 'model_matrixs/mid_matrixB.pt'  

    small_test_matrix_pathA = 'model_matrixs/small_matrixA.pt'  
    small_test_matrix_pathB = 'model_matrixs/small_matrixB.pt'  

    # ----------------- FILE PATHS for dim = 1 split test-----------------
    big_test_matrix_pathA_T = 'model_matrixs/big_matrixA_T.pt'  
    big_test_matrix_pathB_T = 'model_matrixs/big_matrixB_T.pt'  

    mid_test_matrix_pathA_T = 'model_matrixs/mid_matrixA_T.pt'  
    mid_test_matrix_pathB_T = 'model_matrixs/mid_matrixB_T.pt'  

    small_test_matrix_pathA_T = 'model_matrixs/small_matrixA_T.pt'  
    small_test_matrix_pathB_T = 'model_matrixs/small_matrixB_T.pt'  

    
    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
    
    # ----------------- CLUSTER TEST (BIG MATRIX) dim = 0 split/join test-----------------

    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
    percentages = [0.25,0.25,0.25,0.25]  
    CPU_GPU_select_list = [ True, True, True, False ]  
    backend_select_list = ['llama','llama','llama','llama'] 

    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------

    # ----------------- REFERENCE RESULTS for dim = 0 split test-----------------
    
    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    #'''
    big_new_matrixA = cluster_matrix(big_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0)
    big_new_matrixA.convert_to_cluster_matrix_shards()
    big_new_matrixA.save_distribute_matrix_shards_bin()

    big_new_matrixB = cluster_matrix(big_test_matrix_pathA, IP_list, CPU_GPU_select_list, 
                             percentages, backend_select_list, True, 0) 
    big_new_matrixB.convert_to_cluster_matrix_shards()
    big_new_matrixB.save_distribute_matrix_shards_bin()

    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True, 'add')  
    #check_combined_result_values('model_matrixs/big_c_ref.pt',big_new_matrixC)
    


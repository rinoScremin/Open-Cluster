from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
import torch
import numpy as np


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

#######################################------MAIN FUNCTION - TESTING-----######################################  
if __name__ == "__main__":
    
    # ----------------- CREATE MATRICES for dim = 0 split test -----------------
    '''
    A = np.random.rand(10000, 20000)
    B = np.random.rand(15000, 20000)

    A2 = np.random.rand(5000, 7000)
    B2 = np.random.rand(9000, 7000)
    '''
    
    A3 = np.random.rand(1500, 4500)
    B3 = np.random.rand(1000, 4500)

    '''
    torch.save(torch.tensor(A, dtype=torch.float32), 'model_matrixs/big_matrixA.pt')
    torch.save(torch.tensor(B, dtype=torch.float32), 'model_matrixs/big_matrixB.pt')   
    
    torch.save(torch.tensor(A2, dtype=torch.float32), 'model_matrixs/mid_matrixA.pt')
    torch.save(torch.tensor(B2, dtype=torch.float32), 'model_matrixs/mid_matrixB.pt')   
    '''
    
    torch.save(torch.tensor(A3, dtype=torch.float32), 'model_matrixs/small_matrixA.pt')
    torch.save(torch.tensor(B3, dtype=torch.float32), 'model_matrixs/small_matrixB.pt')
      

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

    
    # ----------------- REFERENCE RESULTS for dim = 0 split test-----------------
    '''
    big_a = torch.load(big_test_matrix_pathA)
    big_b = torch.load(big_test_matrix_pathB)
    big_c_ref = big_a @ big_b.T  # A @ B.T
    torch.save(big_c_ref, 'model_matrixs/big_c_ref.pt')

    mid_a = torch.load(mid_test_matrix_pathA)
    mid_b = torch.load(mid_test_matrix_pathB)
    mid_c_ref = mid_a @ mid_b.T
    torch.save(mid_c_ref, 'model_matrixs/mid_c_ref.pt')
    '''
    
    small_a = torch.load(small_test_matrix_pathA)
    small_b = torch.load(small_test_matrix_pathB)
    small_c_ref = small_a @ small_b.T
    torch.save(small_c_ref, 'model_matrixs/small_c_ref.pt')
    

    # ----------------- REFERENCE RESULTS for dim = 1 split test-----------------
    '''
    big_a_T = torch.load(big_test_matrix_pathA_T)
    big_b_T = torch.load(big_test_matrix_pathB_T)
    big_c_ref_T = big_a_T.T @ big_b_T  # A @ B.T
    torch.save(big_c_ref_T, 'model_matrixs/big_c_ref_T.pt')

    mid_a_T = torch.load(mid_test_matrix_pathA)
    mid_b_T = torch.load(mid_test_matrix_pathB)
    mid_c_ref_T = mid_a_T @ mid_b_T.T
    torch.save(mid_c_ref_T, 'model_matrixs/mid_c_ref_T.pt')

    small_a_T = torch.load(small_test_matrix_pathA_T)
    small_b_T = torch.load(small_test_matrix_pathB_T)
    small_c_ref_T = small_a_T.T @ small_b_T
    torch.save(small_c_ref_T, 'model_matrixs/small_c_ref_T.pt')
    '''
    

    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
    
    # ----------------- CLUSTER TEST (BIG MATRIX) dim = 0 split/join test-----------------

    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
    percentages = [0.35,0.35,0.15,0.15]  
    CPU_GPU_select_list = [ True, True, True, True ]  
    backend_select_list = ['llama','llama','llama','llama'] 


    cluster_zmq_obj = cluster_zmq(IP_list)

    # ----------------- CLUSTER TEST (small MATRIX) -----------------
    
    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=1,
                                    auto_set_up=[1, "save"]
                                    )


    small_new_matrixB = cluster_matrix(small_test_matrix_pathB, 
                                    cluster_zmq_object=cluster_zmq_obj,  
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"]
                                    )

    big_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/small_c_ref.pt',big_new_matrixC)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=1,
                                    auto_set_up=[1, "load"]
                                    )
    
    small_new_matrixB = cluster_matrix(small_test_matrix_pathB, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "load"]
                                    )

    big_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/small_c_ref.pt',big_new_matrixC)
    
    
    #############################SYSTEM 1 — 5 SLOT TEST#############################
    # 5 compute slots (duplicates allowed), still System 1 behavior (full A, sharded B)
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.101','192.168.2.104']
    percentages = [0.2, 0.2, 0.2, 0.2, 0.2]
    CPU_GPU_select_list = [ True, True, True, True, True ]
    backend_select_list = ['llama','llama','llama','llama','llama']

    cluster_zmq_obj = cluster_zmq(IP_list)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=1,
                                    auto_set_up=[1, "save"]
                                    )

    small_new_matrixB = cluster_matrix(small_test_matrix_pathB,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"]
                                    )

    big_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)
    check_combined_result_values('model_matrixs/small_c_ref.pt',big_new_matrixC)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=1,
                                    auto_set_up=[1, "load"]
                                    )

    small_new_matrixB = cluster_matrix(small_test_matrix_pathB,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "load"]
                                    )

    big_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)
    check_combined_result_values('model_matrixs/small_c_ref.pt',big_new_matrixC)

    
    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 2#############################
    
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.101','192.168.2.104']    
    CPU_GPU_select_list = [ True, True, True, True, True, True ]  
    backend_select_list = ['llama','llama','llama','llama','llama','llama']

    cluster_zmq_obj = cluster_zmq(IP_list)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA, 
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "save"],
                                    matrix_labeling='a'
                                    )


    small_new_matrixB = cluster_matrix(small_test_matrix_pathB, 
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "save"],
                                    matrix_labeling='b'
                                    )

    small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/small_c_ref.pt',small_new_matrixC)


    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA, 
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "load"],
                                    matrix_labeling='a'
                                    )


    small_new_matrixB = cluster_matrix(small_test_matrix_pathB, 
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "load"],
                                    matrix_labeling='b'
                                    )

    small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)  
    check_combined_result_values('model_matrixs/small_c_ref.pt',small_new_matrixC)
    

    #############################SYSTEM 2 — 5 SLOT TEST#############################
    # 5 compute slots should still run as an 8-op grid (2x4 blocks) via wrap-around.
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.101','192.168.2.104']
    CPU_GPU_select_list = [ True, True, True, True, True ]
    backend_select_list = ['llama','llama','llama','llama','llama']

    cluster_zmq_obj = cluster_zmq(IP_list)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "save"],
                                    matrix_labeling='a'
                                    )

    small_new_matrixB = cluster_matrix(small_test_matrix_pathB,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "save"],
                                    matrix_labeling='b'
                                    )

    small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)
    check_combined_result_values('model_matrixs/small_c_ref.pt',small_new_matrixC)

    small_big_new_matrixA = cluster_matrix(small_test_matrix_pathA,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "load"],
                                    matrix_labeling='a'
                                    )

    small_new_matrixB = cluster_matrix(small_test_matrix_pathB,
                                    cluster_zmq_object=cluster_zmq_obj,
                                    CPU_GPU_select_list=CPU_GPU_select_list,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[2, "load"],
                                    matrix_labeling='b'
                                    )

    small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(small_new_matrixB, False, True, True)
    check_combined_result_values('model_matrixs/small_c_ref.pt',small_new_matrixC)
    
    

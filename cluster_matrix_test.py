
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

    small_a = torch.load(small_test_matrix_pathA)
    small_b = torch.load(small_test_matrix_pathB)
    small_c_ref = small_a @ small_b.T
    torch.save(small_c_ref, 'model_matrixs/small_c_ref.pt')
    '''

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
    percentages = [0.4,0.4,0.1,0.1]  
    CPU_GPU_select_list = [ True, True, True, True ]  
    backend_select_list = ['llama','llama','llama','llama'] 

    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    #'''
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
    #'''

    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
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
    #'''
    
    # ----------------- 5 NODE TEST -----------------
    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    #'''
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
    #'''
    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
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
    #'''

    ############################# TESTING CLUSTER MATRIX OPERATIONS SYSTEM 2 #############################
    ######################MID MATRIX 4 NODE TEST###############
    #'''
    IP_list = [
        '192.168.2.100','192.168.2.100',
        '192.168.2.101','192.168.2.104'
    ]

    percentages =  [0.50,0.50,0,0]
    CPU_GPU_select_list = [True, True, True, True]
    # You already have this variable defined:
    backend_select_list = ['llama', 'llama', 'llama', 'llama'] 
    #'''
    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    #'''
    # Use it instead of empty list:
    big_sys2_matrix_A = cluster_matrix(
        matrix_file_path=big_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    big_sys2_matrix_A.convert_to_cluster_matrix_grid()
    big_sys2_matrix_A.save_distribute_matrixA_grid_bin()
    big_sys2_matrix_B = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_sys2_matrix_B.convert_to_cluster_matrix_grid()
    big_sys2_matrix_B.save_distribute_matrix_shards_bin()
    big_result = big_sys2_matrix_A.cluster_shard_operation(big_sys2_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/big_c_ref.pt',big_result)
    #''' 

    #'''
    big_sys2_load_matrix_A = cluster_matrix(
        matrix_file_path='big_matrixA',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    big_sys2_load_matrix_A.load_cluster_matrixA_grid()
    big_sys2_load_matrix_B = cluster_matrix(
        matrix_file_path='big_matrixB',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_sys2_load_matrix_B.load_cluster_matrix_shards()
    big_sys2_load_result = big_sys2_load_matrix_A.cluster_shard_operation(big_sys2_load_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/big_c_ref.pt',big_sys2_load_result)
    #'''
        #'''
    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
    mid_sys2_matrix_A = cluster_matrix(
        matrix_file_path=mid_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    mid_sys2_matrix_A.convert_to_cluster_matrix_grid()
    mid_sys2_matrix_A.save_distribute_matrixA_grid_bin()
    mid_sys2_matrix_B = cluster_matrix(
        matrix_file_path=mid_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    mid_sys2_matrix_B.convert_to_cluster_matrix_grid()
    mid_sys2_matrix_B.save_distribute_matrix_shards_bin()
    mid_result = mid_sys2_matrix_A.cluster_shard_operation(mid_sys2_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/mid_c_ref.pt',mid_result)
    #'''
    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
    mid_sys2_load_matrix_A = cluster_matrix(
        matrix_file_path='mid_matrixA',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    mid_sys2_load_matrix_A.load_cluster_matrixA_grid()
    mid_sys2_load_matrix_B = cluster_matrix(
        matrix_file_path='mid_matrixB',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    mid_sys2_load_matrix_B.load_cluster_matrix_shards()
    mid_result = mid_sys2_load_matrix_A.cluster_shard_operation(mid_sys2_load_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/mid_c_ref.pt',mid_result)
    #'''

                    ######################MID MATRIX 6 NODE TEST###############
    # Use it instead of empty list:
    
    #'''
    IP_list = [
        '192.168.2.100','192.168.2.100',
        '192.168.2.101','192.168.2.104',
        '192.168.2.100','192.168.2.101'
    ]
    percentages =  [0.50,0.25,0.25,0,0,0]
    CPU_GPU_select_list = [True, True, True, True, True, True]
    # You already have this variable defined:
    backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama'] 
    #'''
    # ----------------- CLUSTER TEST (BIG MATRIX) -----------------
    #'''
    # Use it instead of empty list:
    big_sys2_matrix_A = cluster_matrix(
        matrix_file_path=big_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    big_sys2_matrix_A.convert_to_cluster_matrix_grid()
    big_sys2_matrix_A.save_distribute_matrixA_grid_bin()
    big_sys2_matrix_B = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_sys2_matrix_B.convert_to_cluster_matrix_grid()
    big_sys2_matrix_B.save_distribute_matrix_shards_bin()
    big_result = big_sys2_matrix_A.cluster_shard_operation(big_sys2_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/big_c_ref.pt',big_result)
    #''' 
    #'''
    big_sys2_load_matrix_A = cluster_matrix(
        matrix_file_path='big_matrixA',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    big_sys2_load_matrix_A.load_cluster_matrixA_grid()
    big_sys2_load_matrix_B = cluster_matrix(
        matrix_file_path='big_matrixB',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_sys2_load_matrix_B.load_cluster_matrix_shards()
    big_sys2_load_result = big_sys2_load_matrix_A.cluster_shard_operation(big_sys2_load_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/big_c_ref.pt',big_sys2_load_result)
    #'''

    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
    mid_sys2_matrix_A = cluster_matrix(
        matrix_file_path=mid_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    mid_sys2_matrix_A.convert_to_cluster_matrix_grid()
    mid_sys2_matrix_A.save_distribute_matrixA_grid_bin()
    mid_sys2_matrix_B = cluster_matrix(
        matrix_file_path=mid_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    mid_sys2_matrix_B.convert_to_cluster_matrix_grid()
    mid_sys2_matrix_B.save_distribute_matrix_shards_bin()
    mid_result = mid_sys2_matrix_A.cluster_shard_operation(mid_sys2_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/mid_c_ref.pt',mid_result)
    #'''
    # ----------------- CLUSTER TEST (MID MATRIX) -----------------
    #'''
    mid_sys2_load_matrix_A = cluster_matrix(
        matrix_file_path='mid_matrixA',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    mid_sys2_load_matrix_A.load_cluster_matrixA_grid()
    mid_sys2_load_matrix_B = cluster_matrix(
        matrix_file_path='mid_matrixB',
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,  # Use the actual variable!
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    mid_sys2_load_matrix_B.load_cluster_matrix_shards()
    mid_result = mid_sys2_load_matrix_A.cluster_shard_operation(mid_sys2_load_matrix_B,False,True,True)
    check_combined_result_values('model_matrixs/mid_c_ref.pt',mid_result)
    #'''


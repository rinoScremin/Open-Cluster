from cluster_matrix_v1 import cluster_matrix
import torch

if __name__ == "__main__":
    # ----------------- FILE PATHS for dim = 0 split test-----------------
    big_test_matrix_pathA = '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/layers_1_mlp_down_proj_weight.pt'  
    big_test_matrix_pathB = '/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrixs/layers_1_mlp_down_proj_weight.pt'  

    mid_test_matrix_pathA = 'model_matrixs/mid_matrixA.pt'  
    mid_test_matrix_pathB = 'model_matrixs/mid_matrixB.pt'  

    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
    percentages = [0.5,0.5,0.0,0.0]  
    CPU_GPU_select_list = [True, True, True, True]  
    backend_select_list = ['llama','llama','llama','llama'] 

    big_new_matrixA = cluster_matrix(
        matrix_file_path=big_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=0,
        matrix_labeling='a'
    )
    result_paths_A = big_new_matrixA.remote_save_distribute_matrix_shards_bin()
    
    big_new_matrixB = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_new_matrixB.remote_save_distribute_matrix_shards_bin()
    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True)  

    percentages = [0.4,0.4,0.1,0.1]  
    big_new_matrixA = cluster_matrix(
        matrix_file_path=big_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=False,
        dim=0
    )
    result_paths_A = big_new_matrixA.remote_save_distribute_matrix_shards_bin()
    
    big_new_matrixB = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=0
    )
    big_new_matrixB.remote_save_distribute_matrix_shards_bin()
    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True)  

    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104','192.168.2.100','192.168.2.101']   
    percentages = [0.4,0.4,0.2,0.0,0.0,0.0]  
    CPU_GPU_select_list = [True, True, True, True, True, True]  
    backend_select_list = ['llama','llama','llama','llama','llama','llama'] 

    big_new_matrixB = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=0,
        matrix_labeling='b'
    )
    big_new_matrixB.remote_save_distribute_matrix_shards_bin()
    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True)  


    ######################################### STILL UNDER DEVELOPMENT ###################################### 
    percentages = [0.2,0.2,0.2,0.2,0.1,0.1]  
    big_new_matrixA = cluster_matrix(
        matrix_file_path=big_test_matrix_pathA,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=False,
        dim=0
    )
    result_paths_A = big_new_matrixA.remote_save_distribute_matrix_shards_bin()
    
    big_new_matrixB = cluster_matrix(
        matrix_file_path=big_test_matrix_pathB,
        node_IP_list=IP_list,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=0
    )
    big_new_matrixB.remote_save_distribute_matrix_shards_bin()
    big_new_matrixC = big_new_matrixA.cluster_shard_operation(big_new_matrixB, False, True, True)  

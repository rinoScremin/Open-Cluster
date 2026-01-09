import os
from transformers import AutoModel, AutoTokenizer
from cluster_matrix_v1 import cluster_matrix
import torch
import time
import math

def check_combined_result_values(c_ref, combined):
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

class cluster_llm_transformer:
    def __init__(self, model_path, IP_list, percentages, CPU_GPU_select_list, backend_select_list):
        self.local_project_dir = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/"
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model_matrix_fold_dir = "model_matrixs/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = self.model.config
        
        # Get attention configuration
        self.attention_type, self.num_q_heads, self.num_kv_heads = self.detect_attention_type()
        self.hidden_size = self.config.hidden_size
        
        # Calculate head dimension (should be hidden_size ÷ num_q_heads)
        self.head_dim = self.hidden_size // self.num_q_heads
        
        # KV dimension for GQA/MQA
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Store attention configuration
        self.Model_Attention = self.attention_type
        self.attention_Heads = [self.num_q_heads, self.num_kv_heads]
        self.Hidden_size = self.hidden_size
        self.seq_len = 0
        
        print(f"🔍 Model: {self.config.model_type}")
        print(f"🔍 Attention: {self.attention_type}")
        print(f"🔍 Heads: Q={self.num_q_heads}, KV={self.num_kv_heads}")
        print(f"🔍 Hidden size: {self.hidden_size}")
        print(f"🔍 Head dimension: {self.head_dim}")
        print(f"🔍 KV dimension: {self.kv_dim}")
        
        self.IP_list = IP_list
        self.percentages = percentages
        self.CPU_GPU_select_list = CPU_GPU_select_list
        self.backend_select_list = backend_select_list
        self.tokens = None
        
        # Create directory for matrices
        os.makedirs("model_matrixs", exist_ok=True)
        self.token_embedding_matrix = None
        self.token_embedding_matrix_path = ""
        self.cluster_token_embedding_matrix = None
        self.full_token_embedding_matrix = None

    def detect_attention_type(self):
        """Detect what kind of attention bullshit we're dealing with"""
        config = self.model.config
        
        num_heads = getattr(config, 'num_attention_heads', 32)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        if num_kv_heads == num_heads:
            return "MHA", num_heads, num_kv_heads
        elif num_kv_heads == 1:
            return "MQA", num_heads, num_kv_heads  
        elif num_kv_heads < num_heads:
            return "GQA", num_heads, num_kv_heads
        else:
            return "WTF", num_heads, num_kv_heads

    def list_llm_layer(self):
        for name, param in self.model.named_parameters():
            print("LLM layer --> ", name)

    def tokenize_text(self, text):
        self.tokens = self.tokenizer(text, return_tensors="pt")
        return self.tokens.input_ids

    def save_all_model_layers(self, start_layer=0, end_layer=None, batch_size=4):
        """
        Save model layers in batches to avoid memory crashes
        """
        if end_layer is None:
            end_layer = getattr(self.config, 'num_hidden_layers', 32) - 1
        
        print(f"💾 SAVING MODEL LAYERS SAFELY {start_layer} to {end_layer}")
        print(f"📊 Batch size: {batch_size} layers at a time")
        print("=" * 60)
        
        total_saved = 0
        
        # Process in batches
        for batch_start in range(start_layer, end_layer + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_layer)
            
            print(f"\n🔧 Processing layers {batch_start} to {batch_end}...")
            
            for layer_idx in range(batch_start, batch_end + 1):
                print(f"  📁 Layer {layer_idx}...")
                layer_saved = 0
                
                # Save this layer's matrices
                for name, param in self.model.named_parameters():
                    name_split = name.split(".")
                    try:
                        layer_index = int(name_split[1])
                        if len(param.shape) == 2 and layer_index == layer_idx:
                            safe_name = name.replace('.', '_')
                            path = self.model_matrix_fold_dir + safe_name + '.pt'
                            torch.save(param.float(), path)
                            print(f"    ✅ {safe_name}.pt")
                            layer_saved += 1
                            total_saved += 1
                    except (ValueError, IndexError):
                        continue
                
                print(f"    📊 Saved {layer_saved} matrices")
            
            # Clear memory after each batch
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            # Small delay to let system breathe
            import time
            time.sleep(1)
            print(f"  💤 Batch complete, pausing...")
        
        # Save special layers separately
        print(f"\n📁 Saving special layers...")
        special_saved = 0
        special_layers = []
        
        # First collect all special layer names
        for name, param in self.model.named_parameters():
            if ('norm' in name.lower() and 'weight' in name) or \
            ('lm_head' in name.lower()) or \
            ('embed' in name.lower() and len(param.shape) == 2):
                special_layers.append((name, param))
        
        # Save them one by one
        for name, param in special_layers:
            safe_name = name.replace('.', '_')
            path = self.model_matrix_fold_dir + safe_name + '.pt'
            torch.save(param.float(), path)
            print(f"  ✅ {safe_name}.pt - Shape: {param.shape}")
            special_saved += 1
            total_saved += 1
        
        print(f"\n🎉 SAFELY SAVED {total_saved} matrices")
        print(f"   • Layers: {end_layer - start_layer + 1}")
        print(f"   • Special layers: {special_saved}")
        
        return total_saved

    def get_save_distribute_token_embeddings(self, op='mul'):
        """Get embeddings for the tokenized input using the full model"""
        if self.tokens is None:
            print("❌ No tokens found. Call tokenize_text() first.")
            return None
        
        print("🔍 Getting token embeddings from full model...")
        
        # Check if we already have saved embeddings
        embedding_path = self.model_matrix_fold_dir + 'embed_tokens_weight.pt'
        if not os.path.exists(embedding_path):
            print("❌ Embedding weights not found. Run save_all_model_layers() first.")
            return None
        
        # Load embedding matrix
        embedding_matrix = torch.load(embedding_path)
        print(f"📊 Using embedding matrix: {embedding_matrix.shape}")
        
        # Get embeddings for each token ID
        token_embeddings = []
        for token_id in self.tokens.input_ids[0]:
            # Look up the embedding in the full matrix
            embedding = embedding_matrix[token_id]  # Shape: [hidden_size]
            token_text = self.tokenizer.decode(token_id)
            print(f"🔤 Token {token_id}: '{token_text}' -> embedding shape: {embedding.shape}")
            token_embeddings.append(embedding)
        
        # Stack all embeddings
        all_embeddings = torch.stack(token_embeddings)  # Shape: [seq_len, hidden_size]
        self.seq_len = all_embeddings.shape[0]
        print(f"📦 Combined embeddings shape: {all_embeddings.shape}")
        
        # Save to disk
        path = self.model_matrix_fold_dir + 'input_token_embedding_matrix.pt'
        torch.save(all_embeddings.float(), path)
        self.token_embedding_matrix = all_embeddings
        
        # Create and distribute cluster matrix for token embeddings
        print("🔀 Distributing token embeddings across cluster...")
        if op == 'mul':
            # Create cluster matrix object for token embeddings
            # Based on your working example above, use the correct parameters
            self.cluster_token_embedding_matrix = cluster_matrix(
                matrix_file_path=path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,  
            )
            # Convert and distribute
            self.cluster_token_embedding_matrix.save_distribute_full_matrix_bin()
            #self.cluster_token_embedding_matrix.convert_to_cluster_matrix_grid()
            #self.cluster_token_embedding_matrix.save_distribute_matrixA_grid_bin()

            print("✅ Token embeddings distributed successfully!")
            return self.cluster_token_embedding_matrix
        if op == 'add':
            # Create cluster matrix object for token embeddings
            # Based on your working example above, use the correct parameters
           
            self.cluster_token_embedding_matrix = cluster_matrix(
                matrix_file_path=path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=[True, True, True, False],
                split_matrix=True,
                dim=1,  # Split by rows (sequence dimension)
            )
            
            # Convert and distribute
            self.cluster_token_embedding_matrix.convert_to_cluster_matrix_shards()
            self.cluster_token_embedding_matrix.save_distribute_matrix_shards_bin()
            
            print("✅ Token embeddings distributed successfully!")
            return self.cluster_token_embedding_matrix

    def save_distribute_model_matrices(self, batch_size=4):
        """
        ONE-TIME SETUP: Save and distribute ALL model matrices to cluster
        Handles both 2D matrices and 1D vectors (normalization weights)
        Uses remote_save_distribute_matrix_shards_bin_cpp_server() for direct distribution
        """
        print("=" * 70)
        print("🚀 SAVING AND DISTRIBUTING ALL MODEL MATRICES (REMOTE DISTRIBUTION)")
        print("=" * 70)
        
        num_layers = getattr(self.config, 'num_hidden_layers', 32)
        
        # Dictionary to track all distributed matrices
        self.distributed_matrices = {}
        total_distributed = 0
        
        print(f"\n📦 DISTRIBUTING {num_layers} TRANSFORMER LAYERS...")
        print("-" * 50)
        
        # Process layers in batches to avoid memory issues
        for batch_start in range(0, num_layers, batch_size):
            batch_end = min(batch_start + batch_size - 1, num_layers - 1)
            print(f"\n🔧 Batch {batch_start} to {batch_end}...")
            
            for layer_idx in range(batch_start, batch_end + 1):
                print(f"  📁 Layer {layer_idx}: ", end="")
                
                # List of weight matrices for this layer (2D matrices)
                layer_matrices_2d = [
                    (f'layers_{layer_idx}_self_attn_q_proj_weight', 'b'),
                    (f'layers_{layer_idx}_self_attn_k_proj_weight', 'b'),
                    (f'layers_{layer_idx}_self_attn_v_proj_weight', 'b'),
                    (f'layers_{layer_idx}_self_attn_o_proj_weight', 'b'),
                    (f'layers_{layer_idx}_mlp_up_proj_weight', 'b'),
                    (f'layers_{layer_idx}_mlp_gate_proj_weight', 'b'),
                    (f'layers_{layer_idx}_mlp_down_proj_weight', 'b'),
                ]
                
                # List of normalization weights (1D vectors - handle separately!)
                layer_matrices_1d = [
                    (f'layers_{layer_idx}_input_layernorm_weight', 'norm'),
                    (f'layers_{layer_idx}_post_attention_layernorm_weight', 'norm'),
                ]
                
                # Save and distribute 2D matrices
                for matrix_name, matrix_label in layer_matrices_2d:
                    matrix_path = f'{self.model_matrix_fold_dir}{matrix_name}.pt'
                    
                    if os.path.exists(matrix_path):
                        print(f"{matrix_label} ", end="")
                        
                        try:
                            # Create cluster matrix for 2D weights
                            cluster_mat = cluster_matrix(
                                matrix_file_path=matrix_path,
                                node_IP_list=self.IP_list,
                                CPU_GPU_select_list=self.CPU_GPU_select_list,
                                node_percentages=self.percentages,
                                back_end_select_list=self.backend_select_list,
                                split_matrix=True,
                                dim=0,  # Always split by rows for weights
                                matrix_labeling=matrix_label
                            )
                            
                            # REMOTE DISTRIBUTION: Use remote_save_distribute_matrix_shards_bin_cpp_server()
                            # This directly sends shards to cluster nodes via C++ server
                            result_paths = cluster_mat.remote_save_distribute_matrix_shards_bin_cpp_server()
                            
                            # Store reference with distribution info
                            key = f'layer_{layer_idx}_{matrix_name}'
                            self.distributed_matrices[key] = {
                                'matrix': cluster_mat,
                                'name': matrix_name,
                                'label': matrix_label,
                                'dim': 2,  # Mark as 2D
                                'path': matrix_path,
                                'remote_paths': result_paths,  # Store remote paths
                                'distribution_method': 'remote_cpp_server'
                            }
                            total_distributed += 1
                            print(f"✓ ", end="")
                        except Exception as e:
                            print(f"❌{matrix_label}({str(e)[:30]}) ", end="")
                    else:
                        print(f"❌{matrix_name[0]} ", end="")
                
                # Handle 1D normalization weights (store locally, not in cluster)
                for matrix_name, weight_type in layer_matrices_1d:
                    matrix_path = f'{self.model_matrix_fold_dir}{matrix_name}.pt'
                    
                    if os.path.exists(matrix_path):
                        print(f"N({weight_type[0]}) ", end="")
                        
                        # Load the 1D weight vector
                        norm_weight = torch.load(matrix_path)
                        
                        # Store it locally (1D vectors don't need cluster distribution)
                        key = f'layer_{layer_idx}_{matrix_name}'
                        self.distributed_matrices[key] = {
                            'weight': norm_weight,
                            'name': matrix_name,
                            'type': weight_type,
                            'dim': 1,  # Mark as 1D
                            'path': matrix_path,
                            'shape': norm_weight.shape,
                            'distribution_method': 'local_storage'
                        }
                        total_distributed += 1
                    else:
                        print(f"• ", end="")  # Not all models have all normalization layers
                
                print()  # New line after each layer
        
        # Special layers (norm, lm_head, embeddings)
        print(f"\n📁 DISTRIBUTING SPECIAL LAYERS...")
        print("-" * 50)
        
        # 2D special matrices
        special_matrices_2d = [
            ('lm_head_weight', 'b'),
            ('embed_tokens_weight', 'b'),
        ]
        
        # 1D special vectors
        special_matrices_1d = [
            ('norm_weight', 'norm'),
        ]
        
        # Handle 2D special matrices
        for matrix_name, matrix_label in special_matrices_2d:
            matrix_path = f'{self.model_matrix_fold_dir}{matrix_name}.pt'
            
            if os.path.exists(matrix_path):
                print(f"  ✅ {matrix_name}...", end=" ")
                
                try:
                    cluster_mat = cluster_matrix(
                        matrix_file_path=matrix_path,
                        node_IP_list=self.IP_list,
                        CPU_GPU_select_list=self.CPU_GPU_select_list,
                        node_percentages=self.percentages,
                        back_end_select_list=self.backend_select_list,
                        split_matrix=True,
                        dim=0,
                        matrix_labeling=matrix_label
                    )
                    
                    # REMOTE DISTRIBUTION for special matrices
                    result_paths = cluster_mat.remote_save_distribute_matrix_shards_bin_cpp_server()
                    
                    key = f'special_{matrix_name}'
                    self.distributed_matrices[key] = {
                        'matrix': cluster_mat,
                        'name': matrix_name,
                        'label': matrix_label,
                        'dim': 2,
                        'path': matrix_path,
                        'remote_paths': result_paths,
                        'distribution_method': 'remote_cpp_server'
                    }
                    total_distributed += 1
                    
                    print(f"Distributed ({len(result_paths)} shards)")
                except Exception as e:
                    print(f"Failed: {str(e)[:50]}")
            else:
                print(f"  ❌ {matrix_name}: Not found")
        
        # Handle 1D special vectors
        for matrix_name, weight_type in special_matrices_1d:
            matrix_path = f'{self.model_matrix_fold_dir}{matrix_name}.pt'
            
            if os.path.exists(matrix_path):
                print(f"  📏 {matrix_name} (1D vector)...", end=" ")
                
                # Load 1D vector
                norm_weight = torch.load(matrix_path)
                
                key = f'special_{matrix_name}'
                self.distributed_matrices[key] = {
                    'weight': norm_weight,
                    'name': matrix_name,
                    'type': weight_type,
                    'dim': 1,
                    'path': matrix_path,
                    'shape': norm_weight.shape,
                    'distribution_method': 'local_storage'
                }
                total_distributed += 1
                
                print(f"Stored locally")
            else:
                print(f"  ❌ {matrix_name}: Not found")
        
        # Summary
        print(f"\n🎉 REMOTE DISTRIBUTION COMPLETE!")
        print(f"   • Total matrices distributed: {total_distributed}")
        
        # Count types
        count_2d = sum(1 for v in self.distributed_matrices.values() if v.get('dim') == 2)
        count_1d = sum(1 for v in self.distributed_matrices.values() if v.get('dim') == 1)
        
        print(f"   • 2D matrices (remote cluster): {count_2d}")
        print(f"   • 1D vectors (local storage): {count_1d}")
        print(f"   • Layers processed: {num_layers}")
        print(f"   • All 2D matrices now distributed to cluster nodes via C++ servers!")
        
        # Optional: Verify remote distribution
        print(f"\n📋 DISTRIBUTION VERIFICATION:")
        print(f"   • Distribution method: remote_save_distribute_matrix_shards_bin_cpp_server()")
        print(f"   • Matrix shards sent directly to: {self.IP_list}")
        print(f"   • CPU/GPU devices: {self.CPU_GPU_select_list}")
        print(f"   • Backend selection: {self.backend_select_list}")
        
        return total_distributed


    def _apply_layer_norm(self, x, weight, eps=1e-5):
        """
        Apply layer normalization to input tensor x.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size] or [seq_len, hidden_size]
            weight: Scaling weights of shape [hidden_size]
            eps: Small epsilon for numerical stability
            
        Returns:
            Normalized tensor of same shape as x
        """
        # Ensure x is at least 2D
        if x.dim() == 2:
            # [seq_len, hidden_size] - add batch dimension
            x = x.unsqueeze(0)  # [1, seq_len, hidden_size]
            needs_squeeze = True
        else:
            needs_squeeze = False
        
        # Calculate mean and variance along the last dimension (hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch, seq_len, 1]
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        
        # Apply scaling weights
        # weight shape: [hidden_size] -> reshape to [1, 1, hidden_size] for broadcasting
        weight_expanded = weight.view(1, 1, -1)
        x_normalized = x_normalized * weight_expanded
        
        # Remove batch dimension if we added it
        if needs_squeeze:
            x_normalized = x_normalized.squeeze(0)  # [seq_len, hidden_size]
        
        return x_normalized

    def run_full_model_forward(self, text="fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!"):
        """
        Run the full transformer model through all layers using cluster acceleration.
        Returns predicted text along with intermediate hidden states.
        """
        
        '''
        input_enbedding = torch.load(f'{self.model_matrix_fold_dir}input_token_embedding_matrix.pt')
        q_matrix = torch.load(f'{self.model_matrix_fold_dir}layers_{0}_self_attn_q_proj_weight.pt')
        q_matrix_ref = input_enbedding @ q_matrix

        k_matrix = torch.load(f'{self.model_matrix_fold_dir}layers_{0}_self_attn_k_proj_weight.pt')
        k_matrix_ref = input_enbedding @ k_matrix.T

        v_matrix = torch.load(f'{self.model_matrix_fold_dir}layers_{0}_self_attn_v_proj_weight.pt')
        v_matrix_ref = input_enbedding @ v_matrix.T
        '''

        print("=" * 70)
        print("🚀 STARTING FULL MODEL FORWARD PASS")
        print("=" * 70)

        num_layers = getattr(self.config, 'num_hidden_layers', 32)
        print(f"📊 Model: {getattr(self.config,'model_type','Unknown')}")
        print(f"📊 Total layers: {num_layers}")
        print(f"📊 Attention type: {self.Model_Attention}")
        print(f"📊 Heads: Q={self.attention_Heads[0]}, KV={self.attention_Heads[1]}")
        print(f"📊 Hidden size: {self.Hidden_size}")
        print(f"📊 Text: {text[:50]}...")
        print("-" * 70)

        # -----------------------------
        # STEP 1: TOKENIZE + EMBEDDINGS
        # -----------------------------
        self.tokenize_text(text)
        print(f"Token IDs: {self.tokens.input_ids[0].tolist()}")
        current_hidden_state_mul = self.get_save_distribute_token_embeddings('mul') # set up cluster for mul op

        current_hidden_state_add = self.get_save_distribute_token_embeddings('add') # set up cluster for mul op

        for layer_idx in range(num_layers):
        #for layer_idx in range(1):

            # ============================================================
            # LOAD ATTENTION WEIGHTS FOR THIS LAYER
            # ============================================================
            print(f"   Loading attention weights for layer {layer_idx}...")

            attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt'
            attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt'
            attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt'
            attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt'


            # ============================================================
            # Q, K, V PROJECTIONS WITH CLUSTER
            # ============================================================
            print(f"   Running Q/K/V projections with CLUSTER...")
            cluster_start = time.time()

            # Load and distribute Q projection
            attn_q_proj = cluster_matrix(
                matrix_file_path=attn_q_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1
            )
            attn_q_proj.convert_to_cluster_matrix_shards()
            attn_q_proj.save_distribute_matrix_shards_bin()

            cluster_q_result = current_hidden_state_mul.cluster_shard_operation(
                attn_q_proj, False, False, True
            )
            #check_combined_result_values(q_matrix_ref,cluster_q_result)

            # K projection with CLUSTER
            attn_k_proj = cluster_matrix(
                matrix_file_path=attn_k_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
            )
            attn_k_proj.convert_to_cluster_matrix_shards()
            attn_k_proj.save_distribute_matrix_shards_bin()

            cluster_k_result = current_hidden_state_mul.cluster_shard_operation(
                attn_k_proj, False, True, True
            )
            #check_combined_result_values(k_matrix_ref,cluster_k_result)
            
            # V projection with CLUSTER
            attn_v_proj = cluster_matrix(
                matrix_file_path=attn_v_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
            )
            attn_v_proj.convert_to_cluster_matrix_shards()
            attn_v_proj.save_distribute_matrix_shards_bin()

            cluster_v_result = current_hidden_state_mul.cluster_shard_operation(
                attn_v_proj, False, True, True
            )
            #check_combined_result_values(v_matrix_ref,cluster_v_result)


            # ============================================================
            # GQA ATTENTION WITH TORCH (CORRECT & COMPLETE)
            # ============================================================
            print(f"   Computing GQA attention with TORCH...")
            torch_start = time.time()

            seq_len = cluster_q_result.shape[0]
            device = cluster_q_result.device
            dtype = cluster_q_result.dtype

            # ---- reshape to [seq, heads, head_dim]
            q = cluster_q_result.view(seq_len, self.num_q_heads, self.head_dim)
            k = cluster_k_result.view(seq_len, self.num_kv_heads, self.head_dim)
            v = cluster_v_result.view(seq_len, self.num_kv_heads, self.head_dim)

            # ---- expand KV for GQA
            if self.num_kv_heads < self.num_q_heads:
                repeat_factor = self.num_q_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            # ---- transpose to [heads, seq, head_dim]
            q = q.transpose(0, 1)   # [Hq, S, D]
            k = k.transpose(0, 1)   # [Hq, S, D]
            v = v.transpose(0, 1)   # [Hq, S, D]

            # ---- scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2))
            attn_scores *= (1.0 / (self.head_dim ** 0.5))

            # ---- causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores.masked_fill_(causal_mask, float("-inf"))

            # ---- softmax (numerically stable)
            attn_probs = torch.softmax(attn_scores, dim=-1)

            # ---- attention output
            attn_output = torch.matmul(attn_probs, v)   # [Hq, S, D]

            # ---- restore shape → [S, hidden]
            attn_output = attn_output.transpose(0, 1).contiguous()
            attn_output_flat_torch = attn_output.view(seq_len, self.Hidden_size)

            print(f'attn_output_flat shape : {attn_output_flat_torch.shape}')

            # ------------------------------------------------------------
            # Paths
            # ------------------------------------------------------------
            mlp_gate_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt'
            mlp_up_path   = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt'
            mlp_down_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt'

            attn_output_flat = cluster_matrix(
                matrix_file_path=attn_output_flat_torch,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,  # Split columns
                matrix_name='attn_output_flat'
            )
            attn_output_flat.save_distribute_full_matrix_bin()

            # Load MLP gate weights - shape should be [4096, 11008] for Llama
            mlp_gate_proj = cluster_matrix(
                matrix_file_path=mlp_gate_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0 # Split columns
            )
            mlp_gate_proj.convert_to_cluster_matrix_shards()
            mlp_gate_proj.save_distribute_matrix_shards_bin()

            # Multiply: attn_output @ gate_proj
            cluster_gate_result = attn_output_flat.cluster_shard_operation(
                mlp_gate_proj, False, True, True
            )
            
            # ============================================================
            # MLP UP PROJECTION (parallel to gate)
            # ============================================================
            print(f"   Running MLP up projection...")
            
            # Load MLP up weights - shape should be [4096, 14336] for Llama
            mlp_up_proj = cluster_matrix(
                matrix_file_path=mlp_up_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0  # Split columns
            )
            mlp_up_proj.convert_to_cluster_matrix_shards()
            mlp_up_proj.save_distribute_matrix_shards_bin()

            # Multiply: attn_output @ up_proj
            cluster_up_result = attn_output_flat.cluster_shard_operation(
                mlp_up_proj, False, True, True
            )
            
            print(f"   Up projection result shape: {cluster_up_result.shape}")

            # ============================================================
            # APPLY SILU ACTIVATION TO GATE
            # ============================================================
            print(f"   Applying SiLU activation to gate...")
            gate_silu = torch.nn.functional.silu(cluster_gate_result)
            
            # ============================================================
            # ELEMENT-WISE MULTIPLICATION: gate_silu * up
            # ============================================================
            print(f"   Element-wise multiplication (gate_silu * up)...")
            mlp_intermediate = gate_silu * cluster_up_result
            print(f"   MLP intermediate shape: {mlp_intermediate.shape}")
            
            # ============================================================
            # MLP DOWN PROJECTION
            # ============================================================
            print(f"   Running MLP down projection...")
            
            # Create cluster matrix for intermediate result
            mlp_intermediate_cluster = cluster_matrix(
                matrix_file_path=mlp_intermediate,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,
                matrix_name='mlp_intermediate_cluster'
            )
            mlp_intermediate_cluster.save_distribute_full_matrix_bin()
            
            # Load MLP down weights - shape should be [14336, 4096] for Llama
            mlp_down_proj = cluster_matrix(
                matrix_file_path=mlp_down_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0
            )
            
            mlp_down_proj.convert_to_cluster_matrix_shards()
            mlp_down_proj.save_distribute_matrix_shards_bin()

            # Multiply: mlp_intermediate @ down_proj
            mlp_output = mlp_intermediate_cluster.cluster_shard_operation(
                mlp_down_proj, False, True, True
            )

        

        
            # ============================================================
            # ADD RESIDUAL CONNECTION (MLP output + attention output)
            # ============================================================
            print(f"   Adding residual connection (MLP + attention)...")
            

            # Create cluster matrix for mlp_output
            mlp_output_cluster = cluster_matrix(
                matrix_file_path=mlp_output,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=[True, True, True, False],
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                matrix_name='mlp_output_cluster'
            )
            mlp_output_cluster.convert_to_cluster_matrix_shards()
            mlp_output_cluster.save_distribute_matrix_shards_bin()
            

            # Create cluster matrix for attention output
            attn_output_cluster = cluster_matrix(
                matrix_file_path=attn_output_flat_torch,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=[True, True, True, False],
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                matrix_name='attn_outpuattn_output_clustert_flat'
            )
            attn_output_cluster.convert_to_cluster_matrix_shards()
            attn_output_cluster.save_distribute_matrix_shards_bin()

            # Perform addition using cluster operation
            layer_output_cluster = mlp_output_cluster.cluster_shard_operation(
                attn_output_cluster, False, True, True, 'add'
            )

            # ============================================================
            # APPLY POST-ATTENTION LAYER NORMALIZATION
            # ============================================================
            print(f"   Applying post-attention layer normalization...")

            # Load post-attention layer norm weights
            post_attention_layernorm_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_post_attention_layernorm_weight.pt'

            if os.path.exists(post_attention_layernorm_path):
                try:
                    post_attention_layernorm = torch.load(post_attention_layernorm_path, map_location='cpu')
                    print(f"   Loaded post-attention layer norm weights: shape {post_attention_layernorm.shape}")

                    # layer_output_cluster is already a tensor (not cluster_matrix)
                    # So we apply layer norm directly
                    layer_output_tensor = self._apply_layer_norm(layer_output_cluster, post_attention_layernorm)
                    print(f"   Applied layer normalization")

                    if layer_idx == num_layers:
                        return layer_output_tensor

                    # If the next layer expects a cluster_matrix, wrap it again
                    layer_output_cluster = cluster_matrix(
                        matrix_file_path=layer_output_tensor,
                        node_IP_list=self.IP_list,
                        CPU_GPU_select_list=self.CPU_GPU_select_list,
                        node_percentages=self.percentages,
                        back_end_select_list=self.backend_select_list,
                        split_matrix=True,
                        dim=1,  # split across columns
                        matrix_name=f'layer_{layer_idx}_output_cluster'
                    )
                    layer_output_cluster.convert_to_cluster_matrix_shards()

                except Exception as e:
                    print(f"   Error loading/applying layer norm: {e}")
                    print(f"   Skipping layer normalization for this layer")
            else:
                print(f"   Warning: No post-attention layer norm found at {post_attention_layernorm_path}")
                print(f"   Skipping layer normalization for this layer")


                    


IP_list = [
    '192.168.2.100','192.168.2.100',
    '192.168.2.101','192.168.2.104',
]
percentages = [0.35, 0.35, 0.15, 0.15]
CPU_GPU_select_list = [True, True, True, True]
backend_select_list = ['llama', 'llama', 'llama', 'llama'] 

# Initialize the transformer
test = cluster_llm_transformer(
    '/home/rino/.cache/exo/downloads/mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated',
    IP_list, percentages, CPU_GPU_select_list, backend_select_list
)

print("=" * 70)
print("🚀 SAVING ALL MODEL LAYERS")
print("=" * 70)
# Get total number of layers
num_layers = getattr(test.config, 'num_hidden_layers', 32)
print(f"📊 Model has {num_layers} layers")
# Save ALL layers (0 through num_layers-1)
#test.save_all_model_layers(0, 4)
print("\n" + "=" * 70)
print("✅ ALL MODEL LAYERS SAVED!")
print("=" * 70)
# Now proceed with inference
#print("\n🔤 Tokenizing text and preparing embeddings...")
#test.tokenize_text("fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!")
#print("\n📦 Creating distributed embeddings...")
#test.get_save_distribute_token_embeddings()
#print("\n🏗️ Running full model forward pass...")
#result = test.run_full_model_forward()
#print("\n🎉 INFERENCE COMPLETE!")
#print(f"Final hidden state shape: {result['final_hidden_state'].shape}")

#test.run_GQA_transformer_layer()  # Distribute ALL weights to cluster

# ULTRA-FAST INFERENCE (Run this as many times as you want!):
result = test.run_full_model_forward("fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!")
#print(f"⚡ Inference completed in {result['performance']['total_time']:.2f}s!")

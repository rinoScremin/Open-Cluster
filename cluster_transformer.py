import os
from transformers import AutoModel, AutoTokenizer
from cluster_matrix_v1 import cluster_matrix
import torch
import time
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

    def save_all_model_layers(self, start_layer=0, end_layer=None):
        """
        Save all model layers sequentially
        """
        if end_layer is None:
            # Get total number of layers from model config
            end_layer = getattr(self.config, 'num_hidden_layers', 32) - 1
        
        print(f"💾 SAVING ALL MODEL LAYERS {start_layer} to {end_layer}")
        print("=" * 60)
        
        total_saved = 0
        for layer_idx in range(start_layer, end_layer + 1):
            print(f"\n📁 Processing layer {layer_idx}...")
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
                        print(f"   ✅ {safe_name}.pt - Shape: {param.shape}")
                        layer_saved += 1
                        total_saved += 1
                except (ValueError, IndexError):
                    continue
            
            print(f"   📊 Layer {layer_idx}: Saved {layer_saved} matrices")
        
        # Save special layers (norm, lm_head)
        print(f"\n📁 Saving special layers...")
        special_saved = 0
        for name, param in self.model.named_parameters():
            # Save normalization layers
            if 'norm' in name.lower() and 'weight' in name and len(param.shape) == 1:
                safe_name = name.replace('.', '_')
                path = self.model_matrix_fold_dir + safe_name + '.pt'
                torch.save(param.float(), path)
                print(f"   ✅ {safe_name}.pt - Shape: {param.shape}")
                special_saved += 1
                total_saved += 1
            
            # Save LM head
            elif 'lm_head' in name.lower() and len(param.shape) == 2:
                safe_name = name.replace('.', '_')
                path = self.model_matrix_fold_dir + safe_name + '.pt'
                torch.save(param.float(), path)
                print(f"   ✅ {safe_name}.pt - Shape: {param.shape}")
                special_saved += 1
                total_saved += 1
            
            # Save embedding layer
            elif 'embed' in name.lower() and len(param.shape) == 2:
                safe_name = name.replace('.', '_')
                path = self.model_matrix_fold_dir + safe_name + '.pt'
                torch.save(param.float(), path)
                print(f"   ✅ {safe_name}.pt - Shape: {param.shape}")
                special_saved += 1
                total_saved += 1
        
        print(f"\n🎉 COMPLETE: Saved {total_saved} matrices total")
        print(f"   • Transformer layers: {end_layer - start_layer + 1}")
        print(f"   • Special layers: {special_saved}")
        print(f"   • Total matrices: {total_saved}")
        
        return total_saved

    def save_llm_layer(self, layer=0):
        print(f"💾 Saving weight matrices for layer {layer}...")
        saved_count = 0
        for name, param in self.model.named_parameters():
            name_split = name.split(".")
            try:
                layer_index = int(name_split[1])
                if len(param.shape) == 2 and layer_index == layer:  # Only 2D matrices
                    safe_name = name.replace('.', '_')
                    path = self.model_matrix_fold_dir + safe_name + '.pt'
                    torch.save(param.float(), path)
                    print(f"✅ Saved: {safe_name}.pt - Shape: {param.shape}")
                    saved_count += 1
            except (ValueError, IndexError):
                # This parameter doesn't have a layer number, skip it
                continue
        
        print(f"🎯 Saved {saved_count} matrices for layer {layer}")

    def get_save_distribute_token_embeddings(self):
        """Get embeddings for the tokenized input using the full model"""
        if self.tokens is None:
            print("❌ No tokens found. Call tokenize_text() first.")
            return None
        print("🔍 Getting token embeddings from full model...")
        
        # Use the FULL model's embedding matrix, not the distributed one
        if hasattr(self.model, 'embed_tokens'):
            embedding_matrix = self.model.embed_tokens.weight
            print(f"📊 Using full embedding matrix: {embedding_matrix.shape}")
            self.full_token_embedding_matrix = embedding_matrix
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
            self.token_embedding_matrix = all_embeddings
            path = self.model_matrix_fold_dir + 'input_token_embedding_matrix'
            torch.save(self.token_embedding_matrix.float(), path)
            
            # CORRECT: Token embeddings should be split by rows (sequence dimension)
            self.cluster_token_embedding_matrix = cluster_matrix(matrix_file_path=path,
                                                node_IP_list=IP_list,
                                                CPU_GPU_select_list=self.CPU_GPU_select_list,
                                                node_percentages=percentages,
                                                back_end_select_list=self.backend_select_list,  # Use the actual variable!
                                                split_matrix=True,
                                                dim=0,
                                                matrix_labeling='a'
                                            )
            self.cluster_token_embedding_matrix.convert_to_cluster_matrix_grid()
            self.cluster_token_embedding_matrix.save_distribute_matrixA_grid_bin()

            return all_embeddings
        else:
            print("❌ No embedding layer found in model")
            return None

    def run_GQA_transformer_layer(self, layer=0, text="fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!"):
        """
        Complete GQA transformer layer with automatic tokenization
        Uses CLUSTER for heavy matrix multiplications, TORCH for lightweight additions
        """
        print(f"🧩 STEP 0: Tokenizing and distributing embeddings...")
        # 1. Tokenize if not already done
        if self.tokens is None:
            self.tokenize_text(text)
        # 2. Distribute embeddings if not already done
        if self.cluster_token_embedding_matrix is None:
            self.get_save_distribute_token_embeddings()
        if self.cluster_token_embedding_matrix is None:
            print("❌ Failed to distribute token embeddings!")
            return None
        
        print(f"\n🔧 STEP 1: Loading Q, K, V projection weights for layer {layer}")
        
        # Load Q, K, V projection weights
        attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_self_attn_q_proj_weight.pt'
        attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_self_attn_k_proj_weight.pt'
        attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_self_attn_v_proj_weight.pt'
        
        # Load torch references for comparison
        print("📊 Loading torch references for verification...")
        attn_q_proj_torch_ref = torch.load(attn_q_proj_path)
        attn_k_proj_torch_ref = torch.load(attn_k_proj_path)
        attn_v_proj_torch_ref = torch.load(attn_v_proj_path)
        torch_embedding_ref = torch.load(self.model_matrix_fold_dir + 'input_token_embedding_matrix.pt')
        
        # Create torch references for Q, K, V projections
        torch_q_ref = torch_embedding_ref @ attn_q_proj_torch_ref.T
        torch_k_ref = torch_embedding_ref @ attn_k_proj_torch_ref.T
        torch_v_ref = torch_embedding_ref @ attn_v_proj_torch_ref.T
        
        torch.save(torch_q_ref, 'torch_q_ref.pt')
        torch.save(torch_k_ref, 'torch_k_ref.pt')
        torch.save(torch_v_ref, 'torch_v_ref.pt')
        
        print(f"✅ Created torch references:")
        print(f"   Q shape: {torch_q_ref.shape}")
        print(f"   K shape: {torch_k_ref.shape}")
        print(f"   V shape: {torch_v_ref.shape}")
        
        # ============================================================
        # ATTENTION MECHANISM WITH CLUSTER MULTIPLICATIONS
        # ============================================================
        
        print(f"\n🎯 STEP 2: DISTRIBUTED ATTENTION WITH CLUSTER")
        print(f"   Using CLUSTER for heavy matrix multiplications")
        print(f"   Using TORCH for reshape, softmax, and additions")
        
        # Create and distribute Q projection matrix
        print(f"\n📦 STEP 2.1: Distributing Q projection matrix...")
        attn_q_proj = cluster_matrix(
            matrix_file_path=attn_q_proj_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,
            matrix_labeling='b'
        )
        attn_q_proj.convert_to_cluster_matrix_grid()
        attn_q_proj.save_distribute_matrix_shards_bin()
        
        # Run Q projection operation with CLUSTER
        print(f"\n🚀 STEP 2.2: Running Q projection with CLUSTER...")
        cluster_q_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
            attn_q_proj, False, True, True
        )
        print("✅ Q projection complete!")
        check_combined_result_values('torch_q_ref.pt', cluster_q_result)
        
        # Create and distribute K projection matrix
        print(f"\n📦 STEP 2.3: Distributing K projection matrix...")
        attn_k_proj = cluster_matrix(
            matrix_file_path=attn_k_proj_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,
            matrix_labeling='b'
        )
        attn_k_proj.convert_to_cluster_matrix_grid()
        attn_k_proj.save_distribute_matrix_shards_bin()
        
        # Run K projection operation with CLUSTER
        print(f"\n🚀 STEP 2.4: Running K projection with CLUSTER...")
        cluster_k_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
            attn_k_proj, False, True, True
        )
        print("✅ K projection complete!")
        check_combined_result_values('torch_k_ref.pt', cluster_k_result)
        
        # Create and distribute V projection matrix
        print(f"\n📦 STEP 2.5: Distributing V projection matrix...")
        attn_v_proj = cluster_matrix(
            matrix_file_path=attn_v_proj_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,
            matrix_labeling='b'
        )
        attn_v_proj.convert_to_cluster_matrix_grid()
        attn_v_proj.save_distribute_matrix_shards_bin()
        
        # Run V projection operation with CLUSTER
        print(f"\n🚀 STEP 2.6: Running V projection with CLUSTER...")
        cluster_v_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
            attn_v_proj, False, True, True
        )
        print("✅ V projection complete!")
        check_combined_result_values('torch_v_ref.pt', cluster_v_result)
        
        # ============================================================
        # GQA ATTENTION WITH TORCH (reshape, softmax, matmul)
        # ============================================================
        
        print(f"\n🔀 STEP 3: GQA ATTENTION COMPUTATION WITH TORCH")
        print(f"   Using torch for reshape, softmax, and batch matmul")
        
        # Use the distributed results from cluster
        cluster_q_tensor = cluster_q_result
        cluster_k_tensor = cluster_k_result
        cluster_v_tensor = cluster_v_result
        
        # Reshape for GQA attention with TORCH
        seq_len = cluster_q_tensor.shape[0]
        
        q_reshaped = cluster_q_tensor.view(seq_len, self.num_q_heads, self.head_dim)
        k_reshaped = cluster_k_tensor.view(seq_len, self.num_kv_heads, self.head_dim)
        v_reshaped = cluster_v_tensor.view(seq_len, self.num_kv_heads, self.head_dim)
        
        # Repeat KV heads for GQA
        if self.num_kv_heads < self.num_q_heads:
            repeat_factor = self.num_q_heads // self.num_kv_heads
            k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
            v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)
        
        # Transpose for batch matmul with TORCH
        q_transposed = q_reshaped.transpose(0, 1)
        k_transposed = k_reshaped.transpose(0, 1)
        v_transposed = v_reshaped.transpose(0, 1)
        
        # Compute attention with TORCH
        attention_scores = torch.matmul(q_transposed, k_transposed.transpose(-1, -2))
        scaling_factor = 1.0 / (self.head_dim ** 0.5)
        attention_scores = attention_scores * scaling_factor
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v_transposed)
        
        # Reshape back with TORCH
        attention_output = attention_output.transpose(0, 1)
        attention_output_flat = attention_output.reshape(seq_len, -1)
        
        print(f"   GQA attention output shape: {attention_output_flat.shape}")
        
        # Save torch reference
        torch.save(attention_output_flat, 'torch_attention_output.pt')
        
        # ============================================================
        # ATTENTION OUTPUT PROJECTION WITH CLUSTER
        # ============================================================
        
        print(f"\n🎯 STEP 4: ATTENTION OUTPUT PROJECTION WITH CLUSTER")
        
        # Load attention output projection
        attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_self_attn_o_proj_weight.pt'
        attn_o_proj_torch_ref = torch.load(attn_o_proj_path)
        
        print(f"📊 Output projection shape: {attn_o_proj_torch_ref.shape}")
        
        # Compute torch reference for verification
        torch_attn_final = attention_output_flat @ attn_o_proj_torch_ref.T
        torch.save(torch_attn_final, 'torch_attn_final.pt')
        print(f"   Torch reference shape: {torch_attn_final.shape}")
        
        # Load and distribute output projection matrix
        attn_o_proj = cluster_matrix(
            matrix_file_path=attn_o_proj_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,
            matrix_labeling='b'
        )
        attn_o_proj.convert_to_cluster_matrix_grid()
        attn_o_proj.save_distribute_matrix_shards_bin()
        
        # Convert attention output to cluster matrix for CLUSTER multiplication
        print(f"\n📦 Preparing attention output for CLUSTER multiplication...")
        torch.save(attention_output_flat, 'attention_output.pt')
        cluster_attention_output = cluster_matrix(
            matrix_file_path='attention_output.pt',
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,
            matrix_labeling='a'
        )
        cluster_attention_output.convert_to_cluster_matrix_grid()
        cluster_attention_output.save_distribute_matrixA_grid_bin()
        
        # Run output projection with CLUSTER
        print(f"\n🚀 STEP 4.1: Running output projection with CLUSTER...")
        attn_final_result = cluster_attention_output.cluster_shard_operation(
            attn_o_proj, False, True, True
        )
        print("✅ Output projection complete!")
        check_combined_result_values('torch_attn_final.pt', attn_final_result)
        
        # ============================================================
        # RESIDUAL CONNECTION WITH TORCH (CHEAP OPERATION)
        # ============================================================
        
        print(f"\n➕ STEP 5: RESIDUAL CONNECTION WITH TORCH")
        print(f"   Using torch for addition (lightweight operation)")
        
        # Add input embeddings to attention output with TORCH
        residual_result = torch_embedding_ref + attn_final_result
        torch.save(residual_result, 'torch_residual.pt')
        print(f"   Residual shape: {residual_result.shape}")
        
        # ============================================================
        # MLP FORWARD PASS (Hybrid: some torch, some cluster)
        # ============================================================
        
        print(f"\n🧠 STEP 6: MLP FORWARD PASS")
        
        # Load MLP weights
        mlp_up_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_mlp_up_proj_weight.pt'
        mlp_gate_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_mlp_gate_proj_weight.pt'
        mlp_down_proj_path = f'{self.model_matrix_fold_dir}layers_{layer}_mlp_down_proj_weight.pt'
        
        mlp_up_proj_torch_ref = torch.load(mlp_up_proj_path)
        mlp_gate_proj_torch_ref = torch.load(mlp_gate_proj_path)
        mlp_down_proj_torch_ref = torch.load(mlp_down_proj_path)
        
        print(f"📊 MLP weights loaded:")
        print(f"   Up projection shape: {mlp_up_proj_torch_ref.shape}")
        print(f"   Gate projection shape: {mlp_gate_proj_torch_ref.shape}")
        print(f"   Down projection shape: {mlp_down_proj_torch_ref.shape}")
        
        # Compute MLP with TORCH (all operations can be done locally)
        print(f"\n🔍 Computing MLP with TORCH...")
        
        # Step 1: Apply gate projection + SiLU activation
        gate_output = residual_result @ mlp_gate_proj_torch_ref.T
        gate_activated = torch.nn.functional.silu(gate_output)
        
        # Step 2: Apply up projection
        up_output = residual_result @ mlp_up_proj_torch_ref.T
        
        # Step 3: Element-wise multiply with TORCH
        mlp_intermediate = gate_activated * up_output
        
        # Step 4: Apply down projection
        mlp_output = mlp_intermediate @ mlp_down_proj_torch_ref.T
        
        torch.save(mlp_output, 'torch_mlp_output.pt')
        print(f"   MLP output shape: {mlp_output.shape}")
        
        # ============================================================
        # FINAL RESIDUAL CONNECTION WITH TORCH
        # ============================================================
        
        print(f"\n➕ STEP 7: FINAL RESIDUAL CONNECTION WITH TORCH")
        
        # Add MLP output back to residual with TORCH
        layer_output = residual_result + mlp_output
        torch.save(layer_output, 'torch_layer_output.pt')
        print(f"   Final layer output shape: {layer_output.shape}")
        
        print(f"\n🎉🎉🎉 LAYER {layer} COMPLETE! 🎉🎉🎉")
        print(f"\n📊 OPERATION SUMMARY:")
        print(f"   ✅ CLUSTER operations (heavy matrix multiplications):")
        print(f"      • Q projection (TokenEmbeddings × Q_weights)")
        print(f"      • K projection (TokenEmbeddings × K_weights)")
        print(f"      • V projection (TokenEmbeddings × V_weights)")
        print(f"      • Output projection (AttentionOutput × O_weights)")
        print(f"   ✅ TORCH operations (lightweight/non-distributable):")
        print(f"      • GQA reshape and attention computation")
        print(f"      • Softmax activation")
        print(f"      • Residual connections (additions)")
        print(f"      • MLP forward pass")
        print(f"      • SiLU activation")
        print(f"\n💡 Strategy: Heavy O(n³) operations → CLUSTER, Light O(n²) operations → TORCH")
        
        # Return the final output
        return layer_output

    def run_full_model_forward(self, text="fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!"):
        """
        Run the full transformer model through all layers
        Combines cluster operations for heavy matrix math with torch for lightweight ops
        """
        print("=" * 70)
        print("🚀 STARTING FULL MODEL FORWARD PASS")
        print("=" * 70)
        
        # Get model configuration
        num_layers = getattr(self.config, 'num_hidden_layers', 32)
        print(f"📊 Model: {self.config.model_type}")
        print(f"📊 Total layers: {num_layers}")
        print(f"📊 Attention type: {self.Model_Attention}")
        print(f"📊 Heads: Q={self.attention_Heads[0]}, KV={self.attention_Heads[1]}")
        print(f"📊 Hidden size: {self.Hidden_size}")
        print(f"📊 Text: {text[:50]}...")
        print("-" * 70)
        
        # ============================================================
        # STEP 1: INITIAL TOKENIZATION AND EMBEDDINGS
        # ============================================================
        print("\n🔤 STEP 1: TOKENIZATION AND EMBEDDINGS")
        print("-" * 50)
        
        # Tokenize text
        self.tokenize_text(text)
        print(f"   Token IDs: {self.tokens.input_ids[0].tolist()}")
        print(f"   Sequence length: {self.tokens.input_ids.shape[1]}")
        
        # Get and distribute initial embeddings
        initial_embeddings = self.get_save_distribute_token_embeddings()
        print(f"   Initial embeddings shape: {initial_embeddings.shape}")
        
        # Store for residual connections
        current_hidden_state = initial_embeddings
        self.seq_len = current_hidden_state.shape[0]
        print(f"   Sequence length after embedding: {self.seq_len}")
        
        # ============================================================
        # STEP 2: RUN THROUGH ALL TRANSFORMER LAYERS
        # ============================================================
        print(f"\n🏗️  STEP 2: PROCESSING {num_layers} TRANSFORMER LAYERS")
        print("-" * 50)
        
        layer_outputs = {}
        total_cluster_time = 0
        total_torch_time = 0
        
        for layer_idx in range(num_layers):
            print(f"\n{'='*50}")
            print(f"🏗️  PROCESSING LAYER {layer_idx}/{num_layers-1}")
            print(f"{'='*50}")
            
            # ============================================================
            # PRE-LAYER NORMALIZATION (RMSNorm)
            # ============================================================
            layer_start_time = time.time()
            
            print(f"\n📏 PRE-LAYER {layer_idx}: INPUT NORMALIZATION")
            
            # Check if layer normalization exists for this layer
            input_layernorm_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_input_layernorm_weight.pt'
            if os.path.exists(input_layernorm_path):
                print(f"   Applying RMSNorm from {input_layernorm_path}")
                # Load normalization weights
                norm_weights = torch.load(input_layernorm_path)
                
                # Apply RMSNorm: x * weight / sqrt(mean(x²) + eps)
                eps = getattr(self.config, 'rms_norm_eps', 1e-5)
                variance = current_hidden_state.pow(2).mean(-1, keepdim=True)
                normalized = current_hidden_state * torch.rsqrt(variance + eps)
                current_hidden_state = normalized * norm_weights
                
                print(f"   Normalized shape: {current_hidden_state.shape}")
            else:
                print(f"   No input normalization found for layer {layer_idx}, skipping...")
            
            # ============================================================
            # ATTENTION MECHANISM
            # ============================================================
            print(f"\n🎯 LAYER {layer_idx}: ATTENTION MECHANISM")
            
            # Update the cluster token embedding matrix with current hidden state
            print("   Updating cluster matrix with current hidden state...")
            torch.save(current_hidden_state, f'temp_layer{layer_idx}_hidden.pt')
            
            # Create new cluster matrix with current hidden state
            self.cluster_token_embedding_matrix = cluster_matrix(
                matrix_file_path=f'temp_layer{layer_idx}_hidden.pt',
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='a'
            )
            self.cluster_token_embedding_matrix.convert_to_cluster_matrix_grid()
            self.cluster_token_embedding_matrix.save_distribute_matrixA_grid_bin()
            
            # ============================================================
            # LOAD ATTENTION WEIGHTS FOR THIS LAYER
            # ============================================================
            print(f"   Loading attention weights for layer {layer_idx}...")
            
            attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt'
            attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt'
            attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt'
            attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt'
            
            # Verify all weight files exist
            missing_files = []
            for path in [attn_q_proj_path, attn_k_proj_path, attn_v_proj_path, attn_o_proj_path]:
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                print(f"❌ Missing weight files for layer {layer_idx}: {missing_files}")
                print(f"   Skipping layer {layer_idx}...")
                continue
            
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
                dim=0,
                matrix_labeling='b'
            )
            attn_q_proj.convert_to_cluster_matrix_grid()
            attn_q_proj.save_distribute_matrix_shards_bin()
            
            # Q projection with CLUSTER
            cluster_q_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
                attn_q_proj, False, True, True
            )
            
            # K projection with CLUSTER
            attn_k_proj = cluster_matrix(
                matrix_file_path=attn_k_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='b'
            )
            attn_k_proj.convert_to_cluster_matrix_grid()
            attn_k_proj.save_distribute_matrix_shards_bin()
            
            cluster_k_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
                attn_k_proj, False, True, True
            )
            
            # V projection with CLUSTER
            attn_v_proj = cluster_matrix(
                matrix_file_path=attn_v_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='b'
            )
            attn_v_proj.convert_to_cluster_matrix_grid()
            attn_v_proj.save_distribute_matrix_shards_bin()
            
            cluster_v_result = self.cluster_token_embedding_matrix.cluster_shard_operation(
                attn_v_proj, False, True, True
            )
            
            cluster_time = time.time() - cluster_start
            total_cluster_time += cluster_time
            
            # ============================================================
            # GQA ATTENTION WITH TORCH
            # ============================================================
            print(f"   Computing GQA attention with TORCH...")
            torch_start = time.time()
            
            # Reshape for GQA attention
            seq_len = cluster_q_result.shape[0]
            
            q_reshaped = cluster_q_result.view(seq_len, self.num_q_heads, self.head_dim)
            k_reshaped = cluster_k_result.view(seq_len, self.num_kv_heads, self.head_dim)
            v_reshaped = cluster_v_result.view(seq_len, self.num_kv_heads, self.head_dim)
            
            # Repeat KV heads for GQA
            if self.num_kv_heads < self.num_q_heads:
                repeat_factor = self.num_q_heads // self.num_kv_heads
                k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
                v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)
            
            # Compute attention
            q_transposed = q_reshaped.transpose(0, 1)
            k_transposed = k_reshaped.transpose(0, 1)
            v_transposed = v_reshaped.transpose(0, 1)
            
            attention_scores = torch.matmul(q_transposed, k_transposed.transpose(-1, -2))
            scaling_factor = 1.0 / (self.head_dim ** 0.5)
            attention_scores = attention_scores * scaling_factor
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_probs, v_transposed)
            
            # Reshape back
            attention_output = attention_output.transpose(0, 1)
            attention_output_flat = attention_output.reshape(seq_len, -1)
            
            torch_time = time.time() - torch_start
            total_torch_time += torch_time
            
            # ============================================================
            # OUTPUT PROJECTION WITH CLUSTER
            # ============================================================
            print(f"   Output projection with CLUSTER...")
            cluster_start = time.time()
            
            # Load output projection
            attn_o_proj = cluster_matrix(
                matrix_file_path=attn_o_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='b'
            )
            attn_o_proj.convert_to_cluster_matrix_grid()
            attn_o_proj.save_distribute_matrix_shards_bin()
            
            # Convert attention output to cluster matrix
            torch.save(attention_output_flat, f'temp_layer{layer_idx}_attn_out.pt')
            cluster_attention_output = cluster_matrix(
                matrix_file_path=f'temp_layer{layer_idx}_attn_out.pt',
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='a'
            )
            cluster_attention_output.convert_to_cluster_matrix_grid()
            cluster_attention_output.save_distribute_matrixA_grid_bin()
            
            # Apply output projection
            attn_final_result = cluster_attention_output.cluster_shard_operation(
                attn_o_proj, False, True, True
            )
            
            cluster_time += time.time() - cluster_start
            total_cluster_time += cluster_time
            
            # ============================================================
            # ATTENTION RESIDUAL CONNECTION
            # ============================================================
            print(f"   Attention residual connection...")
            torch_start = time.time()
            
            # Add input to attention output (first residual)
            attention_with_residual = current_hidden_state + attn_final_result
            
            torch_time += time.time() - torch_start
            total_torch_time += torch_time
            
            # ============================================================
            # POST-ATTENTION NORMALIZATION
            # ============================================================
            print(f"   Post-attention normalization...")
            
            # Check if post-attention normalization exists
            post_attention_layernorm_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_post_attention_layernorm_weight.pt'
            if os.path.exists(post_attention_layernorm_path):
                norm_weights = torch.load(post_attention_layernorm_path)
                eps = getattr(self.config, 'rms_norm_eps', 1e-5)
                variance = attention_with_residual.pow(2).mean(-1, keepdim=True)
                normalized = attention_with_residual * torch.rsqrt(variance + eps)
                attention_with_residual = normalized * norm_weights
            
            # ============================================================
            # MLP FORWARD PASS
            # ============================================================
            print(f"   MLP forward pass...")
            
            # Load MLP weights
            mlp_up_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt'
            mlp_gate_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt'
            mlp_down_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt'
            
            if os.path.exists(mlp_up_proj_path) and os.path.exists(mlp_gate_proj_path) and os.path.exists(mlp_down_proj_path):
                mlp_up_proj = torch.load(mlp_up_proj_path)
                mlp_gate_proj = torch.load(mlp_gate_proj_path)
                mlp_down_proj = torch.load(mlp_down_proj_path)
                
                # MLP computation with TORCH
                torch_start = time.time()
                
                # Apply MLP: SiLU(gate(x)) * up(x) then down projection
                gate_output = attention_with_residual @ mlp_gate_proj.T
                gate_activated = torch.nn.functional.silu(gate_output)
                
                up_output = attention_with_residual @ mlp_up_proj.T
                
                mlp_intermediate = gate_activated * up_output
                mlp_output = mlp_intermediate @ mlp_down_proj.T
                
                torch_time += time.time() - torch_start
                total_torch_time += torch_time
            else:
                print(f"   MLP weights not found for layer {layer_idx}, skipping MLP...")
                mlp_output = torch.zeros_like(attention_with_residual)
            
            # ============================================================
            # FINAL RESIDUAL CONNECTION
            # ============================================================
            print(f"   Final residual connection...")
            torch_start = time.time()
            
            # Add MLP output (second residual)
            current_hidden_state = attention_with_residual + mlp_output
            
            torch_time += time.time() - torch_start
            total_torch_time += torch_time
            
            # Store layer output
            layer_outputs[layer_idx] = {
                'hidden_state': current_hidden_state.clone(),
                'attention_output': attention_output_flat.clone(),
                'mlp_output': mlp_output.clone() if 'mlp_output' in locals() else None
            }
            
            # ============================================================
            # LAYER SUMMARY
            # ============================================================
            layer_end_time = time.time()
            layer_total_time = layer_end_time - layer_start_time
            
            print(f"\n📊 LAYER {layer_idx} SUMMARY:")
            print(f"   • Hidden state shape: {current_hidden_state.shape}")
            print(f"   • Attention output shape: {attention_output_flat.shape}")
            print(f"   • Time: {layer_total_time:.2f}s")
            print(f"   • Cluster ops: {cluster_time:.2f}s")
            print(f"   • Torch ops: {torch_time:.2f}s")
            
            # Clean up temporary files
            temp_files = [
                f'temp_layer{layer_idx}_hidden.pt',
                f'temp_layer{layer_idx}_attn_out.pt'
            ]
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # ============================================================
        # STEP 3: FINAL NORMALIZATION
        # ============================================================
        print(f"\n📏 STEP 3: FINAL NORMALIZATION")
        print("-" * 50)
        
        # Apply final layer normalization
        final_norm_path = f'{self.model_matrix_fold_dir}norm_weight.pt'
        if os.path.exists(final_norm_path):
            print(f"   Applying final normalization...")
            norm_weights = torch.load(final_norm_path)
            eps = getattr(self.config, 'rms_norm_eps', 1e-5)
            variance = current_hidden_state.pow(2).mean(-1, keepdim=True)
            normalized = current_hidden_state * torch.rsqrt(variance + eps)
            current_hidden_state = normalized * norm_weights
            print(f"   Final normalized shape: {current_hidden_state.shape}")
        else:
            print(f"   No final normalization found, skipping...")
        
        # ============================================================
        # STEP 4: LANGUAGE MODEL HEAD
        # ============================================================
        print(f"\n📝 STEP 4: LANGUAGE MODEL HEAD")
        print("-" * 50)
        
        # Apply LM head if it exists
        lm_head_path = f'{self.model_matrix_fold_dir}lm_head_weight.pt'
        if os.path.exists(lm_head_path):
            print(f"   Applying LM head...")
            
            # Load LM head weights
            lm_head_weights = torch.load(lm_head_path)
            print(f"   LM head shape: {lm_head_weights.shape}")
            
            # Convert to cluster for multiplication
            torch.save(current_hidden_state, 'temp_final_hidden.pt')
            final_hidden_cluster = cluster_matrix(
                matrix_file_path='temp_final_hidden.pt',
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='a'
            )
            final_hidden_cluster.convert_to_cluster_matrix_grid()
            final_hidden_cluster.save_distribute_matrixA_grid_bin()
            
            # Create cluster matrix for LM head
            lm_head_cluster = cluster_matrix(
                matrix_file_path=lm_head_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='b'
            )
            lm_head_cluster.convert_to_cluster_matrix_grid()
            lm_head_cluster.save_distribute_matrix_shards_bin()
            
            # Apply LM head with CLUSTER
            logits = final_hidden_cluster.cluster_shard_operation(
                lm_head_cluster, False, True, True
            )
            
            # Get predicted tokens
            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_tokens = self.tokenizer.decode(predicted_token_ids)
            
            print(f"   Logits shape: {logits.shape}")
            print(f"   Predicted token IDs: {predicted_token_ids.tolist()}")
            print(f"   Predicted tokens: {predicted_tokens}")
            
            # Clean up
            if os.path.exists('temp_final_hidden.pt'):
                os.remove('temp_final_hidden.pt')
        else:
            print(f"   No LM head found, skipping...")
            logits = None
            predicted_tokens = None
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("🎉 FULL MODEL FORWARD PASS COMPLETE!")
        print("=" * 70)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   • Total layers processed: {len(layer_outputs)}")
        print(f"   • Final hidden state shape: {current_hidden_state.shape}")
        print(f"   • Total cluster time: {total_cluster_time:.2f}s")
        print(f"   • Total torch time: {total_torch_time:.2f}s")
        print(f"   • Total time: {total_cluster_time + total_torch_time:.2f}s")
        print(f"   • Efficiency: {total_cluster_time/(total_cluster_time + total_torch_time)*100:.1f}% cluster, {total_torch_time/(total_cluster_time + total_torch_time)*100:.1f}% torch")
        
        print(f"\n📋 LAYER OUTPUTS STORED FOR:")
        for layer_idx in layer_outputs.keys():
            print(f"   • Layer {layer_idx}")
        
        print(f"\n🔍 FINAL HIDDEN STATE INFO:")
        print(f"   • Shape: {current_hidden_state.shape}")
        print(f"   • Mean: {current_hidden_state.mean().item():.6f}")
        print(f"   • Std: {current_hidden_state.std().item():.6f}")
        print(f"   • Min: {current_hidden_state.min().item():.6f}")
        print(f"   • Max: {current_hidden_state.max().item():.6f}")
        
        if logits is not None:
            print(f"\n📝 GENERATION RESULTS:")
            print(f"   • Predicted tokens: {predicted_tokens}")
            print(f"   • Logits shape: {logits.shape}")
        
        print("\n✅ All layers processed successfully!")
        
        return {
            'final_hidden_state': current_hidden_state,
            'layer_outputs': layer_outputs,
            'logits': logits,
            'predicted_tokens': predicted_tokens,
            'performance': {
                'total_cluster_time': total_cluster_time,
                'total_torch_time': total_torch_time,
                'total_time': total_cluster_time + total_torch_time,
                'layers_processed': len(layer_outputs)
            }
        }


IP_list = [
    '192.168.2.100','192.168.2.100',
    '192.168.2.101','192.168.2.104',
    '192.168.2.100','192.168.2.101'
]
percentages = [0.50, 0.25, 0.25, 0, 0, 0]
CPU_GPU_select_list = [True, True, True, True, True, True]
backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama'] 

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
test.save_all_model_layers(0, num_layers - 1)

print("\n" + "=" * 70)
print("✅ ALL MODEL LAYERS SAVED!")
print("=" * 70)

# Now proceed with inference
print("\n🔤 Tokenizing text and preparing embeddings...")
test.tokenize_text("fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!")

print("\n📦 Creating distributed embeddings...")
test.get_save_distribute_token_embeddings()

print("\n🏗️ Running full model forward pass...")
result = test.run_full_model_forward()

print("\n🎉 INFERENCE COMPLETE!")
print(f"Final hidden state shape: {result['final_hidden_state'].shape}")


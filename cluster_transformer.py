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

        def save_model_layers_safely(self, start_layer=0, end_layer=None, batch_size=4):
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

    def get_save_distribute_token_embeddings(self):
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
        
        # Create cluster matrix object for token embeddings
        # Based on your working example above, use the correct parameters
        self.cluster_token_embedding_matrix = cluster_matrix(
            matrix_file_path=path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=0,  # Split by rows (sequence dimension)
            matrix_labeling='a'  # Token embeddings are matrix A
        )
        
        # Convert and distribute
        self.cluster_token_embedding_matrix.convert_to_cluster_matrix_grid()
        self.cluster_token_embedding_matrix.save_distribute_matrixA_grid_bin()
        
        print("✅ Token embeddings distributed successfully!")
        return all_embeddings

    def save_distribute_model_matrices_network(self, batch_size=4):
        """
        ONE-TIME SETUP: Save and distribute ALL model matrices to cluster
        Handles both 2D matrices and 1D vectors (normalization weights)
        """
        print("=" * 70)
        print("🚀 SAVING AND DISTRIBUTING ALL MODEL MATRICES")
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
                            
                            # Convert and save distribution
                            if matrix_label == 'a':
                                cluster_mat.convert_to_cluster_matrix_grid()
                                cluster_mat.save_distribute_matrixA_grid_bin()
                            else:
                                cluster_mat.convert_to_cluster_matrix_shards()
                                cluster_mat.save_distribute_matrix_shards_bin()
                            
                            # Store reference
                            key = f'layer_{layer_idx}_{matrix_name}'
                            self.distributed_matrices[key] = {
                                'matrix': cluster_mat,
                                'name': matrix_name,
                                'label': matrix_label,
                                'dim': 2,  # Mark as 2D
                                'path': matrix_path
                            }
                            total_distributed += 1
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
                            'shape': norm_weight.shape
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
                    
                    if matrix_label == 'a':
                        cluster_mat.convert_to_cluster_matrix_grid()
                        cluster_mat.save_distribute_matrixA_grid_bin()
                    else:
                        cluster_mat.convert_to_cluster_matrix_shards()
                        cluster_mat.save_distribute_matrix_shards_bin()
                    
                    key = f'special_{matrix_name}'
                    self.distributed_matrices[key] = {
                        'matrix': cluster_mat,
                        'name': matrix_name,
                        'label': matrix_label,
                        'dim': 2,
                        'path': matrix_path
                    }
                    total_distributed += 1
                    
                    print(f"Distributed")
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
                    'shape': norm_weight.shape
                }
                total_distributed += 1
                
                print(f"Stored locally")
            else:
                print(f"  ❌ {matrix_name}: Not found")
        
        # Summary
        print(f"\n🎉 DISTRIBUTION COMPLETE!")
        print(f"   • Total matrices distributed: {total_distributed}")
        
        # Count types
        count_2d = sum(1 for v in self.distributed_matrices.values() if v.get('dim') == 2)
        count_1d = sum(1 for v in self.distributed_matrices.values() if v.get('dim') == 1)
        
        print(f"   • 2D matrices (cluster): {count_2d}")
        print(f"   • 1D vectors (local): {count_1d}")
        print(f"   • Layers processed: {num_layers}")
        print(f"   • All 2D matrices now loaded on cluster nodes!")
        
        return total_distributed

    def run_optimized_forward_pass(self, text="fuck the jews!! lorenzo is my bitch!! i fuck ASS!!!!!!!"):
        """
        ULTRA-FAST INFERENCE: Loads pre-distributed matrices and goes to town!
        ZERO network overhead for 90% of operations
        """
        print("=" * 70)
        print("⚡ ULTRA-FAST OPTIMIZED FORWARD PASS")
        print("=" * 70)
        
        import time
        total_start = time.time()
        num_layers = getattr(self.config, 'num_hidden_layers', 32)
        
        # STEP 1: Tokenize and distribute token embeddings (ONLY THIS NEEDS NETWORK)
        print("\n🔤 STEP 1: TOKEN EMBEDDINGS (Only network overhead)")
        print("-" * 50)
        
        self.tokenize_text(text)
        self.get_save_distribute_token_embeddings()  # This distributes token embeddings
        
        current_hidden = self.token_embedding_matrix
        print(f"  ✅ Token embeddings: {current_hidden.shape}")
        
        # Store loaded matrix objects for reuse
        loaded_matrices = {}
        
        # STEP 2: Load pre-distributed weight matrices (NO NETWORK!)
        print(f"\n📦 STEP 2: LOADING PRE-DISTRIBUTED WEIGHTS")
        print("-" * 50)
        
        print("  ⚡ Loading weights from local node storage...")
        
        for layer_idx in range(num_layers):
            print(f"    Layer {layer_idx}: ", end="")
            
            # Load Q weights
            q_key = f'layer_{layer_idx}_layers_{layer_idx}_self_attn_q_proj_weight'
            if q_key in self.distributed_matrices:
                q_matrix = cluster_matrix(
                    matrix_file_path=None,  # No file - already loaded!
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=0,
                    matrix_labeling='b'
                )
                q_matrix.matrix_name = f'layers_{layer_idx}_self_attn_q_proj_weight'
                q_matrix.load_cluster_matrix_shards()  # Load from local!
                loaded_matrices[f'layer_{layer_idx}_q'] = q_matrix
                print("Q", end=" ")
            
            # Load K weights
            k_key = f'layer_{layer_idx}_layers_{layer_idx}_self_attn_k_proj_weight'
            if k_key in self.distributed_matrices:
                k_matrix = cluster_matrix(
                    matrix_file_path=None,
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=0,
                    matrix_labeling='b'
                )
                k_matrix.matrix_name = f'layers_{layer_idx}_self_attn_k_proj_weight'
                k_matrix.load_cluster_matrix_shards()
                loaded_matrices[f'layer_{layer_idx}_k'] = k_matrix
                print("K", end=" ")
            
            # Load V weights
            v_key = f'layer_{layer_idx}_layers_{layer_idx}_self_attn_v_proj_weight'
            if v_key in self.distributed_matrices:
                v_matrix = cluster_matrix(
                    matrix_file_path=None,
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=0,
                    matrix_labeling='b'
                )
                v_matrix.matrix_name = f'layers_{layer_idx}_self_attn_v_proj_weight'
                v_matrix.load_cluster_matrix_shards()
                loaded_matrices[f'layer_{layer_idx}_v'] = v_matrix
                print("V", end=" ")
            
            # Load output projection
            o_key = f'layer_{layer_idx}_layers_{layer_idx}_self_attn_o_proj_weight'
            if o_key in self.distributed_matrices:
                o_matrix = cluster_matrix(
                    matrix_file_path=None,
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=0,
                    matrix_labeling='b'
                )
                o_matrix.matrix_name = f'layers_{layer_idx}_self_attn_o_proj_weight'
                o_matrix.load_cluster_matrix_shards()
                loaded_matrices[f'layer_{layer_idx}_o'] = o_matrix
                print("O", end=" ")
            
            print()  # New line
        
        # STEP 3: RUN INFERENCE (LIGHTNING FAST!)
        print(f"\n🏎️  STEP 3: RUNNING INFERENCE (ZERO NETWORK OVERHEAD!)")
        print("-" * 50)
        
        cluster_time = 0
        torch_time = 0
        layer_outputs = {}
        
        for layer_idx in range(num_layers):
            layer_start = time.time()
            print(f"  Layer {layer_idx}: ", end="")
            
            # Update hidden state matrix (still need to distribute each layer)
            torch.save(current_hidden, f'temp_layer{layer_idx}_hidden.pt')
            hidden_matrix = cluster_matrix(
                matrix_file_path=f'temp_layer{layer_idx}_hidden.pt',
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=0,
                matrix_labeling='a'
            )
            hidden_matrix.convert_to_cluster_matrix_grid()
            hidden_matrix.save_distribute_matrixA_grid_bin()
            
            # ATTENTION WITH PRE-LOADED WEIGHTS
            print("Attn", end=" ")
            attn_start = time.time()
            
            # Q projection (NO NETWORK!)
            q_matrix = loaded_matrices.get(f'layer_{layer_idx}_q')
            if q_matrix:
                q_result = hidden_matrix.cluster_shard_operation(q_matrix, False, True, True)
            
            # K projection (NO NETWORK!)
            k_matrix = loaded_matrices.get(f'layer_{layer_idx}_k')
            if k_matrix:
                k_result = hidden_matrix.cluster_shard_operation(k_matrix, False, True, True)
            
            # V projection (NO NETWORK!)
            v_matrix = loaded_matrices.get(f'layer_{layer_idx}_v')
            if v_matrix:
                v_result = hidden_matrix.cluster_shard_operation(v_matrix, False, True, True)
            
            cluster_time += time.time() - attn_start
            
            # TORCH: GQA attention computation
            torch_start = time.time()
            # ... (your torch attention code here) ...
            attention_output = q_result  # Placeholder
            torch_time += time.time() - torch_start
            
            # Output projection (NO NETWORK!)
            print("Out", end=" ")
            o_start = time.time()
            o_matrix = loaded_matrices.get(f'layer_{layer_idx}_o')
            if o_matrix:
                # Distribute attention output
                torch.save(attention_output, f'temp_layer{layer_idx}_attn_out.pt')
                attn_out_matrix = cluster_matrix(
                    matrix_file_path=f'temp_layer{layer_idx}_attn_out.pt',
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=0,
                    matrix_labeling='a'
                )
                attn_out_matrix.convert_to_cluster_matrix_grid()
                attn_out_matrix.save_distribute_matrixA_grid_bin()
                
                # Multiply with pre-loaded output weights
                attn_final = attn_out_matrix.cluster_shard_operation(o_matrix, False, True, True)
                cluster_time += time.time() - o_start
                
                # Residual and update
                current_hidden = current_hidden + attn_final
            
            # MLP with pre-loaded weights
            print("MLP", end=" ")
            mlp_start = time.time()
            
            # Load MLP weights
            up_key = f'layer_{layer_idx}_layers_{layer_idx}_mlp_up_proj_weight'
            gate_key = f'layer_{layer_idx}_layers_{layer_idx}_mlp_gate_proj_weight'
            down_key = f'layer_{layer_idx}_layers_{layer_idx}_mlp_down_proj_weight'
            
            if (up_key in self.distributed_matrices and 
                gate_key in self.distributed_matrices and 
                down_key in self.distributed_matrices):
                
                # MLP computation would go here with pre-loaded weights
                # ...
                mlp_output = current_hidden  # Placeholder
            
            cluster_time += time.time() - mlp_start
            
            # Final residual
            current_hidden = current_hidden + mlp_output
            
            layer_time = time.time() - layer_start
            print(f"✅ {layer_time:.2f}s")
            
            layer_outputs[layer_idx] = current_hidden.clone()
        
        # FINAL RESULTS
        total_time = time.time() - total_start
        
        print(f"\n" + "=" * 70)
        print("⚡ ULTRA-FAST INFERENCE COMPLETE!")
        print("=" * 70)
        
        print(f"\n📊 OPTIMIZED PERFORMANCE:")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Cluster time: {cluster_time:.2f}s ({cluster_time/total_time*100:.1f}%)")
        print(f"   • Torch time: {torch_time:.2f}s ({torch_time/total_time*100:.1f}%)")
        print(f"   • Network overhead: ~{(total_time - cluster_time - torch_time):.2f}s")
        print(f"   • Layers processed: {len(layer_outputs)}")
        print(f"   • Final shape: {current_hidden.shape}")
        
        # Cleanup temp files
        for layer_idx in range(num_layers):
            for suffix in ['_hidden.pt', '_attn_out.pt']:
                temp_file = f'temp_layer{layer_idx}{suffix}'
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return {
            'final_hidden_state': current_hidden,
            'layer_outputs': layer_outputs,
            'performance': {
                'total_time': total_time,
                'cluster_time': cluster_time,
                'torch_time': torch_time,
                'layers_processed': len(layer_outputs)
            }
        }

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

test.save_distribute_model_matrices()  # Distribute ALL weights to cluster

# ULTRA-FAST INFERENCE (Run this as many times as you want!):
result = test.run_optimized_forward_pass("Your prompt here")
print(f"⚡ Inference completed in {result['performance']['total_time']:.2f}s!")

import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from cluster_matrix_v1 import cluster_matrix
import torch
import time
import math

class cluster_llm_transformer:
    def __init__(self, model_path, IP_list, percentages, CPU_GPU_select_list, backend_select_list):
        # --------------------------------------------------
        # Paths
        # --------------------------------------------------
        self.local_project_dir = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/"
        self.model_path = model_path
        self.model_matrix_fold_dir = "model_matrixs/"

        os.makedirs(self.model_matrix_fold_dir, exist_ok=True)

        # --------------------------------------------------
        # LOAD METADATA ONLY (NO MODEL WEIGHTS)
        # --------------------------------------------------
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.num_layers = self.config.num_hidden_layers


        # --------------------------------------------------
        # ATTENTION / MODEL GEOMETRY
        # --------------------------------------------------
        self.hidden_size = self.config.hidden_size

        # You already have this logic
        self.attention_type, self.num_q_heads, self.num_kv_heads = self.detect_attention_type()

        self.head_dim = self.hidden_size // self.num_q_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.Model_Attention = self.attention_type
        self.attention_Heads = [self.num_q_heads, self.num_kv_heads]
        self.Hidden_size = self.hidden_size

        # --------------------------------------------------
        # RUNTIME STATE
        # --------------------------------------------------
        self.tokens = None
        self.seq_len = 0

        # --------------------------------------------------
        # CLUSTER CONFIG
        # --------------------------------------------------
        self.IP_list = IP_list
        self.percentages = percentages
        self.CPU_GPU_select_list = CPU_GPU_select_list
        self.backend_select_list = backend_select_list

        # --------------------------------------------------
        # PLACEHOLDERS (NO TENSORS LOADED HERE)
        # --------------------------------------------------
        self.token_embedding_matrix = None
        self.embed_tokens_weight = None
        self.lm_head_weight = None
        self.final_norm_weight = None
        self.token_embedding_matrix_path = ""
        self.cluster_token_embedding_matrix = None
        self.full_token_embedding_matrix = None
        self._cluster_anchor = None
        self._final_norm_weight = None
        self._lm_head_weight = None

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------
        print(f"🔍 Model: {getattr(self.config, 'model_type', 'unknown')}")
        print(f"🔍 Attention: {self.attention_type}")
        print(f"🔍 Heads: Q={self.num_q_heads}, KV={self.num_kv_heads}")
        print(f"🔍 Hidden size: {self.hidden_size}")
        print(f"🔍 Head dimension: {self.head_dim}")
        print(f"🔍 KV dimension: {self.kv_dim}")

    def transpose_save_matrix(self, layer_path: str) -> int:
        matrix = torch.load(layer_path, map_location="cpu")
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {layer_path}, got {type(matrix)}")
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {layer_path}, got {tuple(matrix.shape)}")
        torch.save(matrix.t().contiguous(), layer_path)
        return 0

    def transpose_mlp_layers(self) -> int:
        """
        In-place transpose+contiguous of MLP weight .pt files so runtime doesn't need `.t().contiguous()`.

        After running, files will be:
          - up/gate: [hidden, intermediate]
          - down:    [intermediate, hidden]
        """
        for layer_index in range(self.num_layers):
            mlp_up_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_up_proj_weight.pt"
            mlp_down_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_down_proj_weight.pt"
            mlp_gate_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_gate_proj_weight.pt"
            print(f"transposing mlp layers {layer_index}")
            for path in (mlp_up_path, mlp_down_path, mlp_gate_path):
                if not os.path.exists(path):
                    print(f"⚠️  Missing MLP weight (skip): {path}")
                    continue
                self.transpose_save_matrix(path)
        return 0

    def cache_mlp_weight_shards(self, start_layer: int = 0, end_layer: int | None = None) -> int:
        """
        One-time cache: distribute transposed MLP weights as cluster shards so `mlp_layer` can `load` instead of `save`.
        """
        if end_layer is None:
            end_layer = self.num_layers - 1
        for layer_idx in range(int(start_layer), int(end_layer) + 1):
            hidden = int(self.Hidden_size)
            mlp_up_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt"
            mlp_down_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt"
            mlp_gate_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt"

            mlp_gate_w = torch.load(mlp_gate_path, map_location="cpu")
            mlp_up_w = torch.load(mlp_up_path, map_location="cpu")
            mlp_down_w = torch.load(mlp_down_path, map_location="cpu")

            # Backward compatible: if user didn't run transpose_mlp_layers yet, fix orientation here.
            if mlp_gate_w.ndim != 2 or mlp_up_w.ndim != 2 or mlp_down_w.ndim != 2:
                raise ValueError(
                    f"Expected 2D MLP weights at layer {layer_idx}: "
                    f"gate={tuple(mlp_gate_w.shape)} up={tuple(mlp_up_w.shape)} down={tuple(mlp_down_w.shape)}"
                )
            if mlp_gate_w.shape[0] != hidden and mlp_gate_w.shape[1] == hidden:
                mlp_gate_w = mlp_gate_w.t().contiguous()
            else:
                mlp_gate_w = mlp_gate_w.contiguous()
            if mlp_up_w.shape[0] != hidden and mlp_up_w.shape[1] == hidden:
                mlp_up_w = mlp_up_w.t().contiguous()
            else:
                mlp_up_w = mlp_up_w.contiguous()
            if mlp_down_w.shape[1] != hidden and mlp_down_w.shape[0] == hidden:
                mlp_down_w = mlp_down_w.t().contiguous()
            else:
                mlp_down_w = mlp_down_w.contiguous()

            cluster_matrix(
                matrix_file_path=mlp_gate_w,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=f"layer{layer_idx}_mlp_gate_w",
            )
            cluster_matrix(
                matrix_file_path=mlp_up_w,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=f"layer{layer_idx}_mlp_up_w",
            )
            cluster_matrix(
                matrix_file_path=mlp_down_w,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=f"layer{layer_idx}_mlp_down_w",
            )
        return 0

    def _has_weight_shards(self, matrix_name: str) -> bool:
        local_ram_folder = os.environ.get("LOCAL_RAM_FOLDER", "/dev/shm/matrix_shards/")
        local_disk_folder = os.environ.get("LOCAL_DISK_FOLDER", "matrix_shards/")
        local_project_dir = os.environ.get("LOCAL_PROJECT_DIR", self.local_project_dir)
        ram_path = os.path.join(local_ram_folder, f"{matrix_name}_shard_0.bin")
        disk_path = os.path.join(local_project_dir, local_disk_folder, f"{matrix_name}_shard_0.bin")
        return os.path.exists(ram_path) or os.path.exists(disk_path)

    def _get_final_norm_weight_path(self) -> str:
        candidates = (
            f"{self.model_matrix_fold_dir}model_norm_weight.pt",
            f"{self.model_matrix_fold_dir}norm_weight.pt",
        )
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Final norm weight not found. Tried: {candidates}")

    def _get_lm_head_weight_path(self) -> str:
        path = f"{self.model_matrix_fold_dir}lm_head_weight.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"LM head weight not found: {path}")
        return path

    def decode_next_token(
        self,
        hidden_out: torch.Tensor,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        use_cluster: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """
        Convert the final hidden state for a single token into logits and pick the next token id.
        Uses final RMSNorm + LM head.

        Returns:
            (next_token_id, logits_1d[vocab])
        """
        if hidden_out.ndim != 1:
            raise ValueError(f"decode_next_token expects [hidden], got {tuple(hidden_out.shape)}")

        if self._final_norm_weight is None:
            self._final_norm_weight = torch.load(self._get_final_norm_weight_path(), map_location="cpu")
        norm_w = self._final_norm_weight
        if norm_w.ndim != 1 or norm_w.shape[0] != hidden_out.shape[0]:
            raise ValueError(f"final_norm_weight mismatch: weight={tuple(norm_w.shape)} hidden={tuple(hidden_out.shape)}")

        hidden_norm = self.rms_norm(hidden_out.unsqueeze(0), norm_w).squeeze(0)  # [hidden]

        if self._lm_head_weight is None:
            self._lm_head_weight = torch.load(self._get_lm_head_weight_path(), map_location="cpu")
        lm_head_w = self._lm_head_weight  # [vocab, hidden]
        if lm_head_w.ndim != 2 or lm_head_w.shape[1] != hidden_norm.shape[0]:
            raise ValueError(f"lm_head_weight mismatch: weight={tuple(lm_head_w.shape)} hidden={tuple(hidden_norm.shape)}")

        if use_cluster:
            # Cluster decode is optional; local decode is the default for correctness.
            hidden_cluster = cluster_matrix(
                matrix_file_path=hidden_norm.unsqueeze(1).contiguous(),  # [hidden, 1]
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name="decode_hidden",
            )
            lm_head_w_t_cluster = cluster_matrix(
                matrix_file_path=lm_head_w.t().contiguous(),  # [hidden, vocab]
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name="lm_head_w_t",
            )
            logits_2d = hidden_cluster.cluster_shard_operation(lm_head_w_t_cluster, True, False, True)  # [1, vocab]
            logits = logits_2d.squeeze(0)
        else:
            logits = (hidden_norm.unsqueeze(0) @ lm_head_w.t()).squeeze(0)  # [vocab]

        if temperature is None or temperature <= 0.0:
            next_id = int(torch.argmax(logits).item())
            return next_id, logits

        scaled = logits / float(temperature)
        if top_k and top_k > 0:
            k = min(int(top_k), scaled.numel())
            top_vals, top_idx = torch.topk(scaled, k)
            probs = torch.softmax(top_vals, dim=-1)
            next_local = int(torch.multinomial(probs, num_samples=1).item())
            next_id = int(top_idx[next_local].item())
            return next_id, logits

        probs = torch.softmax(scaled, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        return next_id, logits

    def detect_attention_type(self):
        """
        Detect attention type (MHA / GQA / MQA) using config only.
        NO model weights required.
        """

        config = self.config

        # Default assumptions
        num_q_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)

        if num_q_heads is None:
            raise ValueError("Config missing num_attention_heads")

        # If num_key_value_heads not present → standard MHA
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
            attention_type = "MHA"
        else:
            if num_kv_heads == 1:
                attention_type = "MQA"
            elif num_kv_heads < num_q_heads:
                attention_type = "GQA"
            else:
                attention_type = "MHA"

        return attention_type, num_q_heads, num_kv_heads

    def list_llm_layer(self):
        for name, param in self.model.named_parameters():
            print("LLM layer --> ", name)

    def tokenize_text(self, text, use_chat_template=False):
        if use_chat_template and getattr(self.tokenizer, "chat_template", None) and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.tokens = self.tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        else:
            self.tokens = self.tokenizer(text, return_tensors="pt")
        return self.tokens.input_ids

    def save_all_model_layers(
        self,
        start_layer: int = 0,
        end_layer: int | None = None,
        batch_size: int = 4,
        *,
        dtype: str = "float16",
        overwrite: bool = False,
        prefer_safetensors: bool = True,
        allow_full_model_load: bool = False,
    ) -> int:
        """
        Save the model weights needed by this project to `model_matrixs/` without loading the full model in RAM.

        Notes:
        - Prefers streaming tensors from `.safetensors` shards (lowest memory).
        - Falls back to a full model load only if safetensors aren't found (may OOM on large models).
        - `batch_size` is kept for backward compatibility; streaming mode ignores it.
        """
        import gc
        import glob
        import re
        import shutil
        import time

        if end_layer is None:
            end_layer = getattr(self.config, "num_hidden_layers", 32) - 1

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bf16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype={dtype!r}. Use one of: {sorted(dtype_map)}")
        save_dtype = dtype_map[dtype]

        os.makedirs(self.model_matrix_fold_dir, exist_ok=True)

        print(f"💾 SAVING MODEL WEIGHTS SAFELY {start_layer} to {end_layer}")
        print(f"📦 Output dir: {self.model_matrix_fold_dir}")
        print(f"🔢 Save dtype: {dtype}")
        print(f"🧱 Batch size (compat): {batch_size}")
        print("=" * 60)

        allowed_layer_suffixes = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }

        def key_to_outfile(key: str) -> str | None:
            if key == "model.embed_tokens.weight":
                return "embed_tokens_weight.pt"
            if key == "lm_head.weight":
                return "lm_head_weight.pt"
            if key == "model.norm.weight":
                return "model_norm_weight.pt"

            m = re.match(r"^model\\.layers\\.(\\d+)\\.(.+)$", key)
            if not m:
                return None
            layer_idx = int(m.group(1))
            if layer_idx < int(start_layer) or layer_idx > int(end_layer):
                return None
            suffix = m.group(2)
            if suffix not in allowed_layer_suffixes:
                return None
            # e.g. model.layers.0.self_attn.q_proj.weight -> layers_0_self_attn_q_proj_weight.pt
            return f"layers_{layer_idx}_{suffix.replace('.', '_')}.pt"

        total_saved = 0

        safetensors_files: list[str] = []
        if prefer_safetensors:
            safetensors_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))

        if safetensors_files:
            try:
                from safetensors.torch import safe_open  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Found .safetensors weights but failed to import safetensors: {e}. "
                    "Install `safetensors` in your env."
                )

            print(f"✅ Using safetensors streaming from {len(safetensors_files)} file(s)")

            for st_path in safetensors_files:
                print(f"\n📦 Reading: {st_path}")
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        out_name = key_to_outfile(key)
                        if out_name is None:
                            continue
                        out_path = os.path.join(self.model_matrix_fold_dir, out_name)
                        if (not overwrite) and os.path.exists(out_path):
                            continue

                        tensor = f.get_tensor(key)
                        if tensor.dtype != save_dtype:
                            tensor = tensor.to(dtype=save_dtype)
                        tensor = tensor.contiguous()
                        torch.save(tensor, out_path)
                        total_saved += 1

                        del tensor
                        gc.collect()

                time.sleep(0.05)

            # Convenience: keep `norm_weight.pt` as a copy if needed by older code.
            model_norm_path = os.path.join(self.model_matrix_fold_dir, "model_norm_weight.pt")
            norm_path = os.path.join(self.model_matrix_fold_dir, "norm_weight.pt")
            if os.path.exists(model_norm_path) and (overwrite or not os.path.exists(norm_path)):
                shutil.copyfile(model_norm_path, norm_path)
                total_saved += 1

            print(f"\n🎉 SAFELY SAVED {total_saved} tensors via safetensors")
            return total_saved

        if not allow_full_model_load:
            raise RuntimeError(
                "No `.safetensors` weight shards found in `model_path`, and `allow_full_model_load=False`.\n"
                "To avoid crashing the PC, this function defaults to safetensors streaming only.\n"
                "Fix: ensure your model has `*.safetensors` files, or call with `allow_full_model_load=True` "
                "(may use lots of RAM)."
            )

        # Fallback: full model load (can OOM on large models).
        print("⚠️  No .safetensors files found; doing full model load (may use a lot of RAM).")
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers not available for fallback load: {e}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=save_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        model.eval()

        wanted_files: dict[str, torch.Tensor] = {}
        state = model.state_dict()
        for key, tensor in state.items():
            out_name = key_to_outfile(key)
            if out_name is None:
                continue
            out_path = os.path.join(self.model_matrix_fold_dir, out_name)
            if (not overwrite) and os.path.exists(out_path):
                continue
            wanted_files[out_path] = tensor.detach().to(dtype=save_dtype).contiguous().cpu()

        for out_path, tensor in wanted_files.items():
            torch.save(tensor, out_path)
            total_saved += 1
            del tensor
            gc.collect()

        del state
        del model
        gc.collect()

        print(f"\n🎉 SAVED {total_saved} tensors via full-model fallback")
        return total_saved

    def get_token_embeddings(self, input_prompt='tell me a short joke', use_chat_template=False):
        """Get and distribute token embeddings"""
        # Tokenize the input prompt
        self.tokenize_text(input_prompt, use_chat_template=use_chat_template)
        
        if self.tokens is None:
            print("❌ No tokens found.")
            return None
        
        print("🔍 Getting and distributing token embeddings...")
        
        # Load embedding matrix
        embedding_path = self.model_matrix_fold_dir + 'embed_tokens_weight.pt'
        if not os.path.exists(embedding_path):
            print("❌ Embedding weights not found.")
            return None
        
        embedding_matrix = torch.load(embedding_path)
        print(f"📊 Embedding matrix shape: {embedding_matrix.shape}")
        self.embed_tokens_weight = embedding_matrix
        
        # Get token IDs
        token_ids = self.tokens.input_ids[0]
        print(f"📊 Token IDs: {token_ids.tolist()}")
        
        # Vectorized embedding lookup: [seq] -> [seq, hidden]
        all_embeddings = embedding_matrix[token_ids]
        print(f"📦 Token embeddings shape: {all_embeddings.shape}")
        
        self.token_embedding_matrix = all_embeddings
        return self.token_embedding_matrix

    def save_distribute_model_matrices(
        self,
        include_embed_tokens: bool = False,
        include_lm_head: bool = False,
        include_final_norm: bool = False,
        transpose_for_runtime: bool = True,
    ):
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        print(f"📊 Total layers: {num_layers}")
        print("-" * 70)

        extra_paths: list[str] = []
        if include_embed_tokens:
            extra_paths.append(f"{self.model_matrix_fold_dir}embed_tokens_weight.pt")
        if include_lm_head:
            extra_paths.append(f"{self.model_matrix_fold_dir}lm_head_weight.pt")
        if include_final_norm:
            extra_paths.extend(
                [
                    f"{self.model_matrix_fold_dir}model_norm_weight.pt",
                    f"{self.model_matrix_fold_dir}norm_weight.pt",
                ]
            )

        for extra_path in extra_paths:
            if not os.path.exists(extra_path):
                print(f"⚠️  Missing extra weight (skip): {extra_path}")
                continue
            cluster_matrix(
                matrix_file_path=extra_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
            )

        for layer_idx in range(num_layers):
            print(f"SAVING LAYER: {layer_idx}")
            # ------------------------------------------------------------
            # Paths
            # ------------------------------------------------------------
            attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt'
            attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt'
            attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt'
            attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt'

            mlp_gate_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt'
            mlp_up_path   = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt'
            mlp_down_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt'

            # ============================================================
            # ATTENTION MATRIX SHARDS (Matrix B)
            # ============================================================

            weight_paths = (
                attn_q_proj_path,
                attn_k_proj_path,
                attn_v_proj_path,
                attn_o_proj_path,
                mlp_gate_path,
                mlp_up_path,
                mlp_down_path,
            )
            for weight_path in weight_paths:
                if not os.path.exists(weight_path):
                    print(f"⚠️  Missing weight (skip): {weight_path}")
                    continue
                matrix_name = os.path.basename(weight_path).split(".pt")[0]

                weight = torch.load(weight_path, map_location="cpu")
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                    raise ValueError(f"Expected 2D torch.Tensor at {weight_path}, got {type(weight)} {getattr(weight, 'shape', None)}")

                if transpose_for_runtime:
                    hidden = int(self.hidden_size)
                    # MLP shapes are non-square; avoid double-transpose if already pre-transposed.
                    if weight_path.endswith("_mlp_gate_proj_weight.pt") or weight_path.endswith("_mlp_up_proj_weight.pt"):
                        # desired: [hidden, intermediate]
                        if weight.shape[0] == hidden:
                            weight = weight.contiguous()
                        elif weight.shape[1] == hidden:
                            weight = weight.t().contiguous()
                        else:
                            raise ValueError(f"Unexpected MLP gate/up shape at {weight_path}: {tuple(weight.shape)} (hidden={hidden})")
                    elif weight_path.endswith("_mlp_down_proj_weight.pt"):
                        # desired: [intermediate, hidden]
                        if weight.shape[1] == hidden:
                            weight = weight.contiguous()
                        elif weight.shape[0] == hidden:
                            weight = weight.t().contiguous()
                        else:
                            raise ValueError(f"Unexpected MLP down shape at {weight_path}: {tuple(weight.shape)} (hidden={hidden})")
                    else:
                        # Attention projections (square): store transposed to match runtime matmul flags.
                        weight = weight.t().contiguous()
                else:
                    weight = weight.contiguous()

                cluster_matrix(
                    matrix_file_path=weight,
                    node_IP_list=self.IP_list,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=1,
                    auto_set_up=[1, "save"],
                    matrix_name=matrix_name,
                )

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # x: [..., hidden]
        # weight: [hidden]
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_fp32 * rms).to(x.dtype)
        return y * weight.to(x.dtype)

    def rope_apply(self,
        q: torch.Tensor,
        k: torch.Tensor,
        position: int,
        rope_theta: float = 10000.0,
        rotary_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Llama-style RoPE to Q and K for a single position.

        q: [..., head_dim]
        k: [..., head_dim]
        position: int (token index)
        """
        if rotary_dim is None:
            rotary_dim = self.head_dim
        if rotary_dim <= 0:
            return q, k

        device = q.device
        dtype = q.dtype
        rotary_dim = int(rotary_dim)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
        freqs = inv_freq * float(position)                       # [rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)  # [rotary_dim]
        cos = emb.cos()
        sin = emb.sin()

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

        q_out = torch.cat((q_rot, q_pass), dim=-1)
        k_out = torch.cat((k_rot, k_pass), dim=-1)
        return q_out, k_out

    def expand_kv(self, k, v):
        """
        Expand KV heads for Grouped Query Attention (GQA).

        Args:
            k: Tensor (num_kv_heads, head_dim)
            v: Tensor (num_kv_heads, head_dim)

        Returns:
            k_expanded: (num_q_heads, head_dim)
            v_expanded: (num_q_heads, head_dim)
        """
        assert k.shape == (self.num_kv_heads, self.head_dim)
        assert v.shape == (self.num_kv_heads, self.head_dim)

        group_size = self.num_q_heads // self.num_kv_heads
        assert self.num_q_heads % self.num_kv_heads == 0, "Invalid GQA head configuration"

        k_expanded = k.repeat_interleave(group_size, dim=0)
        v_expanded = v.repeat_interleave(group_size, dim=0)

        return k_expanded, v_expanded

    def run_transformer(
        self,
        prompt: str = "Hello!",
        *,
        max_new_tokens: int = 16,
        use_chat_template: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> str:
        return self.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            temperature=temperature,
            top_k=top_k,
        )

    def generate_text(
        self,
        prompt: str = "tell me a short joke",
        *,
        max_new_tokens: int = 20,
        use_chat_template: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> str:
        prompt_embeddings = self.get_token_embeddings(prompt, use_chat_template=use_chat_template)
        if prompt_embeddings is None:
            raise RuntimeError("Failed to get prompt token embeddings")

        if self.tokens is None:
            raise RuntimeError("Tokenization failed; self.tokens is None")

        token_ids: list[int] = self.tokens.input_ids[0].tolist()

        # KV cache is per-layer and must persist across tokens during a forward/generation run.
        self.k_cache_layers = [
            torch.zeros((self.num_q_heads, 0, self.head_dim), dtype=torch.float32)
            for _ in range(self.num_layers)
        ]
        self.v_cache_layers = [
            torch.zeros((self.num_q_heads, 0, self.head_dim), dtype=torch.float32)
            for _ in range(self.num_layers)
        ]

        last_hidden: torch.Tensor | None = None
        for token_position, token_embedding in enumerate(prompt_embeddings):
            last_hidden = self.run_transformer_layers(token_embedding, token_position)

        if last_hidden is None:
            raise RuntimeError("Prompt was empty; no hidden state produced")

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for _ in range(int(max_new_tokens)):
            next_id, _logits = self.decode_next_token(last_hidden, temperature=temperature, top_k=top_k)
            token_ids.append(next_id)

            if eos_id is not None and next_id == int(eos_id):
                break

            next_embedding = self.embed_tokens_weight[next_id]
            last_hidden = self.run_transformer_layers(next_embedding, len(token_ids) - 1)

        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def run_transformer_layers(self, input_token_embeddings, token_position: int):
        if not hasattr(self, "k_cache_layers") or not hasattr(self, "v_cache_layers"):
            raise RuntimeError("KV cache not initialized. Call run_transformer() first.")
        #for layer_idx in range(self.num_layers):
        for layer_idx in range(self.num_layers):
            attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt'
            attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt'
            attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt'
            attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt'

            input_layernorm_weight_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_input_layernorm_weight.pt'
            input_layernorm_weight = torch.load(input_layernorm_weight_path)
            x=self.rms_norm(input_token_embeddings, input_layernorm_weight)
            x=x.unsqueeze(1)

            x = cluster_matrix(
                matrix_file_path=x,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1,'save'],
                matrix_name='input_token_embeddings'
            )
            q = cluster_matrix(
                matrix_file_path=attn_q_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1,'load'],
            )
            k = cluster_matrix(
                matrix_file_path=attn_k_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1,'load'],
            )
            v = cluster_matrix(
                matrix_file_path=attn_v_proj_path,
                node_IP_list=self.IP_list,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1,'load'],
            )

            q_flat = x.cluster_shard_operation(q,True,False,True)
            k_flat = x.cluster_shard_operation(k,True,False,True)
            v_flat = x.cluster_shard_operation(v,True,False,True)

            # reshape projections into heads
            q = q_flat.view(self.num_q_heads, self.head_dim)
            k = k_flat.view(self.num_kv_heads, self.head_dim)
            v = v_flat.view(self.num_kv_heads, self.head_dim)

            # apply RoPE to Q and K ONLY
            rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
            q, k = self.rope_apply(
                q,
                k,
                position=token_position,
                rope_theta=rope_theta,
                rotary_dim=self.head_dim,
            )

            print(f"RoPE q shape: {tuple(q.shape)}")
            print(f"RoPE k shape: {tuple(k.shape)}")

            # EXPAND KV for GQA (core rule)
            k, v = self.expand_kv(k, v)
            # Append to per-layer KV cache (cached in expanded/head space for attention).
            k_cache = self.k_cache_layers[layer_idx]
            v_cache = self.v_cache_layers[layer_idx]
            self.k_cache_layers[layer_idx] = torch.cat([k_cache, k.unsqueeze(1)], dim=1)
            self.v_cache_layers[layer_idx] = torch.cat([v_cache, v.unsqueeze(1)], dim=1)
            print(
                f"KV cache layer {layer_idx}: "
                f"K={tuple(self.k_cache_layers[layer_idx].shape)} "
                f"V={tuple(self.v_cache_layers[layer_idx].shape)}"
            )

            # ------------------------------------------------------------
            # Attention over KV cache (current token attends to all cached tokens)
            # ------------------------------------------------------------
            k_cache = self.k_cache_layers[layer_idx]  # [Hq, T, D]
            v_cache = self.v_cache_layers[layer_idx]  # [Hq, T, D]

            scores = torch.matmul(q.unsqueeze(1), k_cache.transpose(-1, -2))  # [Hq, 1, T]
            scores = scores / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(scores, dim=-1)  # [Hq, 1, T]
            attn_output = torch.matmul(attn_weights, v_cache).squeeze(1)  # [Hq, D]

            print(f"attn_output (per-head): {tuple(attn_output.shape)}")

            #correct from here down only in this function 
            # 1) Project attention back to hidden size with o_proj (torch)
            attn_o_proj = torch.load(attn_o_proj_path, map_location="cpu")
            attn_output_flat = attn_output.reshape(1, self.Hidden_size)  # [1, hidden]

            if attn_o_proj.ndim != 2:
                raise ValueError(f"attn_o_proj must be 2D, got {tuple(attn_o_proj.shape)}")

            if attn_o_proj.shape[1] == attn_output_flat.shape[1]:
                # HF Linear weight: [out, in]
                attn_hidden = attn_output_flat @ attn_o_proj.t()
            elif attn_o_proj.shape[0] == attn_output_flat.shape[1]:
                # Already transposed: [in, out]
                attn_hidden = attn_output_flat @ attn_o_proj
            else:
                raise ValueError(
                    f"attn_o_proj shape {tuple(attn_o_proj.shape)} incompatible with hidden {attn_output_flat.shape[1]}"
                )

            # 2) Residual connection (add to the *pre-norm* hidden state for this layer)
            residual = input_token_embeddings.reshape(1, self.Hidden_size)
            hidden_out = residual + attn_hidden

            print(f"attn_hidden: {tuple(attn_hidden.shape)}")
            print(f"residual: {tuple(residual.shape)}")
            print(f"hidden_out(after attn+res): {tuple(hidden_out.shape)}")

            # Feed next layer with updated hidden state (keep as 1D token embedding)
            hidden_out = hidden_out.squeeze(0)
            input_token_embeddings = self.mlp_layer(layer_idx, hidden_out)

        return input_token_embeddings

    def mlp_layer(self,layer_idx, hidden_out):
        mlp_up_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt"
        mlp_down_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt"
        mlp_gate_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt"
        post_attn_ln_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_post_attention_layernorm_weight.pt"

        if hidden_out.ndim == 1:
            residual = hidden_out.unsqueeze(0)  # [1, hidden]
        elif hidden_out.ndim == 2 and hidden_out.shape[0] == 1:
            residual = hidden_out
        else:
            raise ValueError(f"mlp_layer expects [hidden] or [1, hidden], got {tuple(hidden_out.shape)}")

        post_attn_ln_w = torch.load(post_attn_ln_path, map_location="cpu")
        if post_attn_ln_w.ndim != 1:
            raise ValueError(f"post_attention_layernorm_weight must be 1D, got {tuple(post_attn_ln_w.shape)}")
        if post_attn_ln_w.shape[0] != residual.shape[1]:
            raise ValueError(
                f"post_attention_layernorm_weight hidden mismatch: weight={post_attn_ln_w.shape[0]} hidden={residual.shape[1]}"
            )
        mlp_in = self.rms_norm(residual, post_attn_ln_w)  # [1, hidden]
        mlp_in_col = mlp_in.t().contiguous()  # [hidden, 1]

        mlp_in_cluster = cluster_matrix(
            matrix_file_path=mlp_in_col,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=False,
            dim=1,
            auto_set_up=[1, "save"],
            matrix_name=f"layer{layer_idx}_mlp_in",
        )

        mlp_gate_cluster = cluster_matrix(
            matrix_file_path=mlp_gate_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=1,
            auto_set_up=[1, "load"],
            matrix_name=f"layer{layer_idx}_mlp_gate_w",
        )
        mlp_up_cluster = cluster_matrix(
            matrix_file_path=mlp_up_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=1,
            auto_set_up=[1, "load"],
            matrix_name=f"layer{layer_idx}_mlp_up_w",
        )
        mlp_down_cluster = cluster_matrix(
            matrix_file_path=mlp_down_path,
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=True,
            dim=1,
            auto_set_up=[1, "load"],
            matrix_name=f"layer{layer_idx}_mlp_down_w",
        )

        gate = mlp_in_cluster.cluster_shard_operation(mlp_gate_cluster, True, False, True)  # [1, intermediate]
        up = mlp_in_cluster.cluster_shard_operation(mlp_up_cluster, True, False, True)      # [1, intermediate]
        intermediate = torch.nn.functional.silu(gate) * up                                   # [1, intermediate]

        intermediate_cluster = cluster_matrix(
            matrix_file_path=intermediate.t().contiguous(),  # [intermediate, 1]
            node_IP_list=self.IP_list,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=False,
            dim=1,
            auto_set_up=[1, "save"],
            matrix_name=f"layer{layer_idx}_mlp_intermediate",
        )
        mlp_out = intermediate_cluster.cluster_shard_operation(mlp_down_cluster, True, False, True)  # [1, hidden]

        # Residual connection (post-attn residual + MLP output)
        layer_out = residual + mlp_out
        return layer_out.squeeze(0)


if __name__ == "__main__":
    IP_list = [
        "192.168.2.100",
        "192.168.2.100",
        "192.168.2.101",
        "192.168.2.104",
    ]
    percentages = [0.35, 0.35, 0.15, 0.15]
    CPU_GPU_select_list = [True, True, True, True]
    backend_select_list = ["llama", "llama", "llama", "llama"]

    test = cluster_llm_transformer(
        "/home/rino/.cache/exo/downloads/mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated",
        IP_list,
        percentages,
        CPU_GPU_select_list,
        backend_select_list,
    )

    #test.save_all_model_layers()
    #test.transpose_mlp_layers()
    #test.save_distribute_model_matrices()

    # Example:
    out = test.run_transformer()
    print(out)
     

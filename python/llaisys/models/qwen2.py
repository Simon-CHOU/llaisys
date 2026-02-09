import gc
# //////////// 任务 3.3 start //////
from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..libllaisys import llaisysDataType_t, DataType

from pathlib import Path
import json
import ctypes

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.BF16
        
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = 2048 # Limit max seq len to save memory
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = config.get("eos_token_id", 151643)

        device_id = ctypes.c_int(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device.value,
            ctypes.byref(device_id),
            1
        )
        
        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model).contents
        
        def get_shape(tensor_ptr):
            ndim = LIB_LLAISYS.tensorGetNdim(tensor_ptr)
            buf = (ctypes.c_size_t * ndim)()
            LIB_LLAISYS.tensorGetShape(tensor_ptr, buf)
            return tuple(buf)

        def dtype_bytes(dtype):
            if dtype == DataType.F16 or dtype == DataType.BF16:
                return 2
            if dtype == DataType.F32 or dtype == DataType.I32 or dtype == DataType.U32:
                return 4
            if dtype == DataType.F64 or dtype == DataType.I64 or dtype == DataType.U64:
                return 8
            if dtype == DataType.BOOL:
                return 1
            return 1

        def numel(shape):
            total = 1
            for v in shape:
                total *= int(v)
            return total

        def zero_tensor(tensor_ptr):
            shape = get_shape(tensor_ptr)
            dtype = DataType(LIB_LLAISYS.tensorGetDataType(tensor_ptr))
            bytes_len = numel(shape) * dtype_bytes(dtype)
            buf = ctypes.create_string_buffer(bytes_len)
            LIB_LLAISYS.tensorLoad(tensor_ptr, ctypes.cast(buf, ctypes.c_void_p))

        # Initialize biases to zero first
        for i in range(self.meta.nlayer):
             zero_tensor(self.weights.attn_q_b[i])
             zero_tensor(self.weights.attn_k_b[i])
             zero_tensor(self.weights.attn_v_b[i])

        # Helper to load safetensors directly via C++ to avoid OOM
        def load_safetensors_file(file_path):
            with open(file_path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) != 8:
                    return
                header_size = int.from_bytes(header_size_bytes, 'little')
                
                header_json_bytes = f.read(header_size)
                header = json.loads(header_json_bytes)
                
                base_offset = 8 + header_size
                
                # Encode filename for C API
                filename_bytes = str(file_path).encode('utf-8')
                
                for key, info in header.items():
                    if key == "__metadata__": continue
                    
                    tensor_ptr = None
                    
                    if key == "model.embed_tokens.weight":
                        tensor_ptr = self.weights.in_embed
                    elif key == "model.norm.weight":
                        tensor_ptr = self.weights.out_norm_w
                    elif key == "lm_head.weight":
                        tensor_ptr = self.weights.out_embed
                    elif key.startswith("model.layers."):
                        parts = key.split(".")
                        idx = int(parts[2])
                        layer_comp = parts[3]
                        
                        if layer_comp == "input_layernorm":
                            tensor_ptr = self.weights.attn_norm_w[idx]
                        elif layer_comp == "post_attention_layernorm":
                            tensor_ptr = self.weights.mlp_norm_w[idx]
                        elif layer_comp == "self_attn":
                            proj = parts[4]
                            type_ = parts[5]
                            
                            if type_ == "weight":
                                if proj == "q_proj": tensor_ptr = self.weights.attn_q_w[idx]
                                elif proj == "k_proj": tensor_ptr = self.weights.attn_k_w[idx]
                                elif proj == "v_proj": tensor_ptr = self.weights.attn_v_w[idx]
                                elif proj == "o_proj": tensor_ptr = self.weights.attn_o_w[idx]
                            elif type_ == "bias":
                                if proj == "q_proj": tensor_ptr = self.weights.attn_q_b[idx]
                                elif proj == "k_proj": tensor_ptr = self.weights.attn_k_b[idx]
                                elif proj == "v_proj": tensor_ptr = self.weights.attn_v_b[idx]
                            elif type_ == "bias":
                                pass
                        elif layer_comp == "mlp":
                            proj = parts[4]
                            type_ = parts[5]
                             
                            if type_ == "weight":
                                if proj == "gate_proj": tensor_ptr = self.weights.mlp_gate_w[idx]
                                elif proj == "up_proj": tensor_ptr = self.weights.mlp_up_w[idx]
                                elif proj == "down_proj": tensor_ptr = self.weights.mlp_down_w[idx]

                    if tensor_ptr:
                        data_offsets = info['data_offsets']
                        start = data_offsets[0]
                        end = data_offsets[1]
                        length = end - start
                        abs_offset = base_offset + start
                        
                        LIB_LLAISYS.tensorLoadFromFile(
                            tensor_ptr, 
                            filename_bytes, 
                            ctypes.c_size_t(abs_offset), 
                            ctypes.c_size_t(length)
                        )

        for file in sorted(model_path.glob("*.safetensors")):
            load_safetensors_file(file)


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 100

        tokens = list(inputs)
        
        curr_tokens = (ctypes.c_int64 * len(tokens))(*tokens)
        
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            curr_tokens,
            ctypes.c_size_t(len(tokens))
        )
        tokens.append(next_token)
        
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
                
            curr_in = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                curr_in,
                ctypes.c_size_t(1)
            )
            tokens.append(next_token)
            
        return tokens

    def __del__(self):
        if hasattr(self, "model"):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
# //////////// 任务 3.3 end ////////////////

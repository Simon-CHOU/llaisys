//////////// 任务 3.1 start //////
#include "model.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils/check.hpp"
#include <cmath>

namespace llaisys::models {

Qwen2::Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id) {
    
    init_weights();
    
    // 初始化 KV Cache
    // 形状: [batch=1, max_seq_len, n_kv_head, head_dim]
    size_t head_dim = meta.hs / meta.nh;
    std::vector<size_t> kv_shape = {1, meta.maxseq, meta.nkvh, head_dim};
    
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto k = Tensor::create(kv_shape, meta.dtype, device_type, device_id);
        auto v = Tensor::create(kv_shape, meta.dtype, device_type, device_id);
        _kv_cache.push_back({k, v});
    }
}

Qwen2::~Qwen2() {
    if (_weights.in_embed) delete _weights.in_embed;
    if (_weights.out_embed) delete _weights.out_embed;
    if (_weights.out_norm_w) delete _weights.out_norm_w;
    
    auto free_array = [&](llaisysTensor_t *arr, size_t n) {
        if (arr) {
            for (size_t i = 0; i < n; ++i) {
                if (arr[i]) delete arr[i];
            }
            delete[] arr;
        }
    };
    
    free_array(_weights.attn_norm_w, _meta.nlayer);
    free_array(_weights.attn_q_w, _meta.nlayer);
    free_array(_weights.attn_q_b, _meta.nlayer);
    free_array(_weights.attn_k_w, _meta.nlayer);
    free_array(_weights.attn_k_b, _meta.nlayer);
    free_array(_weights.attn_v_w, _meta.nlayer);
    free_array(_weights.attn_v_b, _meta.nlayer);
    free_array(_weights.attn_o_w, _meta.nlayer);
    free_array(_weights.mlp_norm_w, _meta.nlayer);
    free_array(_weights.mlp_gate_w, _meta.nlayer);
    free_array(_weights.mlp_up_w, _meta.nlayer);
    free_array(_weights.mlp_down_w, _meta.nlayer);
}

void Qwen2::init_weights() {
    auto create_weight = [&](const std::vector<size_t>& shape) -> llaisysTensor_t {
        auto t = Tensor::create(shape, _meta.dtype, _device_type, _device_id);
        return new LlaisysTensor{t};
    };
    
    _weights.in_embed = create_weight({_meta.voc, _meta.hs});
    _weights.out_norm_w = create_weight({_meta.hs});
    _weights.out_embed = create_weight({_meta.voc, _meta.hs});
    
    _weights.attn_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_q_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_q_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_k_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_k_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_v_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_v_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_o_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_gate_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_up_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_down_w = new llaisysTensor_t[_meta.nlayer];
    
    size_t head_dim = _meta.hs / _meta.nh;
    size_t kv_dim = _meta.nkvh * head_dim;
    size_t q_dim = _meta.nh * head_dim;
    size_t inter_dim = _meta.di;
    
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _weights.attn_norm_w[i] = create_weight({_meta.hs});
        
        _weights.attn_q_w[i] = create_weight({q_dim, _meta.hs});
        _weights.attn_q_b[i] = create_weight({q_dim});
        
        _weights.attn_k_w[i] = create_weight({kv_dim, _meta.hs});
        _weights.attn_k_b[i] = create_weight({kv_dim});
        
        _weights.attn_v_w[i] = create_weight({kv_dim, _meta.hs});
        _weights.attn_v_b[i] = create_weight({kv_dim});
        
        _weights.attn_o_w[i] = create_weight({_meta.hs, q_dim});
        
        _weights.mlp_norm_w[i] = create_weight({_meta.hs});
        
        _weights.mlp_gate_w[i] = create_weight({inter_dim, _meta.hs});
        _weights.mlp_up_w[i] = create_weight({inter_dim, _meta.hs});
        _weights.mlp_down_w[i] = create_weight({_meta.hs, inter_dim});
    }
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    std::vector<size_t> input_shape = {1, ntoken};
    auto input_tensor = Tensor::create(input_shape, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_tensor->load(token_ids);
    
    auto logits = forward(input_tensor);
    
    auto last_logits = logits->slice(0, ntoken - 1, ntoken);
    
    std::vector<size_t> flat_shape = {_meta.voc};
    auto flat_logits = last_logits->reshape(flat_shape);
    
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    
    ops::argmax(max_idx, max_val, flat_logits);
    
    int64_t result_token;
    if (_device_type == LLAISYS_DEVICE_CPU) {
        result_token = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        auto cpu_idx = max_idx->to(LLAISYS_DEVICE_CPU);
        result_token = *reinterpret_cast<int64_t*>(cpu_idx->data());
    }
    
    _kv_pos += ntoken;
    
    return result_token;
}

tensor_t Qwen2::forward(tensor_t input_ids) {
    size_t seq_len = input_ids->shape()[1];
    
    auto input_flat = input_ids->reshape({seq_len});
    auto hidden = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(hidden, input_flat, _weights.in_embed->tensor);
    
    auto x = hidden;
    size_t head_dim = _meta.hs / _meta.nh;
    size_t q_dim = _meta.nh * head_dim;
    size_t kv_dim = _meta.nkvh * head_dim;
    size_t inter_dim = _meta.di;
    
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = x;
        
        // 1. RMS Norm
        auto x_norm = Tensor::create(x->shape(), x->dtype(), x->deviceType(), x->deviceId());
        ops::rms_norm(x_norm, x, _weights.attn_norm_w[i]->tensor, _meta.epsilon);
        
        // 2. Attention
        auto q = Tensor::create({seq_len, q_dim}, x->dtype(), _device_type, _device_id);
        auto k = Tensor::create({seq_len, kv_dim}, x->dtype(), _device_type, _device_id);
        auto v = Tensor::create({seq_len, kv_dim}, x->dtype(), _device_type, _device_id);
        
        ops::linear(q, x_norm, _weights.attn_q_w[i]->tensor, _weights.attn_q_b[i]->tensor);
        ops::linear(k, x_norm, _weights.attn_k_w[i]->tensor, _weights.attn_k_b[i]->tensor);
        ops::linear(v, x_norm, _weights.attn_v_w[i]->tensor, _weights.attn_v_b[i]->tensor);
        
        auto q_3d = q->reshape({seq_len, _meta.nh, head_dim});
        auto k_3d = k->reshape({seq_len, _meta.nkvh, head_dim});
        auto v_3d = v->reshape({seq_len, _meta.nkvh, head_dim});
        
        // RoPE
        auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, _device_type, _device_id);
        if (_device_type == LLAISYS_DEVICE_CPU) {
            int64_t *pos_ptr = reinterpret_cast<int64_t*>(pos_ids->data());
            for (size_t p = 0; p < seq_len; ++p) {
                pos_ptr[p] = static_cast<int64_t>(_kv_pos + p);
            }
        } else {
             std::vector<int64_t> pos_vec(seq_len);
             for (size_t p = 0; p < seq_len; ++p) pos_vec[p] = _kv_pos + p;
             pos_ids->load(pos_vec.data());
        }
        
        auto q_rope = Tensor::create(q_3d->shape(), q_3d->dtype(), _device_type, _device_id);
        ops::rope(q_rope, q_3d, pos_ids, _meta.theta);
        
        // KV Cache
        auto k_cache = _kv_cache[i].first;
        auto v_cache = _kv_cache[i].second;
        
        auto k_slot = k_cache->slice(1, _kv_pos, _kv_pos + seq_len);
        auto v_slot = v_cache->slice(1, _kv_pos, _kv_pos + seq_len);
        
        auto k_slot_3d = k_slot->reshape({seq_len, _meta.nkvh, head_dim});
        auto v_slot_3d = v_slot->reshape({seq_len, _meta.nkvh, head_dim});
        
        ops::rope(k_slot_3d, k_3d, pos_ids, _meta.theta);
        
        // Copy V to cache. Since we computed V in linear (v), we need to copy to v_slot_3d.
        // We can use a trick: use `add` with zero.
        // Or better: pass `v_slot_3d` (reshaped to 2D) as output to `linear` above!
        // But `linear` was already called.
        // To be safe and efficient, we should use `linear` output as `v` temp, then copy.
        // `ops::add` works for copy if we add 0.
        // Or assume `v_slot_3d` is destination.
        // Let's use `add(out, in, zero_tensor)`. Or just `add(out, in, in)` is 2*in.
        // Wait, `add(c, a, b)` -> `c = a + b`.
        // If we don't have copy, and `add` is available...
        // We can create a zero tensor?
        // Or maybe just `swiglu` or something? No.
        // Actually, `linear` output could have been directly `v_slot` if we reshaped `v_slot` to 2D.
        // Let's refactor to write directly to cache for V!
        
        auto v_slot_2d = v_slot_3d->reshape({seq_len, kv_dim});
        ops::linear(v_slot_2d, x_norm, _weights.attn_v_w[i]->tensor, _weights.attn_v_b[i]->tensor);
        // Done for V.
        
        auto k_hist = k_cache->slice(1, 0, _kv_pos + seq_len);
        auto v_hist = v_cache->slice(1, 0, _kv_pos + seq_len);
        
        auto k_hist_3d = k_hist->reshape({_kv_pos + seq_len, _meta.nkvh, head_dim});
        auto v_hist_3d = v_hist->reshape({_kv_pos + seq_len, _meta.nkvh, head_dim});
        
        auto attn_out = Tensor::create({seq_len, _meta.nh, head_dim}, x->dtype(), _device_type, _device_id);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        ops::self_attention(attn_out, q_rope, k_hist_3d, v_hist_3d, scale);
        
        auto attn_out_2d = attn_out->reshape({seq_len, _meta.hs});
        auto proj_out = Tensor::create({seq_len, _meta.hs}, x->dtype(), _device_type, _device_id);
        ops::linear(proj_out, attn_out_2d, _weights.attn_o_w[i]->tensor, nullptr);
        
        ops::add(x, x, proj_out);
        
        // 3. MLP
        residual = x;
        auto x_mlp_norm = Tensor::create(x->shape(), x->dtype(), _device_type, _device_id);
        ops::rms_norm(x_mlp_norm, x, _weights.mlp_norm_w[i]->tensor, _meta.epsilon);
        
        auto gate = Tensor::create({seq_len, inter_dim}, x->dtype(), _device_type, _device_id);
        auto up = Tensor::create({seq_len, inter_dim}, x->dtype(), _device_type, _device_id);
        
        ops::linear(gate, x_mlp_norm, _weights.mlp_gate_w[i]->tensor, nullptr);
        ops::linear(up, x_mlp_norm, _weights.mlp_up_w[i]->tensor, nullptr);
        
        auto swiglu_out = Tensor::create({seq_len, inter_dim}, x->dtype(), _device_type, _device_id);
        ops::swiglu(swiglu_out, gate, up);
        
        auto down = Tensor::create({seq_len, _meta.hs}, x->dtype(), _device_type, _device_id);
        ops::linear(down, swiglu_out, _weights.mlp_down_w[i]->tensor, nullptr);
        
        ops::add(x, x, down);
    }
    
    auto x_final = Tensor::create(x->shape(), x->dtype(), _device_type, _device_id);
    ops::rms_norm(x_final, x, _weights.out_norm_w->tensor, _meta.epsilon);
    
    auto logits = Tensor::create({seq_len, _meta.voc}, x->dtype(), _device_type, _device_id);
    ops::linear(logits, x_final, _weights.out_embed->tensor, nullptr);
    
    return logits;
}

} // namespace llaisys::models
//////////// 任务 3.1 end ////////////////

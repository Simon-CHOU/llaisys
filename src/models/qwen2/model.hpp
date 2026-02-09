//////////// 任务 3.1 start //////
#pragma once
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include <vector>
#include <memory>
#include <utility>

namespace llaisys::models {

class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2();

    int64_t infer(const int64_t *token_ids, size_t ntoken);
    LlaisysQwen2Weights *weights() { return &_weights; }

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights;
    
    // KV Cache: [layer][0=k, 1=v]
    // Shape: [batch, max_seq_len, n_kv_head, head_dim]
    std::vector<std::pair<tensor_t, tensor_t>> _kv_cache;
    
    // Current position in KV cache (number of tokens processed so far)
    size_t _kv_pos = 0;
    
    // Device info
    llaisysDeviceType_t _device_type;
    int _device_id;

    // Helpers
    void init_weights();
    tensor_t forward(tensor_t input_ids);
};

} // namespace llaisys::models
//////////// 任务 3.1 end ////////////////

#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {
////////// 作业 #2.1 start //////
namespace {
template <typename T>
void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    const T *vals_t = reinterpret_cast<const T *>(vals); // 将输入视为具体类型指针
    int64_t *max_idx_t = reinterpret_cast<int64_t *>(max_idx); // 输出索引为 int64
    T *max_val_t = reinterpret_cast<T *>(max_val); // 输出最大值为输入同类型

    float best = llaisys::utils::cast<float>(vals_t[0]); // 以第一个元素初始化最大值
    size_t best_idx = 0; // 初始化最大值索引
    for (size_t i = 1; i < numel; i++) { // 遍历剩余元素
        float v = llaisys::utils::cast<float>(vals_t[i]); // 转为 float 便于比较
        if (v > best) { // 若当前值更大则更新
            best = v; // 更新最大值
            best_idx = i; // 更新最大索引
        }
    }
    max_idx_t[0] = static_cast<int64_t>(best_idx); // 写入最大值索引
    max_val_t[0] = llaisys::utils::cast<T>(best); // 写入最大值
}
} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals); // 设备一致性检查
    CHECK_ARGUMENT(vals->ndim() == 1, "argmax expects 1d tensor"); // 仅支持 1D 输入
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1, "argmax output size must be 1"); // 输出必须是单元素
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax index dtype must be int64"); // 索引类型固定为 int64
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype()); // 最大值类型与输入一致
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous."); // 仅支持连续内存

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) { // CPU 路径
        switch (vals->dtype()) { // 根据数据类型分发
        case LLAISYS_DTYPE_F32:
            return argmax_impl<float>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // float32 实现
        case LLAISYS_DTYPE_F16:
            return argmax_impl<llaisys::fp16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // float16 实现
        case LLAISYS_DTYPE_BF16:
            return argmax_impl<llaisys::bf16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // bfloat16 实现
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype()); // 其他类型不支持
        }
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId()); // 切换到目标设备
    switch (vals->deviceType()) { // 设备分发
    case LLAISYS_DEVICE_CPU:
        switch (vals->dtype()) { // CPU 下按类型分发
        case LLAISYS_DTYPE_F32:
            return argmax_impl<float>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // float32 实现
        case LLAISYS_DTYPE_F16:
            return argmax_impl<llaisys::fp16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // float16 实现
        case LLAISYS_DTYPE_BF16:
            return argmax_impl<llaisys::bf16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel()); // bfloat16 实现
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype()); // 其他类型不支持
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED(); // CUDA 版本未实现
        return; // 保持控制流完整
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE; // 其他设备不支持
    }
}
////////// 作业 #2.1 end ////////////////
} // namespace llaisys::ops

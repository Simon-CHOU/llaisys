#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

////////// 作业 #2.7 start //////
namespace llaisys::ops {
namespace {
// 模板实现：SwiGLU 激活
template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    const T *gate_t = reinterpret_cast<const T *>(gate); // gate指针转换
    const T *up_t = reinterpret_cast<const T *>(up); // up指针转换
    T *out_t = reinterpret_cast<T *>(out); // 输出指针转换

    for (size_t i = 0; i < numel; i++) { // 遍历元素
        float g = llaisys::utils::cast<float>(gate_t[i]); // gate值转float
        float u = llaisys::utils::cast<float>(up_t[i]); // up值转float
        float denom = 1.0f + std::exp(-g); // sigmoid分母
        out_t[i] = llaisys::utils::cast<T>(u * g / denom); // 写回输出
    }
}
} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up); // 检查设备一致性
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype()); // 检查数据类型一致性
    CHECK_ARGUMENT(out->shape() == gate->shape() && out->shape() == up->shape(), "swiglu shape mismatch"); // 检查形状匹配
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous."); // 检查连续性

    if (out->deviceType() == LLAISYS_DEVICE_CPU) { // CPU设备处理
        switch (out->dtype()) { // 根据数据类型调用实现
        case LLAISYS_DTYPE_F32:
            return swiglu_impl<float>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_F16:
            return swiglu_impl<llaisys::fp16_t>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_BF16:
            return swiglu_impl<llaisys::bf16_t>(out->data(), gate->data(), up->data(), out->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype()); // 不支持的数据类型报错
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); // 设置设备上下文
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_impl<float>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_F16:
            return swiglu_impl<llaisys::fp16_t>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_BF16:
            return swiglu_impl<llaisys::bf16_t>(out->data(), gate->data(), up->data(), out->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED(); // NVIDIA设备未实现
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE; // 不支持的设备报错
    }
}
} // namespace llaisys::ops
////////// 作业 #2.7 end ////////////////

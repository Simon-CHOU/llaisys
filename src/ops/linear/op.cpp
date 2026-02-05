#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
////////// 作业 #2.3 start //////
namespace {
template <typename T>
void linear_impl(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                 size_t m, size_t n, size_t k) {
    const T *in_t = reinterpret_cast<const T *>(in); // 输入指针
    const T *w_t = reinterpret_cast<const T *>(weight); // 权重指针
    const T *b_t = reinterpret_cast<const T *>(bias); // bias 指针（可能为空）
    T *out_t = reinterpret_cast<T *>(out); // 输出指针

    for (size_t i = 0; i < m; i++) { // 遍历 batch 维
        for (size_t j = 0; j < n; j++) { // 遍历输出维
            float acc = 0.0f; // 累加器用 float
            for (size_t kk = 0; kk < k; kk++) { // 内积维度
                float a = llaisys::utils::cast<float>(in_t[i * k + kk]); // 输入值
                float b = llaisys::utils::cast<float>(w_t[j * k + kk]); // 权重值（等价于 W^T）
                acc += a * b; // 乘加
            }
            if (b_t != nullptr) { // 若有 bias
                acc += llaisys::utils::cast<float>(b_t[j]); // 加上 bias
            }
            out_t[i * n + j] = llaisys::utils::cast<T>(acc); // 写回输出
        }
    }
}
} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) { // 有 bias 时检查四个张量设备
        CHECK_SAME_DEVICE(out, in, weight, bias);
    } else { // 无 bias 时检查三个张量设备
        CHECK_SAME_DEVICE(out, in, weight);
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype()); // 数据类型一致性
    if (bias) { // bias 类型一致性
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "linear expects 2d tensors"); // 仅支持 2D
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "linear output batch mismatch"); // batch 对齐
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "linear output dim mismatch"); // 输出维度匹配
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "linear input dim mismatch"); // 输入维度匹配
    if (bias) { // bias 形状检查
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0], "linear bias shape mismatch");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: tensors must be contiguous."); // 仅支持连续内存
    if (bias) { // bias 需要连续
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    const size_t m = out->shape()[0]; // batch
    const size_t n = out->shape()[1]; // 输出维度
    const size_t k = in->shape()[1]; // 输入维度
    const std::byte *bias_ptr = bias ? bias->data() : nullptr; // bias 指针

    if (out->deviceType() == LLAISYS_DEVICE_CPU) { // CPU 路径
        switch (out->dtype()) { // 按类型分发
        case LLAISYS_DTYPE_F32:
            return linear_impl<float>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // float32 实现
        case LLAISYS_DTYPE_F16:
            return linear_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // float16 实现
        case LLAISYS_DTYPE_BF16:
            return linear_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // bfloat16 实现
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype()); // 其他类型不支持
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); // 切换设备
    switch (out->deviceType()) { // 设备分发
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) { // CPU 下按类型分发
        case LLAISYS_DTYPE_F32:
            return linear_impl<float>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // float32 实现
        case LLAISYS_DTYPE_F16:
            return linear_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // float16 实现
        case LLAISYS_DTYPE_BF16:
            return linear_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, n, k); // bfloat16 实现
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype()); // 其他类型不支持
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
////////// 作业 #2.3 end ////////////////
} // namespace llaisys::ops

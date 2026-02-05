#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

////////// 作业 #2.4 start //////
namespace llaisys::ops {
namespace {
// 模板实现：RMSNorm 计算
template <typename T>
void rms_norm_impl(std::byte *out, const std::byte *in, const std::byte *weight, size_t m, size_t d, float eps) {
    const T *in_t = reinterpret_cast<const T *>(in); // 输入指针转换
    const T *w_t = reinterpret_cast<const T *>(weight); // 权重指针转换
    T *out_t = reinterpret_cast<T *>(out); // 输出指针转换

    for (size_t i = 0; i < m; i++) { // 遍历每一行
        float mean = 0.0f; // 均方初始化
        const T *row = in_t + i * d; // 当前行起始指针
        for (size_t j = 0; j < d; j++) { // 计算均方
            float v = llaisys::utils::cast<float>(row[j]); // 转为float便于累加
            mean += v * v; // 累加平方
        }
        mean = mean / static_cast<float>(d); // 求均方
        float scale = 1.0f / std::sqrt(mean + eps); // 归一化因子
        for (size_t j = 0; j < d; j++) { // 应用归一化与权重
            float v = llaisys::utils::cast<float>(row[j]); // 输入值转float
            float w = llaisys::utils::cast<float>(w_t[j]); // 权重转float
            out_t[i * d + j] = llaisys::utils::cast<T>(v * scale * w); // 写回输出
        }
    }
}
} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight); // 检查设备一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype()); // 检查数据类型一致性
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "rms_norm expects 2d tensors"); // 检查输入维度
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm weight must be 1d"); // 检查权重维度
    CHECK_ARGUMENT(out->shape() == in->shape(), "rms_norm shape mismatch"); // 检查形状匹配
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "rms_norm weight size mismatch"); // 检查权重长度
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RmsNorm: all tensors must be contiguous."); // 检查连续性

    const size_t m = in->shape()[0]; // 行数
    const size_t d = in->shape()[1]; // 列数

    if (out->deviceType() == LLAISYS_DEVICE_CPU) { // CPU设备处理
        switch (out->dtype()) { // 根据数据类型调用实现
        case LLAISYS_DTYPE_F32:
            return rms_norm_impl<float>(out->data(), in->data(), weight->data(), m, d, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), m, d, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), m, d, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype()); // 不支持的数据类型报错
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); // 设置设备上下文
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_impl<float>(out->data(), in->data(), weight->data(), m, d, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), m, d, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), m, d, eps);
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
////////// 作业 #2.4 end ////////////////

#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

////////// 作业 #2.5 start //////
namespace llaisys::ops {
namespace {
// 模板实现：RoPE 旋转位置编码
template <typename T>
void rope_impl(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t seq_len, size_t nhead, size_t dim,
               float theta) {
    const T *in_t = reinterpret_cast<const T *>(in); // 输入指针转换
    T *out_t = reinterpret_cast<T *>(out); // 输出指针转换
    const int64_t *pos_t = reinterpret_cast<const int64_t *>(pos_ids); // 位置索引指针转换
    size_t half = dim / 2; // 头维度一分为二

    for (size_t s = 0; s < seq_len; s++) { // 遍历序列位置
        float pos = static_cast<float>(pos_t[s]); // 当前位置
        for (size_t h = 0; h < nhead; h++) { // 遍历头
            const T *in_row = in_t + (s * nhead + h) * dim; // 输入行指针
            T *out_row = out_t + (s * nhead + h) * dim; // 输出行指针
            for (size_t j = 0; j < half; j++) { // 遍历半维度
                float a = llaisys::utils::cast<float>(in_row[j]); // 实部
                float b = llaisys::utils::cast<float>(in_row[j + half]); // 虚部
                float freq = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(dim)); // 旋转频率
                float s_val = std::sin(freq); // 正弦值
                float c_val = std::cos(freq); // 余弦值
                out_row[j] = llaisys::utils::cast<T>(a * c_val - b * s_val); // 写回实部
                out_row[j + half] = llaisys::utils::cast<T>(b * c_val + a * s_val); // 写回虚部
            }
        }
    }
}
} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids); // 检查设备一致性
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope pos_ids dtype must be int64"); // 检查索引类型
    CHECK_SAME_DTYPE(out->dtype(), in->dtype()); // 检查数据类型一致性
    CHECK_ARGUMENT(out->shape() == in->shape(), "rope output shape mismatch"); // 检查形状匹配
    CHECK_ARGUMENT(out->ndim() == 3, "rope expects 3d tensor"); // 检查输入维度
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "rope pos_ids must be 1d"); // 检查位置索引维度
    CHECK_ARGUMENT(pos_ids->shape()[0] == out->shape()[0], "rope pos_ids length mismatch"); // 检查位置长度
    CHECK_ARGUMENT(out->shape()[2] % 2 == 0, "rope head dim must be even"); // 检查头维度偶数
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous."); // 检查连续性

    const size_t seq_len = out->shape()[0]; // 序列长度
    const size_t nhead = out->shape()[1]; // 头数量
    const size_t dim = out->shape()[2]; // 头维度

    if (out->deviceType() == LLAISYS_DEVICE_CPU) { // CPU设备处理
        switch (out->dtype()) { // 根据数据类型调用实现
        case LLAISYS_DTYPE_F32:
            return rope_impl<float>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_impl<llaisys::fp16_t>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_impl<llaisys::bf16_t>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype()); // 不支持的数据类型报错
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); // 设置设备上下文
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_impl<float>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_impl<llaisys::fp16_t>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_impl<llaisys::bf16_t>(out->data(), in->data(), pos_ids->data(), seq_len, nhead, dim, theta);
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
////////// 作业 #2.5 end ////////////////

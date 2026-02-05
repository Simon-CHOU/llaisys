#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

////////// 作业 #2.6 start //////
namespace llaisys::ops {
namespace {
// 模板实现：自注意力计算
template <typename T>
void self_attention_impl(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, size_t qlen, size_t kvlen,
                         size_t nh, size_t nkvh, size_t hd, float scale) {
    const T *q_t = reinterpret_cast<const T *>(q); // Q指针转换
    const T *k_t = reinterpret_cast<const T *>(k); // K指针转换
    const T *v_t = reinterpret_cast<const T *>(v); // V指针转换
    T *out_t = reinterpret_cast<T *>(out); // 输出指针转换

    const size_t repeat = nh / nkvh; // 头重复次数
    const size_t attn_size = kvlen; // 注意力长度
    std::vector<float> attn(attn_size); // 注意力分数缓冲
    std::vector<float> exp_buf(attn_size); // softmax缓冲

    const float neg_inf = -std::numeric_limits<float>::infinity(); // 负无穷
    const int64_t diag = static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen); // 掩码对角线偏移

    for (size_t h = 0; h < nh; h++) { // 遍历Q头
        size_t kh = h / repeat; // 对应K/V头
        for (size_t i = 0; i < qlen; i++) { // 遍历查询位置
            float max_val = neg_inf; // softmax最大值
            for (size_t j = 0; j < kvlen; j++) { // 遍历键值位置
                if (static_cast<int64_t>(j) > static_cast<int64_t>(i) + diag) { // 因果遮罩
                    attn[j] = neg_inf; // 被遮罩位置
                    continue;
                }
                float dot = 0.0f; // 点积累加
                const T *q_row = q_t + (i * nh + h) * hd; // 当前Q向量
                const T *k_row = k_t + (j * nkvh + kh) * hd; // 当前K向量
                for (size_t d = 0; d < hd; d++) { // 遍历头维度
                    float a = llaisys::utils::cast<float>(q_row[d]); // Q值
                    float b = llaisys::utils::cast<float>(k_row[d]); // K值
                    dot += a * b; // 累加点积
                }
                float val = dot * scale; // 缩放得分
                attn[j] = val; // 记录注意力分数
                if (val > max_val) { // 更新最大值
                    max_val = val;
                }
            }
            float sum = 0.0f; // softmax和
            for (size_t j = 0; j < kvlen; j++) { // 计算softmax
                float vj = attn[j]; // 当前分数
                if (std::isinf(vj) && vj < 0.0f) { // 遮罩位置
                    exp_buf[j] = 0.0f; // 置零
                    continue;
                }
                float e = std::exp(vj - max_val); // 指数
                exp_buf[j] = e; // 记录指数
                sum += e; // 累加
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f; // 归一化系数
            for (size_t d = 0; d < hd; d++) { // 计算加权和
                float acc = 0.0f; // 输出累加
                for (size_t j = 0; j < kvlen; j++) { // 遍历键值
                    float w = exp_buf[j] * inv_sum; // softmax权重
                    const T *v_row = v_t + (j * nkvh + kh) * hd; // 当前V向量
                    acc += w * llaisys::utils::cast<float>(v_row[d]); // 加权累加
                }
                out_t[(i * nh + h) * hd + d] = llaisys::utils::cast<T>(acc); // 写回输出
            }
        }
    }
}
} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v); // 检查设备一致性
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype()); // 检查数据类型一致性
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "self_attention expects 3d tensors"); // 检查维度
    CHECK_ARGUMENT(attn_val->shape() == q->shape(), "self_attention output shape mismatch"); // 检查输出形状
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1] && k->shape()[2] == v->shape()[2],
                   "self_attention kv shape mismatch"); // 检查KV形状
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "self_attention head dim mismatch"); // 检查头维度
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "self_attention head repeat mismatch"); // 检查头重复
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous."); // 检查连续性

    const size_t qlen = q->shape()[0]; // Q长度
    const size_t kvlen = k->shape()[0]; // K/V长度
    const size_t nh = q->shape()[1]; // Q头数量
    const size_t nkvh = k->shape()[1]; // K/V头数量
    const size_t hd = q->shape()[2]; // 头维度

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) { // CPU设备处理
        switch (attn_val->dtype()) { // 根据数据类型调用实现
        case LLAISYS_DTYPE_F32:
            return self_attention_impl<float>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_impl<llaisys::fp16_t>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_impl<llaisys::bf16_t>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype()); // 不支持的数据类型报错
        }
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId()); // 设置设备上下文
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            return self_attention_impl<float>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_impl<llaisys::fp16_t>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_impl<llaisys::bf16_t>(attn_val->data(), q->data(), k->data(), v->data(), qlen, kvlen, nh, nkvh, hd, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
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
////////// 作业 #2.6 end ////////////////

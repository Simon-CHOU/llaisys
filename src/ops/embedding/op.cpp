#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
////////// 作业 #2.2 start //////
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight); // 设备一致性检查
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding index dtype must be int64"); // index 必须是 int64
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype()); // 输出与权重类型一致
    CHECK_ARGUMENT(index->ndim() == 1, "embedding index must be 1d"); // index 必须是 1D
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding weight must be 2d"); // weight 必须是 2D
    CHECK_ARGUMENT(out->ndim() == 2, "embedding output must be 2d"); // out 必须是 2D
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "embedding output rows mismatch"); // 行数与 index 长度一致
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "embedding output dim mismatch"); // 列数与词向量维度一致
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous."); // 仅支持连续内存

    const size_t vocab = weight->shape()[0]; // 词表大小
    const size_t dim = weight->shape()[1]; // 向量维度
    const size_t row_bytes = dim * llaisys::utils::dsize(out->dtype()); // 每行字节数

    if (out->deviceType() == LLAISYS_DEVICE_CPU) { // CPU 路径
        const int64_t *idx = reinterpret_cast<const int64_t *>(index->data()); // index 指针
        const std::byte *w = weight->data(); // weight 指针
        std::byte *o = out->data(); // out 指针
        for (size_t i = 0; i < index->shape()[0]; i++) { // 逐行拷贝
            int64_t id = idx[i]; // 当前词 id
            CHECK_ARGUMENT(id >= 0 && static_cast<size_t>(id) < vocab, "embedding index out of range"); // 越界检查
            std::memcpy(o + i * row_bytes, w + static_cast<size_t>(id) * row_bytes, row_bytes); // 拷贝一整行
        }
        return; // 完成
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); // 切换到目标设备
    switch (out->deviceType()) { // 设备分发
    case LLAISYS_DEVICE_CPU: { // CPU 兜底路径
        const int64_t *idx = reinterpret_cast<const int64_t *>(index->data()); // index 指针
        const std::byte *w = weight->data(); // weight 指针
        std::byte *o = out->data(); // out 指针
        for (size_t i = 0; i < index->shape()[0]; i++) { // 逐行拷贝
            int64_t id = idx[i]; // 当前词 id
            CHECK_ARGUMENT(id >= 0 && static_cast<size_t>(id) < vocab, "embedding index out of range"); // 越界检查
            std::memcpy(o + i * row_bytes, w + static_cast<size_t>(id) * row_bytes, row_bytes); // 拷贝一整行
        }
        return; // 完成
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
////////// 作业 #2.2 end ////////////////
} // namespace llaisys::ops

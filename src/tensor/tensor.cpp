#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 如果形状为空（标量），认为是连续的
    if (_meta.shape.empty()) {
        return true;
    }
    // 期望的 stride，从 1 开始（最内层维度）
    ptrdiff_t expected = 1;
    // 从最后一个维度向前遍历
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        size_t dim = _meta.shape[i];
        // 维度为 0 的张量被认为是连续的
        if (dim == 0) {
            return true;
        }
        // 维度为 1 的轴不影响 stride 计算，跳过
        if (dim == 1) {
            continue;
        }
        // 检查当前维度的 stride 是否等于期望值
        if (_meta.strides[i] != expected) {
            return false;
        }
        // 更新期望的 stride：当前 stride * 当前维度大小
        expected *= static_cast<ptrdiff_t>(dim);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查维度数量是否匹配
    CHECK_ARGUMENT(order.size() == _meta.shape.size(), "permute order ndim mismatch");
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    std::vector<bool> seen(order.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        size_t dim = order[i];
        // 检查维度索引是否越界
        CHECK_ARGUMENT(dim < _meta.shape.size(), "permute order out of range");
        // 检查维度是否重复
        CHECK_ARGUMENT(!seen[dim], "permute order duplicated");
        seen[dim] = true;
        // 重新排列 shape 和 strides
        new_shape[i] = _meta.shape[dim];
        new_strides[i] = _meta.strides[dim];
    }
    // 创建新的元数据，共享底层存储
    TensorMeta meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    // 检查元素总数是否一致
    CHECK_ARGUMENT(new_numel == this->numel(), "view shape numel mismatch");
    // view 操作通常要求原张量是连续的
    CHECK_ARGUMENT(this->isContiguous(), "view requires contiguous tensor");

    // 计算新的连续 strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= shape.size(); i++) {
        new_strides[shape.size() - i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[shape.size() - i]);
    }

    // 创建新 Tensor，共享存储
    TensorMeta meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 参数校验
    CHECK_ARGUMENT(dim < _meta.shape.size(), "slice dim out of range");
    CHECK_ARGUMENT(start <= end, "slice start must be <= end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end out of range");

    // 复制元数据并修改切片维度的形状
    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;

    // 计算切片后的内存偏移量：原偏移 + start * 该维度的stride * 元素大小
    size_t byte_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();
    // 返回新 Tensor，共享存储，但有新的偏移和形状
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, byte_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "src is null");
    // 切换到当前 Tensor 所在的设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto api = core::context().runtime().api();
    // 计算需要拷贝的字节数
    size_t bytes = this->numel() * this->elementSize();
    // 确定拷贝方向：如果是 Host 存储则是 H2H，否则假设 src 在 Host，进行 H2D
    llaisysMemcpyKind_t kind = _storage->isHost() ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    // 执行同步拷贝
    api->memcpy_sync(this->data(), src_, bytes, kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys

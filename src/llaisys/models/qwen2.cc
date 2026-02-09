//////////// 任务 3.2 start //////
#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/model.hpp"

using namespace llaisys;

struct LlaisysQwen2Model {
    models::Qwen2 *model;
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    int dev_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    
    auto *wrapper = new LlaisysQwen2Model();
    wrapper->model = new models::Qwen2(*meta, device, dev_id);
    return wrapper;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) {
        if (model->model) delete model->model;
        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model || !model->model) return nullptr;
    return model->model->weights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model || !model->model) return -1;
    return model->model->infer(token_ids, ntoken);
}

}
//////////// 任务 3.2 end ////////////////

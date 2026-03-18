#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include <ATen/record_function.h>
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

using namespace torch_ext;

namespace {

void runWriteCacheStore(DeviceBase*        device,
                        torch::Tensor      input_lengths,
                        torch::Tensor      prefix_lengths,
                        torch::Tensor      kv_cache_block_id_host,
                        PyCacheStoreInputs cache_store_inputs,
                        torch::Tensor      kv_cache_base,
                        torch::Tensor      kv_scale_base,
                        int                layer_id) {

    auto layer_to_group_buf = (cache_store_inputs.kv_cache_layer_to_group.defined()
                               && cache_store_inputs.kv_cache_layer_to_group.numel() > 0) ?
                                  torchTensor2Buffer(cache_store_inputs.kv_cache_layer_to_group) :
                                  nullptr;
    auto group_types_buf =
        (cache_store_inputs.kv_cache_group_types.defined() && cache_store_inputs.kv_cache_group_types.numel() > 0) ?
            torchTensor2Buffer(cache_store_inputs.kv_cache_group_types) :
            nullptr;

    auto cache_keys_buf = (cache_store_inputs.cache_keys.defined() && cache_store_inputs.cache_keys.numel() > 0) ?
                              torchTensor2Buffer(cache_store_inputs.cache_keys) :
                              nullptr;

    CacheStoreInputs inputs{torchTensor2Buffer(input_lengths),
                            torchTensor2Buffer(prefix_lengths),
                            torchTensor2Buffer(kv_cache_block_id_host),
                            layer_to_group_buf,
                            group_types_buf,
                            cache_store_inputs.context_batch_size,
                            cache_store_inputs.decoder_batch_size,
                            torchTensor2Buffer(cache_store_inputs.request_id),
                            torchTensor2Buffer(cache_store_inputs.request_pd_separation),
                            cache_keys_buf,
                            cache_store_inputs.tokens_per_block,
                            cache_store_inputs.kv_block_stride_bytes,
                            cache_store_inputs.kv_scale_stride_bytes,
                            cache_store_inputs.pd_separation,
                            cache_store_inputs.model_id,
                            cache_store_inputs.decode_entrance,
                            cache_store_inputs.warmup,
                            layer_id};

    KvCacheInfo kv_cache_info;
    kv_cache_info.kv_cache_buffer = torchTensor2Buffer(kv_cache_base);
    kv_cache_info.kv_scale_buffer =
        (kv_scale_base.defined() && kv_scale_base.numel() > 0) ? torchTensor2Buffer(kv_scale_base) : nullptr;

    device->writeCacheStore(inputs, kv_cache_info, cache_store_inputs.mla_kvcache);
}

}  // anonymous namespace

void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::KVCache>            kv_cache) {
    RECORD_USER_SCOPE("rtp_llm::WriteCacheStoreOp");
    if (!kv_cache.has_value() || !cache_store_member.has_value()) {
        return;
    }

    auto device = DeviceFactory::getDefaultDevice();

    auto captured_input_lengths          = input_lengths;
    auto captured_prefix_lengths         = prefix_lengths;
    auto captured_kv_cache_block_id_host = kv_cache_block_id_host;
    auto captured_cache_store            = std::move(*cache_store_member);

    auto captured_kv_base  = kv_cache->kv_cache_base;
    auto captured_kv_scale = kv_cache->kv_scale_base;
    int  captured_layer_id = kv_cache->layer_id;

    device->cache_store_async_writer_->submit([device,
                                               captured_input_lengths,
                                               captured_prefix_lengths,
                                               captured_kv_cache_block_id_host,
                                               captured_cache_store = std::move(captured_cache_store),
                                               captured_kv_base,
                                               captured_kv_scale,
                                               captured_layer_id]() {
        runWriteCacheStore(device,
                           captured_input_lengths,
                           captured_prefix_lengths,
                           captured_kv_cache_block_id_host,
                           captured_cache_store,
                           captured_kv_base,
                           captured_kv_scale,
                           captured_layer_id);
    });
}

}  // namespace rtp_llm

// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef ODINANN_CONFIG_H
#define ODINANN_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

namespace {

constexpr const CFG_INT::value_type kSearchListSizeMinValue = 16;
constexpr const CFG_INT::value_type kDefaultSearchListSizeForBuild = 128;

}  // namespace

class OdinANNConfig : public BaseConfig {
 public:
    // The degree of the vamana graph, typically between 60 and 150.
    // Larger values result in larger indices and longer indexing times but better search quality.
    CFG_INT max_degree;
    
    // The search list size during build and search operations.
    // Typical values are between 75 to 200.
    CFG_INT search_list_size;
    
    // The ratio of PQ code size to raw vector data size.
    CFG_FLOAT pq_code_budget_gb_ratio;
    
    // Limit the size of the PQ code after compression (in GB).
    CFG_FLOAT pq_code_budget_gb;
    
    // Limit on memory allowed for building the index (in GB).
    // If insufficient, the index is built using divide and conquer approach.
    CFG_FLOAT build_dram_budget_gb;
    
    // PQ compression dimension. Use 0 for no compression (full-precision).
    CFG_INT disk_pq_dims;
    
    // Enable fast build mode (~30% faster with ~1% recall regression).
    CFG_BOOL accelerate_build;
    
    // The ratio of search cache size to raw vector data size.
    CFG_FLOAT search_cache_budget_gb_ratio;
    
    // Search cache budget in GB for faster query performance.
    CFG_FLOAT search_cache_budget_gb;
    
    // Enable warmup before searching.
    CFG_BOOL warm_up;
    
    // Use BFS strategy for cache generation.
    CFG_BOOL use_bfs_cache;
    
    // Beam width for search operations.
    CFG_INT beamwidth;
    
    // Minimum K for range search.
    CFG_INT min_k;
    
    // Maximum K for range search.
    CFG_INT max_k;
    
    // Filter threshold for PQ refinement strategy.
    CFG_FLOAT filter_threshold;

    // Sampling rate for generating memory index (0.0 ~ 1.0)
    CFG_FLOAT sampling_rate;

    // Alpha parameter for memory index construction (affects graph pruning)
    CFG_FLOAT mem_index_alpha;

    // mem_L: number of neighbors cached in memory for faster search
    CFG_INT mem_L;
    
    // Data path for building the index (optional in Milvus context)
    CFG_STRING data_path;
    
    // Index prefix for the saved index files (optional in Milvus context)
    CFG_STRING index_prefix;

    KNOHWERE_DECLARE_CONFIG(OdinANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(max_degree)
            .description("the degree of the vamana graph index.")
            .set_default(48)
            .set_range(1, 2048)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_list_size)
            .description("the size of search list during index build or search.")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train()
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb_ratio)
            .description("the ratio of PQ code size to raw vector data size")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .description("limit the size of PQ code after compression in GB.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(build_dram_budget_gb)
            .description("limit on the memory allowed for building the index in GB.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(disk_pq_dims)
            .description("PQ compression dimension, 0 for no compression.")
            .set_default(0)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(accelerate_build)
            .description("enable fast build mode for faster index construction.")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb_ratio)
            .description("the ratio of search cache size to raw vector data size")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb)
            .description("search cache budget in GB for faster query performance.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(warm_up)
            .description("enable warmup before searching.")
            .set_default(false)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(use_bfs_cache)
            .description("use BFS strategy for cache generation.")
            .set_default(false)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(beamwidth)
            .description("beam width for search operations.")
            .set_default(8)
            .set_range(1, 128)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(min_k)
            .description("minimum K for range search.")
            .set_default(100)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_k)
            .description("maximum K for range search.")
            .set_default(std::numeric_limits<CFG_INT::value_type>::max())
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(filter_threshold)
            .description("filter threshold for PQ refinement strategy.")
            .set_default(-1.0f)
            .set_range(-1.0f, 1.0f)
            .for_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(sampling_rate)
            .description("sampling rate for generating memory index (0.0 ~ 1.0).")
            .set_default(0.01f)
            .set_range(0.0f, 1.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(mem_index_alpha)
            .description("alpha parameter for memory index construction, affects graph pruning.")
            .set_default(1.2f)
            .set_range(1.0f, 2.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(mem_L)
            .description("mem_L parameter: number of neighbors cached in memory for faster search. 0 means disabled.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(data_path)
            .description("data path for building the index")
            .allow_empty_without_default()
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(index_prefix)
            .description("index prefix for the saved index files")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                if (!search_list_size.has_value()) {
                    search_list_size = kDefaultSearchListSizeForBuild;
                }
                
                // Validate max_degree if provided
                if (max_degree.has_value() && (max_degree.value() < 1 || max_degree.value() > 2048)) {
                    std::string msg = "max_degree(" + std::to_string(max_degree.value()) + 
                                      ") should be between 1 and 2048";
                    return HandleError(err_msg, msg, Status::invalid_param_in_json);
                }
                
                pq_code_budget_gb =
                    std::max(pq_code_budget_gb.value(), pq_code_budget_gb_ratio.value() * vec_field_size_gb.value());
                search_cache_budget_gb = std::max(search_cache_budget_gb.value(),
                                                  search_cache_budget_gb_ratio.value() * vec_field_size_gb.value());
                break;
            }
            case PARAM_TYPE::SEARCH: {
                if (!search_list_size.has_value()) {
                    search_list_size = std::max(k.value(), kSearchListSizeMinValue);
                } else if (k.value() > search_list_size.value()) {
                    std::string msg = "search_list_size(" + std::to_string(search_list_size.value()) +
                                      ") should be larger than k(" + std::to_string(k.value()) + ")";
                    return HandleError(err_msg, msg, Status::out_of_range_in_json);
                }
                break;
            }
            default:
                break;
        }
        return Status::success;
    }
};
}  // namespace knowhere
#endif /* ODINANN_CONFIG_H */

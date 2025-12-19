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

#include "knowhere/feder/OdinANN.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <thread>

#include "aux_utils.h"
#include "filemanager/FileManager.h"
#include "fmt/core.h"
#include "index/odinann/odinann_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"
#include "knowhere/comp/task.h"
#include <fstream>

// OdinANN headers
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "ssd_index.h"
#include "partition_and_pq.h"


namespace knowhere {

// Global memory index cache removed to avoid thread safety issues in multi-threaded environments
// Original implementation had global state that caused race conditions
// static std::mutex g_mem_index_lock;
// static std::condition_variable g_mem_index_cv;
// static std::string g_mem_index_path;  // Cached global mem_index path (shared by all indices)
// static bool g_mem_index_building = false;  // Flag to indicate if mem_index is being built
// static std::shared_ptr<void> g_global_mem_index;  // holds std::shared_ptr<pipeann::Index<...>> casted to void

static constexpr float kCacheExpansionRate = 1.2;

// Forward declarations for functions used in template
std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix);

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix);

bool
CheckMetric(const std::string& metric);

knowhere::Status
TryOdinANNCall(std::function<void()>&& odinann_call);

inline bool
AnyIndexFileExist(const std::string& index_prefix);

template <typename DataType>
class OdinANNIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "OdinANN only support floating point data type(float32, float16, bfloat16)");

 public:
    using DistType = float;
    OdinANNIndexNode(const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {

    }

    OdinANNIndexNode(const int& version, const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
        if (typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>)) {
            auto odinann_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
            if (odinann_index_pack != nullptr) {
                file_manager_ = odinann_index_pack->GetPack();
            }
        }
        LOG_KNOWHERE_INFO_ << "filemanager: " << file_manager_ <<  " OdinANNIndexNode created.";
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        auto build_conf = static_cast<const OdinANNConfig&>(*cfg);
        if (!CheckMetric(build_conf.metric_type.value())) {
            return Status::invalid_metric_type;
        }
        
        // Validate required parameters for OdinANN
        if (!build_conf.max_degree.has_value()) {
            LOG_KNOWHERE_ERROR_ << "max_degree is required for OdinANN build";
            return Status::invalid_param_in_json;
        }
        
        if (!build_conf.search_list_size.has_value()) {
            LOG_KNOWHERE_ERROR_ << "search_list_size is required for OdinANN build";
            return Status::invalid_param_in_json;
        }
        
        // For OdinANN, the actual index building is done externally
        // In Milvus context, the Build method validates parameters and sets up metadata
        // The actual index files are built later by external tools
        LOG_KNOWHERE_INFO_ << "OdinANN Build called - parameters validated: max_degree=" 
                           << build_conf.max_degree.value() 
                           << ", search_list_size=" << build_conf.search_list_size.value();
        
        // Set dim and count from dataset
        dim_.store(dataset->GetDim());
        count_.store(dataset->GetRows());
        
        return Status::success;
    }



    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!is_prepared_.load()) {
            LOG_KNOWHERE_ERROR_ << "OdinANN index not loaded.";
            // Check if loading has already been attempted and failed
            if (!index_prefix_.empty() && ssd_index_ == nullptr) {
                LOG_KNOWHERE_ERROR_ << "OdinANN index loading has been attempted but failed, search not possible";
                return expected<DataSetPtr>::Err(Status::empty_index, "OdinANN index loading failed");
            }
            return expected<DataSetPtr>::Err(Status::empty_index, "OdinANN not loaded");
        }

        auto search_conf = static_cast<const OdinANNConfig&>(*cfg);
        if (!CheckMetric(search_conf.metric_type.value())) {
            return expected<DataSetPtr>::Err(Status::invalid_metric_type, "unsupported metric type");
        }

        if (ssd_index_ == nullptr) {
            LOG_KNOWHERE_ERROR_ << "OdinANN search backend not initialized.";
            return expected<DataSetPtr>::Err(Status::not_implemented, "OdinANN search backend not initialized");
        }

        auto k = static_cast<uint64_t>(search_conf.k.value());
        auto lsearch = static_cast<unsigned>(search_conf.search_list_size.value());
        auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value_or(8));
        auto nq = dataset->GetRows();
        auto dim = dataset->GetDim();
        auto xq = static_cast<const DataType*>(dataset->GetTensor());
        LOG_KNOWHERE_INFO_ << "k,lsearch,beamwidth: " << k << " " << lsearch << " " << beamwidth << std::endl;
        if (nq <= 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be >= 1");
        }

        // allocate output buffers
        auto p_id = std::make_unique<int64_t[]>(k * nq);
        auto p_dist = std::make_unique<DistType[]>(k * nq);

        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(nq);
        
        // For safety in multi-instance environments, disable mem_index usage
        // and set mem_L to 0 to avoid global state issues
        uint32_t mem_L = 0;
        std::shared_ptr<pipeann::Index<DataType>> shared_mem_index = nullptr;
        
        // If mem_L was configured, warn that it's not supported in this implementation
        if (search_conf.mem_L.has_value() && search_conf.mem_L.value() > 0) {
            LOG_KNOWHERE_WARNING_ << "mem_L parameter is not supported in this implementation, setting to 0";
        }
        for (int64_t row = 0; row < nq; ++row) {
            futures.emplace_back(
                search_pool_->push([this, index = row, k, beamwidth, lsearch, dim, xq, mem_L, shared_mem_index,
                                   p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
                    try {
                        std::unique_ptr<uint32_t[]> res_tags(new uint32_t[k]);
                        std::unique_ptr<float[]> res_dists(new float[k]);
                        
                        const DataType* query = xq + (index * dim);
                        
                        // Use beam_search to avoid complex memory index dependencies
                        size_t returned;
                        returned = ssd_index_->beam_search(query, k, mem_L, lsearch, res_tags.get(),
                                                          res_dists.get(), beamwidth, nullptr);
                        
                        for (size_t i = 0; i < returned && i < k; ++i) {
                            p_id_ptr[index * k + i] = static_cast<int64_t>(res_tags[i]);
                            p_dist_ptr[index * k + i] = static_cast<DistType>(res_dists[i]);
                        }
                        // fill rest with -1 / INF
                        for (size_t i = returned; i < k; ++i) {
                            p_id_ptr[index * k + i] = -1;
                            p_dist_ptr[index * k + i] = std::numeric_limits<DistType>::infinity();
                        }
                    } catch (const std::exception& e) {
                        LOG_KNOWHERE_ERROR_ << "OdinANN search failed: " << e.what();
                        throw;
                    }
                }));
        }

        if (TryOdinANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
            return expected<DataSetPtr>::Err(Status::odinann_inner_error, "some odinann search failed");
        }

        auto res = GenResultDataSet(nq, k, std::move(p_id), std::move(p_dist));
        return res;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        if (!is_prepared_.load()) {
            LOG_KNOWHERE_ERROR_ << "OdinANN index not loaded.";
            // Check if loading has already been attempted and failed
            if (!index_prefix_.empty() && ssd_index_ == nullptr) {
                LOG_KNOWHERE_ERROR_ << "OdinANN index loading has been attempted but failed, GetVectorByIds not possible";
                return expected<DataSetPtr>::Err(Status::empty_index, "OdinANN index loading failed");
            }
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        if (ssd_index_ == nullptr) {
            LOG_KNOWHERE_ERROR_ << "OdinANN backend not initialized.";
            return expected<DataSetPtr>::Err(Status::not_implemented, "OdinANN backend not initialized");
        }

        // Note: SSDIndex doesn't support get_vector_by_tag directly.
        // For now, return not_implemented as this is a limitation of the disk index.
        LOG_KNOWHERE_WARNING_ << "OdinANN SSDIndex does not support GetVectorByIds operation.";
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "OdinANN disk index does not support GetVectorByIds");
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        LOG_KNOWHERE_INFO_ << "OdinANN AnnIterator is not supported yet.";
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                  "OdinANN does not support AnnIterator");
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        knowhere::MetricType metric_type = config.metric_type.has_value() ? config.metric_type.value() : "";
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        auto odinann_conf = static_cast<const OdinANNConfig&>(*cfg);
        auto count = Count();
        // If count is 0 and we have an index prefix, try to get the count from the index files if available
        if (count == 0 && !index_prefix_.empty() && ssd_index_ != nullptr) {
            // If SSD index is loaded, get the count directly from it
            count = static_cast<int64_t>(ssd_index_->return_nd());
        }
        feder::odinann::OdinANNMeta meta(
            odinann_conf.data_path.value_or(""), odinann_conf.max_degree.value_or(48),
            odinann_conf.search_list_size.value_or(128), odinann_conf.pq_code_budget_gb.value_or(0),
            odinann_conf.build_dram_budget_gb.value_or(0), odinann_conf.disk_pq_dims.value_or(0),
            odinann_conf.accelerate_build.value_or(false), odinann_conf.sampling_rate.value_or(0.01f), 
            odinann_conf.mem_index_alpha.value_or(1.2f), count, std::vector<int64_t>());
        LOG_KNOWHERE_INFO_ << " data_path:" << odinann_conf.data_path.value() << std::endl;
        Json json_meta;
        nlohmann::to_json(json_meta, meta);
        return GenResultDataSet(json_meta.dump(), "");
    }

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "OdinANN does nothing for serialize";
        return Status::success;
    }

    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& version) {
        return Resource{file_size_in_bytes / 4, file_size_in_bytes};
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        std::lock_guard<std::mutex> lock(preparation_lock_);
        auto prep_conf = static_cast<const OdinANNConfig&>(*cfg);

        if (!CheckMetric(prep_conf.metric_type.value())) {
            return Status::invalid_metric_type;
        }

        if (!prep_conf.index_prefix.has_value()) {
            LOG_KNOWHERE_ERROR_ << "OdinANN file path for deserialize is empty." << std::endl;
            return Status::invalid_param_in_json;
        }

        index_prefix_ = prep_conf.index_prefix.value();
        LOG_KNOWHERE_INFO_ << "indexprefix: " << index_prefix_ << std::endl;
        
        // If file manager is not available yet, we'll load the files later when it is set
        if (file_manager_ == nullptr) {
            LOG_KNOWHERE_WARNING_ << "File manager not yet available, will load files when available";
            // Mark that we have an index prefix but it's not loaded yet
            is_prepared_.store(false);
            return Status::success;
        }
        
        return LoadIndexWithConfig(cfg);
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "OdinANN doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<OdinANNConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    Status
    SetFileManager(std::shared_ptr<milvus::FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        
        // If we have an index prefix but haven't loaded the index yet, try to load it now
        if (!index_prefix_.empty() && !is_prepared_.load()) {
            auto status = LoadIndexFiles(nullptr);  // No config available at this point
            if (status != Status::success) {
                LOG_KNOWHERE_ERROR_ << "Failed to load OdinANN index files, will not retry";
                // Set prepared to true to prevent further attempts to load
                is_prepared_.store(true);
                return status;
            }
            return status;
        }
        
        return Status::success;
    }
    
    // Method to handle loading when config is available after file manager is set
    Status
    LoadIndexWithConfig(std::shared_ptr<Config> cfg) {
        if (file_manager_ == nullptr || index_prefix_.empty()) {
            return Status::invalid_param_in_json;
        }
        
        return LoadIndexFiles(cfg);
    }
    
    // Helper method to load index files when file manager becomes available
    Status LoadIndexFiles(std::shared_ptr<Config> cfg = nullptr) {
        if (file_manager_ == nullptr || index_prefix_.empty()) {
            return Status::invalid_param_in_json;
        }
        
        // Check if necessary files exist before attempting to load
        for (auto& filename : GetNecessaryFilenames(index_prefix_)) {
            auto is_exist_op = file_manager_->IsExisted(filename);
            if (!is_exist_op.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
                return Status::disk_file_error;
            }
            if (!is_exist_op.value()) {
                LOG_KNOWHERE_ERROR_ << "Necessary file does not exist: " << filename;
                return Status::disk_file_error;
            }
        }
        
        // Load files from file manager
        for (auto& filename : GetNecessaryFilenames(index_prefix_)) {
            if (!LoadFile(filename)) {
                LOG_KNOWHERE_ERROR_ << "Failed to load necessary file: " << filename;
                return Status::disk_file_error;
            }
        }

        for (auto& filename : GetOptionalFilenames(index_prefix_)) {
            auto is_exist_op = file_manager_->IsExisted(filename);
            if (!is_exist_op.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
                // Don't fail on file existence check error for optional files
            }
            if (is_exist_op.value() && !LoadFile(filename)) {
                LOG_KNOWHERE_WARNING_ << "Failed to load optional file: " << filename;
                // Don't fail on optional files
            }
        }
        
        // Set thread pool
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
        
        // Determine metric type - use config if available, otherwise use a default approach
        std::string metric_type = knowhere::metric::L2; // default
        if (cfg != nullptr) {
            auto prep_conf = static_cast<const OdinANNConfig&>(*cfg);
            metric_type = prep_conf.metric_type.value();
        } else {
            // If no config provided, we might need to extract metric from index files or use a default approach
            // For now, we'll need to pass the metric type somehow - this is a limitation of the current approach
            // We'll try to extract it from the index prefix or assume it was stored in the index files
            LOG_KNOWHERE_WARNING_ << "No config provided to LoadIndexFiles, using default metric L2";
        }
        
        auto load_status = TryOdinANNCall([&]() {
            // choose metric
            pipeann::Metric metric = pipeann::Metric::L2;
            if (IsMetricType(metric_type, knowhere::metric::IP)) {
                metric = pipeann::Metric::INNER_PRODUCT;
            } else if (IsMetricType(metric_type, knowhere::metric::COSINE)) {
                metric = pipeann::Metric::COSINE;
            }
            LOG_KNOWHERE_INFO_ << "Using metric: " << (int)metric;

            file_reader_ = std::make_shared<pipeann::LinuxAlignedFileReader>();
            ssd_index_ = std::make_unique<pipeann::SSDIndex<DataType>>(metric, file_reader_, true, false, nullptr);
            
            int load_result = ssd_index_->load(index_prefix_.c_str(), static_cast<uint32_t>(search_pool_->size()),
                                               true,   // new_index_format
                                               false   // use_page_search
            );
            if (load_result != 0) {
                throw std::runtime_error("Failed to load SSDIndex: " + std::to_string(load_result));
            }
            
            // update dim_ and count_ from loaded index
            uint64_t num_pts = ssd_index_->return_nd();
            count_.store(static_cast<int64_t>(num_pts));
            dim_.store(static_cast<int64_t>(ssd_index_->data_dim));
            LOG_KNOWHERE_INFO_ << "Loaded OdinANN index with " << num_pts << " points, dim " << ssd_index_->data_dim;
            
            // Load global memory index if available (single global instance, not per-instance)
            {
                // Skip loading global memory index - not implemented in knowhere
                // Global memory index loading is not part of knowhere's standard implementation
                LOG_KNOWHERE_INFO_ << "Skipping global memory index loading - not supported in knowhere";
            }
        });
        
        if (load_status != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load OdinANN SSD index, will not retry";
            // Mark as prepared to prevent further loading attempts
            is_prepared_.store(true);
            return load_status;
        }
        
        is_prepared_.store(true);
        LOG_KNOWHERE_INFO_ << "OdinANN index files loaded successfully";
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (dim_.load() == -1) {
            // If index is not loaded yet, return 0
            if (!is_prepared_.load() && !index_prefix_.empty()) {
                LOG_KNOWHERE_INFO_ << "Dim() function called before index is loaded, returning 0";
                return 0;
            }
            LOG_KNOWHERE_ERROR_ << "Dim() function is not supported when index is not ready yet.";
            return 0;
        }
        return dim_.load();
    }

    int64_t
    Size() const override {
        // OdinANN index size calculation
        if (index_prefix_.empty()) {
            return 0;
        }
        
        // If file manager is not available, use direct file size calculation
        int64_t total_size = 0;
        
        // Add sizes of necessary files
        for (const auto& filename : GetNecessaryFilenames(index_prefix_)) {
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (file.good()) {
                total_size += file.tellg();
            }
        }
        
        // Add sizes of optional files
        for (const auto& filename : GetOptionalFilenames(index_prefix_)) {
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (file.good()) {
                total_size += file.tellg();
            }
        }
        
        return total_size;
    }

    int64_t
    Count() const override {
        if (count_.load() == -1) {
            // If index is not loaded yet, return 0
            if (!is_prepared_.load() && !index_prefix_.empty()) {
                LOG_KNOWHERE_INFO_ << "Count() function called before index is loaded, returning 0";
                return 0;
            }
            LOG_KNOWHERE_ERROR_ << "Count() function is not supported when index is not ready yet.";
            return 0;
        }
        return count_.load();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_ODINANN;
    }

 private:
    bool
    LoadFile(const std::string& filename) {
        if (file_manager_ == nullptr) {
            LOG_KNOWHERE_ERROR_ << "File manager is not set for OdinANN index";
            return false;
        }
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    bool
    AddFile(const std::string& filename) {
        if (file_manager_ == nullptr) {
            LOG_KNOWHERE_ERROR_ << "File manager is not set for OdinANN index";
            return false;
        }
        if (!file_manager_->AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return false;
        }
        return true;
    }

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> search_pool_;
    // Underlying OdinANN SSDIndex instance for disk-resident index
    std::unique_ptr<pipeann::SSDIndex<DataType>> ssd_index_;
    // File reader for disk access
    std::shared_ptr<pipeann::AlignedFileReader> file_reader_;

    // Ensure proper template instantiation for OdinANN API calls
    template<typename T, typename TagT>
    friend bool build_disk_index(const char *dataPath, const char *indexFilePath, const char *indexBuildParameters,
                                pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file);

    friend Status TryOdinANNCall(std::function<void()>&& odinann_call);
};

// Define the function outside the class
Status
TryOdinANNCall(std::function<void()>&& odinann_call) {
    try {
        odinann_call();
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "OdinANN Exception: " << e.what();
        return Status::odinann_inner_error;
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    // OdinANN build_disk_index generates:
    // - prefix_pq_pivots.bin
    // - prefix_pq_compressed.bin
    // - prefix_disk.index
    // - prefix_disk.index_medoids.bin
    // - prefix_disk.index_centroids.bin
    filenames.push_back(prefix + "_pq_pivots.bin");
    filenames.push_back(prefix + "_pq_compressed.bin");
    filenames.push_back(prefix + "_disk.index");
    filenames.push_back(prefix + "_disk.index_medoids.bin");
    filenames.push_back(prefix + "_disk.index_centroids.bin");
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    // Optional files for OdinANN (warmup sample file)
    filenames.push_back(prefix + "_sample.bin");
    // mem index file generated when mem_L > 0 during build
    filenames.push_back(prefix + "_mem.index");
    // tags file
    filenames.push_back(prefix + "_disk.index.tags");
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (pipeann::file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    return file_exist(GetNecessaryFilenames(index_prefix)) || file_exist(GetOptionalFilenames(index_prefix));
}

inline bool
CheckMetric(const std::string& metric) {
    if (metric != knowhere::metric::L2 && metric != knowhere::metric::IP && metric != knowhere::metric::COSINE) {
        LOG_KNOWHERE_ERROR_ << "OdinANN currently only supports floating point data for L2, IP, and COSINE metrics.";
        return false;
    }
    return true;
}






#ifdef KNOWHERE_WITH_CARDINAL
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(ODINANN_DEPRECATED, OdinANNIndexNode, knowhere::feature::DISK)
#else
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(ODINANN, OdinANNIndexNode, knowhere::feature::DISK)
#endif

// Explicit template instantiations for OdinANN API calls
// This is causing compilation issues, commenting out to avoid duplicate instantiation errors
/* namespace pipeann {
    // Additional template instantiations for other OdinANN functions
    template<typename T>
    int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_file_index,
                                  unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                  std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                  const char *tag_file);
} */

}  // namespace knowhere

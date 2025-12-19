#include "aux_utils_ext.h"
#include <stdexcept>
#include <thread>

namespace pipeann {

template<typename T, typename TagT>
BuildResult build_disk_index_from_config(const DiskIndexBuildConfig& config, Metric _compareMetric) {
    BuildResult result;
    
    try {
        // Convert new config structure to old API parameters
        bool success = build_disk_index_py<T, TagT>(
            config.data_path.c_str(),
            config.index_prefix.c_str(),
            config.max_degree,
            config.search_list_size,
            0, // M parameter (not used in OdinANN)
            config.num_threads == 0 ? std::thread::hardware_concurrency() : config.num_threads,
            config.pq_dims,
            _compareMetric,
            config.single_file_index,
            nullptr // tag_file (not used in current implementation)
        );
        
        result.success = success;
        if (!success) {
            result.error_message = "Build failed with unknown error";
        }
        // TODO: Populate num_points, dimension, and index_size from the built index
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception during build: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error_message = "Unknown exception during build";
    }
    
    return result;
}

template<typename T, typename TagT>
BuildResult build_disk_index_with_result(const char* dataPath, const char* indexFilePath, 
                                       uint32_t R, uint32_t L, uint32_t M,
                                       uint32_t num_threads, uint32_t PQ_bytes,
                                       Metric _compareMetric, bool single_file_index,
                                       const char* tag_file) {
    BuildResult result;
    
    try {
        bool success = build_disk_index_py<T, TagT>(
            dataPath, indexFilePath, R, L, M, 
            num_threads == 0 ? std::thread::hardware_concurrency() : num_threads,
            PQ_bytes, _compareMetric, single_file_index, tag_file
        );
        
        result.success = success;
        if (!success) {
            result.error_message = "Build failed with unknown error";
        }
        // TODO: Populate num_points, dimension, and index_size from the built index
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception during build: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error_message = "Unknown exception during build";
    }
    
    return result;
}

// Explicit template instantiations for common types
template BuildResult build_disk_index_from_config<float, uint32_t>(const DiskIndexBuildConfig&, Metric);
template BuildResult build_disk_index_from_config<uint8_t, uint32_t>(const DiskIndexBuildConfig&, Metric);

template BuildResult build_disk_index_with_result<float, uint32_t>(const char*, const char*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, Metric, bool, const char*);
template BuildResult build_disk_index_with_result<uint8_t, uint32_t>(const char*, const char*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, Metric, bool, const char*);

}  // namespace pipeann
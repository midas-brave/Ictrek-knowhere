#pragma once
#include <chrono>
#include <string>
#include <thread>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "tsl/robin_set.h"
#include "utils.h"
#include "parameters.h"

namespace pipeann {
  const size_t MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 1000000;
  const double PQ_TRAINING_SET_FRACTION = 0.1;
  const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t WARMUP_L = 20;

  template<typename T, typename TagT>
  class SSDIndex;

  // ============================================================================
  // NEW: Structured build configuration
  // ============================================================================
  
  struct DiskIndexBuildConfig {
      std::string data_path;              // Input: path to raw data binary file
      std::string index_prefix;           // Output: prefix for index files
      uint32_t max_degree;                // R: max graph degree (60-150 typical)
      uint32_t search_list_size;          // L: search list size during build (75-200)
      uint32_t pq_dims;                   // PQ compression dims (0=no compression)
      float build_dram_budget_gb;         // Memory budget in GB (0=unlimited)
      uint32_t num_threads;               // Number of threads for parallel build
      Metric metric_type;                 // Distance metric (L2, IP, COSINE)
      bool accelerate_build;              // Enable fast build (~30% faster, -1% recall)
      bool single_file_index;             // Save all in one file vs multiple files
      const char* tag_file;               // Optional: path to tag/id mapping file
      
      // Default constructor with sensible defaults
      DiskIndexBuildConfig()
          : max_degree(48),
            search_list_size(128),
            pq_dims(0),
            build_dram_budget_gb(0),
            num_threads(std::thread::hardware_concurrency()),
            metric_type(Metric::L2),
            accelerate_build(false),
            single_file_index(true),
            tag_file(nullptr) {
      }
      
      // Validation method
      bool validate(std::string& error_msg) const {
          if (data_path.empty()) {
              error_msg = "data_path cannot be empty";
              return false;
          }
          if (index_prefix.empty()) {
              error_msg = "index_prefix cannot be empty";
              return false;
          }
          if (max_degree < 1 || max_degree > 2048) {
              error_msg = "max_degree must be in range [1, 2048]";
              return false;
          }
          if (search_list_size < 1) {
              error_msg = "search_list_size must be >= 1";
              return false;
          }
          if (num_threads < 1) {
              error_msg = "num_threads must be >= 1";
              return false;
          }
          return true;
      }
  };
  
  // ============================================================================
  // NEW: Detailed build result with metrics
  // ============================================================================
  
  struct BuildResult {
      bool success;                       // Whether build completed successfully
      std::string error_message;          // Error message if build failed
      int64_t num_points;                 // Number of points indexed
      uint32_t dimension;                 // Vector dimension
      double build_time_seconds;          // Total build time
      double index_size_mb;               // Final index size in MB
      std::string index_file_path;        // Path to the built index
      
      BuildResult() 
          : success(false), num_points(0), dimension(0), 
            build_time_seconds(0), index_size_mb(0) {
      }
      
      explicit BuildResult(bool s) : success(s), num_points(0), dimension(0), 
                                     build_time_seconds(0), index_size_mb(0) {
      }
  };
  
  double get_memory_budget(const std::string &mem_budget_str);
  double get_memory_budget(double search_ram_budget_in_gb);
  void add_new_file_to_single_index(std::string index_file, std::string new_file);

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags);

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs);

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file);

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_index_file,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                        pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file = nullptr);
  template<typename T, typename TagT = uint32_t>
  bool build_disk_index_py(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                           uint32_t num_threads, uint32_t PQ_bytes, pipeann::Metric _compareMetric,
                           bool single_file_index, const char *tag_file);
  
  // ============================================================================
  // NEW: Modern config-based API
  // ============================================================================
  
  template<typename T, typename TagT = uint32_t>
  bool build_disk_index_from_config(const DiskIndexBuildConfig& config);
  
  template<typename T, typename TagT = uint32_t>
  BuildResult build_disk_index_with_result(const DiskIndexBuildConfig& config);
  
  template<typename T, typename TagT = uint32_t>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file,
                          bool single_file_index, const std::string &output_file);
}  // namespace pipeann

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

#ifndef ODINANN_H
#define ODINANN_H

#include <memory>
#include <string>
#include <vector>

#include "knowhere/index/index_node.h"
#include "knowhere/index/odinann/odinann_config.h"

namespace knowhere {

template <typename DataType>
class OdinANNIndexNode : public IndexNode {
 public:
    OdinANNIndexNode(const Object& object);
    ~OdinANNIndexNode() override;

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override;

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override;

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override;

    Status
    Serialize(BinarySet& binset) const override;

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override;

    int64_t
    Dim() const override;

    int64_t
    Size() const override;

    int64_t
    Count() const override;

    std::string
    Type() const override;

 private:
    // Implementation details would go here
};

}  // namespace knowhere

#endif /* ODINANN_H */
/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_DEPTH_FEATURE_CONSTRAINTS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_DEPTH_FEATURE_CONSTRAINTS_HPP_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace LightGBM {

using json11_internal_lightgbm::Json;

/*! \brief One depth stage: from min_depth inclusive onward until a later stage. */
struct DepthFeatureStage {
  int min_depth = 0;
  /*! \brief Inner feature indices allowed at this stage. */
  std::vector<int8_t> inner_mask;
};

/*! \brief True when a forced-split node has both feature and threshold. */
inline bool ForcedSplitNodeHasFeatureAndThreshold(const Json& node) {
  if (!node.is_object()) {
    return false;
  }
  const auto& items = node.object_items();
  return items.count("feature") > 0 && items.count("threshold") > 0;
}

inline bool ForcedSplitJsonHasDepthConstraints(const Json& root) {
  if (!root.is_object()) {
    return false;
  }
  return root.object_items().count("depth_feature_constraints") > 0;
}

inline int RequireExactNonNegativeInt(const Json& value, const char* field_name) {
  if (!value.is_number()) {
    Log::Fatal("%s must be a number", field_name);
  }
  const double raw = value.number_value();
  if (!std::isfinite(raw)) {
    Log::Fatal("%s must be a finite number", field_name);
  }
  if (raw < 0.0) {
    Log::Fatal("%s must be >= 0, got %g", field_name, raw);
  }
  if (raw > static_cast<double>(std::numeric_limits<int>::max())) {
    Log::Fatal("%s exceeds maximum int value, got %g", field_name, raw);
  }
  const int as_int = static_cast<int>(raw);
  if (static_cast<double>(as_int) != raw) {
    Log::Fatal("%s must be an integer, got %g", field_name, raw);
  }
  return as_int;
}

/*!
 * \brief Validate and parse depth_feature_constraints from forced-splits JSON root.
 * \return Sorted stages by increasing min_depth; empty if key absent.
 */
inline std::vector<DepthFeatureStage> ParseDepthFeatureConstraints(
    const Json& root, const Dataset* train_data, int max_feature_idx) {
  std::vector<DepthFeatureStage> stages;
  if (!ForcedSplitJsonHasDepthConstraints(root)) {
    return stages;
  }
  const Json& arr = root["depth_feature_constraints"];
  if (!arr.is_array()) {
    Log::Fatal("Forced splits field depth_feature_constraints must be a JSON array");
  }
  if (arr.array_items().empty()) {
    Log::Fatal("Forced splits field depth_feature_constraints must be a non-empty array");
  }
  std::set<int> seen_depths;
  const int num_inner = train_data->num_features();
  for (const auto& item : arr.array_items()) {
    if (!item.is_object()) {
      Log::Fatal("Each depth_feature_constraints entry must be a JSON object");
    }
    const auto& obj = item.object_items();
    if (obj.count("min_depth") == 0 || obj.count("features") == 0) {
      Log::Fatal(
          "Each depth_feature_constraints entry must contain min_depth and features");
    }
    const int min_depth =
        RequireExactNonNegativeInt(item["min_depth"], "depth_feature_constraints min_depth");
    if (seen_depths.count(min_depth) > 0) {
      Log::Fatal("Duplicate depth_feature_constraints min_depth %d", min_depth);
    }
    seen_depths.insert(min_depth);
    if (!item["features"].is_array()) {
      Log::Fatal("depth_feature_constraints features must be a JSON array");
    }
    const auto& feats = item["features"].array_items();
    if (feats.empty()) {
      Log::Fatal("depth_feature_constraints features must be non-empty at min_depth %d",
                 min_depth);
    }
    DepthFeatureStage stage;
    stage.min_depth = min_depth;
    stage.inner_mask.assign(num_inner, 0);
    int allowed_count = 0;
    for (const auto& f : feats) {
      const int raw_idx =
          RequireExactNonNegativeInt(f, "depth_feature_constraints feature index");
      if (raw_idx > max_feature_idx) {
        Log::Fatal(
            "Forced splits depth_feature_constraints includes feature index %d, "
            "but maximum feature index in dataset is %d",
            raw_idx, max_feature_idx);
      }
      const int inner = train_data->InnerFeatureIndex(raw_idx);
      if (inner < 0) {
        Log::Fatal(
            "Forced splits depth_feature_constraints feature index %d is not usable "
            "(ignored or non-splittable)",
            raw_idx);
      }
      if (stage.inner_mask[inner] == 0) {
        stage.inner_mask[inner] = 1;
        ++allowed_count;
      }
    }
    if (allowed_count == 0) {
      Log::Fatal("depth_feature_constraints at min_depth %d has no usable features",
                 min_depth);
    }
    stages.push_back(std::move(stage));
  }
  std::sort(stages.begin(), stages.end(),
            [](const DepthFeatureStage& a, const DepthFeatureStage& b) {
              return a.min_depth < b.min_depth;
            });
  return stages;
}

/*! \brief Greatest matching min_depth stage, or nullptr if none. */
inline const DepthFeatureStage* FindDepthFeatureStage(
    const std::vector<DepthFeatureStage>& stages, int leaf_depth) {
  const DepthFeatureStage* best = nullptr;
  for (const auto& stage : stages) {
    if (stage.min_depth <= leaf_depth) {
      best = &stage;
    } else {
      break;
    }
  }
  return best;
}

inline void CheckDepthConstraintsBackendSupport(const Config* config, bool has_depth_constraints) {
  if (!has_depth_constraints) {
    return;
  }
  if (config->device_type != std::string("cpu") || config->tree_learner != std::string("serial")) {
    Log::Fatal("Don't support depth_feature_constraints in tree_learner=%s device_type=%s",
               config->tree_learner.c_str(), config->device_type.c_str());
  }
  if (config->use_quantized_grad) {
    Log::Fatal("Don't support depth_feature_constraints with use_quantized_grad=true");
  }
}

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_DEPTH_FEATURE_CONSTRAINTS_HPP_

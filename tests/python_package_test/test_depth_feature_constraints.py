# coding: utf-8
"""Tests for depth_feature_constraints on forcedsplits_filename JSON."""

import json

import numpy as np
import pytest

import lightgbm as lgb

from .utils import make_synthetic_regression


def _collect_split_features_by_depth(node, depth=0, out=None):
    """Walk dump_model tree_structure; record (depth, split_feature) for internal nodes."""
    if out is None:
        out = []
    if "split_feature" not in node:
        return out
    out.append((depth, node["split_feature"]))
    if "left_child" in node:
        _collect_split_features_by_depth(node["left_child"], depth + 1, out)
    if "right_child" in node:
        _collect_split_features_by_depth(node["right_child"], depth + 1, out)
    return out


def _assert_branch_features_after_split(node, parent_feat, allowed):
    """After a split on parent_feat, all deeper splits on that subtree must be in allowed."""
    if "split_feature" not in node:
        return
    if node["split_feature"] == parent_feat:
        for child_key in ("left_child", "right_child"):
            child = node.get(child_key)
            if child is None:
                continue
            for _, feat in _collect_split_features_by_depth(child, depth=1):
                assert feat in allowed, f"after split on {parent_feat}, found {feat}"
        return
    for child_key in ("left_child", "right_child"):
        child = node.get(child_key)
        if child is not None:
            _assert_branch_features_after_split(child, parent_feat, allowed)


def _write_forced_json(tmp_path, payload, name="forced.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _serial_cpu_params(**overrides):
    params = {
        "objective": "regression",
        "verbosity": -1,
        "tree_learner": "serial",
        "device_type": "cpu",
    }
    params.update(overrides)
    return params


def test_depth_feature_constraints_stages(tmp_path):
    rng = np.random.RandomState(0)
    n = 2000
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] > 0).astype(float) * 3.0 + (X[:, 1] > 0).astype(float) * 2.0
    y += (X[:, 2] > 0).astype(float) * 1.5 + (X[:, 3] > 0).astype(float) * 1.0
    y += rng.normal(scale=0.05, size=n)
    split_file = _write_forced_json(
        tmp_path,
        {
            "depth_feature_constraints": [
                {"min_depth": 2, "features": [0, 1]},
                {"min_depth": 3, "features": [2, 3]},
            ]
        },
    )
    bst = lgb.train(
        _serial_cpu_params(
            num_leaves=31,
            min_data_in_leaf=5,
            learning_rate=1.0,
            seed=1,
            forcedsplits_filename=split_file,
        ),
        lgb.Dataset(X, y),
        num_boost_round=3,
    )
    saw_deep = False
    for tree in bst.dump_model()["tree_info"]:
        for depth, feat in _collect_split_features_by_depth(tree["tree_structure"]):
            if depth >= 3:
                assert feat in (2, 3)
                saw_deep = True
            elif depth >= 2:
                assert feat in (0, 1)
    assert saw_deep


def test_depth_feature_constraints_no_shallow_future_stage_leak(tmp_path):
    rng = np.random.RandomState(29)
    n = 4000
    X = rng.normal(size=(n, 30))
    y = 20.0 * X[:, 2] + 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(scale=0.01, size=n)
    depth_file = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 2, "features": [2, 3]}]},
        "depth_no_leak.json",
    )
    common = _serial_cpu_params(
        num_leaves=8,
        feature_fraction=0.1,
        feature_fraction_bynode=1.0,
        feature_fraction_seed=29,
        learning_rate=1.0,
        seed=29,
    )
    baseline = lgb.train(common, lgb.Dataset(X, y), num_boost_round=1)
    staged = lgb.train(
        {**common, "forcedsplits_filename": depth_file},
        lgb.Dataset(X, y),
        num_boost_round=1,
    )
    baseline_root = baseline.dump_model()["tree_info"][0]["tree_structure"]["split_feature"]
    staged_root = staged.dump_model()["tree_info"][0]["tree_structure"]["split_feature"]
    assert staged_root != 2
    assert staged_root == baseline_root


def test_depth_feature_constraints_feature_fraction_and_interaction(tmp_path):
    rng = np.random.RandomState(3)
    n = 1500
    X = rng.normal(size=(n, 10))
    y = X[:, 7] * 5.0 + X[:, 8] * 4.0 + rng.normal(scale=0.1, size=n)
    ff_file = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [7, 8]}]},
        "ff.json",
    )
    bst = lgb.train(
        _serial_cpu_params(
            feature_fraction=0.2,
            feature_fraction_bynode=0.2,
            feature_fraction_seed=99,
            num_leaves=15,
            forcedsplits_filename=ff_file,
            seed=99,
        ),
        lgb.Dataset(X, y),
        num_boost_round=5,
    )
    for tree in bst.dump_model()["tree_info"]:
        for _, feat in _collect_split_features_by_depth(tree["tree_structure"]):
            assert feat in (7, 8)

    rng = np.random.RandomState(15)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] > 0).astype(float) * 5.0 + X[:, 1] * 2.0 + X[:, 2] * 2.0 + X[:, 3]
    y += rng.normal(scale=0.05, size=n)
    inter_file = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [0, 1, 2, 3]}]},
        "inter.json",
    )
    bst = lgb.train(
        _serial_cpu_params(
            interaction_constraints=[[0, 1], [2, 3]],
            num_leaves=31,
            min_data_in_leaf=5,
            learning_rate=1.0,
            forcedsplits_filename=inter_file,
            seed=15,
        ),
        lgb.Dataset(X, y),
        num_boost_round=3,
    )
    saw_root0 = False
    for tree in bst.dump_model()["tree_info"]:
        for _, feat in _collect_split_features_by_depth(tree["tree_structure"]):
            assert feat in (0, 1, 2, 3)
        root = tree["tree_structure"]
        if root.get("split_feature") == 0:
            saw_root0 = True
            _assert_branch_features_after_split(root, parent_feat=0, allowed=(0, 1))
    assert saw_root0


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"depth_feature_constraints": [{"min_depth": 0, "features": [999]}]}, "feature index"),
        (
            {
                "depth_feature_constraints": [
                    {"min_depth": 1, "features": [0]},
                    {"min_depth": 1, "features": [1]},
                ]
            },
            "Duplicate depth_feature_constraints",
        ),
        ({"depth_feature_constraints": [{"min_depth": 0, "features": []}]}, "features must be non-empty"),
        ({"depth_feature_constraints": []}, "non-empty array"),
        ({"depth_feature_constraints": [{"min_depth": 1.5, "features": [0]}]}, "must be an integer"),
        ({"depth_feature_constraints": [{"min_depth": 1e100, "features": [0]}]}, "exceeds maximum int"),
        (
            {
                "feature": 0,
                "threshold": 0.0,
                "depth_feature_constraints": [{"min_depth": 0, "features": [1, 2]}],
            },
            "not allowed by depth_feature_constraints",
        ),
    ],
)
def test_depth_feature_constraints_schema_errors(tmp_path, payload, match):
    X, y = make_synthetic_regression()
    split_file = _write_forced_json(tmp_path, payload)
    params = _serial_cpu_params(forcedsplits_filename=split_file, min_data_in_leaf=1, num_leaves=4)
    with pytest.raises(lgb.basic.LightGBMError, match=match):
        lgb.train(params, lgb.Dataset(X, y), num_boost_round=1)


def test_depth_feature_constraints_backend_and_reset_safety(tmp_path):
    rng = np.random.RandomState(42)
    n = 1500
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 3.0 + X[:, 4] * 2.0 + rng.normal(scale=0.05, size=n)
    depth_file = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [1]}]},
        "depth.json",
    )
    depth_b = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [4]}]},
        "depth_b.json",
    )
    depth_bad = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [999]}]},
        "depth_bad.json",
    )
    dtrain = lgb.Dataset(X, y)
    bst = lgb.Booster(
        params=_serial_cpu_params(
            num_leaves=8,
            learning_rate=1.0,
            seed=42,
            forcedsplits_filename=depth_file,
        ),
        train_set=dtrain,
    )
    bst.update()
    for _, feat in _collect_split_features_by_depth(bst.dump_model()["tree_info"][0]["tree_structure"]):
        assert feat == 1
    with pytest.raises(lgb.basic.LightGBMError, match="feature index"):
        bst.reset_parameter({"forcedsplits_filename": depth_bad})
    bst.reset_parameter({"forcedsplits_filename": depth_b})
    bst.update()
    for _, feat in _collect_split_features_by_depth(bst.dump_model()["tree_info"][1]["tree_structure"]):
        assert feat == 4

    depth_only = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [0, 1]}]},
        "depth_gpu.json",
    )
    bst2 = lgb.Booster(
        params=_serial_cpu_params(
            num_leaves=8,
            learning_rate=1.0,
            seed=22,
            forcedsplits_filename=depth_only,
        ),
        train_set=lgb.Dataset(X[:, :4], y),
    )
    bst2.update()
    with pytest.raises(lgb.basic.LightGBMError, match="Don't support depth_feature_constraints"):
        bst2.reset_parameter({"device_type": "gpu"})
    bst2.update()
    assert "[device_type: cpu]" in bst2.model_to_string()

    with pytest.raises(lgb.basic.LightGBMError, match="Don't support depth_feature_constraints"):
        lgb.train(
            _serial_cpu_params(
                use_quantized_grad=True,
                forcedsplits_filename=depth_only,
                num_leaves=8,
            ),
            lgb.Dataset(X[:, :4], y),
            num_boost_round=1,
        )


def test_depth_feature_constraints_single_machine_parallel_learner_fallback(tmp_path):
    rng = np.random.RandomState(9)
    n = 1000
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 3.0 + X[:, 1] * 2.0 + rng.normal(scale=0.05, size=n)
    split_file = _write_forced_json(
        tmp_path,
        {"depth_feature_constraints": [{"min_depth": 0, "features": [0, 1]}]},
    )
    bst = lgb.train(
        _serial_cpu_params(
            forcedsplits_filename=split_file,
            tree_learner="feature",
            num_leaves=8,
            seed=9,
        ),
        lgb.Dataset(X, y),
        num_boost_round=2,
    )
    assert "[tree_learner: serial]" in bst.model_to_string()
    for tree in bst.dump_model()["tree_info"]:
        for _, feat in _collect_split_features_by_depth(tree["tree_structure"]):
            assert feat in (0, 1)


def test_depth_feature_constraints_histogram_memory_stress(tmp_path):
    rng = np.random.RandomState(51)
    n, p = 1500, 80
    X = rng.normal(size=(n, p))
    y = 8.0 * X[:, 70] + 6.0 * X[:, 71] + 3.0 * X[:, 0] + rng.normal(scale=0.05, size=n)
    split_file = _write_forced_json(
        tmp_path,
        {
            "depth_feature_constraints": [
                {"min_depth": 1, "features": [0, 1]},
                {"min_depth": 2, "features": [70, 71]},
            ]
        },
    )
    bst = lgb.train(
        _serial_cpu_params(
            num_leaves=31,
            feature_fraction=0.05,
            feature_fraction_seed=51,
            forcedsplits_filename=split_file,
            seed=51,
        ),
        lgb.Dataset(X, y),
        num_boost_round=3,
    )
    for tree in bst.dump_model()["tree_info"]:
        for depth, feat in _collect_split_features_by_depth(tree["tree_structure"]):
            if depth >= 2:
                assert feat in (70, 71)
            elif depth >= 1:
                assert feat in (0, 1)

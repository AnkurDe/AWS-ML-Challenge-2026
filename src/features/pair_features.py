"""Feature extraction for a Source 1 / candidate pair."""

from __future__ import annotations

from .address_features import address_features
from .name_features import name_features


def build_pair_features(source_row, target_row) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(name_features(source_row.name_norm_core, target_row["name_norm_core"]))
    features.update(address_features(source_row.address_norm, target_row["address_norm"]))
    features["country_equal"] = float(
        source_row.country_norm == target_row["country_norm"]
    )
    features["source_is_s2"] = float(target_row["source"] == "S2")
    features["source_is_s3"] = float(target_row["source"] == "S3")
    features["name_address_combo"] = (
        0.65 * features["name_token_set"] + 0.35 * features["address_token_set"]
    )
    return features


def build_feature_matrix(source1, candidate_map, target_by_id):
    import pandas as pd

    rows: list[dict[str, float]] = []
    keys: list[tuple[str, str]] = []
    for row in source1.itertuples(index=False):
        s1_id = str(row.entity_id)
        for candidate_id in candidate_map.get(s1_id, []):
            target = target_by_id[candidate_id]
            rows.append(build_pair_features(row, target))
            keys.append((s1_id, candidate_id))
    if not rows:
        return pd.DataFrame(), keys
    return pd.DataFrame(rows).fillna(0.0), keys

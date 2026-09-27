"""End-to-end Amazon ML Challenge entity-resolution pipeline.

The implementation deliberately keeps the pipeline small:
1. Normalize names/addresses.
2. Generate multi-pass candidates.
3. Build pairwise similarity features.
4. Train a binary matcher.
5. Tune F0.5 threshold on a Source-1-level validation split.
6. Refit and generate the two required TSV outputs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.blocking import BlockingConfig, CandidateGenerator
from src.features import build_feature_matrix
from src.model import PairMatcher, build_predictions
from src.model.threshold import macro_f05, tune_threshold
from src.preprocessing import normalize_address, normalize_name_core

SEED = 42


class PipelineConfig:
    max_candidates_per_entity = 80
    max_block_frequency = 250
    max_name_ngram_frequency = 500
    max_train_negatives_per_entity = 20
    validation_fraction = 0.20


def read_tsv(path: Path, columns: tuple[str, ...], source: str | None = None) -> pd.DataFrame:
    # Parse bounded chunks and show progress instead of creating a parser-sized
    # temporary allocation for the full multi-GB input.
    chunks = pd.read_csv(
        path,
        sep="\t",
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    )
    loaded = []
    for chunk in tqdm(chunks, desc=f"Loading {path.name}", unit="chunk"):
        loaded.append(prepare_frame(chunk, source) if source else chunk)
    return pd.concat(loaded, ignore_index=True, copy=False)


def prepare_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # The frame is freshly loaded by load_dataset; copying it here needlessly
    # doubles its memory while the multi-hundred-MB source files are prepared.
    frame["source"] = source
    frame["country_norm"] = frame["country"].fillna("").map(lambda x: str(x).strip().casefold())
    frame["name_norm_core"] = frame["business_name"].map(normalize_name_core)
    frame["address_norm"] = frame["business_address"].map(normalize_address)
    # Downstream retrieval and matching use only these normalized values.
    # Discard their larger raw inputs and unused country/name-normalized fields.
    frame.drop(columns=["business_name", "business_address", "country"], inplace=True)
    return frame


def load_dataset(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    directory = data_dir / split
    source_columns = ("entity_id", "business_name", "business_address", "country")
    s1 = read_tsv(directory / f"{split}_source1.tsv", source_columns, "S1")
    s2 = read_tsv(directory / f"{split}_source2.tsv", source_columns, "S2")
    s3 = read_tsv(directory / f"{split}_source3.tsv", source_columns, "S3")
    truth_path = directory / f"{split}_ground_truth.tsv"
    truth = read_tsv(truth_path, ("source1_entity_id", "matched_entity_ids")) if truth_path.exists() else None
    return s1, s2, s3, truth


def truth_map(truth: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in truth.itertuples(index=False):
        value = str(row.matched_entity_ids).strip()
        result[str(row.source1_entity_id)] = {item for item in value.split(",") if item}
    return result


def quick_pair_score(source_name: str, source_address: str, target: dict) -> float:
    from rapidfuzz import fuzz

    name_score = fuzz.token_set_ratio(source_name, target["name_norm_core"])
    address_score = fuzz.token_set_ratio(source_address, target["address_norm"])
    return 0.65 * name_score + 0.35 * address_score


def cap_training_candidates(
    source1: pd.DataFrame,
    candidates: dict[str, list[str]],
    truth: dict[str, set[str]],
    targets: dict[str, dict],
    max_negatives_per_entity: int,
) -> dict[str, list[str]]:
    """Keep all generated positives and only the strongest generated negatives."""
    source_rows = {
        str(row.entity_id): (row.name_norm_core, row.address_norm)
        for row in source1.itertuples(index=False)
    }
    result: dict[str, list[str]] = {}
    rng = random.Random(SEED)

    for s1_id, candidate_ids in candidates.items():
        actual = truth.get(s1_id, set())
        positives = [cid for cid in candidate_ids if cid in actual]
        negatives = [cid for cid in candidate_ids if cid not in actual]
        source_name, source_address = source_rows[s1_id]
        negatives.sort(
            key=lambda cid: quick_pair_score(source_name, source_address, targets[cid]),
            reverse=True,
        )
        if len(negatives) > max_negatives_per_entity:
            # Deterministic hard-negative sampling. A small random tie-break is not necessary.
            negatives = negatives[:max_negatives_per_entity]
        result[s1_id] = sorted(set(positives + negatives))

    # Keep the variable intentionally used so the function remains deterministic if sampling is added later.
    rng.random()
    return result


def build_labeled_data(source1, candidate_map, truth_map_data, targets):
    features, keys = build_feature_matrix(source1, candidate_map, targets)
    labels = [int(candidate_id in truth_map_data.get(s1_id, set())) for s1_id, candidate_id in keys]
    return features, keys, labels


def candidate_recall(candidate_map: dict[str, list[str]], truth: dict[str, set[str]]) -> float:
    total = 0
    found = 0
    for s1_id, actual in truth.items():
        total += len(actual)
        found += len(actual & set(candidate_map.get(s1_id, [])))
    return found / total if total else 1.0


def split_source1(source1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = source1["entity_id"].astype(str).tolist()
    train_ids, val_ids = train_test_split(
        ids,
        test_size=PipelineConfig.validation_fraction,
        random_state=SEED,
        shuffle=True,
    )
    train_set = set(train_ids)
    val_set = set(val_ids)
    return source1[source1["entity_id"].isin(train_set)].copy(), source1[source1["entity_id"].isin(val_set)].copy()


def make_generator() -> CandidateGenerator:
    return CandidateGenerator(
        BlockingConfig(
            max_candidates_per_entity=PipelineConfig.max_candidates_per_entity,
            max_name_block_frequency=PipelineConfig.max_block_frequency,
            max_address_block_frequency=PipelineConfig.max_block_frequency,
            max_name_ngram_frequency=PipelineConfig.max_name_ngram_frequency,
        )
    )


def train(data_dir: Path, model_dir: Path) -> None:
    s1, s2, s3, truth_frame = load_dataset(data_dir, "train")
    if truth_frame is None:
        raise FileNotFoundError("train_ground_truth.tsv is required for training.")
    truth = truth_map(truth_frame)

    train_s1, val_s1 = split_source1(s1)
    generator = make_generator().fit(s2, s3)
    # Candidate indexes and compact target fields are now built; raw target
    # frames are no longer needed and otherwise remain live throughout fitting.
    del s2, s3
    targets = generator._target_by_id

    train_candidates = generator.generate(train_s1)
    val_candidates = generator.generate(val_s1)

    print(f"[train] S1 train/val: {len(train_s1):,}/{len(val_s1):,}")
    print(f"[train] candidate recall: train={candidate_recall(train_candidates, truth):.4f} val={candidate_recall(val_candidates, truth):.4f}")
    print(f"[train] avg candidates: train={_avg_candidates(train_candidates):.1f} val={_avg_candidates(val_candidates):.1f}")

    train_truth = {s1_id: truth.get(s1_id, set()) for s1_id in train_s1["entity_id"].astype(str)}
    val_truth = {s1_id: truth.get(s1_id, set()) for s1_id in val_s1["entity_id"].astype(str)}

    capped_train = cap_training_candidates(
        train_s1,
        train_candidates,
        train_truth,
        targets,
        PipelineConfig.max_train_negatives_per_entity,
    )
    x_train, _, y_train = build_labeled_data(train_s1, capped_train, train_truth, targets)

    matcher = PairMatcher().fit(x_train, y_train)
    del x_train, y_train, capped_train, train_candidates

    x_val, val_keys, _ = build_labeled_data(val_s1, val_candidates, val_truth, targets)
    val_probabilities = matcher.predict_proba(x_val)
    threshold, score = tune_threshold(val_probabilities.tolist(), val_keys, val_truth)
    val_predictions = build_predictions(
        val_keys,
        val_probabilities,
        threshold,
        val_s1["entity_id"].astype(str).tolist(),
    )
    score = macro_f05(val_predictions, val_truth)
    del x_val, val_probabilities, val_keys, val_predictions, val_candidates

    # Refit using all training Source 1 entities with hard negatives selected from the full candidate set.
    full_candidates = generator.generate(s1)
    capped_full = cap_training_candidates(s1, full_candidates, truth, targets, PipelineConfig.max_train_negatives_per_entity)
    del full_candidates
    x_full, _, y_full = build_labeled_data(s1, capped_full, truth, targets)
    matcher = PairMatcher().fit(x_full, y_full)
    del x_full, y_full, capped_full

    model_dir.mkdir(parents=True, exist_ok=True)
    matcher.save(model_dir / "pair_matcher.joblib")
    metadata = {
        "threshold": threshold,
        "validation_f05": score,
        "feature_columns": matcher.feature_columns,
        "blocking": {
            "max_candidates_per_entity": PipelineConfig.max_candidates_per_entity,
            "max_block_frequency": PipelineConfig.max_block_frequency,
            "max_name_ngram_frequency": PipelineConfig.max_name_ngram_frequency,
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[train] validation F0.5: {score:.4f} @ threshold={threshold:.3f}")
    print(f"[train] saved model to {model_dir}")


def predict(data_dir: Path, model_dir: Path, output_dir: Path) -> None:
    s1, s2, s3, _ = load_dataset(data_dir, "test")
    matcher = PairMatcher.load(model_dir / "pair_matcher.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    generator = make_generator().fit(s2, s3)
    del s2, s3
    candidates = generator.generate(s1)
    targets = generator._target_by_id
    features, keys = build_feature_matrix(s1, candidates, targets)
    probabilities = matcher.predict_proba(features)
    predictions = build_predictions(keys, probabilities, threshold, s1["entity_id"].astype(str).tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_output(s1, candidates, output_dir / "candidate_pairs.tsv")
    write_match_output(s1, predictions, output_dir / "matching_results.tsv")

    final_pairs = sum(len(v) for v in predictions.values())
    print(f"[predict] test S1: {len(s1):,} | avg candidates: {_avg_candidates(candidates):.1f} | predicted links: {final_pairs:,}")
    print(f"[predict] wrote {output_dir / 'candidate_pairs.tsv'}")
    print(f"[predict] wrote {output_dir / 'matching_results.tsv'}")


def write_candidate_output(source1: pd.DataFrame, candidates: dict[str, list[str]], path: Path) -> None:
    rows = [
        {
            "source1_entity_id": str(entity_id),
            "candidate_entity_ids": ",".join(candidates.get(str(entity_id), [])),
        }
        for entity_id in source1["entity_id"].astype(str)
    ]
    pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(path, sep="\t", index=False)


def write_match_output(source1: pd.DataFrame, predictions: dict[str, set[str]], path: Path) -> None:
    rows = [
        {
            "source1_entity_id": str(entity_id),
            "matched_entity_ids": ",".join(sorted(predictions.get(str(entity_id), set()))),
        }
        for entity_id in source1["entity_id"].astype(str)
    ]
    pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(path, sep="\t", index=False)


def _avg_candidates(candidate_map: dict[str, list[str]]) -> float:
    if not candidate_map:
        return 0.0
    return sum(len(v) for v in candidate_map.values()) / len(candidate_map)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon ML Challenge entity-resolution pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "run"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--model-dir", type=Path, default=Path("models"))
        p.set_defaults(func=train if command == "train" else run_all)

    p = sub.add_parser("predict")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.set_defaults(func=predict)
    return parser


def run_all(data_dir: Path, model_dir: Path) -> None:
    train(data_dir, model_dir)
    predict(data_dir, model_dir, Path("output"))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        args.func(args.data_dir, args.model_dir)
    elif args.command == "train":
        args.func(args.data_dir, args.model_dir)
    else:
        args.func(args.data_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()

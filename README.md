# Amazon ML Challenge 2026 — Business Entity Resolution

A compact, reproducible baseline for the Business Entity Resolution challenge.

The pipeline has two explicit stages:

1. **Blocking / candidate generation** — reduce the Source 2/Source 3 search space for every Source 1 record.
2. **Matching / ranking** — score each generated pair and keep only sufficiently strong matches.

The final pipeline writes the two required files:

- `output/matching_results.tsv` — final predicted links.
- `output/candidate_pairs.tsv` — the exact candidate set passed to the matching model.

## 1. Project structure

```text
amazon_ml_entity_resolution/
├── output/
├── models/
├── src/
│   ├── preprocessing/
│   │   ├── name_normalizer.py
│   │   ├── address_normalizer.py
│   │   └── tokenization.py
│   ├── blocking/
│   │   ├── name_blocking.py
│   │   ├── address_blocking.py
│   │   └── candidate_generator.py
│   ├── features/
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   └── pair_features.py
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── threshold.py
│   └── pipeline.py
├── Documentation_template.md
├── README.md
└── requirements.txt
```

The modules have single responsibilities so the main pipeline can remain small and easy to change.

## 2. Requirements

Python **3.12** is the target runtime.

Create an environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Expected dataset layout

The challenge data should be available in this form:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

The files are read with `sep="\t"`.

## 4. Run the complete pipeline

From the project root:

```bash
python -m src.pipeline run --data-dir dataset
```

This performs:

```text
train data
   ↓
normalization
   ↓
blocking
   ↓
pair features
   ↓
classifier training
   ↓
validation threshold tuning using macro F0.5
   ↓
refit on all training data
   ↓
test candidate generation
   ↓
test matching
   ↓
output/*.tsv
```

The terminal output is intentionally compact and reports only the important run statistics.

## 5. Train only

```bash
python -m src.pipeline train --data-dir dataset --model-dir models
```

The trained model and metadata are written to:

```text
models/
├── pair_matcher.joblib
└── metadata.json
```

## 6. Predict using an existing model

```bash
python -m src.pipeline predict \
  --data-dir dataset \
  --model-dir models \
  --output-dir output
```

## 7. What the blocker does

Candidate generation uses several local blocking signals:

- country-aware name blocks
- normalized name prefix / first-token blocks
- rare name-token blocks
- postal-code and numeric-address blocks
- rare address-token blocks
- character 3-gram name blocks for typo-tolerant retrieval

The candidate sets are capped by `max_candidates_per_entity` after lightweight fuzzy reranking.

No external database, API, geocoder, web lookup, or external business data is used.

## 8. What the matcher does

For every generated pair, the feature layer computes compact string and structural signals such as:

- name edit similarity
- name token-set similarity
- name token-sort similarity
- name token Jaccard overlap
- address edit similarity
- address token similarity
- address Jaccard overlap
- numeric and postal-code overlap
- country equality
- source indicator
- combined name/address similarity

The default classifier is `HistGradientBoostingClassifier` from scikit-learn.

## 9. Validation strategy

The training Source 1 entities are split at the **entity level**, not at the pair level. This keeps all candidate pairs for a Source 1 entity on the same side of the split.

The validation threshold is selected against the challenge's macro-averaged F0.5 objective, including singleton entities.

## 10. Output files

### `matching_results.tsv`

```text
source1_entity_id<TAB>matched_entity_ids
```

Example:

```text
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### `candidate_pairs.tsv`

```text
source1_entity_id<TAB>candidate_entity_ids
```

Every final match must be present in the candidate list.

## 11. Tuning the baseline

The most useful parameters are in `src/pipeline.py`:

```python
class PipelineConfig:
    max_candidates_per_entity = 80
    max_block_frequency = 250
    fuzzy_rescue_top_k = 20
    max_train_negatives_per_entity = 20
    validation_fraction = 0.20
```

The first parameters to experiment with are:

- `max_candidates_per_entity` — candidate-set size / blocking recall trade-off.
- `max_block_frequency` — controls how selective rare-token blocks are.
- `max_name_ngram_frequency` — controls how selective character n-gram blocks are.
- `max_train_negatives_per_entity` — controls training-set size while keeping hard negatives.

Before every submission, validate the generated TSVs with the challenge-provided validator.

## 12. Recommended next improvements

The code is intentionally a clean baseline rather than a large framework. A natural next step is to improve candidate recall with character n-gram TF-IDF retrieval and then add stronger pair features without changing the rest of the pipeline contract.

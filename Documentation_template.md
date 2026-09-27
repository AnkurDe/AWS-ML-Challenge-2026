# Methodology — Business Entity Resolution

## 1. Overview

Describe the overall entity-resolution pipeline and the motivation for using a two-stage design:

```text
Source 1
  ↓
Blocking / candidate generation
  ↓
Candidate pairs
  ↓
Pairwise feature engineering
  ↓
Binary matching model
  ↓
Threshold selection
  ↓
Final entity matches
```

Replace this section with the final method used in the submission.

## 2. Data preprocessing

### 2.1 Business names

Describe:

- Unicode normalization / transliteration
- case normalization
- punctuation handling
- business/legal suffix normalization
- tokenization
- treatment of empty values

### 2.2 Business addresses

Describe:

- abbreviation normalization
- tokenization
- numeric components
- postal-code extraction
- handling of missing address components
- treatment of landmarks and component reordering

No external address lookup or geocoding is used.

## 3. Candidate generation / blocking

Explain how the search space is reduced before matching.

### Blocking rules

Document every blocking rule used, for example:

- country-aware blocks
- normalized name prefix
- first-name-token blocks
- rare name tokens
- postal-code / numeric address blocks
- rare address tokens
- fuzzy rescue retrieval

### Candidate-set controls

Document:

- maximum candidates per Source 1 entity
- block-frequency limits
- fuzzy rescue size
- deduplication rules

### Blocking quality

Report at least:

- candidate recall
- average candidates per Source 1 entity
- median candidates per Source 1 entity
- candidate reduction ratio, if calculated

Explain the validation procedure used to measure candidate recall.

## 4. Pairwise feature engineering

Document the features used by the matcher.

### Name features

Examples:

- edit/rational similarity
- token-set similarity
- token-sort similarity
- Jaccard overlap
- exact normalized match
- length ratio

### Address features

Examples:

- edit/rational similarity
- token-set similarity
- token-sort similarity
- Jaccard overlap
- exact normalized match
- numeric overlap
- postal-code overlap
- length ratio

### Cross-field features

Examples:

- exact country equality
- source indicator
- combined name/address similarity

## 5. Matching model

Describe:

- model architecture
- hyperparameters
- training labels
- positive/negative construction
- negative sampling strategy
- class imbalance handling, if applicable
- reproducibility seed

## 6. Validation and threshold selection

Explain:

- entity-level train/validation split
- macro F0.5 calculation
- singleton handling
- threshold search range
- selected threshold

Report validation metrics such as:

```text
Candidate recall:
Average candidates:
Precision:
Recall:
Macro F0.5:
```

## 7. Final inference

Describe how the trained model is applied to the complete test set.

Confirm that:

- every Source 1 test entity receives exactly one output row
- duplicate IDs are removed
- final matched IDs are a subset of the candidate IDs
- unmatched entities receive an empty ID list

## 8. Output generation

### `matching_results.tsv`

Document the exact schema and ordering used.

### `candidate_pairs.tsv`

Document the exact schema and explain that this file contains the final candidate set actually scored by the matching model.

## 9. Reproducibility

Document:

- Python version
- dependency versions
- command used for training
- command used for inference
- random seed
- relevant configuration values

## 10. Fair-play / data usage

State that the system uses only the challenge-provided datasets and does not perform external business-identity lookup, geocoding, web search, API calls, or external data augmentation.

## 11. Limitations and future work

Discuss remaining limitations such as:

- transliteration ambiguity
- sparse addresses
- businesses with very generic names
- large country-level blocks
- missed candidates caused by blocking

Also document any future improvements tested but not included in the final submission.

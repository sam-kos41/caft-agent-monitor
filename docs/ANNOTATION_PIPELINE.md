# Annotation Pipeline Architecture

This document describes the first-class annotation system in agentdiag, covering the data model, label lifecycle, storage, disagreement tracking, and integration with evaluation and calibration.

## Why Annotation is First-Class

Rule-based detectors plateau at ~67% precision without NLU (confirmed by V3 experiments). To improve, we need trusted labels for:

- **Calibration**: Tuning detector thresholds on verified data
- **Evaluation**: Computing precision/recall/F1 on gold-standard labels
- **Active learning**: Prioritizing which sessions to annotate next
- **Disagreement analysis**: Understanding where detectors vs. humans diverge

The annotation pipeline makes labels a structured, versioned, non-destructive artifact that flows through the entire system.

## The 4 Layers of Truth

Every session can have up to 4 annotation layers, each preserved independently:

| Layer | Source | Trust Rank | Status |
|-------|--------|------------|--------|
| **Detector** | Heuristic CAFT detectors | 1 (lowest) | `unlabeled` |
| **Auto** | LLM annotation (e.g., Sonnet 4.5) | 2 | `auto_labeled` |
| **Human** | Human annotator review | 3 | `human_reviewed` |
| **Adjudicated** | Final resolved gold label | 4 (highest) | `adjudicated` |

**Key principle**: Raw detector output is never overwritten by annotation. Each layer is append-only. Multiple humans can annotate the same session.

## Label Lifecycle

```
unlabeled --> auto_labeled --> human_reviewed --> adjudicated
                                                      |
                                                      v
                                                held_out_test
```

- **unlabeled**: Detector prediction exists, no review yet
- **auto_labeled**: LLM has annotated (can be replaced by newer prompt versions)
- **human_reviewed**: A human annotator has reviewed the trace
- **adjudicated**: Final gold label after disagreement resolution
- **held_out_test**: Reserved for evaluation, excluded from threshold tuning

## Data Model

### AnnotationRecord (`annotation_models.py`)

The universal annotation artifact with 25+ fields:

```python
AnnotationRecord(
    # Identity
    annotation_id="a1b2c3d4e5f6",  # auto-generated UUID
    session_id="abc12345",
    trace_id="abc12345",

    # Source
    annotator_type="human",        # detector | auto | human | adjudicated
    annotator_id="reviewer_a",     # model name, username, or detector name

    # Versioning
    caft_version="1.0",
    codebook_version="1.0",
    annotation_prompt_version="1.0",

    # Classification
    label_status="human_reviewed",
    has_failure=True,
    primary_caft_code="2.2",       # CAFT taxonomy code
    secondary_caft_codes=["2.1"],
    onset_step=15,
    evidence_steps=[15, 16, 17],
    severity=4,                    # 1-5
    confidence=3,                  # 1-5
    free_text_rationale="...",
)
```

**Deduplication key**: `(session_id, annotator_type, annotator_id, caft_version)`. This means the same annotator reviewing the same session under the same CAFT version produces one record (newer replaces older).

### Builder Functions

Four builder functions enforce correct defaults per layer:

- `build_detector_annotation(session_id, diagnosis)` - From CaftDiagnosis
- `build_auto_annotation(session_id, ...)` - From LLM output
- `build_human_annotation(session_id, annotator_id, ...)` - From human review
- `build_adjudicated_annotation(session_id, adjudicator_id, ...)` - Gold label

### Legacy Conversion

`from_ground_truth_file(gt_dict)` converts existing `ground_truth_*.json` files into AnnotationRecords, preserving backward compatibility.

## Storage

### AnnotationLedger (`annotation_store.py`)

JSONL-backed persistent store with:

- **Append with dedup**: Same key replaces older record
- **Lifecycle filters**:
  - `get_gold_annotations()` - Adjudicated only (final metrics)
  - `get_trainable_annotations()` - Adjudicated + human_reviewed (threshold tuning)
  - `get_eval_annotations()` - Adjudicated + held_out_test
- **Trust hierarchy**: `get_best_label(session_id)` returns highest-trust record
- **Merge**: `merge_from(other_ledger)` combines ledgers with dedup

### OpenViking Integration (`context/openviking.py`)

Annotations are stored in two places:

1. **Local JSONL ledger** (always) - `{db_path}/annotation_ledger.jsonl`
2. **OpenViking session** (best-effort) - Structured tool message

Methods: `record_annotation()`, `get_annotations_for_session()`, `find_annotation_needed_cases()`, `record_adjudicated_label()`

All OpenViking calls are wrapped in try/except for graceful degradation.

## Disagreement Tracking (`disagreement.py`)

### Pairwise Comparison

`compare_annotations(a, b)` produces a `DisagreementSummary`:
- Binary agreement (has_failure match)
- Code agreement (primary_caft_code match)
- Severity delta, onset delta, confidence delta

### Session Bundle

`compute_session_disagreement_bundle(session_id, records)` computes all pairwise comparisons:
- detector vs auto
- auto vs human
- human vs adjudicated
- detector vs human

### Annotation Priority Queue

`rank_annotation_queue(records_by_session)` scores sessions by urgency:

| Factor | Scale | Signal |
|--------|-------|--------|
| Severity | 0-10 | High-severity findings need review first |
| Uncertainty | 0-8 | Low confidence needs human judgment |
| Disagreement | 0-12 | Conflicting layers = hard case |
| Novelty | 0-5 | Rare failure types need examples |
| Unlabeled | 0-10 | Completely unlabeled = highest priority |

Already-adjudicated sessions get score 0 (excluded from queue).

## Integration with Evaluation

### `load_annotation_ledger_for_eval()` (`evaluate.py`)

Bridges the annotation ledger with the existing evaluation pipeline:

```python
from agentdiag.evaluate import load_annotation_ledger_for_eval

# Only adjudicated labels for final metrics
annotations = load_annotation_ledger_for_eval("ledger.jsonl", "gold")

# Adjudicated + human for threshold tuning
annotations = load_annotation_ledger_for_eval("ledger.jsonl", "trainable")
```

Returns `{session_id: annotation_dict}` compatible with `_compute_annotation_metrics`.

### `fit_from_annotated()` (`baselines.py`)

Fits calibration baselines only on annotation-verified sessions:

```python
pipeline = CalibrationPipeline()
profile = pipeline.fit_from_annotated(
    traces_path="~/.claude/projects",
    annotation_ledger_path="ledger.jsonl",
    label_filter="trainable",
)
```

## CLI Commands

```bash
# View annotation queue (highest priority first)
python -m agentdiag annotate queue --ledger ledger.jsonl --limit 10

# Show all layers for a session
python -m agentdiag annotate show <session_id> --ledger ledger.jsonl

# Record adjudicated gold label
python -m agentdiag annotate adjudicate <session_id> \
    --failure --code 2.2 --severity 4 --rationale "Confirmed" \
    --ledger ledger.jsonl

# Export gold annotations
python -m agentdiag annotate export-gold --status adjudicated --ledger ledger.jsonl

# Show ledger statistics
python -m agentdiag annotate stats --ledger ledger.jsonl

# Import legacy ground_truth_*.json
python -m agentdiag annotate import-gt ground_truth_50.json --ledger ledger.jsonl
```

## Version Pinning

Three version constants in `annotation_models.py`:

| Constant | Current | Tracks |
|----------|---------|--------|
| `CAFT_VERSION` | "1.0" | CAFT taxonomy (32 types, 8 categories) |
| `CODEBOOK_VERSION` | "1.0" | Annotation criteria/rules |
| `ANNOTATION_PROMPT_VERSION` | "1.0" | LLM annotation prompt template |

Bump these when definitions change. Records carry versions so old and new annotations can coexist.

## File Map

| File | Purpose |
|------|---------|
| `annotation_models.py` | AnnotationRecord, enums, builders, legacy conversion |
| `annotation_store.py` | AnnotationLedger (JSONL persistence, filters, merge) |
| `disagreement.py` | Pairwise comparison, session bundles, priority queue |
| `context/openviking.py` | OpenViking integration (record/retrieve annotations) |
| `evaluate.py` | `load_annotation_ledger_for_eval()` bridge |
| `baselines.py` | `fit_from_annotated()` calibration filtering |
| `__main__.py` | CLI `annotate` subcommand (6 sub-commands) |
| `tests/test_annotation_pipeline.py` | 104 tests covering all components |

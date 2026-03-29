# CAFT Evaluation Report

## Overall Results

| Metric | Value |
|--------|-------|
| True Positive Rate | 0.942 |
| False Positive Rate | 0.0868 |
| Detection Latency (median) | 26.0 steps |
| Detection Latency (mean) | 28.3 +/- 13.7 steps |
| Failure Traces | 52 |
| Detected | 49 |
| Clean Traces | 13 |
| Clean False Alarms | 345 |

## Detection by Failure Type

| Failure Type | TPR | Sig. Accuracy | Latency (med) | Expected Signature |
|-------------|-----|---------------|---------------|-------------------|
| loop | 1.000 | 0.923 | 38.0 | mechanical_repetition |
| drift | 1.000 | 1.000 | 14.0 | distributional_shift |
| thrash | 1.000 | 1.000 | 19.0 | context_thrashing |
| stall | 0.769 | 1.000 | 46.0 | distributional_anomaly |

## Cross-Task Generalization

| Domain | TPR | Detected/Total |
|--------|-----|----------------|
| testing | 1.000 | 8/8 |
| devops | 1.000 | 8/8 |
| web_app | 0.875 | 14/16 |
| cli_tool | 1.000 | 4/4 |
| data_pipeline | 1.000 | 8/8 |
| game | 0.750 | 3/4 |
| docs | 1.000 | 4/4 |

## Task x Failure Detection Matrix

| Task | loop | drift | thrash | stall |
|------|--------|--------|--------|--------|
| api_test_harness | Y | Y | Y | Y |
| bash_utility | Y | Y | Y | Y |
| chat_app | Y | Y | Y | Y |
| ci_cd_setup | Y | Y | Y | Y |
| cli_data_processor | Y | Y | Y | Y |
| etl_pipeline | Y | Y | Y | Y |
| fullstack_auth | Y | Y | Y | Y |
| game_physics | Y | Y | Y | N |
| markdown_docs | Y | Y | Y | Y |
| ml_pipeline | Y | Y | Y | Y |
| react_dashboard | Y | Y | Y | N |
| rest_api | Y | Y | Y | N |
| unit_test_suite | Y | Y | Y | Y |

## Clean Trace False Alarms

| Task | Anomalies |
|------|-----------|
| api_test_harness | 0 |
| bash_utility | 0 |
| chat_app | 56 |
| ci_cd_setup | 73 |
| cli_data_processor | 0 |
| etl_pipeline | 0 |
| fullstack_auth | 133 |
| game_physics | 0 |
| markdown_docs | 0 |
| ml_pipeline | 83 |
| react_dashboard | 0 |
| rest_api | 0 |
| unit_test_suite | 0 |

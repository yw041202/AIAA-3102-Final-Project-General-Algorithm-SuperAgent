# Integration verification — 2026-11-22

## Results

- Current complete A/B/C/D suite: **142 passed, 0 failed, 0 skipped**, 295.36 seconds. Includes school API configuration, key isolation and final-evaluation checks.
- Windows executor-only verification outside the restricted sandbox: **8 passed**, 4.06 seconds; included again in the complete suite.
- `git diff --check`: passed.
- Released tasks: **12 successful live planning calls**, course-accounted cost **RMB 0.022983**. Connectivity adds one successful call (RMB 0.000059) and one uncertain network attempt with RMB 0.000888 reserved. Platform account deductions were not inspected.

Environment: Windows, Python 3.11.5, NumPy 1.26.4, scikit-learn 1.4.2, joblib 1.4.2, pandas 3.0.6, SciPy 1.17.1, Pillow 12.3.0.

Current evidence: [complete suite XML](evidence/released-20261122/test-results.xml), [independent verification](evidence/released-20261122/verification.json), [live results and review checklist](LIVE_RESULTS_ZH.md). Earlier XML files are retained as historical checks.

## Behaviors exercised

- Real B data loading/preprocessing, real C subprocess execution, real D training and prediction under A orchestration.
- Tabular accuracy maximization and text log-loss minimization across two candidate configurations.
- Automatic selection and export of model plus preprocessing; 12 independent predictions for each synthetic fixture.
- Invalid negative logistic-regression C produces a real D training failure, which enters the next planning context; a repaired candidate completes prediction.
- Test labels deliberately contain invalid sentinels. They never enter planning or prediction; train/validation alone determine selection.
- Malformed decisions, patience stop, no-candidate stop, duplicate run-directory rejection and immutable data/metric fields.
- Budget rejection and uncertain/error responses stop further execution; real C client with a mocked network backend persists accounting across planner instances.
- AIAA3102_API_KEY overrides the legacy key setting and is stripped from training environments.
- Windows timeout terminates the test parent and its spawned child process.
- Existing B/C/D tests, including environment-enabled B/D/C integrations, sparse text, NPY input, artifact tamper checks and reloaded predictions.

## Issues found and resolved during verification

1. Windows sandbox temporary paths exceeded legacy path limits. A candidate artifacts now avoid duplicated task/run segments, and test fixtures use short names with an explicit local pytest base directory.
2. C used POSIX process-group termination on Windows. Added Windows taskkill tree termination plus parent fallback; the PID-check test now uses the Windows process API.
3. The restricted execution sandbox cannot run taskkill successfully (authentication error). Full verification was rerun with approved execution outside that sandbox and passed. This is an execution-environment limitation, not a reason to suppress the child cleanup test.
4. D's legacy bundled artifact records Python 3.12.4. On this Python 3.11.5 runtime, its test now verifies the intended environment-mismatch rejection instead of incorrectly expecting loading to succeed. No artifact integrity/version check was weakened.

## Reproduction

From the repository root after installing requirements:

```powershell
$env:PARTB_ROOT = (Resolve-Path partb).Path
$env:PARTC_ROOT = (Resolve-Path partc).Path
.\.venv\Scripts\python.exe -m pytest parta/tests partb/tests partc/tests partd/tests --import-mode=importlib --basetemp=runs/tests -q -o addopts=
```

`runs/tests` is disposable and will be cleared by pytest. An environment permitting cleanup of the tests' own process trees is required on Windows.

## Live acceptance and remaining team review

The user authorized RMB 50 total for validation and specified the Windows user variable AIAA3102_API_KEY. The key's existence was verified without displaying it. Official per-task accounting remains capped at RMB 20.

Both instructor tasks completed autonomous runs using the same source fingerprint and school DeepSeek-V4.1-Flash endpoint. Selection used validation accuracy only; designated test accuracy was evaluated after selection. Independent new-process predictions are byte-identical to both original outputs. Archived model bundles were checked byte-for-byte against these models. Team cross-review, historical pricing evidence, final version freeze, report and presentation remain outstanding. Image training is still unsupported by D. Results retain `official_result: false` until the team completes final acceptance; do not describe this flag as an execution failure.

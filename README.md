# AIAA3102-Final Project-General Algorithm SuperAgent
Autonomous LLM-powered SuperAgent for end-to-end algorithm development on unseen classification tasks — data analysis, solution design, experimentation, and iterative refinement with logged, reproducible runs.

## Integrated entry point (Part A)

See [Part A setup, execution and architecture](parta/README.md),
[integration review and plan](parta/REVIEW_AND_PLAN_ZH.md), and
[verification results](parta/VERIFICATION.md).

Released-task live runs completed on 2026-11-22: coupon test accuracy **75.13%**,
headlines **65.84%**, with **142 passing tests**. See [results, saved model bundles,
execution evidence and cross-review instructions](parta/LIVE_RESULTS_ZH.md).

```powershell
python -m venv .venv  # CPython 3.11.5 for saved-model reproduction
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe run_superagent.py demo --output runs/demo-tabular
```

The demo uses synthetic data and an offline planner, with real B/C/D training and
prediction. Live runs require the instructor dataset/configuration and a compliant
model API configured locally. Image training is not yet supported by Part D.

Exact environment setup and test commands: [ENVIRONMENT.md](ENVIRONMENT.md).

## Group Members:

* Anqi ZHAO
* Wenjun LU
* Yilin WU
* Yingwen PENG

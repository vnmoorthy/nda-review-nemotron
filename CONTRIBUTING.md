# Contributing

Issues and pull requests are welcome.

**Good first contributions:** a new checklist domain (vendor security review, RFP compliance), a new baseline model, or a better span splitter for the "your NDA" box in the demo.

1. Fork and branch from `main`.
2. `pip install pytest && pytest -q tests`. These tests need no AWS account.
3. If you change anything under `lab/baselines/` or `lab/contractnli*.py`, rebuild the demo data with `python tools/build_demo_data.py` (see the README for the dataset download). The build fails if a stored score does not reproduce.
4. If you touch the demo's prompt builder, run `node tests/test_prompt_parity.mjs`. The browser prompt must stay byte-identical to the training prompt.

Please keep results honest: every number in the README and the demo must reproduce from files in this repo.

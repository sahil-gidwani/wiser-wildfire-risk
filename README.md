# wiser-wildfire-risk
Research on uncertainty-aware wildfire spread prediction, conformal risk control, and exposure-aware evaluation for reliable spatial risk assessment.

## Status

Planning is done. The data step (`python -m wiser.data`) and the calibration code are written and tested. No experiments have been run yet.

**WISER** picks the cutoff for a next-day wildfire map so that the expected share of missed fire pixels stays below a target. It holds misses in populated pixels to a stricter target. We test it on the NDWS dataset and measure how much of the map it has to flag.

Deliverables: the code, a demo and a final report.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Where things are

- `docs/` holds the plan, method and data notes. Start with [docs/README.md](docs/README.md).
- `wiser/` holds the code, one module per job. [docs/engineering.md](docs/engineering.md) explains the layout and how to run it.

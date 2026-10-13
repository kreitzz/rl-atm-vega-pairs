# Options relative value: report and code submission

Start with [the written report](../options_relative_value.pdf). The accompanying
[Python implementation](paper_code/README.md) contains the retained strategy,
monthly tree/CEM sizing, funded daily backtester, risk checks and metrics.

This folder is self-contained for the main four-policy experiment: no parent
repository, WRDS login, or data downloads are needed. Selected-contract quotes,
stocks, dividends, rates, training inputs, separate completed-outcome labels,
saved monthly models and reference ledgers are included.

## Reproduce

From the `code` folder, using Python 3.11:

On Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m paper_code test
.\.venv\Scripts\python.exe -m paper_code verify
.\.venv\Scripts\python.exe -m paper_code metrics
.\.venv\Scripts\python.exe -m paper_code fit --month 2018-01-01
.\.venv\Scripts\python.exe -m paper_code replay --policy learned
```

On macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r paper_code/requirements.txt
.venv/bin/python -m paper_code test
.venv/bin/python -m paper_code verify
.venv/bin/python -m paper_code metrics
```

`verify` reproduces all 608 saved sizing choices, the four policies' reported
validation/evaluation metrics, and the paired bootstrap interval. `metrics`
exports period and annual results to `artifacts/paper_code/`.

For a fresh initial model fit or account replay:

```sh
.venv/bin/python -m paper_code fit --month 2018-01-01
.venv/bin/python -m paper_code replay --policy learned
```

Replay controls are `fixed50`, `fixed75`, and `fixed100`. Fit and replay run one
worker at 50% of one CPU core with a 384 MiB RSS stop and macOS memory checks.
No full research search runs by default.

Python and the packages in `requirements.txt` must be installed separately;
they are not vendored in this folder. After installation, the main experiment
runs offline. On Windows, `psutil` provides process supervision; macOS/Linux
use the operating system's process controls.

## Contents

- `../options_relative_value.pdf`: the current ten-page written report.
- `report_source.tex`, `figures/`: editable report source and its chart.
- `paper_code/`: focused implementation, configuration, documentation and tests.
- `data/`: required main-experiment data and saved models, with data-use notice.
- `reports/`: main result tables and archived supporting-comparison evidence.
- `manifest.json`: SHA-256 hashes for the submission files.

The main implementation was verified against all four original funded ledgers
with zero daily NAV differences. Secondary research pipelines remain linked
from the code guide; their result tables are included as supporting evidence.
The retrospective universe and historical-selection qualifications described
in the report remain unchanged. See `data/DATA_LICENSE.txt` for data-use terms.

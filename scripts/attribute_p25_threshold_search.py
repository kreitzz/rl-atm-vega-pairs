"""Run the existing audited pair attribution on direct-threshold deployments.

Run serially after the funded replay worker stops. The unchanged shared helper
enforces its 384 MiB memory guard and reports unavailable attribution explicitly.
"""
from pathlib import Path

from scripts import attribute_p25_opportunity_filter as shared


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/contracts/p25_threshold_search"
REPORT = ROOT / "reports/p25_threshold_search/funded"
CASES = ["q20", "selected_fixed", "selected_rolling", "selected_context"]


def main():
    shared.OUT = OUT
    shared.REPORT = REPORT
    shared.CASES = CASES
    shared.main()


if __name__ == "__main__":
    main()

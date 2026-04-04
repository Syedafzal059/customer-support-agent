"""Compare eval JSONL report to regression_baseline.json (routing only)."""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def load_jsonl_by_case_id(path:Path) -> dict[str, dict]:
    out: dict[str, dict] ={}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = row.get("case_id")
        if cid:
            out[str(cid)] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Regression check: route_ok vs baseline")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    cases_b = baseline.get("cases") or {}
    report = load_jsonl_by_case_id(args.report)
    failed = False
    for case_id, expected in cases_b.items():
        exp_route = expected.get("route_ok")
        if exp_route is None:
            continue
        row = report.get(case_id)
        if row is None:
            print(f"MISSING in report: {case_id}", file=sys.stderr)
            failed = True
            continue
        actual = row.get("route_ok")
        if actual != exp_route:
            print(
                f"REGRESSION {case_id}: route_ok expected {exp_route!r} got {actual!r}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0

    
if __name__ == "__main__":
    raise SystemExit(main())

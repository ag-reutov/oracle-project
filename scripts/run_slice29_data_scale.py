#!/usr/bin/env python
"""Run Slice 29 data-scale learning-curve audit.

Usage::

    uv run python scripts/run_slice29_data_scale.py \\
        --output data/interim/slice29_data_scale.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dota_predictor.features.duckdb_layer import connect
from dota_predictor.training.data_scale_diagnostics import (
    run_slice29_data_scale_benchmark,
    slice29_report_to_jsonable,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END
from dota_predictor.training.walk_forward import WalkForwardConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice29_data_scale.json"),
    )
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--skip-track-a", action="store_true")
    parser.add_argument("--skip-track-b", action="store_true")
    args = parser.parse_args()

    wf = WalkForwardConfig(n_blocks=args.n_blocks)

    with connect() as store:
        report = run_slice29_data_scale_benchmark(
            store,
            development_end=FROZEN_DEVELOPMENT_END,
            walk_forward_config=wf,
            run_track_a=not args.skip_track_a,
            run_track_b=not args.skip_track_b,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(slice29_report_to_jsonable(report), f, indent=2, default=str)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

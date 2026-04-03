#!/usr/bin/env python3
"""Run all thesis modules in sequence.

Module 1: Preprocessing (already executed, artifacts in data/processed/)
Module 2: Train final models with best hyperparameters
Module 3: Composite risk scores
Module 4: Stakeholder-tailored explanations (SHAP + DAE + NLG)
Module 5: Closed-loop response recommendations
Module 6: Human evaluation artifacts

Usage:
    python run_all_modules.py              # run all
    python run_all_modules.py --from 3     # resume from Module 3
    python run_all_modules.py --only 4     # run only Module 4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

MODULES = [
    {
        "id": 2,
        "name": "Train Final Models",
        "script": "pipeline/module2_detection/module2_train_models.py",
        "description": "Retrain XGBoost/RF/DT/DAE with best hyperparameters",
    },
    {
        "id": 3,
        "name": "Composite Risk Scores",
        "script": "pipeline/module3_risk_scoring/module3_risk_scores.py",
        "description": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*A_patient",
    },
    {
        "id": 4,
        "name": "Explanations (Batch)",
        "script": "pipeline/module4_explanations/module4_explanations.py",
        "description": "TreeSHAP + DAE decomposition + stakeholder outputs + validation",
    },
    {
        "id": 5,
        "name": "Response Recommendations",
        "script": "pipeline/module5_responses/module5_responses.py",
        "description": "Adaptive mitigation + audit trail + effectiveness analysis",
    },
    {
        "id": "5b",
        "name": "Response Pipeline Integration",
        "script": "pipeline/module5_responses/module5_pipeline.py",
        "description": "PolicyEngine + clinical safety + feedback loop",
    },
    {
        "id": 6,
        "name": "Evaluation Artifacts",
        "script": "pipeline/module6_evaluation/module6_evaluation.py",
        "description": "Curate alerts, simulated responses, statistical analysis, thesis figures",
    },
]


def run_module(module: dict) -> bool:
    """Run a single module script."""
    sep = "=" * 72
    logger.info(sep)
    logger.info("MODULE %s — %s", module["id"], module["name"])
    logger.info("  %s", module["description"])
    logger.info(sep)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, module["script"]],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = round(time.perf_counter() - t0, 1)

    if result.returncode == 0:
        logger.info("  Module %s PASSED (%.1fs)\n", module["id"], elapsed)
        return True
    else:
        logger.error("  Module %s FAILED (exit %d, %.1fs)\n",
                     module["id"], result.returncode, elapsed)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all thesis modules")
    parser.add_argument("--from", type=int, dest="from_module", default=2,
                        help="Start from this module number (default: 2)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this module number")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    sep = "=" * 72

    logger.info(sep)
    logger.info("IoMT IDS — THESIS MODULE PIPELINE")
    logger.info(sep)

    results = {}
    for module in MODULES:
        mod_id = module["id"]

        if args.only is not None:
            if str(mod_id) != str(args.only):
                continue
        elif isinstance(mod_id, int) and mod_id < args.from_module:
            logger.info("Skipping Module %s (--from %d)", mod_id, args.from_module)
            continue

        success = run_module(module)
        results[str(mod_id)] = success
        if not success:
            logger.error("Pipeline halted at Module %s", mod_id)
            break

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("PIPELINE COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    for mod_id, success in results.items():
        logger.info("  Module %-3s: %s", mod_id, "PASS" if success else "FAIL")
    logger.info(sep)


if __name__ == "__main__":
    main()

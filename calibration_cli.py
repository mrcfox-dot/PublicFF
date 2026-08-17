#!/usr/bin/env python3
"""Stage 4.5 single-command launcher: does Stage 4 actually predict real
manager behaviour better than simple baselines?

    python3 calibration_cli.py                          # synthetic dataset (no real data supplied)
    python3 calibration_cli.py --dataset /path/to/data.csv
    python3 calibration_cli.py --dataset /path/to/dir --long-format
    python3 calibration_cli.py --fit --report-dir ./calibration_reports

No dataset is ever downloaded or scraped - ``--dataset`` must be a local
file or directory already on disk. Without it, a synthetic dataset is
generated from Stage 4's own behavioural personas and loaded through the
same CSV importer a real dataset would use (see
fpl_rival/fixtures/calibration_fixtures.py) - every section of the report
says so loudly when that's what happened.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from fpl_rival.calibration import ablation, baselines, harness, history_depth, predictability, splits
from fpl_rival.calibration.calibration_report import expected_calibration_error, fit_temperature
from fpl_rival.calibration.fitting import coordinate_search
from fpl_rival.calibration.importer import load_dataset, records_by_manager_and_season, summarize_dataset
from fpl_rival.calibration.models import TEST, VALIDATION
from fpl_rival.calibration.report_text import (
    render_ablation,
    render_calibration,
    render_dataset_summary,
    render_history_depth,
    render_performance_table,
    render_predictability,
    render_recommendation,
    render_split,
    render_stage3_impact,
)
from fpl_rival.fixtures import calibration_fixtures as cfx
from fpl_rival.fixtures import prediction_fixtures as pfx
from fpl_rival.prediction.calibration import compute_calibration_buckets
from fpl_rival.prediction.evaluation import evaluate
from fpl_rival.prediction.models import PredictionConfig
from fpl_rival.calibration.stage3_impact import Stage3Scenario, compare_stage3_impact
from fpl_rival.simulation.models import SimulationConfig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Rival Stage 4.5 - historical calibration of the rival prediction model.")
    parser.add_argument("--dataset", type=str, default=None, help="Local file or directory. Omit to use a synthetic dataset.")
    parser.add_argument("--long-format", action="store_true", help="Dataset CSV is one row per (manager, gameweek, player) - use the example adaptor.")
    parser.add_argument("--train-through", type=str, default=None, help="'season:gameweek', e.g. '2026-27:8'.")
    parser.add_argument("--validation-through", type=str, default=None, help="'season:gameweek'.")
    parser.add_argument("--test-through", type=str, default=None, help="'season:gameweek'.")
    parser.add_argument("--max-managers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the Stage 3 impact demo's simulations.")
    parser.add_argument("--fit", action="store_true", help="Run coordinate-search parameter fitting on the validation split.")
    parser.add_argument("--report-dir", type=str, default=None, help="If set, also writes a JSON summary here.")
    parser.add_argument("--num-gameweeks", type=int, default=30, help="Synthetic dataset only: gameweeks per manager.")
    parser.add_argument("--managers-per-persona", type=int, default=2, help="Synthetic dataset only.")
    return parser.parse_args(argv)


def _load_dataset(args: argparse.Namespace):
    if args.dataset:
        records = load_dataset(Path(args.dataset), long_format=args.long_format)
        return records, None, False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic_dataset.csv"
        expected_points = cfx.write_synthetic_dataset_csv(path, num_gameweeks=args.num_gameweeks, managers_per_persona=args.managers_per_persona)
        records = load_dataset(path)  # exercises the real importer, even for synthetic data
    return records, expected_points, True


def _build_split(records, args: argparse.Namespace):
    if args.train_through and args.validation_through and args.test_through:
        season_order = tuple(sorted({r.season for r in records}))
        return splits.explicit_split(args.train_through, args.validation_through, args.test_through, list(season_order))
    return splits.infer_default_split(records)


def main(argv=None) -> int:
    args = parse_args(argv)
    records, expected_points, is_synthetic = _load_dataset(args)
    summary = summarize_dataset(records, is_synthetic=is_synthetic)
    split = _build_split(records, args)

    print(render_dataset_summary(summary))
    print()
    print(render_split(split))
    print()

    test_total = len([r for r in records if split.phase_of(r.season, r.gameweek) == TEST])

    # -- baselines -----------------------------------------------------------------
    baseline_rows = []
    for name in baselines.BASELINE_REGISTRY:
        results = harness.run_baseline_over_dataset(records, name, split=split, expected_points=expected_points, max_managers=args.max_managers)
        test_results = harness.filter_phase(results, TEST)
        metrics = evaluate(harness.to_eval_pairs(test_results))
        label, _, _, _ = baselines.BASELINE_REGISTRY[name]
        baseline_rows.append((label, metrics))
    print(render_performance_table("BASELINE PERFORMANCE (test split)", baseline_rows, total=test_total))
    print()

    # -- manual Stage 4 --------------------------------------------------------------
    manual_config = PredictionConfig()
    manual_results = harness.run_stage4_over_dataset(records, manual_config, split=split, expected_points=expected_points, max_managers=args.max_managers)
    manual_test_metrics = evaluate(harness.to_eval_pairs(harness.filter_phase(manual_results, TEST)))
    print(render_performance_table("MANUAL STAGE 4 PERFORMANCE (test split)", [("Manual Stage 4", manual_test_metrics)], total=test_total))
    print()

    final_config = manual_config
    final_results = manual_results
    fit_result = None
    if args.fit:
        fit_result = coordinate_search(records, split, base_config=manual_config, expected_points=expected_points, max_managers=args.max_managers)
        fitted_results = harness.run_stage4_over_dataset(records, fit_result.fitted_config, split=split, expected_points=expected_points, max_managers=args.max_managers)
        fitted_test_metrics = evaluate(harness.to_eval_pairs(harness.filter_phase(fitted_results, TEST)))
        print(render_performance_table("FITTED MODEL PERFORMANCE (test split)", [("Fitted Stage 4", fitted_test_metrics)], total=test_total))
        print(f"\nFitted parameters: {fit_result.fitted_parameters}")
        print(f"Validation log loss: base={fit_result.base_validation_log_loss:.4f} -> fitted={fit_result.final_validation_log_loss:.4f}")
        print()
        final_config = fit_result.fitted_config
        final_results = fitted_results

    # -- calibration (final model, test split) --------------------------------------
    test_pairs = harness.to_eval_pairs(harness.filter_phase(final_results, TEST))
    buckets = compute_calibration_buckets(test_pairs, num_buckets=10)
    ece = expected_calibration_error(buckets)
    print(render_calibration(buckets, ece))
    print()

    temperature_result = None
    if args.fit:
        validation_with_predictions = harness.filter_phase(final_results, VALIDATION)
        if validation_with_predictions:
            temperature_result = fit_temperature(validation_with_predictions)
            print("OPTIONAL CALIBRATION CORRECTION (temperature scaling, fit on validation)")
            print("-" * 40)
            print(f"Best temperature: {temperature_result.temperature}  (T=1.0 = no correction)")
            print(f"Validation log loss: {temperature_result.baseline_validation_log_loss:.4f} -> {temperature_result.fitted_validation_log_loss:.4f}")
            print("Justified (applied)" if temperature_result.improved else "Not applied - no meaningful validation improvement over T=1.")
            print()

    # -- history depth ------------------------------------------------------------
    print(render_history_depth(history_depth.bucket_by_history_depth(final_results)))
    print()

    # -- ablation (fitted/final config, validation split) ---------------------------
    ablation_rows = ablation.run_ablation(records, split, final_config, expected_points=expected_points, max_managers=args.max_managers)
    print(render_ablation(ablation_rows))
    print()

    # -- manager predictability -----------------------------------------------------
    histories_by_manager = {
        manager_id: tuple(r.to_captain_observation() for r in recs)
        for (season, manager_id), recs in records_by_manager_and_season(records).items()
    }
    names_by_manager = {r.manager_id: r.manager_name for r in records if r.manager_name}
    predictability_rows = predictability.compute_all_manager_predictability(final_results, histories_by_manager, names_by_manager)
    print(render_predictability(predictability_rows))
    print()

    # -- Stage 3 impact demo ----------------------------------------------------------
    chris, s3_league, projections, candidates, dave, dave_history, dave_ep = pfx.full_pipeline_scenario(pfx.dave_salah_loyal_history())
    scenario_a = Stage3Scenario(
        label="dave_salah_loyal", chris=chris, league=s3_league, projections=projections,
        candidate_captain_ids=candidates, rival=dave, rival_history=dave_history,
        target_gameweek=11, expected_points=dave_ep,
    )
    sim_config = SimulationConfig(num_simulations=20_000, random_seed=args.seed, rival_captain_mode="probabilistic")
    impact_rows = compare_stage3_impact([scenario_a], manual_config, final_config, sim_config)
    print(render_stage3_impact(impact_rows))
    print()

    # -- recommendation -----------------------------------------------------------
    recommendation_lines = []
    if summary.is_synthetic:
        recommendation_lines.append(
            "No real historical manager dataset was supplied - this run is on SYNTHETIC data and validates the "
            "INFRASTRUCTURE only (Outcome C: insufficient real-world data to judge whether Stage 4 beats baselines "
            "on real behaviour). See the Stage 4.5 completion report for what real data would be needed."
        )
    best_baseline_label, best_baseline_metrics = min(
        ((label, m) for label, m in baseline_rows if m["num_predictions"] > 0), key=lambda lm: lm[1]["log_loss"]
    )
    recommendation_lines.append(
        f"Best baseline on test: {best_baseline_label} (log loss {best_baseline_metrics['log_loss']:.3f}) "
        f"vs Manual Stage 4 (log loss {manual_test_metrics['log_loss']:.3f})."
    )
    if manual_test_metrics["log_loss"] < best_baseline_metrics["log_loss"]:
        recommendation_lines.append("Manual Stage 4 beats the best baseline on this data.")
    else:
        recommendation_lines.append("Manual Stage 4 does NOT beat the best baseline on this data - do not hide this.")
    if fit_result:
        fitted_test_metrics = evaluate(harness.to_eval_pairs(harness.filter_phase(final_results, TEST)))
        if fitted_test_metrics["log_loss"] < manual_test_metrics["log_loss"] - 0.01:
            recommendation_lines.append(
                f"Fitted parameters improved held-out log loss ({manual_test_metrics['log_loss']:.3f} -> "
                f"{fitted_test_metrics['log_loss']:.3f}) - worth adopting if this generalises to real data."
            )
        else:
            recommendation_lines.append("Fitted parameters did not meaningfully improve held-out performance over the manual defaults.")
    print(render_recommendation(recommendation_lines))

    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset_summary": summary.__dict__,
            "split": split.describe(),
            "baselines": {label: m for label, m in baseline_rows},
            "manual_stage4_test": manual_test_metrics,
            "fitted_stage4_test": (evaluate(harness.to_eval_pairs(harness.filter_phase(final_results, TEST))) if fit_result else None),
            "ece": ece,
        }
        (report_dir / "calibration_summary.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved JSON summary to {report_dir / 'calibration_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

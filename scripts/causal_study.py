"""The causal study of packet P7, on SYNTHETIC lives from development seeds (1000-1999) only.

Two parts.

``mdiag``
    M-DIAG (0003 A3, C-DIAG): the per-alert accuracy of the structural diagnoser (``scm``) and the
    correlational one (``corr``) when the training correlation between heavy load and failure mode
    is reversed at test time, and in the unshifted setting. Training lives come from seeds
    ``1000 ..`` with ``mode_load_tilt = +T``; test lives from seeds ``1500 ..`` with ``-T``
    (reversed) or ``+T`` (unshifted). Nobody acts in these worlds. ``--tilts`` runs the same design
    at several strengths ``T`` (the first is the primary one; the others are an exploratory
    sensitivity sweep). No decision rule is applied: the thresholds belong to the pre-registered
    run.

``effect``
    Q1, Q2 and Q4 of the three effect models (correlational, SCM, misspecified SCM) on first
    lives of the F1+D plant under the confounded operator policy, scored against twin-world truth.
    C-CAUSE, decided by Q1, is **favoured by construction**: the SCM is handed the structure of
    the plant. M-DIAG's SCM is handed the graph but estimates every coefficient from training lives.

    python scripts/causal_study.py mdiag --train-seeds 400 --test-seeds 400 --tilts 8 [--out PATH]
    python scripts/causal_study.py effect --train-lives 200 --test-seeds 40 [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.causal import mdiag  # noqa: E402
from closed_loop.causal.diagnoser import make_diagnoser  # noqa: E402
from closed_loop.causal.queries import (  # noqa: E402
    build_cases,
    make_effect_model,
    paired_difference,
    per_life_scores,
    records_from_scores,
)
from closed_loop.causal.twin import causal_params, generate_observational  # noqa: E402

TRAIN_START, TEST_START = 1000, 1500


def run_mdiag(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {
        "train_seeds": f"{TRAIN_START}..{TRAIN_START + args.train_seeds - 1} (dev)",
        "test_seeds": f"{TEST_START}..{TEST_START + args.test_seeds - 1} (dev)",
        "alpha": args.alpha,
        "tilts": {},
    }
    for T in args.tilts:
        t0 = time.perf_counter()
        train = mdiag.collect_lives(
            range(TRAIN_START, TRAIN_START + args.train_seeds), mdiag.diagnosis_params(+T)
        )
        diagnosers = {k: make_diagnoser(k) for k in ("scm", "corr", "oracle")}
        for k in ("scm", "corr"):
            diagnosers[k].fit(train)
        result: dict[str, object] = {
            "train_lives": len(train),
            "train_alerts": sum(len(lf.alerts) for lf in train),
        }
        for setting, tilt in (("reversed", -T), ("unshifted", +T)):
            lives = mdiag.collect_lives(
                range(TEST_START, TEST_START + args.test_seeds), mdiag.diagnosis_params(tilt)
            )
            cases = mdiag.alert_cases(lives)
            outcomes = {k: mdiag.evaluate(d, cases) for k, d in diagnosers.items()}
            where = (
                f"{len(cases)} alerts on {len(lives)} first lives, dev seeds "
                f"{TEST_START}..{TEST_START + args.test_seeds - 1}, F1 plant, mode_load_tilt "
                f"train {T:+g} test {tilt:+g} ({setting}), nobody acting"
            )
            recs = {
                name: r.to_dict()
                for name, r in mdiag.records(
                    setting, cases, outcomes, alpha=args.alpha, measured_on=where
                ).items()
            }
            bonf = mdiag.records(setting, cases, outcomes, alpha=0.05 / 9, measured_on=where)[
                "scm-corr"
            ].to_dict()
            result[setting] = {
                "records": recs,
                "scm_minus_corr_at_alpha_0.05_over_9": bonf,
                "breakdown": mdiag.breakdown(cases, outcomes),
            }
        result["seconds"] = round(time.perf_counter() - t0, 1)
        out["tilts"][f"{T:g}"] = result  # type: ignore[index]
        _print_mdiag(T, result)
    return out


def _print_mdiag(T: float, result: dict[str, object]) -> None:
    print(
        f"\n=== tilt T = {T:g}: train lives {result['train_lives']}, alerts "
        f"{result['train_alerts']}"
    )
    for setting in ("reversed", "unshifted"):
        block = result[setting]
        recs, bd = block["records"], block["breakdown"]  # type: ignore[index]
        scm, corr, diff = recs["scm"], recs["corr"], recs["scm-corr"]
        print(
            f"  {setting:9s} n={scm['n']:5d}  scm {scm['value']:.4f} "
            f"[{scm['ci_lo']:.3f},{scm['ci_hi']:.3f}]"
            f"  corr {corr['value']:.4f} [{corr['ci_lo']:.3f},{corr['ci_hi']:.3f}]"
            f"  scm-corr {diff['value']:+.4f} CS95 [{diff['ci_lo']:+.4f},{diff['ci_hi']:+.4f}]"
            f"  (CS at 0.05/9: [{block['scm_minus_corr_at_alpha_0.05_over_9']['ci_lo']:+.4f},"  # type: ignore[index]
            f"{block['scm_minus_corr_at_alpha_0.05_over_9']['ci_hi']:+.4f}])"  # type: ignore[index]
        )
        print(
            f"            mode|degradation scm {bd['scm']['mode_on_degradation']:.4f} "  # type: ignore[index]
            f"corr {bd['corr']['mode_on_degradation']:.4f}; cause alone scm "  # type: ignore[index]
            f"{bd['scm']['cause_only']:.4f} corr {bd['corr']['cause_only']:.4f}; "  # type: ignore[index]
            f"truth causes {bd['n_by_cause']}; oracle {bd['oracle']['both']:.3f}"  # type: ignore[index]
        )


def run_effect(args: argparse.Namespace) -> dict[str, object]:
    params = causal_params()
    t0 = time.perf_counter()
    train = generate_observational(args.train_lives, TRAIN_START, "pi_obs", params=params)
    cases, skipped = build_cases(range(TEST_START, TEST_START + args.test_seeds), params)
    print(
        f"train lives {len(train)} (failed {sum(lf.failed for lf in train)}, derated "
        f"{sum(lf.derate_start is not None for lf in train)}); test cases {len(cases)}, "
        f"lives without an alert {skipped}; built in {time.perf_counter() - t0:.0f} s"
    )
    where = (
        f"{len(cases)} test lives from dev seeds {TEST_START}..{TEST_START + args.test_seeds - 1}, "
        f"F1+D plant, operator policy pi_obs, train {len(train)} lives from seeds {TRAIN_START}.., "
        "twin-world truth"
    )
    scores, fitted_info = {}, {}
    for kind in ("correlational", "scm", "scm_misspecified"):
        model = make_effect_model(kind)
        t = time.perf_counter()
        model.fit(train)
        scores[kind] = per_life_scores(model, cases)
        info = {"fit_and_score_s": round(time.perf_counter() - t, 1)}
        if hasattr(model, "beta"):
            info.update(
                beta=float(model.beta),
                beta_se=float(model.beta_se),
                derate_factor=float(model.derate_factor),
                h0=float(model.h0),
                c_h=float(model.c_h),
            )
        fitted_info[kind] = info
        true_eff = np.array([k.truth_derate - k.truth_nothing for k in cases])
        est = np.array(
            [
                model.counterfactual_failure_day(k.life, float(k.t0), 0.7)
                - model.counterfactual_failure_day(k.life, float(k.t0), 1.0)
                for k in cases
            ]
        )
        info["derate_effect_bias_days"] = float(np.mean(est - true_eff))
        info["true_derate_effect_mean_days"] = float(true_eff.mean())
    # The same correlational model on lives where the operator's choice was a coin flip: the bias
    # that remains is the model's own (noisy outcome, 150-day window), not the confounding.
    randomised = generate_observational(args.train_lives, TRAIN_START, "randomised", params=params)
    rmodel = make_effect_model("correlational")
    rmodel.fit(randomised)
    rest = np.array(
        [
            rmodel.counterfactual_failure_day(k.life, float(k.t0), 0.7)
            - rmodel.counterfactual_failure_day(k.life, float(k.t0), 1.0)
            for k in cases
        ]
    )
    true_eff = np.array([k.truth_derate - k.truth_nothing for k in cases])
    rscores = per_life_scores(rmodel, cases)
    fitted_info["correlational_trained_on_randomised_lives"] = {
        "q1_mean_abs_error_days": float(rscores.q1_err_days.mean()),
        "q1_scaled": float(rscores.q1_scaled.mean()),
        "derate_effect_bias_days": float(np.mean(rest - true_eff)),
        "derate_effect_mae_days": float(np.mean(np.abs(rest - true_eff))),
        "derated_share_of_training": float(
            np.mean([lf.derate_start is not None for lf in randomised])
        ),
        "true_derate_effect_mean_days": float(true_eff.mean()),
    }
    rbias = fitted_info["correlational_trained_on_randomised_lives"]["derate_effect_bias_days"]
    rq1 = fitted_info["correlational_trained_on_randomised_lives"]["q1_mean_abs_error_days"]
    print(
        f"  correlational on randomised lives: derate-effect bias {rbias:+.2f} d, "
        f"Q1 error {rq1:.2f} d"
    )
    result: dict[str, object] = {"fitted": fitted_info, "records": {}, "paired": {}}
    for kind, sc in scores.items():
        result["records"][kind] = {  # type: ignore[index]
            k: r.to_dict() for k, r in records_from_scores(sc, kind, where).items()
        }
    for metric in ("q1", "q2_brier", "q4"):
        for a, b in (
            ("scm", "correlational"),
            ("scm", "scm_misspecified"),
            ("scm_misspecified", "correlational"),
        ):
            r = paired_difference(
                scores[a],
                scores[b],
                metric,
                a_name=a,
                b_name=b,
                alpha=args.alpha,
                measured_on=where,
            )
            result["paired"][r.name] = r.to_dict()  # type: ignore[index]
    for kind in scores:
        recs = result["records"][kind]  # type: ignore[index]
        print(
            f"  {kind:17s} Q1 {recs['Q1_mcf']['value']:.4f} "
            f"({recs['Q1_mcf']['extra']['mean_abs_error_days']:.2f} d)"
            f"  Q2 brier {recs['Q2_brier']['value']:.4f} slope "
            f"{recs['Q2_calibration_slope']['value']:.2f}"
            f"  Q4 regret {recs['Q4_regret']['extra']['mean_regret_days']:.2f} d"
            f"  derate-effect bias {fitted_info[kind]['derate_effect_bias_days']:+.2f} d"
        )
    for name, r in result["paired"].items():  # type: ignore[attr-defined]
        print(f"  {name:36s} {r['value']:+.4f} CS [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="part", required=True)
    m = sub.add_parser("mdiag")
    m.add_argument("--train-seeds", type=int, default=300)
    m.add_argument("--test-seeds", type=int, default=300)
    m.add_argument("--tilts", type=float, nargs="+", default=[mdiag.DEFAULT_TILT])
    m.add_argument("--alpha", type=float, default=0.05)
    m.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("effect")
    e.add_argument("--train-lives", type=int, default=300)
    e.add_argument("--test-seeds", type=int, default=75)
    e.add_argument("--alpha", type=float, default=0.05)
    e.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = run_mdiag(args) if args.part == "mdiag" else run_effect(args)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, chi2


def read_predictions(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return {
        row["path"]: (int(row["true_index"]), int(row["pred_index"]))
        for row in rows
    }


def bootstrap_accuracy(correct, repetitions, seed, confidence):
    generator = np.random.default_rng(seed)
    values = np.asarray(correct, dtype=np.float64)
    size = len(values)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, size, size=size)
        estimates[index] = values[sample].mean()
    alpha = 1.0 - confidence
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lower, upper


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    baseline = read_predictions(arguments.baseline)
    proposed = read_predictions(arguments.proposed)
    common = sorted(set(baseline).intersection(proposed))
    if not common:
        raise ValueError("No common sample paths were found")
    baseline_correct = []
    proposed_correct = []
    for path in common:
        baseline_true, baseline_prediction = baseline[path]
        proposed_true, proposed_prediction = proposed[path]
        if baseline_true != proposed_true:
            raise ValueError(f"Different true labels for {path}")
        baseline_correct.append(baseline_true == baseline_prediction)
        proposed_correct.append(proposed_true == proposed_prediction)
    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    proposed_correct = np.asarray(proposed_correct, dtype=bool)
    baseline_only = int(np.logical_and(baseline_correct, np.logical_not(proposed_correct)).sum())
    proposed_only = int(np.logical_and(np.logical_not(baseline_correct), proposed_correct).sum())
    disagreements = baseline_only + proposed_only
    exact_p = (
        float(
            binomtest(
                min(baseline_only, proposed_only),
                disagreements,
                0.5,
                alternative="two-sided"
            ).pvalue
        )
        if disagreements > 0
        else 1.0
    )
    corrected_statistic = (
        (abs(baseline_only - proposed_only) - 1.0) ** 2 / disagreements
        if disagreements > 0
        else 0.0
    )
    asymptotic_p = float(chi2.sf(corrected_statistic, 1))
    lower, upper = bootstrap_accuracy(
        proposed_correct,
        arguments.bootstrap_repetitions,
        arguments.seed,
        arguments.confidence
    )
    output = {
        "samples": len(common),
        "baseline_accuracy": float(baseline_correct.mean()),
        "proposed_accuracy": float(proposed_correct.mean()),
        "baseline_correct_proposed_wrong": baseline_only,
        "baseline_wrong_proposed_correct": proposed_only,
        "mcnemar_continuity_corrected_statistic": float(corrected_statistic),
        "mcnemar_asymptotic_p_value": asymptotic_p,
        "mcnemar_exact_p_value": exact_p,
        "bootstrap_confidence": arguments.confidence,
        "proposed_accuracy_ci": [lower, upper],
        "bootstrap_repetitions": arguments.bootstrap_repetitions,
        "seed": arguments.seed
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

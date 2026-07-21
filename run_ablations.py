import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "baseline": [],
    "se_fine": ["fine"],
    "se_mid": ["mid"],
    "se_global": ["global"],
    "se_all": ["fine", "mid", "global"]
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with Path(arguments.base_config).open("r", encoding="utf-8") as file:
        base_configuration = json.load(file)
    result_rows = []
    for variant, levels in VARIANTS.items():
        variant_directory = output_root / variant
        configuration = dict(base_configuration)
        configuration["se_levels"] = levels
        configuration["output_dir"] = str(variant_directory)
        configuration_path = output_root / f"{variant}.json"
        with configuration_path.open("w", encoding="utf-8") as file:
            json.dump(configuration, file, indent=2)
        command = [arguments.python, "train.py", "--config", str(configuration_path)]
        print(" ".join(command))
        if arguments.dry_run:
            continue
        subprocess.run(command, check=True)
        metrics_path = variant_directory / "best_validation_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        result_rows.append(
            {
                "variant": variant,
                "se_fine": int("fine" in levels),
                "se_mid": int("mid" in levels),
                "se_global": int("global" in levels),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"]
            }
        )
    if result_rows:
        output_path = output_root / "ablation_results.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)
        print(output_path)


if __name__ == "__main__":
    main()

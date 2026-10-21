import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .audit import audit_csv


def main():
    parser = argparse.ArgumentParser(description="Offline defensive DNS research. Synthetic demo metrics are not research results.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect runtime and local project readiness")
    audit = sub.add_parser("audit", help="Stream a local CSV inventory without training")
    audit.add_argument("path", type=Path)
    audit.add_argument("--label", required=True)
    audit.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="Run one explicit experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, default=Path("results/runs"))
    predict = sub.add_parser("predict", help="Score a local unlabelled feature CSV using a saved run model")
    predict.add_argument("run_dir", type=Path)
    predict.add_argument("path", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--variant", default="A_all")
    predict.add_argument("--id", dest="id_column")
    demo = sub.add_parser("demo", help="Generate and evaluate synthetic toy data; no scientific efficacy claim")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--rows", type=int, default=800)
    demo.add_argument("--model", choices=["lr", "knn", "svm", "rf", "xgb", "mlp"], default="rf")
    demo.add_argument("--all-models", action="store_true")
    demo.add_argument("--all-k", action="store_true", help="Evaluate k=10,15,20 on the same synthetic splits")
    demo.add_argument("--output", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            versions = {}
            for package in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "pytest"]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "not installed"
            print(json.dumps({"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "machine": platform.machine(),
                              "packages": versions, "working_directory": str(Path.cwd()),
                              "paper_exists": Path("research/ICS_Research_Paper.pdf").is_file(),
                              "genuine_datasets": "not acquired automatically"}, indent=2))
        elif args.command == "audit":
            report = audit_csv(args.path, args.label)
            text = json.dumps(report, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    stream.write(text)
            print(text)
        elif args.command == "run":
            from .experiments import run_experiment
            config = json.loads(args.config.read_text())
            print(run_experiment(config, args.output))
        elif args.command == "predict":
            from .inference import predict_csv
            report = predict_csv(args.run_dir, args.path, args.output, args.variant, args.id_column)
            print(json.dumps(report, indent=2))
        elif args.command == "demo":
            from .demo import create_demo
            from .experiments import run_experiment
            config, folder = create_demo(Path("data/demo"), args.seed, args.rows)
            models = ["lr", "knn", "svm", "rf", "xgb", "mlp"] if args.all_models else [args.model]
            ks = [10, 15, 20] if args.all_k else [10]
            paths = []
            for model in models:
                for k in ks:
                    for mode in ["within", "zero_shot", "adaptation_assisted"]:
                        cfg = copy.deepcopy(config)
                        cfg["mode"], cfg["top_k"] = mode, k
                        if mode == "within":
                            cfg.pop("target")
                            cfg.pop("compatible_features")
                        cfg["model"] = {"name": model, "params": {}}
                        if model == "rf":
                            cfg["model"]["params"] = {"n_estimators": 60, "n_jobs": 1}
                        if model == "xgb":
                            cfg["model"]["params"] = {"n_estimators": 60, "n_jobs": 1}
                        if mode == "adaptation_assisted":
                            # Hold the SAME target test fraction and seed across modes.
                            # The runner reserves test first, then adaptation from the remainder.
                            cfg["target"]["split"] = {"train": 0.4, "validation": 0.2, "test": 0.2, "adaptation": 0.2}
                        path = run_experiment(cfg, args.output)
                        paths.append(str(path.resolve()))
                        print(f"SYNTHETIC {model} k={k} {mode}: {path}", flush=True)
            (folder / "run-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
            print("Synthetic integration demo complete. These metrics do not validate real DNS/DoH detection.")
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")

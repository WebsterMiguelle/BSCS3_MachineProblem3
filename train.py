"""Developer-only training script. The web app never trains; it only loads versions saved here.

Usage:
    python train.py --notes "why this version exists"              # ensemble on base data
    python train.py --model rf --notes "baseline"                   # class-weighted random forest
    python train.py --include-submissions --notes "added Q3 data"   # also use data/submissions/*.csv
    python train.py --no-activate                                   # save without making it live

Each run writes models/vN.joblib (model + metrics + scored training data) and updates
models/registry.json. Commit both files to git to publish the version on the deployed app.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import pandas as pd

from features import RAW_FEATURES, TARGET
from model_utils import LABEL_MAP, clean, deduplicate, evaluate, importances, train

BASE = Path(__file__).parent
MODELS = BASE / "models"
REGISTRY = MODELS / "registry.json"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"active": None, "versions": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ensemble", "rf"], default="ensemble")
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--min-leaf", type=int, default=1)
    ap.add_argument("--include-submissions", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-activate", action="store_true")
    args = ap.parse_args()

    frames, sources = [pd.read_csv(BASE / "data" / "students.csv")], ["students.csv"]
    if args.include_submissions:
        for f in sorted((BASE / "data" / "submissions").glob("*.csv")):
            frames.append(pd.read_csv(f))
            sources.append(f"submissions/{f.name}")
    raw = clean(pd.concat([clean(f) for f in frames], ignore_index=True))
    df, dedup_info = deduplicate(raw)

    params = {"kind": args.model, "n_estimators": args.trees, "min_samples_leaf": args.min_leaf}
    print(f"Training {args.model} on {len(df):,} unique students from {len(sources)} file(s)…")
    metrics = evaluate(df, params)
    model = train(df, params)

    scored = df.copy()
    scored["Outcome"] = scored[TARGET].map(LABEL_MAP)
    scored["Pass probability"] = metrics.pop("oof_proba")

    reg = load_registry()
    nums = [int(v["version"][1:]) for v in reg["versions"]]
    version = f"v{max(nums, default=0) + 1}"
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bundle = {
        "version": version, "created": created, "notes": args.notes, "params": params,
        "model_name": "Soft-voting ensemble (HistGradientBoosting + class-weighted RF + Extra Trees)"
                      if args.model == "ensemble" else "Class-weighted Random Forest",
        "features": RAW_FEATURES, "sources": sources, "data_info": dedup_info,
        # plain Python/numpy types only, so the bundle loads under any pandas version
        "metrics": {**metrics, "roc": metrics["roc"].to_dict("list")},
        "importances": importances(model).to_dict("list"),
        "scored": scored.to_dict("list"),
        "sklearn_version": sklearn.__version__,
        "model": model,
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS / f"{version}.joblib", compress=3)

    reg["versions"].append({"version": version, "created": created, "notes": args.notes,
                            "model_name": bundle["model_name"], "records": len(df),
                            **{k: round(float(metrics[k]), 4) for k in METRIC_KEYS}})
    if not args.no_activate or reg["active"] is None:
        reg["active"] = version
    REGISTRY.write_text(json.dumps(reg, indent=2))
    print(f"Saved models/{version}.joblib  active={reg['active']}")
    print({k: round(float(metrics[k]), 3) for k in METRIC_KEYS})


if __name__ == "__main__":
    main()

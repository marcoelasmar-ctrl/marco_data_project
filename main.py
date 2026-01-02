import os
import warnings
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_loader import load_and_process_data
from src.models import train_random_forest, train_logistic_regression, train_xgboost, tune_model
from src.evaluation import evaluate_model, evaluate_bookmaker_baseline
from src.analysis import (
    plot_feature_importance,
    simulate_betting_strategy,
    plot_confusion_matrix,
    plot_multiclass_calibration,
    plot_bet_frequency_by_odds,
    plot_edge_distribution,
)

def _metrics_reset(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def _mwrite(path: str, line: str = ""):
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(line) + "\n")


def _fmt_eval(m: dict) -> str:
    # evaluate_model returns dict with keys: "Accuracy", "Log Loss", "Brier Score"
    return f"Acc: {m['Accuracy']:.4f} | LogLoss: {m['Log Loss']:.4f} | Brier: {m['Brier Score']:.4f}"


def main():
    # -------------------------
    # 0) Setup
    # -------------------------
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    RESULTS_DIR = "results"
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

   
    for f in os.listdir(FIG_DIR):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(FIG_DIR, f))

    SHOW_FIGS = False  

    metrics_file = os.path.join(RESULTS_DIR, "metrics.txt")
    _metrics_reset(metrics_file)

    def log(line=""):
        # write to metrics file
        _mwrite(metrics_file, line)

    print("=" * 60)
    print("Serie A Prediction: Advanced Pipeline")
    print("=" * 60)

    log("=" * 70)
    log("Serie A Prediction — Pipeline Output (Key Metrics)")
    log(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log("")

    # -------------------------
    # 1) Load Data
    # -------------------------
    print("\n1) Loading data...")
    log("1) Loading data...")

    out = load_and_process_data()
    if isinstance(out, tuple) and len(out) == 6:
        X_train_full, X_test_full, y_train, y_test, odds_test, data_meta = out
        if isinstance(data_meta, dict):
            if "raw_dir" in data_meta:
                log(f"Looking for files in: {data_meta['raw_dir']}")
            if "files" in data_meta:
                log(f"Found files: {data_meta['files']}")
            if "train_size" in data_meta and "test_size" in data_meta:
                log(f"   Train size: {data_meta['train_size']}, Test size: {data_meta['test_size']}")
    else:
        X_train_full, X_test_full, y_train, y_test, odds_test = out

    print(f"   Train size: {X_train_full.shape[0]}, Test size: {X_test_full.shape[0]}")
    log(f"   Train size: {X_train_full.shape[0]}, Test size: {X_test_full.shape[0]}")
    log("")

    # -------------------------
    # 2) Feature Ablation Study (RF baseline)
    # -------------------------
    experiments = {
        "1. Basic Form": [
            "Diff_AvgPointsLast5",
            "Diff_AvgGoalsForLast5",
            "Diff_AvgGoalsAgainstLast5",
            "Diff_AvgPointsSeason",
        ],
        "2. + Match Stats": [
            "Diff_AvgPointsLast5",
            "Diff_AvgGoalsForLast5",
            "Diff_AvgGoalsAgainstLast5",
            "Diff_AvgPointsSeason",
            "Diff_AvgShotsOnTargetLast5",
            "Diff_AvgCornersLast5",
            "Diff_AvgFoulsLast5",
        ],
        "3. + ELO & Context": [
            "Diff_AvgPointsLast5",
            "Diff_AvgGoalsForLast5",
            "Diff_AvgGoalsAgainstLast5",
            "Diff_AvgPointsSeason",
            "Diff_AvgShotsOnTargetLast5",
            "Diff_AvgCornersLast5",
            "Diff_AvgFoulsLast5",
            "Diff_Elo",
            "HomeElo",
            "AwayElo",
            "RestDays_Home",
            "RestDays_Away",
        ],
    }

    print("\n2) Running feature experiments (RF baseline)...")
    log("2) Running feature experiments (RF baseline)...")
    log("")

    best_feat_logloss = float("inf")
    best_features = None
    best_X_train_scaled = None
    best_X_test_scaled = None
    best_feat_metrics = None
    best_feat_name = None

    for name, features in experiments.items():
        X_tr = X_train_full[features]
        X_te = X_test_full[features]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        model = train_random_forest(X_tr_scaled, y_train)
        metrics = evaluate_model(model, X_te_scaled, y_test, f"RF - {name}")

        
        print(
            f"   [{name}] Acc={metrics['Accuracy']:.4f}  "
            f"LogLoss={metrics['Log Loss']:.4f}  "
            f"Brier={metrics['Brier Score']:.4f}"
        )

        log(f"--- RF - {name} ---")
        log(f"Accuracy:    {metrics['Accuracy']:.4f}")
        log(f"Log Loss:    {metrics['Log Loss']:.4f}")
        log(f"Brier Score: {metrics['Brier Score']:.4f}")
        log(f"   [{name}] Acc={metrics['Accuracy']:.4f}  LogLoss={metrics['Log Loss']:.4f}  Brier={metrics['Brier Score']:.4f}")
        log("")

        if metrics["Log Loss"] < best_feat_logloss:
            best_feat_logloss = metrics["Log Loss"]
            best_features = features
            best_X_train_scaled = X_tr_scaled
            best_X_test_scaled = X_te_scaled
            best_feat_metrics = metrics
            best_feat_name = name

    if best_features is None:
        raise RuntimeError("No feature set selected. Check feature names vs dataset columns.")

    print("\nChosen features:", best_features)
    print(
        f"   -> Winner: {len(best_features)} features "
        f"(LogLoss: {best_feat_logloss:.4f}, Acc: {best_feat_metrics['Accuracy']:.4f}, "
        f"Brier: {best_feat_metrics['Brier Score']:.4f})"
    )

    log(f"Chosen features: {best_features}")
    log(
        f"   -> Winner: {best_feat_name} ({len(best_features)} features) "
        f"(LogLoss: {best_feat_logloss:.4f}, Acc: {best_feat_metrics['Accuracy']:.4f}, "
        f"Brier: {best_feat_metrics['Brier Score']:.4f})"
    )
    log("")

    # -------------------------
    # 3) Model Comparison
    # -------------------------
    print("\n3) Comparing models on best feature set...")
    log("3) Comparing models on best feature set...")
    log("")

    lr_base = train_logistic_regression(best_X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr_base, best_X_test_scaled, y_test, "Logistic Regression")
    log("--- Logistic Regression ---")
    log(f"Accuracy:    {lr_metrics['Accuracy']:.4f}")
    log(f"Log Loss:    {lr_metrics['Log Loss']:.4f}")
    log(f"Brier Score: {lr_metrics['Brier Score']:.4f}")
    log("")

    rf_base = train_random_forest(best_X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf_base, best_X_test_scaled, y_test, "Random Forest")
    log("--- Random Forest ---")
    log(f"Accuracy:    {rf_metrics['Accuracy']:.4f}")
    log(f"Log Loss:    {rf_metrics['Log Loss']:.4f}")
    log(f"Brier Score: {rf_metrics['Brier Score']:.4f}")
    log("")

    xgb_base = train_xgboost(best_X_train_scaled, y_train)
    xgb_metrics = evaluate_model(xgb_base, best_X_test_scaled, y_test, "XGBoost")
    log("--- XGBoost ---")
    log(f"Accuracy:    {xgb_metrics['Accuracy']:.4f}")
    log(f"Log Loss:    {xgb_metrics['Log Loss']:.4f}")
    log(f"Brier Score: {xgb_metrics['Brier Score']:.4f}")
    log("")

    models = {
        LogisticRegression: lr_metrics["Log Loss"],
        RandomForestClassifier: rf_metrics["Log Loss"],
        XGBClassifier: xgb_metrics["Log Loss"],
    }
    best_model_class = min(models, key=models.get)

    # -------------------------
    # 4) Hyperparameter Tuning
    # -------------------------
    print(f"\n4) Tuning Hyperparameters for {best_model_class.__name__}...")
    log(f"4) Tuning Hyperparameters for {best_model_class.__name__}...")

    param_grids = {
        RandomForestClassifier: {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        XGBClassifier: {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
        LogisticRegression: {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga"],
        },
    }

    tuned_out = tune_model(
        best_model_class,
        param_grids[best_model_class],
        best_X_train_scaled,
        y_train,
    )


    tuned_best_params = None
    if isinstance(tuned_out, tuple) and len(tuned_out) == 2:
        tuned_model, tuned_best_params = tuned_out
    else:
        tuned_model = tuned_out
        tuned_best_params = getattr(tuned_model, "best_params_", None)

    if tuned_best_params is not None:
        log(f"   ✓ Best Params: {tuned_best_params}")
    log("")

    baseline_models = {
        LogisticRegression: lr_base,
        RandomForestClassifier: rf_base,
        XGBClassifier: xgb_base,
    }
    baseline_champion = baseline_models[best_model_class]

    baseline_metrics = evaluate_model(baseline_champion, best_X_test_scaled, y_test, "Baseline Champion")
    tuned_metrics = evaluate_model(tuned_model, best_X_test_scaled, y_test, "Tuned Champion")

    log("--- Baseline Champion ---")
    log(_fmt_eval(baseline_metrics))
    log("")
    log("--- Tuned Champion ---")
    log(_fmt_eval(tuned_metrics))
    log("")

    if tuned_metrics["Log Loss"] >= baseline_metrics["Log Loss"]:
        print("\nTuning did not improve Log Loss — using BASELINE champion for deep dive.")
        log("Tuning did not improve Log Loss — using BASELINE champion for deep dive.")
        champion_model = baseline_champion
        champion_metrics = baseline_metrics
    else:
        log("Tuning improved Log Loss — using TUNED champion for deep dive.")
        champion_model = tuned_model
        champion_metrics = tuned_metrics

    log("")

    # -------------------------
    # 5) Deep Dive + Save Figures
    # -------------------------
    print("\n5) Deep Dive Analysis on Champion Model...")
    log("5) Deep Dive Analysis on Champion Model...")
    log("")

    plot_feature_importance(
        champion_model,
        best_X_test_scaled,
        y_test,
        best_features,
        save_path=os.path.join(FIG_DIR, "01_feature_importance.png"),
        show=SHOW_FIGS,
    )

    plot_multiclass_calibration(
        champion_model,
        best_X_test_scaled,
        y_test,
        save_dir=FIG_DIR,
        show=SHOW_FIGS,
    )

    # Betting simulation
    bet_out = simulate_betting_strategy(
        champion_model,
        best_X_test_scaled,
        y_test,
        odds_test,
        bankroll=1000.0,
        stake=10.0,
        margin_threshold=0.05,
        save_path=os.path.join(FIG_DIR, "03_bankroll.png"),
        show=SHOW_FIGS,
    )

    bet_mask, bet_side = bet_out[0], bet_out[1]
    bet_summary = None
    if isinstance(bet_out, tuple) and len(bet_out) >= 3 and isinstance(bet_out[2], dict):
        bet_summary = bet_out[2]

    if bet_summary is not None:
        log("Betting strategy summary:")
        for k, v in bet_summary.items():
            log(f"   {k}: {v}")
        log("")

    plot_bet_frequency_by_odds(
        odds_test,
        bet_mask,
        bet_side,
        save_path=os.path.join(FIG_DIR, "04_bet_frequency_by_odds.png"),
        show=SHOW_FIGS,
    )

    # Bookmaker baseline
    bm = evaluate_bookmaker_baseline(odds_test, y_test)
    if isinstance(bm, dict) and all(k in bm for k in ["Accuracy", "Log Loss", "Brier Score"]):
        log("--- Baseline: Bookmaker Implied Probabilities ---")
        log(_fmt_eval(bm))
        log("")

    plot_confusion_matrix(
        champion_model,
        best_X_test_scaled,
        y_test,
        save_path=os.path.join(FIG_DIR, "05_confusion_matrix.png"),
        show=SHOW_FIGS,
    )

    plot_edge_distribution(
        champion_model,
        best_X_test_scaled,
        odds_test,
        save_path=os.path.join(FIG_DIR, "06_edge_distribution.png"),
        show=SHOW_FIGS,
    )

    # -------------------------
    # 6) Save Metrics (final table)
    # -------------------------
    all_metrics = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "Champion Model (used)": champion_metrics,
    }

    print("\n=== Final Model Comparison (Test Set) ===")
    for name, m in all_metrics.items():
        print(
            f"{name:25s} | "
            f"Acc: {m['Accuracy']:.3f}  "
            f"LogLoss: {m['Log Loss']:.3f}  "
            f"Brier: {m['Brier Score']:.3f}"
        )

    # overwrite metrics file with full content already written + append final table
    log("=== Final Model Comparison (Test Set) ===")
    for name, m in all_metrics.items():
        log(
            f"{name:25s} | "
            f"Acc: {m['Accuracy']:.3f}  "
            f"LogLoss: {m['Log Loss']:.3f}  "
            f"Brier: {m['Brier Score']:.3f}"
        )
    log("")

    print(f"\nSaved: {metrics_file}")
    print(f"Saved figures in: {FIG_DIR}")

    log(f"Saved: {metrics_file}")
    log(f"(Figures saved in: {FIG_DIR})")


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibrationDisplay


CLASS_NAMES = ("Home", "Draw", "Away")
IDX_HOME = 0
IDX_DRAW = 1
IDX_AWAY = 2


def _finalize_plot(save_path=None, show=True):
    plt.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def _coerce_odds_df(odds_df):
    odds = odds_df.copy().reset_index(drop=True)
    for col in ["B365H", "B365D", "B365A"]:
        odds[col] = np.asarray(odds[col], dtype=float)
    return odds


def plot_feature_importance(
    model,
    X_test,
    y_test,
    feature_names,
    save_path=None,
    show=True,
):
    print("\n   Generating Feature Importance Plot...")

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    plt.title("Permutation Importances (Test Set)")
    plt.xlabel("Decrease in score (higher = more important)")

    _finalize_plot(save_path, show)


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    classes=CLASS_NAMES,
    save_path=None,
    show=True,
):
    print("\n   Generating Confusion Matrix...")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=30, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    _finalize_plot(save_path, show)


def plot_multiclass_calibration(
    model,
    X_test,
    y_test,
    class_names=CLASS_NAMES,
    classes_to_plot=(IDX_HOME, IDX_AWAY),
    save_dir=None,
    show=True,
):
    probs = model.predict_proba(X_test)

    for idx in classes_to_plot:
        name = class_names[idx]

        y_true_bin = (np.asarray(y_test) == idx).astype(int)
        prob = probs[:, idx]

        plt.figure(figsize=(6, 6))
        CalibrationDisplay.from_predictions(
            y_true_bin,
            prob,
            n_bins=10,
            name=name,
        )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.title(f"Calibration Curve – {name}")

        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"02_calibration_{name.lower()}.png")

        _finalize_plot(save_path, show)


def simulate_betting_strategy(
    model,
    X_test,
    y_test,
    odds_test,
    bankroll=1000.0,
    stake=10.0,                 
    margin_threshold=0.05,
    save_path=None,
    show=True,
):
    print("\n   Simulating betting strategy...")

    probs = model.predict_proba(X_test)

    odds = _coerce_odds_df(odds_test)
    y = np.asarray(y_test).reshape(-1)

    n = len(y)
    history = [bankroll]
    current = float(bankroll)

    bet_mask = [False] * n
    bet_side = [None] * n

    bets = 0
    wins = 0
    total_staked = 0.0

    for i in range(n):
        p_home = probs[i, IDX_HOME]
        p_draw = probs[i, IDX_DRAW]
        p_away = probs[i, IDX_AWAY]

        h_odd = float(odds.loc[i, "B365H"])
        d_odd = float(odds.loc[i, "B365D"])
        a_odd = float(odds.loc[i, "B365A"])

        inv = np.array([1.0 / h_odd, 1.0 / d_odd, 1.0 / a_odd], dtype=float)
        imp = inv / inv.sum()

        edge_home = p_home - imp[IDX_HOME]
        edge_away = p_away - imp[IDX_AWAY]

        # Choose best bet among Home/Away only (simple + stable)
        if edge_home > margin_threshold and edge_home >= edge_away:
            side = "H"
            win_label = IDX_HOME
            odd = h_odd
        elif edge_away > margin_threshold and edge_away > edge_home:
            side = "A"
            win_label = IDX_AWAY
            odd = a_odd
        else:
            history.append(current)
            continue

        # Place bet
        bets += 1
        total_staked += stake
        bet_mask[i] = True
        bet_side[i] = side

        if y[i] == win_label:
            # Profit = stake * (odd - 1)
            current += stake * (odd - 1.0)
            wins += 1
        else:
            current -= stake

        history.append(current)

    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.axhline(bankroll, linestyle="--")
    plt.title(f"Bankroll | Bets={bets} Wins={wins}")
    plt.xlabel("Matches")
    plt.ylabel("Bankroll")

    _finalize_plot(save_path, show)

    roi_bankroll = (current - bankroll) / bankroll * 100 if bankroll > 0 else 0.0
    roi_on_staked = (current - bankroll) / total_staked * 100 if total_staked > 0 else 0.0
    hit_rate = wins / bets * 100 if bets > 0 else 0.0

    print(f"   Final bankroll: {current:.2f}")
    print(f"   ROI (on bankroll): {roi_bankroll:.2f}%")
    print(f"   ROI (on amount staked): {roi_on_staked:.2f}%")
    print(f"   Hit rate: {hit_rate:.2f}% | Total staked: {total_staked:.2f}")


    summary = {
        "Final bankroll": f"{current:.2f}",
        "ROI (on bankroll)": f"{roi_bankroll:.2f}%",
        "ROI (on amount staked)": f"{roi_on_staked:.2f}%",
        "Hit rate": f"{hit_rate:.2f}%",
        "Total staked": f"{total_staked:.2f}",
        "Bets": str(bets),
        "Wins": str(wins),
        "Margin threshold": str(margin_threshold),
        "Stake": str(stake),
        "Starting bankroll": str(bankroll),
    }

    return bet_mask, bet_side, summary


def plot_bet_frequency_by_odds(
    odds_test,
    bet_mask,
    bet_side,
    bins=(1, 1.5, 2, 3, 5, 10, 100),
    save_path=None,
    show=True,
):
    odds = _coerce_odds_df(odds_test)

    chosen = []
    for i in range(len(odds)):
        if not bet_mask[i]:
            continue
        chosen.append(odds.loc[i, "B365H"] if bet_side[i] == "H" else odds.loc[i, "B365A"])

    plt.figure(figsize=(8, 4))
    plt.hist(chosen, bins=bins)
    plt.title("Bet Frequency by Odds")
    plt.xlabel("Odds")
    plt.ylabel("Count")

    _finalize_plot(save_path, show)

def plot_edge_distribution(
    model,
    X_test,
    odds_test,
    save_path=None,
    show=True,
):
    probs = model.predict_proba(X_test)
    odds = _coerce_odds_df(odds_test)

    edges_home = []
    edges_away = []

    for i in range(len(odds)):
        h = float(odds.loc[i, "B365H"])
        d = float(odds.loc[i, "B365D"])
        a = float(odds.loc[i, "B365A"])

        inv = np.array([1.0 / h, 1.0 / d, 1.0 / a], dtype=float)  # Home,Draw,Away
        imp = inv / inv.sum()

        edges_home.append(probs[i, IDX_HOME] - imp[IDX_HOME])
        edges_away.append(probs[i, IDX_AWAY] - imp[IDX_AWAY])

    plt.figure(figsize=(8, 4))
    plt.hist(edges_home, bins=30, alpha=0.7, label="Home")
    plt.hist(edges_away, bins=30, alpha=0.7, label="Away")
    plt.legend()
    plt.title("Model Edge vs Bookmaker")

    _finalize_plot(save_path, show)

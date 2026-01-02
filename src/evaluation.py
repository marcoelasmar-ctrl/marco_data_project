"""Model evaluation and baseline comparison."""
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

def calculate_multiclass_brier(y_true, y_prob):
    """
    Calculates the multi-class Brier score.
    Formula: (1/N) * sum((f_ij - o_ij)^2) for all i classes and j samples.
    """
    # Create one-hot encoding for true labels
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    

    # Calculate MSE
    return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate ML model with Proposal metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = calculate_multiclass_brier(y_test, y_prob)
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:    {acc:.4f}") # Accuracy: the percentage of matches the model predicted correctly.
    print(f"Log Loss:    {ll:.4f}") # Log loss: how bad your predictions are when you are confident but wrong lower is better.
    print(f"Brier Score: {bs:.4f}") # Brier Score: how close your probability predictions are to the real result. Lower is better.
    
    return {'Accuracy': acc, 'Log Loss': ll, 'Brier Score': bs}

def evaluate_bookmaker_baseline(odds_test, y_test):
    """
    Evaluates the Bookmaker Baseline using implied probabilities.
    Implied Prob = 1 / Odd (normalized to sum to 1 to remove vigorish).
    """
    probs = 1 / odds_test
    probs_norm = probs.div(probs.sum(axis=1), axis=0)
    
    y_pred = probs_norm.idxmax(axis=1).map({'B365H': 0, 'B365D': 1, 'B365A': 2})
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, probs_norm)
    bs = calculate_multiclass_brier(y_test, probs_norm.values)
    
    print(f"\n--- Baseline: Bookmaker Implied Probabilities ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log Loss:    {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    
    return {'Accuracy': acc, 'Log Loss': ll, 'Brier Score': bs}


"""Model definitions, training, and hyperparameter tuning."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import RandomizedSearchCV 
from xgboost import XGBClassifier 
import numpy as np

def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest model with default/baseline parameters."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42): 
    """Train Logistic Regression model."""
    model = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """Train XGBoost model with baseline parameters."""
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05, 
        max_depth=3, 
        eval_metric='mlogloss', 
        random_state=random_state,
        n_jobs=-1 
    )
    model.fit(X_train, y_train)
    return model

def tune_model(model_class, param_dist, X_train, y_train, random_state=42):
    """
    Performs Randomized Search to find the best hyperparameters.

    Returns:
        tuned_model, best_params
    """
    print(f"   Tuning {model_class.__name__}...")

    if model_class == XGBClassifier:
        model = model_class(
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    else:
        model = model_class(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_log_loss",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"   ✓ Best Params: {best_params}")

    tuned_model = search.best_estimator_
    return tuned_model, best_params


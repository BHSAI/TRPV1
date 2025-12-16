"""
Machine learning model factory functions.

This module provides factory functions for creating configured ML models:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes (Bernoulli/Gaussian)
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
"""

import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

# ============================================================================
# Model Factory Functions
# ============================================================================

def create_knn(random_state=42, n_neighbors=5):
    """Create K-Nearest Neighbors classifier."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def create_svm(random_state=42, kernel='rbf', probability=True):
    """Create Support Vector Machine classifier."""
    return SVC(
        kernel=kernel,
        probability=probability,
        random_state=random_state
    )


def create_bayesian_bernoulli(random_state=42, alpha=1.0):
    """Create Bernoulli Naive Bayes classifier (for binary features)."""
    return BernoulliNB(alpha=alpha)


def create_bayesian_gaussian(random_state=42):
    """Create Gaussian Naive Bayes classifier (for continuous features)."""
    return GaussianNB()


def create_logistic_regression(random_state=42, max_iter=2000, class_weight='balanced'):
    """Create Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )


def create_random_forest(random_state=42, n_estimators=500, class_weight='balanced', n_jobs=-1):
    """Create Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state
    )


def create_xgboost(random_state=42, n_estimators=500, learning_rate=0.1,
                   max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1):
    """Create XGBoost classifier."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        use_label_encoder=False,
    )


def create_lightgbm(random_state=42, n_estimators=500, n_jobs=-1):
    """Create LightGBM classifier."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1
    )


# ============================================================================
# Model Factory Registry
# ============================================================================

MODEL_FACTORIES = {
    "KNN": create_knn,
    "SVM": create_svm,
    "Bayesian": create_bayesian_bernoulli,  # Default to Bernoulli for fingerprints
    "LogisticRegression": create_logistic_regression,
    "RandomForest": create_random_forest,
}

# Add gradient boosting if available
if XGBOOST_AVAILABLE:
    MODEL_FACTORIES["XGBoost"] = create_xgboost

if LIGHTGBM_AVAILABLE:
    MODEL_FACTORIES["LightGBM"] = create_lightgbm


# Standard order for reporting
MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost",
]


def get_available_models():
    """Get list of available model names."""
    return list(MODEL_FACTORIES.keys())


def create_model(model_name, random_state=42):
    """
    Create a machine learning model by name.

    Args:
        model_name: Name of the model ('KNN', 'SVM', 'Bayesian', 'LogisticRegression',
                    'RandomForest', 'XGBoost', 'LightGBM')
        random_state: Random seed for reproducibility

    Returns:
        Configured sklearn/xgboost/lightgbm estimator

    Raises:
        ValueError: If model_name is not recognized
        ImportError: If model requires unavailable package
    """
    if model_name not in MODEL_FACTORIES:
        available = get_available_models()
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )

    factory = MODEL_FACTORIES[model_name]
    return factory(random_state=random_state)


def get_model_factories_dict():
    """
    Get dictionary of model factories for cross-validation.

    Returns:
        dict: Mapping of model_name -> factory_function(random_state)
    """
    factories = {}

    for model_name, factory_func in MODEL_FACTORIES.items():
        # Wrap factory to accept only random_state
        factories[model_name] = lambda rs, fn=factory_func: fn(random_state=rs)

    return factories


def check_model_availability():
    """
    Check which models are available and log warnings for missing packages.

    Returns:
        dict: Availability status for each model type
    """
    availability = {
        "XGBoost": XGBOOST_AVAILABLE,
        "LightGBM": LIGHTGBM_AVAILABLE,
    }

    if not XGBOOST_AVAILABLE:
        logging.warning("XGBoost not installed. Install with: pip install xgboost")

    if not LIGHTGBM_AVAILABLE:
        logging.warning("LightGBM not installed. Install with: pip install lightgbm")

    return availability

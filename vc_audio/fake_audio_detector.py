import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import random

def train_fake_audio_detector(X, y, sampling_strategy=0.7, test_size=0.3):
    """
    Trains a fake audio detection model using XGBoost and SMOTE.
    Returns the model and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba)
    }

    return model, X_test, metrics

def explain_with_shap(model, X_test, sample_idx=None):
    """
    Generates SHAP explanations for a model's predictions.
    """
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    if sample_idx is None:
        sample_idx = random.randint(0, len(X_test) - 1)

    shap.plots.waterfall(shap_values[sample_idx])
    shap.plots.beeswarm(shap_values)

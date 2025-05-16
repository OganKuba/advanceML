import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_log_likelihood(y_true, p_pred):
    eps = 1e-15
    p_pred = np.clip(p_pred, eps, 1 - eps)
    loglik = np.sum(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
    return loglik


def fit_logistic_model(X, y, regularization=True):
    if regularization:
        model = LogisticRegression(penalty='l2', solver='lbfgs')
    else:
        model = LogisticRegression(penalty='none', solver='lbfgs')

    model.fit(X, y)
    p_hat = model.predict_proba(X)[:, 1]
    return model, p_hat

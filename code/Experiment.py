import pandas as pd
from Generator import generate_data, accuracy_score
from LDA import LDA
from QDA import QDA
from NB import NB
import numpy as np

def run_experiments_scheme(scheme, vary='a', values=None, n_splits=20):
    results = {
        'LDA': {val: [] for val in values},
        'QDA': {val: [] for val in values},
        'NB': {val: [] for val in values},
    }

    for val in values:
        for _ in range(n_splits):
            if vary == 'a':
                X, y = generate_data(scheme, n=1000, a=val, rho=0.5)
            else:
                X, y = generate_data(scheme, n=1000, a=2.0, rho=val)

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        train_size = 700
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit LDA
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)

        # Fit QDA
        qda = QDA()
        qda.fit(X_train, y_train)
        y_pred_qda = qda.predict(X_test)
        acc_qda = accuracy_score(y_test, y_pred_qda)

        # Fit NB
        nb = NB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        acc_nb = accuracy_score(y_test, y_pred_nb)

        # Store
        results['LDA'][val].append(acc_lda)
        results['QDA'][val].append(acc_qda)
        results['NB'][val].append(acc_nb)

    return results


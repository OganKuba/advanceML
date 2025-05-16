import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_X_y, check_array
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class AdaBoostBeta(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators: int = 50,
                 base_estimator=None,
                 random_state: int | None = None):
        self.n_estimators = n_estimators
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else DecisionTreeClassifier(max_depth=1, random_state=random_state)
        )
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        w = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []
        self.betas_ = []

        rng = np.random.RandomState(self.random_state)

        for m in range(self.n_estimators):
            est = clone(self.base_estimator)
            if hasattr(est, "random_state"):
                est.set_params(random_state=rng.randint(0, np.iinfo(np.int32).max))
            est.fit(X, y, sample_weight=w)

            y_pred = est.predict(X)
            miss = (y_pred != y)
            eps = np.dot(w, miss) / np.sum(w)

            if eps >= 0.5:
                break;

            beta = eps / (1 - eps + 1e-16)
            self.estimators_.append(est)
            self.betas_.append(beta)

            correct = ~(miss)
            w[correct] *= beta

            w /= w.sum()

        self.betas_ = np.array(self.betas_)
        return self

    def score_matrix(self, X):
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))

        log_inv_beta = np.log(1 / self.betas_)
        for est, weight in zip(self.estimators_, log_inv_beta):
            pred = est.predict(X)

            for idx, cls in enumerate(self.classes_):
                scores[:, idx] += weight * (pred == cls)
        return scores

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        scores = self.score_matrix(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        scores = self.score_matrix(X)

        scores = np.maximum(scores, 0)
        norm = scores.sum(axis=1, keepdims=True) + 1e-16
        return scores / norm

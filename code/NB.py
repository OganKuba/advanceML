import numpy as np

class NB:

    def __init__(self):
        self.mu_0 = None
        self.mu_1 = None
        self.var_0 = None
        self.var_1 = None
        self.pi_0 = None
        self.pi_1 = None

    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]

        n0 = X0.shape[0]
        n1 = X1.shape[0]
        n = float(n0 + n1)

        self.mu_0 = np.mean(X0, axis=0)
        self.mu_1 = np.mean(X1, axis=0)

        self.var_0 = np.var(X0, axis=0)
        self.var_1 = np.var(X1, axis=0)

        self.pi_0 = n0 / n
        self.pi_1 = n1 / n

    def predict_proba(self, Xtest):
        log_pi_0 = np.log(self.pi_0)
        log_pi_1 = np.log(self.pi_1)

        def log_gaussian(x, mean, var):
            return -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)

        res = []
        for row in Xtest:
            log_like_0 = log_pi_0
            log_like_1 = log_pi_1
            for j in range(len(row)):
                log_like_0 += log_gaussian(row[j], self.mu_0[j], self.var_0[j])
                log_like_1 += log_gaussian(row[j], self.mu_1[j], self.var_1[j])

            d= log_like_1 - log_like_0
            res.append(d)

        res = np.array(res)
        prob_class1 = 1.0 / (1.0 + np.exp(-res))
        return prob_class1

    def predict(self, Xtest):
        probability = self.predict_proba(Xtest)
        result = (probability > 0.5).astype(int)

        return result

    def get_params(self):
        return {
            'mu_0': self.mu_0,
            'mu_1': self.mu_1,
            'var_0': self.var_0,
            'var_1': self.var_1,
            'pi_0': self.pi_0,
            'pi_1': self.pi_1
        }



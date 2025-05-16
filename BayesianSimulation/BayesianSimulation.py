from Experiment import run_experiments_scheme
from Generator import generate_data
from Plot import boxplot_results
from QDA import QDA
from LDA import LDA
import matplotlib.pyplot as plt
import numpy as np

def bayesian_simulation():
    a_values = [0.1, 0.5, 1, 2, 3, 5]

    results_s1 = run_experiments_scheme(scheme=1, vary='a', values=a_values, n_splits=20)
    boxplot_results(results_s1, parameter_name='a', filename='BayesianSimulatedData1_scheme1.pdf')

    results_s2 = run_experiments_scheme(scheme=2, vary='a', values=a_values, n_splits=20)
    boxplot_results(results_s2, parameter_name='a', filename='BayesianSimulatedData1_scheme2.pdf')

    rho_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    results_s1_rho = run_experiments_scheme(scheme=1, vary='rho', values=rho_values, n_splits=20)
    boxplot_results(results_s1_rho, parameter_name='rho', filename='BayesianSimulatedData2_scheme1.pdf')

    results_s2_rho = run_experiments_scheme(scheme=2, vary='rho', values=rho_values, n_splits=20)
    boxplot_results(results_s2_rho, parameter_name='rho', filename='BayesianSimulatedData2_scheme2.pdf')

    a_chosen = 2
    rho_chosen = 0.5
    X, y = generate_data(scheme=2, a=a_chosen, rho=rho_chosen, n=1000, random_state=123)
    lda = LDA()
    lda.fit(X, y)
    qda = QDA()
    qda.fit(X, y)

    plt.figure()
    X0 = X[y == 0]
    X1 = X[y == 1]
    plt.scatter(X0[:, 0], X0[:, 1], marker='o', label="Class 0", alpha=0.6)
    plt.scatter(X1[:, 0], X1[:, 1], marker='x', label="Class 1", alpha=0.6)


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    lda_scores = lda.predict_proba(grid_points) - 0.5  # positive => class 1
    lda_scores = lda_scores.reshape(xx.shape)


    qda_scores = qda.predict_proba(grid_points) - 0.5
    qda_scores = qda_scores.reshape(xx.shape)


    plt.contour(xx, yy, lda_scores, levels=[0], linestyles=['-'], colors='red', linewidths=2)
    plt.contour(xx, yy, qda_scores, levels=[0], linestyles=['-'], colors='green', linewidths=2)

    plt.title(f"Scheme=2, a={a_chosen}, rho={rho_chosen}\nLDA (red) & QDA (green) boundaries")
    plt.legend()
    plt.savefig("BayesianSimulatedData3.pdf")
    plt.close()
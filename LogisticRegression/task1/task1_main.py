from Lab4.task1.compute import fit_logistic_model, compute_log_likelihood
from Lab4.task1.data_loader import load_earthquake_data
from Lab4.task1.plot import plot_body_surface


def task1_earthquake(file_path):
    df = load_earthquake_data(file_path)

    plot_body_surface(df)

    X = df[['body', 'surface']]
    y = df['popn']


    model_no_reg, p_no_reg = fit_logistic_model(X, y, regularization=False)

    # Współczynniki (pierwszy to bias, w scikit-learn w 'intercept_')
    print("=== Model bez regularyzacji ===")
    print("Intercept:", model_no_reg.intercept_)
    print("Coef body/surface:", model_no_reg.coef_)

    # Prawdopodobieństwa przewidywane
    print("Przewidywane prawdopodobieństwa (pierwsze 5):", p_no_reg[:5])

    # Log-likelihood
    ll_no_reg = compute_log_likelihood(y, p_no_reg)
    print("Log-likelihood (bez regularyzacji):", ll_no_reg)
    print()

    # ------ Model Z regularyzacją L2 ------
    model_l2, p_l2 = fit_logistic_model(X, y, regularization=True)

    print("=== Model z regularyzacją L2 ===")
    print("Intercept:", model_l2.intercept_)
    print("Coef body/surface:", model_l2.coef_)

    # Prawdopodobieństwa przewidywane
    print("Przewidywane prawdopodobieństwa (pierwsze 5):", p_l2[:5])

    # Log-likelihood
    ll_l2 = compute_log_likelihood(y, p_l2)
    print("Log-likelihood (z L2):", ll_l2)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    RBF kernel for 1D or 2D inputs.
    """
    # Convert 1D inputs to 2D (n_samples, n_features)
    if X1.ndim == 1:
        X1 = X1[:, None]
    if X2.ndim == 1:
        X2 = X2[:, None]

    # Squared Euclidean distance
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)


def gp_predict(X_train, Y_train, X_test, length_scale=1.0, sigma_f=1.0, sigma_y=1e-4):
    """
    Gaussian Process prediction using the RBF kernel.
    Supports 1D or 2D feature input.
    Returns predictive mean and standard deviation.
    """
    # Convert 1D inputs to 2D if necessary
    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    K = rbf_kernel(X_train, X_train, length_scale, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale, sigma_f)
    K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f) + 1e-6 * np.eye(len(X_test))  # for stability

    # Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

    # Predictive mean
    mu_s = K_s.T @ alpha

    # Predictive covariance
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T @ v
    std_s = np.sqrt(np.clip(np.diag(cov_s), 0, np.inf))  # Avoid small negative values

    return mu_s, std_s

if __name__ == "__main__":
    # Load your price series
    df = pd.read_csv("sample_data/round-1-island-data-bottle/prices_round_1_day_0.csv", sep=";")
    df["product"] = df["product"].str.strip().str.upper()
    series = df[df["product"] == "SQUID_INK"]["mid_price"].dropna().reset_index(drop=True)

    feature_size = 1
    # Assume series is a pandas Series with the time series data
    y = series.values

    # Efficient lagged feature construction
    n_samples = len(y) - feature_size
    X = np.column_stack([y[i: i + n_samples] for i in range(feature_size)])
    Y = y[feature_size:]
    time_indices = np.arange(feature_size, len(y))      
    # Random train/test split (75/25)
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    split = int(0.75 * n_samples)      
    train_idx = indices[:split]
    test_idx = indices[split:]     
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    time_test = time_indices[test_idx]     
    mu, std = gp_predict(X_train, Y_train, X_test)

    # Step 4: Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_test, Y_test, 'k.', label='True values')
    plt.plot(time_test, mu, 'b-', label='Predicted mean')
    plt.fill_between(time_test, mu - 2*std, mu + 2*std,
                     color='blue', alpha=0.2, label='±2 std. dev.')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title('Gaussian Process Regression (custom) – One-step-ahead Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

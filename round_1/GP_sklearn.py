import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("sample_data/round-1-island-data-bottle/prices_round_1_day_0.csv", sep=";")
    df["product"] = df["product"].str.strip().str.upper()
    series = df[df["product"] == "SQUID_INK"]["mid_price"].dropna().reset_index(drop=True)

    # Feature size: use last 5 values to predict the next one
    feature_size = 5
    y = series.values

    # Create lagged feature matrix
    n_samples = len(y) - feature_size
    X = np.column_stack([y[i: i + n_samples] for i in range(feature_size)])
    Y = y[feature_size:]
    time_indices = np.arange(feature_size, len(y))

    # Train-test split (75/25)
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    split = int(0.75 * n_samples)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx][::10]
    Y_train = Y[train_idx][::10]
    # test_idx = np.argsort(test_idx)
    print(test_idx[np.argsort(test_idx)])
    test_idx = test_idx[np.argsort(test_idx)]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    time_test = time_indices[test_idx]
    print(time_test)

    # Kernel: RBF over 5-dimensional inputs + noise
    kernel = RBF(length_scale=np.ones(feature_size), length_scale_bounds=(1e-2, 1e3)) + \
             WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e1))

    # Fit Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    start_time = time.time()
    gp.fit(X_train, Y_train)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    # Predict
    mu, std = gp.predict(X_test, return_std=True)
    var = std**2

    # Print model details
    print(">>> Posterior mean (first 5 predictions):")
    print(mu[:5])
    print("\n>>> Posterior variance (first 5):")
    print(var[:5])
    print("\n>>> Learned kernel and hyperparameters:")
    print(gp.kernel_)

    # Plot predictions with uncertainty
    plt.figure(figsize=(12, 6))

    # True values as black dots
    plt.plot(time_test, Y_test, 'blue', label='True values')

    # Predicted mean as scatter plot (blue dots)
    plt.plot(time_test, mu, color='red', label='Predicted mean')

    # Standard deviation as error bars (±2 std dev)
    plt.fill_between(time_test, mu - 2*std, mu + 2*std, alpha=0.2, color='blue', label='±2 std. dev.')

    # Labels and title
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title('GP Regression (scikit-learn) – Lag-5 Prediction')

    # Show legend and grid
    # Include training data length in the legend
    plt.legend(title=f'Training data length: {len(Y_test)}', loc='best')
    plt.grid(True)

    # Show the plot
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load and preprocess ---
def load_mid_price_series(df, product_name):
    df_product = df[df['product'] == product_name].sort_values('timestamp')
    series = df_product['mid_price'].dropna().reset_index(drop=True)
    return series

# --- Extract sliding windows ---
def extract_windows(series, window_size):
    return [series[i:i + window_size] for i in range(len(series) - window_size)]

# --- Pure NumPy DTW implementation ---
def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1] # match
            )

    return dtw_matrix[n, m]

# --- Find top matches by DTW ---
def find_similar_patterns(series, pattern, window_size, top_n=5):
    pattern = np.array(pattern)
    distances = []

    for i in range(len(series) - window_size):
        candidate = np.array(series[i:i + window_size])
        dist = dtw_distance(pattern, candidate)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_n]

# --- Plot matches ---
def plot_matches(series, matches, pattern, window_size):
    plt.figure(figsize=(10, 5))
    plt.plot(range(window_size), pattern, label='Reference Pattern', linewidth=2)

    for i, (idx, dist) in enumerate(matches):
        match = np.array(series[idx:idx+window_size].reset_index(drop=True))
        plt.plot(range(window_size), match, alpha=0.5, label=f'Match {i+1} (DTW={dist:.1f})')

    plt.legend()
    plt.title("DTW Pattern Matches")
    plt.xlabel("Timestep")
    plt.ylabel("Mid Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Example usage ---
if __name__ == "__main__":

    base_path = "sample_data"
    # Load the data
    for root, _, files in os.walk(base_path):
        for file in files:
            # if the file starts with "prices"
            if file.startswith("prices"):
                df = pd.read_csv(os.path.join(root, file), sep=';')
                # Process the file
                series = load_mid_price_series(df, product_name='SQUID_INK')

                window_size = 50
                windows = extract_windows(series, window_size)
                pattern = np.array(windows[100])  # You can also pick a hand-crafted one

                matches = find_similar_patterns(series, pattern, window_size=window_size, top_n=5)
                plot_matches(series, matches, pattern, window_size)

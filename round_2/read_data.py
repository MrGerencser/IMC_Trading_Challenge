import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
seconds = 24 * 3600
no_samples = 1e6 / 100
sampling_frequency = no_samples / seconds

def plot_product_data(product_dfs):
    for product_name, df in product_dfs.items():
        print(f"\n📊 Plotting for: {product_name}")
        
        for col in df.columns:
            if col in ['day', 'product']:
                continue  # skip non-numeric/categorical identifiers
            
            plt.figure(figsize=(10, 4))
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.plot(df['timestamp'], df[col], marker='o', label=col)
                plt.title(f"{product_name} - {col}")
                plt.xlabel("Timestamp")
                plt.ylabel(col)
                plt.grid(True)
                plt.legend()
            else:
                df[col].value_counts().plot(kind='bar')
                plt.title(f"{product_name} - {col} (Bar Chart)")
                plt.xlabel(col)
                plt.ylabel("Count")
            
            plt.tight_layout()
            plt.show()

def plot_moving_average(df, window_size=5):
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        series = df[col].dropna()
        print(series)
        smoothed = series.rolling(window=window_size, center=True).mean()

        plt.figure(figsize=(10, 4))
        plt.plot(series.index, series, label='Original', alpha=0.5)
        plt.plot(smoothed.index, smoothed, label=f'Moving Avg (window={window_size})', linewidth=2)
        plt.title(f"{col} - Moving Average Filter")
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compute_and_plot_fft(df, fs=sampling_frequency, bin_width=0.001):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        signal = df[col].dropna().values
        signal = signal - np.mean(signal)  # De-mean

        N = len(signal)
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, d=1/fs)

        magnitudes = np.abs(fft_vals)
        
        # Use only the positive frequencies
        mask = freqs >= 0
        freqs = freqs[mask]
        magnitudes = magnitudes[mask]

        # Bin the magnitudes
        max_freq = np.max(freqs)
        bins = np.arange(0, max_freq + bin_width, bin_width)
        bin_indices = np.digitize(freqs, bins)

        binned_mags = []
        binned_freqs = []

        for i in range(1, len(bins)):
            in_bin = bin_indices == i
            if np.any(in_bin):
                avg_mag = np.mean(magnitudes[in_bin])
                binned_mags.append(avg_mag)
                binned_freqs.append((bins[i-1] + bins[i]) / 2)

        total_mag = np.sum(binned_mags)
        rel_weights = [(m / total_mag) * 100 for m in binned_mags]

        # Plot: Binned Magnitude Spectrum
        plt.figure(figsize=(10, 4))
        plt.plot(binned_freqs, binned_mags, drawstyle='steps-mid')
        plt.title(f"{col} - Binned Magnitude Spectrum (Δf = {bin_width} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Avg Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot: Relative Weight of Each Bin
        plt.figure(figsize=(10, 4))
        plt.bar(binned_freqs, rel_weights, width=bin_width * 0.9, align='center')
        plt.title(f"{col} - Relative Weight per Frequency Bin")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Relative Weight (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()




def process_and_split_csv(file_path):
    print(f"\nReading: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=';')
        
        print("Loaded CSV:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Ensure consistent column order
        base_columns = ['day', 'timestamp', 'product']
        other_columns = [col for col in df.columns if col not in base_columns]
        ordered_columns = base_columns + other_columns
        df = df[ordered_columns]

        # Create dict of product -> sorted DataFrame
        product_dfs = {}
        for product in df['product'].unique():
            product_df = df[df['product'] == product].sort_values(by='timestamp').reset_index(drop=True)
            product_dfs[product] = product_df

        return product_dfs  # Dictionary with keys like 'KELP', 'SQUID_INK', etc.

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def walk_and_process(base_path):
    dataframes = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = process_and_split_csv(file_path)
                dataframes.append(df)

    return dataframes

if __name__ == "__main__":
    # Set your base directory here
    base_dir = "sample_data"
    dataframe_dicts = walk_and_process(base_dir)
    #vplot_product_data(dataframe_dicts[0])
    # compute_and_plot_fft(dataframe_dicts[0]["KELP"])  # Example: compute FFT for the first dataframe
    max_freq = 0.01
    window_size = int(0.5 * sampling_frequency/max_freq)
    print(f"Window size: {window_size}")
    # window_size = 10
    plot_moving_average(dataframe_dicts[0]["KELP"], window_size=window_size)  # Example: plot moving average for the first dataframe
    print(window_size)

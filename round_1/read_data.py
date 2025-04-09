import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
seconds = 24 * 3600
no_samples = 1e5
sampling_frequency = no_samples / seconds

def compute_and_plot_fft(df, fs=sampling_frequency, bin_width=0.05):
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




def process_csv(file_path):
    print(f"\nReading: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=';')

        print("Structure:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        return df
        # Plot each column ATTENTION THE PLOTTING IS PRETTY SUS
        for col in df.columns:
            plt.figure(figsize=(8, 4))
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.plot(df[col], label=col)
                plt.title(f"{col} (Line Plot)")
            else:
                df[col].value_counts().plot(kind='bar')
                plt.title(f"{col} (Bar Plot)")
            plt.xlabel("Index")
            plt.ylabel(col)
            plt.legend()
            plt.tight_layout()
            plt.show()
#
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def walk_and_process(base_path):
    dataframes = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = process_csv(file_path)
                dataframes.append(df)

    return dataframes

if __name__ == "__main__":
    # Set your base directory here
    base_dir = "sample_data"
    dataframes = walk_and_process(base_dir)
    compute_and_plot_fft(dataframes[0])  # Example: compute FFT for the first dataframe

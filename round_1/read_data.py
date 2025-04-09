import os
import pandas as pd
import matplotlib.pyplot as plt

# Set your base directory here
base_dir = "sample_data"

def process_csv(file_path):
    print(f"\nReading: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=)

        print("Structure:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Plot each column
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
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_csv(file_path)

if __name__ == "__main__":
    walk_and_process(base_dir)

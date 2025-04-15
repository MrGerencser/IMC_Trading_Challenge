import subprocess
import itertools
import re
import os
import shutil
import uuid
import statistics # Import statistics module

# --- Configuration ---
original_algo_file_path = "round_2/algo_market_making_and_arb.py" # Target the combined algo file
temp_file_dir = "round_2"
days_to_test = ["1--2", "2--1", "3-0", "3-1", "3-2"] # Adjust days as needed

# Define the parameter grid for the Basket Arbitrage strategy
param_grid = {
    'arb_ewma_span': [20, 40, 60, 80],
    'arb_entry_z_threshold': [1.5, 2.0, 2.5, 3.0, 3.5],
    'arb_exit_z_threshold': [0.2, 0.5, 0.8, 1.0, 1.5],
    'arb_stop_loss_z_threshold': [4.0, 6.0, 8.0, 10.0, 12.0]
}

results = []

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# --- Main Loop ---
for i, params in enumerate(param_combinations):
    print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}", flush=True)
    unique_id = uuid.uuid4()
    # Use a distinct temp file name pattern
    temp_algo_file_path = os.path.join(temp_file_dir, f"algo_market_making_and_arb_temp_{unique_id}.py")
    print(f"  -> Using temp file: {temp_algo_file_path}", flush=True)
    current_temp_file = None

    # 1. Modify a UNIQUE COPY of the algo file
    try:
        shutil.copyfile(original_algo_file_path, temp_algo_file_path)
        current_temp_file = temp_algo_file_path
        with open(temp_algo_file_path, 'r') as f:
            lines = f.readlines()
        modified_lines = []
        modified_count = 0
        for line in lines:
            new_line = line
            # Update parameter modification logic for Basket Arbitrage parameters
            if 'self.arb_ewma_span =' in line:
                new_line = f"        self.arb_ewma_span = {params['arb_ewma_span']} # Window for spread EWMA/EWMSD\n"
                if new_line != line: modified_count += 1
            elif 'self.arb_entry_z_threshold =' in line:
                 new_line = f"        self.arb_entry_z_threshold = {params['arb_entry_z_threshold']} # Z-score to enter\n"
                 if new_line != line: modified_count += 1
            elif 'self.arb_exit_z_threshold =' in line:
                 new_line = f"        self.arb_exit_z_threshold = {params['arb_exit_z_threshold']} # Z-score to exit (closer to 0)\n"
                 if new_line != line: modified_count += 1
            elif 'self.arb_stop_loss_z_threshold =' in line:
                 new_line = f"        self.arb_stop_loss_z_threshold = {params['arb_stop_loss_z_threshold']} # Z-score stop loss\n"
                 if new_line != line: modified_count += 1
            modified_lines.append(new_line)

        # Ensure the correct number of parameters were modified
        if modified_count != 4:
             print(f"  -> WARNING: Expected 4 parameter modifications, but only {modified_count} were made.", flush=True)

        with open(temp_algo_file_path, 'w') as f:
            f.writelines(modified_lines)
        print("  -> Algo file modified successfully.", flush=True)
    except Exception as e:
        print(f"  -> ERROR modifying file: {e}", flush=True)
        if current_temp_file and os.path.exists(current_temp_file):
            try: os.remove(current_temp_file)
            except OSError: pass
        continue

    # 2. Run backtester (remains the same)
    try:
        command = ["prosperity3bt", temp_algo_file_path] + days_to_test
        print(f"  -> Running command: {' '.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300) # 5 min timeout
        stdout = result.stdout
        print("  -> Backtester finished.", flush=True)
        # print("--- Backtester Output Start ---\n", stdout, "\n--- Backtester Output End ---", flush=True) # Optional debug

        # 3. Parse Final PnL (remains the same)
        final_pnl = None
        final_matches = re.findall(r"Total profit:\s*([+-]?[\d,]+(?:\.\d+)?)", stdout)
        if final_matches:
            try:
                final_pnl = float(final_matches[-1].replace(',', ''))
                print(f"  -> Final PnL Found: {final_pnl}", flush=True)
            except ValueError:
                print(f"  -> ERROR converting Final PnL string '{final_matches[-1]}' to float.", flush=True)
        else:
            print("  -> Final PnL string 'Total profit:' not found in output!", flush=True)

        # 4. Parse Daily PnLs from summary (remains the same)
        daily_pnls = []
        daily_matches = re.findall(r"Round \d+ day [-\d]+:\s*([+-]?[\d,]+(?:\.\d+)?)", stdout)
        if daily_matches:
            for pnl_str in daily_matches:
                try:
                    daily_pnls.append(float(pnl_str.replace(',', '')))
                except ValueError:
                    print(f"  -> ERROR converting Daily PnL string '{pnl_str}' to float.", flush=True)
                    daily_pnls = [] # Invalidate if any conversion fails
                    break
            if daily_pnls:
                 print(f"  -> Daily PnLs Found: {daily_pnls}", flush=True)
        else:
             print("  -> Daily PnL summary section not found or no matches.", flush=True)

        # 5. Calculate Sharpe-like Ratio (remains the same)
        sharpe_ratio = None
        if final_pnl is not None and daily_pnls and len(daily_pnls) > 1:
            try:
                std_dev = statistics.stdev(daily_pnls)
                print(f"  -> Daily PnL Std Dev: {std_dev:.2f}", flush=True)
                if std_dev > 1e-6:
                    sharpe_ratio = final_pnl / std_dev
                    print(f"  -> Sharpe-like Ratio: {sharpe_ratio:.4f}", flush=True)
                else:
                    sharpe_ratio = final_pnl * 1e6 # Arbitrarily large number if profitable, else negative
                    print(f"  -> Std Dev near zero, using scaled PnL for ratio: {sharpe_ratio:.4f}", flush=True)
            except statistics.StatisticsError as e:
                 print(f"  -> ERROR calculating standard deviation: {e}", flush=True)
        elif final_pnl is not None:
             print("  -> Not enough data points for std dev, cannot calculate Sharpe ratio.", flush=True)

        results.append({
            'params': params,
            'final_pnl': final_pnl,
            'daily_pnls': daily_pnls,
            'sharpe_ratio': sharpe_ratio
        })

    except subprocess.CalledProcessError as e:
        print(f"  -> ERROR running backtester: Exit code {e.returncode}", flush=True)
        print(f"  -> Stderr: {e.stderr}", flush=True)
    except subprocess.TimeoutExpired:
         print("  -> ERROR: Backtester timed out.", flush=True)
    except Exception as e:
        print(f"  -> ERROR during backtesting/parsing: {e}", flush=True)

    # Clean up temp file (remains the same)
    if current_temp_file and os.path.exists(current_temp_file):
        try:
            os.remove(current_temp_file)
        except OSError as e:
            print(f"  -> Warning: Could not remove temporary file {current_temp_file}: {e}", flush=True)


# --- Find Best (remains the same) ---
if results:
    valid_results = [r for r in results if r['sharpe_ratio'] is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['sharpe_ratio'])
        print("\n--- Best Result (based on Sharpe-like Ratio) ---", flush=True)
        print(f"Parameters: {best_result['params']}", flush=True)
        print(f"Sharpe-like Ratio: {best_result['sharpe_ratio']:.4f}", flush=True)
        print(f"Final PnL: {best_result['final_pnl']}", flush=True)
        print(f"Daily PnLs: {best_result['daily_pnls']}", flush=True)
    else:
        print("\nCould not calculate Sharpe ratio for any combination. Falling back to PnL.", flush=True)
        valid_pnl_results = [r for r in results if r['final_pnl'] is not None]
        if valid_pnl_results:
             best_result = max(valid_pnl_results, key=lambda x: x['final_pnl'])
             print("\n--- Best Result (based on Final PnL) ---", flush=True)
             print(f"Parameters: {best_result['params']}", flush=True)
             print(f"Final PnL: {best_result['final_pnl']}", flush=True)
        else:
             print("\nNo valid results found (all runs failed or PnL parsing failed).", flush=True)
else:
    print("\nNo results generated (check for errors during loop).", flush=True)

# Final cleanup (optional, could remove all temp files matching pattern)
print("\nOptimization finished.", flush=True)

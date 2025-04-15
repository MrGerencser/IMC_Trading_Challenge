import subprocess
import itertools
import re
import os
import shutil
import uuid
import statistics # Import statistics module

# --- Configuration ---
original_algo_file_path = "round_2/algo_market_making.py"
temp_file_dir = "round_2"
days_to_test = ["1--2", "2--1", "3-0", "3-1", "3-2"]

param_grid = {
    'squid_ewma_span': [10, 20, 30, 40, 50, 60],
    'squid_entry_threshold_pct_hv': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    'squid_exit_threshold_pct_hv': [0.2, 0.5, 0.8],
    'squid_stop_loss_pct_hv': [2.0, 2.5, 3.0]
}

results = []

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# --- Main Loop ---
for i, params in enumerate(param_combinations):
    print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}", flush=True)
    unique_id = uuid.uuid4()
    temp_algo_file_path = os.path.join(temp_file_dir, f"algo_market_making_temp_{unique_id}.py")
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
            if 'self.squid_ewma_span =' in line:
                new_line = f"        self.squid_ewma_span = {params['squid_ewma_span']} # Span for EWMA/EWMSD calculation (similar to old window)\n"
                if new_line != line: modified_count += 1
            elif 'self.squid_entry_threshold_pct_hv =' in line:
                 new_line = f"        self.squid_entry_threshold_pct_hv = {params['squid_entry_threshold_pct_hv']} # Enter if deviation > X * EWMSD\n"
                 if new_line != line: modified_count += 1
            elif 'self.squid_exit_threshold_pct_hv =' in line:
                 new_line = f"        self.squid_exit_threshold_pct_hv = {params['squid_exit_threshold_pct_hv']} # Exit if deviation < X * EWMSD (and profitable)\n"
                 if new_line != line: modified_count += 1
            elif 'self.squid_stop_loss_pct_hv =' in line:
                 new_line = f"        self.squid_stop_loss_pct_hv = {params['squid_stop_loss_pct_hv']} # Stop loss if loss exceeds X * EWMSD from entry\n"
                 if new_line != line: modified_count += 1
            modified_lines.append(new_line)
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

    # 2. Run backtester
    try:
        command = ["prosperity3bt", temp_algo_file_path] + days_to_test
        print(f"  -> Running command: {' '.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        stdout = result.stdout
        print("  -> Backtester finished.", flush=True)
        # print("--- Backtester Output Start ---\n", stdout, "\n--- Backtester Output End ---", flush=True) # Optional debug

        # 3. Parse Final PnL
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

        # 4. Parse Daily PnLs from summary
        daily_pnls = []
        # Regex to find lines like "Round X day Y: <pnl>" in the summary section
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

        # 5. Calculate Sharpe-like Ratio
        sharpe_ratio = None
        if final_pnl is not None and daily_pnls and len(daily_pnls) > 1:
            try:
                std_dev = statistics.stdev(daily_pnls)
                print(f"  -> Daily PnL Std Dev: {std_dev:.2f}", flush=True)
                # Avoid division by zero or near-zero std dev
                if std_dev > 1e-6:
                    sharpe_ratio = final_pnl / std_dev
                    print(f"  -> Sharpe-like Ratio: {sharpe_ratio:.4f}", flush=True)
                else:
                    # Handle zero std dev case (e.g., assign high score if PnL > 0, low if PnL <= 0)
                    sharpe_ratio = final_pnl * 1e6 # Arbitrarily large number if profitable, else negative
                    print(f"  -> Std Dev near zero, using scaled PnL for ratio: {sharpe_ratio:.4f}", flush=True)
            except statistics.StatisticsError as e:
                 print(f"  -> ERROR calculating standard deviation: {e}", flush=True)
        elif final_pnl is not None:
             # If only one day or std dev calculation failed, maybe just use PnL or a default low score
             print("  -> Not enough data points for std dev, cannot calculate Sharpe ratio.", flush=True)
             # sharpe_ratio = final_pnl # Option: Fallback to using PnL

        results.append({
            'params': params,
            'final_pnl': final_pnl,
            'daily_pnls': daily_pnls,
            'sharpe_ratio': sharpe_ratio # Store the new metric
        })

    except subprocess.CalledProcessError as e:
        print(f"  -> ERROR running backtester: Exit code {e.returncode}", flush=True)
        print(f"  -> Stderr: {e.stderr}", flush=True)
    except subprocess.TimeoutExpired:
         print("  -> ERROR: Backtester timed out.", flush=True)
    except Exception as e:
        print(f"  -> ERROR during backtesting/parsing: {e}", flush=True)

    # Clean up temp file
    if current_temp_file and os.path.exists(current_temp_file):
        try:
            os.remove(current_temp_file)
            # print(f"  -> Cleaned up temp file: {current_temp_file}", flush=True) # Less verbose
        except OSError as e:
            print(f"  -> Warning: Could not remove temporary file {current_temp_file}: {e}", flush=True)


# --- Find Best ---
if results:
    # Filter results where sharpe_ratio could be calculated
    valid_results = [r for r in results if r['sharpe_ratio'] is not None]
    if valid_results:
        # Find the result with the maximum sharpe_ratio
        best_result = max(valid_results, key=lambda x: x['sharpe_ratio'])
        print("\n--- Best Result (based on Sharpe-like Ratio) ---", flush=True)
        print(f"Parameters: {best_result['params']}", flush=True)
        print(f"Sharpe-like Ratio: {best_result['sharpe_ratio']:.4f}", flush=True)
        print(f"Final PnL: {best_result['final_pnl']}", flush=True)
        print(f"Daily PnLs: {best_result['daily_pnls']}", flush=True)
    else:
        # Fallback if no Sharpe ratio could be calculated
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

# Final cleanup (just in case)
# ...

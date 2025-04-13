import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For handling potential NaN values

# --- Configuration ---
# List of price data files to load
PRICE_DATA_FILES = [
    'sample_data/round-2-island-data-bottle/prices_round_2_day_-1.csv',
    'sample_data/round-2-island-data-bottle/prices_round_2_day_0.csv',
    'sample_data/round-2-island-data-bottle/prices_round_2_day_1.csv'
]

# Basket definitions (copied from your Trader class)
BASKET_COMPONENTS = {
    "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
}

# Minimum profit threshold (optional, for visualization)
MIN_ARBITRAGE_PROFIT = 2.0

# --- Load and Prepare Data ---
all_dfs = []
print("Loading data files...")
for file_path in PRICE_DATA_FILES:
    try:
        df_day = pd.read_csv(file_path, delimiter=';')
        # Keep only necessary columns early, including 'day'
        df_day = df_day[['day', 'timestamp', 'product', 'bid_price_1', 'ask_price_1']]
        all_dfs.append(df_day)
        print(f"  Loaded {file_path} successfully.")
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
    except Exception as e:
        print(f"Warning: Error loading {file_path}: {e}. Skipping.")

if not all_dfs:
    print("Error: No data files were loaded successfully. Exiting.")
    exit()

# Concatenate all loaded dataframes
df_prices = pd.concat(all_dfs, ignore_index=True)
print(f"\nCombined data loaded. Total rows: {len(df_prices)}")

# Pivot the table to have products as columns, using day and timestamp as index
print("Pivoting data...")
try:
    df_pivot = df_prices.pivot_table(index=['day', 'timestamp'], columns='product', values=['bid_price_1', 'ask_price_1'])
except Exception as e:
    print(f"Error during pivoting: {e}")
    # Check for duplicate entries for the same product at the same day/timestamp
    duplicates = df_prices[df_prices.duplicated(subset=['day', 'timestamp', 'product'], keep=False)]
    if not duplicates.empty:
        print("Found duplicate product entries at the same day/timestamp:")
        print(duplicates.sort_values(['day', 'timestamp', 'product']).head())
    exit()


# Flatten the multi-level column index (e.g., ('bid_price_1', 'CROISSANTS') -> 'CROISSANTS_bid')
df_pivot.columns = [f'{col[1]}_{col[0].replace("_price_1", "")}' for col in df_pivot.columns]

# Sort index for cleaner plotting and analysis
df_pivot.sort_index(inplace=True)
print("Data pivoted and sorted.")

# --- Calculate Implied Prices and Profits ---

results = {} # To store calculated data for plotting

for basket_name, components in BASKET_COMPONENTS.items():
    print(f"\nAnalyzing {basket_name}...")
    results[basket_name] = pd.DataFrame(index=df_pivot.index)

    # Get actual basket prices
    basket_bid_col = f'{basket_name}_bid'
    basket_ask_col = f'{basket_name}_ask'
    if basket_bid_col not in df_pivot.columns or basket_ask_col not in df_pivot.columns:
        print(f"Warning: Price data missing for {basket_name} in the combined dataset. Skipping.")
        continue

    results[basket_name]['actual_bid'] = df_pivot[basket_bid_col]
    results[basket_name]['actual_ask'] = df_pivot[basket_ask_col]

    # Calculate implied prices based on components
    implied_cost_to_buy_components = pd.Series(0.0, index=df_pivot.index)
    implied_value_to_sell_components = pd.Series(0.0, index=df_pivot.index)
    components_available = True

    for component, quantity in components.items():
        comp_bid_col = f'{component}_bid'
        comp_ask_col = f'{component}_ask'

        if comp_bid_col not in df_pivot.columns or comp_ask_col not in df_pivot.columns:
            print(f"Warning: Price data missing for component {component}. Cannot calculate implied prices accurately for {basket_name}.")
            components_available = False
            break # Stop calculating for this basket if a component is missing

        # Cost to BUY components uses component ASK prices
        # Use .fillna(0) temporarily if a component price is missing at a specific timestamp,
        # although this might skew results slightly. A better approach might be to drop rows with NaNs later.
        implied_cost_to_buy_components += df_pivot[comp_ask_col].fillna(np.inf) * quantity # Use inf to make cost invalid if ask is missing
        # Value to SELL components uses component BID prices
        implied_value_to_sell_components += df_pivot[comp_bid_col].fillna(0) * quantity # Use 0 to make value invalid if bid is missing

    if not components_available:
        results[basket_name]['implied_cost_buy_comps'] = np.nan
        results[basket_name]['implied_value_sell_comps'] = np.nan
        results[basket_name]['profit_strat1'] = np.nan # Buy Basket, Sell Components
        results[basket_name]['profit_strat2'] = np.nan # Sell Basket, Buy Components
        continue

    # Replace inf placeholders with NaN for proper calculations/plotting
    implied_cost_to_buy_components.replace([np.inf, -np.inf], np.nan, inplace=True)

    results[basket_name]['implied_cost_buy_comps'] = implied_cost_to_buy_components
    results[basket_name]['implied_value_sell_comps'] = implied_value_to_sell_components

    # --- Calculate Arbitrage Profits ---
    # Strategy 1: Buy Basket (at ask), Sell Components (at bid)
    # Profit = Value from Selling Components - Cost of Buying Basket
    results[basket_name]['profit_strat1'] = results[basket_name]['implied_value_sell_comps'] - results[basket_name]['actual_ask']

    # Strategy 2: Sell Basket (at bid), Buy Components (at ask)
    # Profit = Value from Selling Basket - Cost of Buying Components
    results[basket_name]['profit_strat2'] = results[basket_name]['actual_bid'] - results[basket_name]['implied_cost_buy_comps']

    # Drop rows where profit calculation resulted in NaN (due to missing prices)
    results[basket_name].dropna(subset=['profit_strat1', 'profit_strat2'], inplace=True)


    # Print some summary stats for the combined data
    print(f"  --- Combined Stats (Days -1, 0, 1) ---")
    if not results[basket_name].empty:
        print(f"  Average Actual Basket Price (Mid): {((results[basket_name]['actual_bid'] + results[basket_name]['actual_ask']) / 2).mean():.2f}")
        print(f"  Average Implied Basket Price (Mid): {((results[basket_name]['implied_cost_buy_comps'] + results[basket_name]['implied_value_sell_comps']) / 2).mean():.2f}")
        print(f"  Average Profit (Strat 1 - Buy Basket, Sell Comps): {results[basket_name]['profit_strat1'].mean():.2f}")
        print(f"  Average Profit (Strat 2 - Sell Basket, Buy Comps): {results[basket_name]['profit_strat2'].mean():.2f}")
        print(f"  Max Profit (Strat 1): {results[basket_name]['profit_strat1'].max():.2f}")
        print(f"  Max Profit (Strat 2): {results[basket_name]['profit_strat2'].max():.2f}")
        print(f"  Timestamps Analyzed: {len(results[basket_name])}")
    else:
        print("  No valid arbitrage opportunities found after handling missing data.")


# --- Plotting ---
valid_results = {k: v for k, v in results.items() if not v.empty}
num_baskets = len(valid_results)

if num_baskets == 0:
    print("\nNo valid data to plot after processing.")
    exit()

print("\nGenerating plots...")
fig, axes = plt.subplots(num_baskets, 2, figsize=(18, 6 * num_baskets), sharex=True)
fig.suptitle('Basket Arbitrage Analysis (Days -1, 0, 1)', fontsize=16)

# Ensure axes is always a 2D array for consistent indexing
if num_baskets == 1:
    axes = np.array([axes])

plot_idx = 0
for basket_name, df_result in valid_results.items():

    # Reset index to plot against a simple numerical index, but keep day/timestamp
    df_result_plot = df_result.reset_index()

    ax1 = axes[plot_idx, 0]
    ax2 = axes[plot_idx, 1]

    # --- Plot 1: Prices ---
    ax1.plot(df_result_plot.index, df_result_plot['actual_ask'], label='Actual Basket Ask', color='red', alpha=0.7, linewidth=1)
    ax1.plot(df_result_plot.index, df_result_plot['implied_cost_buy_comps'], label='Implied Cost (Buy Comps)', color='darkred', linestyle='--', linewidth=1)

    ax1.plot(df_result_plot.index, df_result_plot['actual_bid'], label='Actual Basket Bid', color='green', alpha=0.7, linewidth=1)
    ax1.plot(df_result_plot.index, df_result_plot['implied_value_sell_comps'], label='Implied Value (Sell Comps)', color='darkgreen', linestyle='--', linewidth=1)

    ax1.set_title(f'{basket_name} - Actual vs Implied Prices')
    ax1.set_ylabel('Price')
    ax1.legend(fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Profits ---
    ax2.plot(df_result_plot.index, df_result_plot['profit_strat1'], label='Profit: Buy Basket, Sell Comps', color='blue', linewidth=1)
    ax2.plot(df_result_plot.index, df_result_plot['profit_strat2'], label='Profit: Sell Basket, Buy Comps', color='orange', linewidth=1)

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(MIN_ARBITRAGE_PROFIT, color='gray', linestyle='--', linewidth=0.8, label=f'Min Profit ({MIN_ARBITRAGE_PROFIT})')
    ax2.axhline(-MIN_ARBITRAGE_PROFIT, color='gray', linestyle='--', linewidth=0.8)

    ax2.set_title(f'{basket_name} - Potential Arbitrage Profit per Basket')
    ax2.set_ylabel('Profit')
    ax2.legend(fontsize='small')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- Add Day Separators ---
    day_changes = df_result_plot[df_result_plot['day'].diff() != 0].index
    unique_days = df_result_plot['day'].unique()
    day_labels = {} # Store position for day labels

    for i, change_idx in enumerate(day_changes):
         if change_idx != 0: # Don't draw line at the very beginning
            ax1.axvline(change_idx - 0.5, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
            ax2.axvline(change_idx - 0.5, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
            # Position for label: midpoint between this change and the previous one (or start)
            prev_idx = day_changes[i-1] if i > 0 else 0
            label_pos = prev_idx + (change_idx - prev_idx) / 2
            day_labels[label_pos] = f"Day {unique_days[i-1]}"

    # Add label for the last day segment
    last_change_idx = day_changes[-1] if len(day_changes)>0 else 0
    label_pos = last_change_idx + (len(df_result_plot) - last_change_idx) / 2
    day_labels[label_pos] = f"Day {unique_days[-1]}"

    # Add day labels below the plot (on the last row)
    if plot_idx == num_baskets - 1:
        for pos, label in day_labels.items():
             ax1.text(pos, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, label, ha='center', va='top', fontsize=9, color='dimgray')
             ax2.text(pos, ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.1, label, ha='center', va='top', fontsize=9, color='dimgray')


    plot_idx += 1

# Common X axis label and formatting
for i in range(num_baskets):
    axes[i, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=(i == num_baskets - 1))
    axes[i, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=(i == num_baskets - 1))
    if i == num_baskets - 1:
        axes[i, 0].set_xlabel('Time Index (across days)')
        axes[i, 1].set_xlabel('Time Index (across days)')
        # Remove numeric labels from x-axis as they are less meaningful now
        axes[i, 0].set_xticklabels([])
        axes[i, 1].set_xticklabels([])


plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
print("Displaying plot...")
plt.show()

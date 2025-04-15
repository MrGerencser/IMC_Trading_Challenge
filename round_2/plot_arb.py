import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # For handling potential NaN values
import os # Added for path joining

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
        # Ensure 'day' column exists or extract it if needed
        if 'day' not in df_day.columns:
             # Attempt to extract day from filename if not present
             try:
                 day_str = file_path.split('_day_')[1].split('.')[0]
                 df_day['day'] = int(day_str)
                 print(f"  Extracted day {df_day['day'].iloc[0]} from filename for {file_path}")
             except Exception:
                 print(f"Warning: Could not determine 'day' for {file_path}. Skipping.")
                 continue

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
        results.pop(basket_name) # Remove entry if basket data is missing
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
        implied_cost_to_buy_components += df_pivot[comp_ask_col].fillna(np.inf) * quantity # Use inf to make cost invalid if ask is missing
        # Value to SELL components uses component BID prices
        implied_value_to_sell_components += df_pivot[comp_bid_col].fillna(0) * quantity # Use 0 to make value invalid if bid is missing

    if not components_available:
        print(f"Skipping {basket_name} due to missing component data.")
        results.pop(basket_name) # Remove entry if components are missing
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
    results[basket_name].dropna(subset=['profit_strat1', 'profit_strat2'], how='all', inplace=True) # Drop if BOTH are NaN


    # Print some summary stats for the combined data
    print(f"  --- Combined Stats (Days {', '.join(map(str, sorted(df_prices['day'].unique())))}) ---")
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


# --- Plotting with Plotly ---
valid_results = {k: v for k, v in results.items() if not v.empty}
num_baskets = len(valid_results)

if num_baskets == 0:
    print("\nNo valid data to plot after processing.")
    exit()

print("\nGenerating Plotly plots...")
# Create subplots: one row per basket, two columns (Prices, Profits)
fig = make_subplots(
    rows=num_baskets, cols=2,
    shared_xaxes=True,
    subplot_titles=[f"{name} - Prices" if i % 2 == 0 else f"{name} - Profits"
                    for name in valid_results.keys() for i in range(2)],
    vertical_spacing=0.1 / num_baskets if num_baskets > 1 else 0.1 # Adjust spacing based on number of rows
)

plot_row = 1 # Plotly rows are 1-indexed
for basket_name, df_result in valid_results.items():

    # Reset index to get a continuous numerical index for plotting, keep day/timestamp for hover
    df_result_plot = df_result.reset_index()
    # Create a combined hover text column
    df_result_plot['hover_text'] = 'Day: ' + df_result_plot['day'].astype(str) + '<br>Timestamp: ' + df_result_plot['timestamp'].astype(str)

    # --- Plot 1: Prices (Column 1) ---
    # Actual Ask
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['actual_ask'],
        mode='lines', name='Actual Basket Ask', legendgroup=f'{basket_name}_price',
        line=dict(color='red', width=1), opacity=0.7,
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Actual Ask: %{y:.2f}<extra></extra>'
    ), row=plot_row, col=1)
    # Implied Cost (Buy Comps)
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['implied_cost_buy_comps'],
        mode='lines', name='Implied Cost (Buy Comps)', legendgroup=f'{basket_name}_price',
        line=dict(color='darkred', dash='dash', width=1),
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Implied Cost: %{y:.2f}<extra></extra>'
    ), row=plot_row, col=1)
    # Actual Bid
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['actual_bid'],
        mode='lines', name='Actual Basket Bid', legendgroup=f'{basket_name}_price',
        line=dict(color='green', width=1), opacity=0.7,
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Actual Bid: %{y:.2f}<extra></extra>'
    ), row=plot_row, col=1)
    # Implied Value (Sell Comps)
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['implied_value_sell_comps'],
        mode='lines', name='Implied Value (Sell Comps)', legendgroup=f'{basket_name}_price',
        line=dict(color='darkgreen', dash='dash', width=1),
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Implied Value: %{y:.2f}<extra></extra>'
    ), row=plot_row, col=1)

    # --- Plot 2: Profits (Column 2) ---
    # Profit Strat 1
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['profit_strat1'],
        mode='lines', name='Profit: Buy Basket, Sell Comps', legendgroup=f'{basket_name}_profit',
        line=dict(color='blue', width=1),
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Profit (Strat 1): %{y:.2f}<extra></extra>'
    ), row=plot_row, col=2)
    # Profit Strat 2
    fig.add_trace(go.Scatter(
        x=df_result_plot.index, y=df_result_plot['profit_strat2'],
        mode='lines', name='Profit: Sell Basket, Buy Comps', legendgroup=f'{basket_name}_profit',
        line=dict(color='orange', width=1),
        customdata=df_result_plot['hover_text'],
        hovertemplate='%{customdata}<br>Profit (Strat 2): %{y:.2f}<extra></extra>'
    ), row=plot_row, col=2)

    # Add horizontal lines for profit thresholds
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='solid'), row=plot_row, col=2)
    fig.add_hline(y=MIN_ARBITRAGE_PROFIT, line=dict(color='grey', width=1, dash='dash'), row=plot_row, col=2, annotation_text=f"Min Profit ({MIN_ARBITRAGE_PROFIT})", annotation_position="bottom right")
    fig.add_hline(y=-MIN_ARBITRAGE_PROFIT, line=dict(color='grey', width=1, dash='dash'), row=plot_row, col=2)


    # --- Add Day Separators and Labels ---
    day_changes = df_result_plot[df_result_plot['day'].diff() != 0].index
    unique_days = df_result_plot['day'].unique()
    day_labels = {} # Store position and text for day labels

    # Calculate positions for vertical lines and labels
    last_idx = 0
    for i, change_idx in enumerate(day_changes):
        if change_idx != 0:
            # Add vertical line
            fig.add_vline(x=change_idx - 0.5, line=dict(color='grey', dash='dot', width=1.5), opacity=0.8, row=plot_row, col=1)
            fig.add_vline(x=change_idx - 0.5, line=dict(color='grey', dash='dot', width=1.5), opacity=0.8, row=plot_row, col=2)
            # Calculate label position (midpoint of the previous segment)
            label_pos = last_idx + (change_idx - last_idx) / 2
            day_labels[label_pos] = f"Day {unique_days[i-1]}"
            last_idx = change_idx

    # Add label for the last day segment
    label_pos = last_idx + (len(df_result_plot) - last_idx) / 2
    day_labels[label_pos] = f"Day {unique_days[-1]}"

    # Add day labels as annotations below the x-axis (only for the last row)
    if plot_row == num_baskets:
        for pos, label in day_labels.items():
            fig.add_annotation(
                x=pos, y=0, # Y position relative to axis
                xref=f"x{plot_row*2-1}", yref=f"y{plot_row*2-1} domain", # Refer to x-axis of the last row, y relative to domain
                text=label, showarrow=False,
                yshift=-30, # Shift label down
                font=dict(size=10, color="dimgray")
            )
            fig.add_annotation(
                x=pos, y=0,
                xref=f"x{plot_row*2}", yref=f"y{plot_row*2} domain",
                text=label, showarrow=False,
                yshift=-30,
                font=dict(size=10, color="dimgray")
            )

    plot_row += 1

# --- Update Layout ---
fig.update_layout(
    title_text=f'Basket Arbitrage Analysis (Days {", ".join(map(str, sorted(df_prices["day"].unique())))})',
    height=350 * num_baskets + 100, # Adjust height based on number of baskets
    hovermode='x unified', # Show hover for all traces at a timestamp
    legend_tracegroupgap=180, # Add gap between legend groups
    margin=dict(t=100, b=80) # Add top/bottom margin for title/labels
)

# Update axes properties
fig.update_xaxes(
    showticklabels=False, # Hide numerical x-axis labels
    showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot'
)
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot'
)

# Set y-axis titles (iterate through all potential axes)
for i in range(1, num_baskets + 1):
    fig.update_yaxes(title_text="Price", row=i, col=1)
    fig.update_yaxes(title_text="Profit", row=i, col=2)

# Set x-axis title only for the bottom-most plots
fig.update_xaxes(title_text="Time Index (across days)", row=num_baskets, col=1)
fig.update_xaxes(title_text="Time Index (across days)", row=num_baskets, col=2)


print("Displaying plot...")
fig.show()

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Configuration ---
# Set the directory where your CSV files are located.
# Use '.' if the script is in the same directory as the data files.
# Note: The paths in PRICE_DATA_FILES and TRADES_DATA_FILES are relative to this script's location.
DATA_DIR = '.' # Keep as '.' if paths below are relative to the script

# List of price data files to load
PRICE_DATA_FILES = [
    'sample_data/round-3-island-data-bottle/prices_round_3_day_0.csv',
    'sample_data/round-3-island-data-bottle/prices_round_3_day_1.csv',
    'sample_data/round-3-island-data-bottle/prices_round_3_day_2.csv'
]

TRADES_DATA_FILES = [
    'sample_data/round-3-island-data-bottle/trades_round_3_day_0.csv',
    'sample_data/round-3-island-data-bottle/trades_round_3_day_1.csv',
    'sample_data/round-3-island-data-bottle/trades_round_3_day_2.csv'
]

# Optional: Specify a User ID from the trades file to highlight their trades
# Look inside your trades files for buyer/seller names like 'Rhianna', 'Vinnie', etc.
# Set to None if you don't want to highlight any specific user.
USER_ID_TO_HIGHLIGHT = 'Rhianna' # Example, change if needed or set to None

# --- Load Data ---
def load_all_data(price_files, trade_files, base_dir='.'):
    """Loads and concatenates price and trade data from multiple CSV files."""
    all_prices = []
    all_trades = []
    global_timestamp_offset = 0 # To handle timestamps across multiple days

    if len(price_files) != len(trade_files):
        print("Error: The number of price files and trade files must be the same.")
        return None, None

    max_timestamp_prev_day = 0

    for prices_path_rel, trades_path_rel in zip(price_files, trade_files):
        prices_path_abs = os.path.join(base_dir, prices_path_rel)
        trades_path_abs = os.path.join(base_dir, trades_path_rel)
        print(f"\n--- Loading data from: ---")
        print(f"Prices: {prices_path_abs}")
        print(f"Trades: {trades_path_abs}")

        try:
            # Read prices, handling potential comments and semicolon delimiter
            prices_df = pd.read_csv(prices_path_abs, delimiter=';', comment='/')
            print(f"Loaded prices data: {prices_df.shape[0]} rows")

            # Read trades, handling potential comments and semicolon delimiter
            trades_df = pd.read_csv(trades_path_abs, delimiter=';', comment='/')
            print(f"Loaded trades data: {trades_df.shape[0]} rows")

            # --- Basic Data Cleaning ---
            prices_df.columns = prices_df.columns.str.strip()
            trades_df.columns = trades_df.columns.str.strip()

            # Add day identifier based on filename (optional but helpful)
            try:
                day = int(prices_path_rel.split('_day_')[1].split('.')[0])
                prices_df['day'] = day
                trades_df['day'] = day
            except Exception:
                print("Warning: Could not parse day number from filename.")
                prices_df['day'] = -1 # Assign a default day if parsing fails
                trades_df['day'] = -1


            # Ensure numeric types
            numeric_cols_prices = ['bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', 'mid_price']
            for col in numeric_cols_prices:
                if col in prices_df.columns:
                    prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')

            numeric_cols_trades = ['price', 'quantity']
            for col in numeric_cols_trades:
                 if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

            # Ensure timestamp is integer and adjust for multi-day plotting
            prices_df['original_timestamp'] = prices_df['timestamp'].astype(int)
            trades_df['original_timestamp'] = trades_df['timestamp'].astype(int)

            # Adjust timestamp to be continuous across days
            # Assumes each day starts at timestamp 0 and has 1,000,000 steps (adjust if needed)
            # Find max timestamp of the current day's prices to calculate offset for the next day
            current_max_ts = prices_df['original_timestamp'].max() if not prices_df.empty else 0

            prices_df['timestamp'] = prices_df['original_timestamp'] + global_timestamp_offset
            trades_df['timestamp'] = trades_df['original_timestamp'] + global_timestamp_offset

            # Update the offset for the next file
            # Add a small buffer (e.g., 100) just in case timestamps aren't perfectly aligned at 1M
            global_timestamp_offset += (current_max_ts + 100)


            # Drop rows with NaNs in essential columns
            prices_df.dropna(subset=['timestamp', 'mid_price', 'bid_price_1', 'ask_price_1'], inplace=True)
            trades_df.dropna(subset=['timestamp', 'price', 'quantity'], inplace=True)

            all_prices.append(prices_df)
            all_trades.append(trades_df)

        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            print(f"Please ensure the file exists at the specified path.")
            # Decide if you want to stop or continue if one file is missing
            # return None, None # Stop if any file is missing
            print("Skipping this day's data.")
            continue # Continue with the next day
        except Exception as e:
            print(f"An unexpected error occurred during data loading for {prices_path_rel}/{trades_path_rel}: {e}")
            # return None, None # Stop on any error
            print("Skipping this day's data.")
            continue # Continue with the next day


    if not all_prices or not all_trades:
        print("Error: No data could be loaded successfully.")
        return None, None

    # Concatenate all loaded dataframes
    final_prices_df = pd.concat(all_prices, ignore_index=True)
    final_trades_df = pd.concat(all_trades, ignore_index=True)

    # Sort final dataframes by the adjusted timestamp
    final_prices_df.sort_values('timestamp', inplace=True)
    final_trades_df.sort_values('timestamp', inplace=True)

    print(f"\nSuccessfully loaded and concatenated data from {len(all_prices)} day(s).")
    print(f"Total price rows: {final_prices_df.shape[0]}")
    print(f"Total trade rows: {final_trades_df.shape[0]}")

    return final_prices_df, final_trades_df


# --- Plotting Function ---
# (Keep the plot_product_prices_and_trades function as it is, it works on the combined dataframe)
def plot_product_prices_and_trades(product_symbol, prices_df, trades_df, user_id=None):
    """
    Generates an interactive Plotly chart showing prices and trades for a product.
    (Assumes prices_df and trades_df contain data potentially from multiple days
     with adjusted continuous timestamps).

    Args:
        product_symbol (str): The product symbol (e.g., 'CROISSANTS').
        prices_df (pd.DataFrame): DataFrame containing price data.
        trades_df (pd.DataFrame): DataFrame containing trade data.
        user_id (str, optional): User ID to highlight in trades. Defaults to None.
    """
    filtered_prices = prices_df[prices_df['product'] == product_symbol].copy()
    filtered_trades = trades_df[trades_df['symbol'] == product_symbol].copy()

    if filtered_prices.empty:
        print(f"No price data found for {product_symbol}. Skipping plot.")
        return

    # Create figure with secondary y-axis if needed (e.g., for volume)
    # For now, keep it simple with one y-axis for price.
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # --- Add Price Traces ---
    # Mid Price
    fig.add_trace(
        go.Scatter(
            x=filtered_prices['timestamp'],
            y=filtered_prices['mid_price'],
            mode='lines',
            name='Mid Price',
            line=dict(color='royalblue', width=2),
            customdata=filtered_prices[['day', 'original_timestamp']],
             hovertemplate=(
                '<b>Mid Price</b><br>'
                'Day: %{customdata[0]} | Timestamp: %{customdata[1]} (%{x})<br>'
                'Price: %{y}<extra></extra>'
            )
        ),
        secondary_y=False,
    )

    # Best Bid
    fig.add_trace(
        go.Scatter(
            x=filtered_prices['timestamp'],
            y=filtered_prices['bid_price_1'],
            mode='lines',
            name='Best Bid',
            line=dict(color='green', width=1, dash='dot'),
            customdata=filtered_prices[['day', 'original_timestamp']],
             hovertemplate=(
                '<b>Best Bid</b><br>'
                'Day: %{customdata[0]} | Timestamp: %{customdata[1]} (%{x})<br>'
                'Price: %{y}<extra></extra>'
            )
        ),
        secondary_y=False,
    )

    # Best Ask
    fig.add_trace(
        go.Scatter(
            x=filtered_prices['timestamp'],
            y=filtered_prices['ask_price_1'],
            mode='lines',
            name='Best Ask',
            line=dict(color='red', width=1, dash='dot'),
            customdata=filtered_prices[['day', 'original_timestamp']],
             hovertemplate=(
                '<b>Best Ask</b><br>'
                'Day: %{customdata[0]} | Timestamp: %{customdata[1]} (%{x})<br>'
                'Price: %{y}<extra></extra>'
            )
        ),
        secondary_y=False,
    )

    # --- Add Trade Traces ---
    if not filtered_trades.empty:
        # Separate trades for clarity
        market_trades = filtered_trades.copy()
        user_buys = pd.DataFrame()
        user_sells = pd.DataFrame()

        if user_id:
            user_buys = filtered_trades[filtered_trades['buyer'] == user_id]
            user_sells = filtered_trades[filtered_trades['seller'] == user_id]
            # Filter market trades to exclude user trades if highlighting
            market_trades = filtered_trades[
                (filtered_trades['buyer'] != user_id) & (filtered_trades['seller'] != user_id)
            ]

        # Plot Market Trades
        if not market_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=market_trades['timestamp'],
                    y=market_trades['price'],
                    mode='markers',
                    name='Market Trades',
                    marker=dict(color='grey', size=5, symbol='circle', opacity=0.7),
                    customdata=market_trades[['quantity', 'buyer', 'seller', 'day', 'original_timestamp']],
                    hovertemplate=(
                        '<b>Market Trade</b><br>'
                        'Day: %{customdata[3]} | Timestamp: %{customdata[4]} (%{x})<br>'
                        'Price: %{y}<br>'
                        'Qty: %{customdata[0]}<br>'
                        'Buyer: %{customdata[1]}<br>'
                        'Seller: %{customdata[2]}<extra></extra>' # <extra></extra> hides the trace name from hover
                    )
                ),
                secondary_y=False,
            )

        # Plot User Buys (if user_id provided and trades exist)
        if not user_buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=user_buys['timestamp'],
                    y=user_buys['price'],
                    mode='markers',
                    name=f'{user_id} Buys',
                    marker=dict(color='lime', size=8, symbol='triangle-up', line=dict(color='black', width=1)),
                    customdata=user_buys[['quantity', 'seller', 'day', 'original_timestamp']],
                    hovertemplate=(
                        f'<b>{user_id} Buy</b><br>'
                        'Day: %{customdata[2]} | Timestamp: %{customdata[3]} (%{x})<br>'
                        'Price: %{y}<br>'
                        'Qty: %{customdata[0]}<br>'
                        'From Seller: %{customdata[1]}<extra></extra>'
                    )
                ),
                secondary_y=False,
            )

        # Plot User Sells (if user_id provided and trades exist)
        if not user_sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=user_sells['timestamp'],
                    y=user_sells['price'],
                    mode='markers',
                    name=f'{user_id} Sells',
                    marker=dict(color='tomato', size=8, symbol='triangle-down', line=dict(color='black', width=1)),
                    customdata=user_sells[['quantity', 'buyer', 'day', 'original_timestamp']],
                    hovertemplate=(
                        f'<b>{user_id} Sell</b><br>'
                        'Day: %{customdata[2]} | Timestamp: %{customdata[3]} (%{x})<br>'
                        'Price: %{y}<br>'
                        'Qty: %{customdata[0]}<br>' # Quantity is positive in Trade object
                        'To Buyer: %{customdata[1]}<extra></extra>'
                    )
                ),
                secondary_y=False,
            )

    # --- Update Layout ---
    fig.update_layout(
        title=f'Price and Trades for {product_symbol} (Across {len(PRICE_DATA_FILES)} Days)',
        xaxis_title='Combined Timestamp (Continuous Across Days)',
        yaxis_title='Price',
        hovermode='x unified',  # Show hover info for all traces at a given x (timestamp)
        legend_title_text='Trace Type',
        template='plotly_white' # Use a clean template
    )

    # Add range slider for better navigation
    fig.update_xaxes(rangeslider_visible=True)

    # Show the plot
    fig.show()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting multi-day plotting script...")

    # Load and concatenate data from all specified files
    prices_df, trades_df = load_all_data(PRICE_DATA_FILES, TRADES_DATA_FILES, base_dir=DATA_DIR)

    if prices_df is not None and trades_df is not None:
        # Get unique products from the combined price data
        available_products = sorted(prices_df['product'].unique())
        print("\nAvailable products found across all days:")
        print(", ".join(available_products))

        # --- Select which products to plot ---
        # Option 1: Plot all available products
        # products_to_plot = available_products

        # Option 2: Plot only specific products of interest
        # Example: products_to_plot = ['AMETHYSTS', 'STARFRUIT']
        products_to_plot = available_products # Defaulting to plot all found products

        # Filter list to only include products actually available in the data (redundant if using available_products)
        # products_to_plot = [p for p in products_to_plot if p in available_products]

        print(f"\nWill generate plots for: {', '.join(products_to_plot)}")

        # Generate plot for each selected product using the combined data
        for product in products_to_plot:
            print(f"\nGenerating plot for {product}...")
            plot_product_prices_and_trades(product, prices_df, trades_df, user_id=USER_ID_TO_HIGHLIGHT)

        print("\nPlotting finished.")
    else:
        print("\nExiting due to data loading errors.")

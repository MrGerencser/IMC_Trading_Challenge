import json
# collections and statistics are no longer needed for this simplified strategy
# import collections
# import statistics
from typing import Any, Dict, List # Deque is no longer needed

# Assuming datamodel.py defines these classes:
# Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    # ... (Logger class remains unchanged, including flush and compression methods) ...
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjusted based on typical limits

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This flush method is provided for context and remains unchanged.
        # It handles compressing the state and logs to fit within the output limits.
        output = [
            self.compress_state(state, trader_data),
            self.compress_orders(orders),
            conversions,
        ]
        output_json = self.to_json(output)

        # Check if the output is too long
        if len(output_json) > self.max_log_length:
            # Calculate the base length without logs and trader data
            base_output = [
                self.compress_state(state, ""), # Empty trader data for length calculation
                self.compress_orders(orders),
                conversions,
            ]
            base_length = len(self.to_json(base_output))
            # Add length for keys and structure "traderData": "", "logs": ""
            base_length += 24 # Approximate length for keys and quotes

            # Available length for trader_data and logs
            available_length = self.max_log_length - base_length
            max_item_length = available_length // 2 # Divide remaining space between trader_data and logs

            if max_item_length < 0: max_item_length = 0 # Ensure non-negative length

            # Truncate trader_data and logs
            truncated_trader_data = self.truncate(trader_data, max_item_length)
            truncated_logs = self.truncate(self.logs, max_item_length)

            # Recompress state with truncated trader_data
            final_output = [
                self.compress_state(state, truncated_trader_data),
                self.compress_orders(orders),
                conversions,
                truncated_logs, # Add truncated logs separately for final JSON
            ]
            print(self.to_json(final_output))

        else:
             # Original output fits, add logs to the end
             final_output = [
                self.compress_state(state, trader_data),
                self.compress_orders(orders),
                conversions,
                self.logs, # Add logs here
            ]
             print(self.to_json(final_output))


        self.logs = "" # Clear logs after flushing

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Helper for flush - unchanged
        return [
            state.timestamp,
            trader_data, # Use potentially truncated trader_data
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        # Helper for flush - unchanged
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        # Helper for flush - unchanged
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        # Helper for flush - unchanged
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        # Helper for flush - unchanged
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        # Helper for flush - unchanged
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        # Helper for flush - unchanged
        # Use dumps with ProsperityEncoder for custom objects if needed, ensure separators for compactness
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))


    def truncate(self, value: str, max_length: int) -> str:
        # Helper for flush - unchanged
        if len(value) <= max_length:
            return value
        # Ensure max_length is not too small for ellipsis
        if max_length < 3:
            return value[:max_length]
        return value[: max_length - 3] + "..."

logger = Logger()

# Removed SpreadTradingState class as it's related to the removed Stat Arb strategy

class Trader:
    def __init__(self):
        # Define position limits for each product - Keep relevant ones
        self.position_limits = {
            "PICNIC_BASKET1": 60, # Kept for context, but not used in logic
            "PICNIC_BASKET2": 100, # Kept for context, but not used in logic
            "CROISSANTS": 250,    # Kept for context, but not used in logic
            "JAMS": 350,          # Kept for context, but not used in logic
            "RAINFOREST_RESIN": 50, # Used by MM strategy
            "KELP": 50,           # Kept for context, but not used in logic
            "SQUID_INK": 50,      # Kept for context, but not used in logic
            "DJEMBES": 60         # Kept for context, but not used in logic
        }

        # Removed basket_components as it's related to the removed Stat Arb strategy

        # Track previous positions for logging changes
        self.previous_positions = {}

        # Removed Stat Arb Strategy Parameters

        # --- MM Strategy Parameters ---
        self.mm_spread_threshold = 3
        self.mm_base_volume = 10 # Base volume for RAINFOREST_RESIN MM

    # Removed _deserialize_trader_data as no complex state is needed
    # Removed _serialize_trader_data as no complex state is needed

    # Helper function to check position limits before placing an order - Kept as it's used by MM
    def _can_place_order(self, symbol: Symbol, quantity: int, current_pos: int, pending_pos_delta: Dict[Symbol, int]) -> bool:
        limit = self.position_limits.get(symbol)
        if limit is None:
            logger.print(f"Warning: No position limit found for {symbol}. Order rejected.")
            return False

        pending_delta = pending_pos_delta.get(symbol, 0)
        final_pos = current_pos + pending_delta + quantity

        if abs(final_pos) <= limit:
            return True
        else:
            # logger.print(f"Order rejected for {symbol}: Quantity {quantity} would exceed limit {limit}. Current: {current_pos}, Pending Delta: {pending_delta}, Resulting: {final_pos}")
            return False

    # Removed _calculate_basket_component_metrics as it's related to Stat Arb
    # Removed _calculate_rolling_stats as it's related to Stat Arb

    # Standard market making logic (with inventory skewing) - Kept as it's the core strategy now
    def get_market_making_orders(self, symbol: Symbol, depth: OrderDepth, position: int) -> list[Order]:
        """
        Get market making orders for a single symbol, respecting position limits
        and skewing quotes based on inventory.
        Uses self.mm_spread_threshold and self.mm_base_volume.
        """
        orders = []
        position_limit = self.position_limits.get(symbol)
        if position_limit is None:
             logger.print(f"Warning: No position limit found for {symbol}. Skipping MM.")
             return [] # Skip MM if limit not defined

        best_bid_price = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask_price = min(depth.sell_orders.keys()) if depth.sell_orders else None

        # Need both bid and ask to make a market
        if best_bid_price is None or best_ask_price is None:
            return orders

        # Calculate remaining capacity based on the *effective* position passed in
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position # Capacity to sell (towards negative limit)

        # Define base volume
        base_volume = self.mm_base_volume

        # Place orders only if spread is wide enough and we have capacity
        spread = best_ask_price - best_bid_price
        if spread >= self.mm_spread_threshold:
            # --- Inventory Skewing ---
            # Default: Place 1 tick inside spread
            buy_price = best_bid_price + 1
            sell_price = best_ask_price - 1
            buy_volume = base_volume
            sell_volume = base_volume

            # If significantly long, make buy quote passive, sell quote aggressive, adjust volume
            if position > base_volume / 2:
                buy_price = best_bid_price # Quote at best bid (passive)
                # sell_price remains best_ask - 1 (aggressive)
                buy_volume = max(0, base_volume - position // 2) # Reduce buy volume
                # logger.print(f"MM Skew LONG for {symbol}: Buy@Bid({buy_volume}), Sell@Ask-1({sell_volume})")

            # If significantly short, make sell quote passive, buy quote aggressive, adjust volume
            elif position < -base_volume / 2:
                sell_price = best_ask_price # Quote at best ask (passive)
                # buy_price remains best_bid + 1 (aggressive)
                sell_volume = max(0, base_volume - abs(position) // 2) # Reduce sell volume
                # logger.print(f"MM Skew SHORT for {symbol}: Buy@Bid+1({buy_volume}), Sell@Ask({sell_volume})")

            # else: # Near flat, quote aggressively on both sides with base volume
                # logger.print(f"MM Skew FLAT for {symbol}: Buy@Bid+1({buy_volume}), Sell@Ask-1({sell_volume})")


            # --- Place Orders ---
            # Place Buy Order
            final_buy_volume = min(buy_volume, buy_capacity)
            if final_buy_volume > 0:
                # Ensure our buy price is still below the best ask
                if buy_price < best_ask_price:
                    orders.append(Order(symbol, buy_price, final_buy_volume))
                    # logger.print(f"Potential MM Buy Order: {final_buy_volume} {symbol}@{buy_price}")
                # else:
                    # logger.print(f"MM Buy price {buy_price} crossed ask {best_ask_price} for {symbol}, skipped.")


            # Place Sell Order
            final_sell_volume = min(sell_volume, sell_capacity)
            if final_sell_volume > 0:
                 # Ensure our sell price is still above the best bid
                if sell_price > best_bid_price:
                    orders.append(Order(symbol, sell_price, -final_sell_volume)) # Negative volume for sell
                    # logger.print(f"Potential MM Sell Order: {-final_sell_volume} {symbol}@{sell_price}")
                # else:
                    # logger.print(f"MM Sell price {sell_price} crossed bid {best_bid_price} for {symbol}, skipped.")

        return orders


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic: Market Make ONLY on RAINFOREST_RESIN
        """
        result = {symbol: [] for symbol in state.listings.keys()} # Initialize result dict

        # --- 0. Log Position Changes ---
        self.log_position_changes(state.position, state.timestamp) # Log changes from previous state

        # --- 1. Market Making for RAINFOREST_RESIN ---
        symbol = "RAINFOREST_RESIN"
        if symbol in state.order_depths:
            depth = state.order_depths[symbol]
            current_position = state.position.get(symbol, 0)
            # Use a local dict for limit checking within MM for this symbol
            pending_pos_delta_mm = {}

            potential_mm_orders = self.get_market_making_orders(symbol, depth, current_position)

            for order in potential_mm_orders:
                # Check limit before adding MM order, considering potential orders already added this tick
                if self._can_place_order(order.symbol, order.quantity, current_position, pending_pos_delta_mm):
                    result[order.symbol].append(order)
                    # Update local pending delta for subsequent checks *within this MM block*
                    pending_pos_delta_mm[order.symbol] = pending_pos_delta_mm.get(order.symbol, 0) + order.quantity
                # else:
                    # logger.print(f"MM Order skipped for {symbol} due to limit.")
        # else:
            # logger.print(f"No order depth data for {symbol}, skipping MM.")


        # --- Final Steps ---
        conversions = 0 # No conversions logic implemented
        trader_data = "" # No state to persist for this simple strategy

        # Log final orders placed
        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log:
             logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")
        # else: # Reduce logging noise
             # logger.print(f"Timestamp: {state.timestamp}, No orders placed.")

        # Update previous positions AFTER all logic for the current timestamp is done
        self.previous_positions = state.position.copy()

        # Flush the logs, orders, conversions, and trader data to standard output
        logger.flush(state, result, conversions, trader_data)

        # Return the orders, conversions, and trader data
        return result, conversions, trader_data

    # Kept log_position_changes for debugging/monitoring
    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        """
        Log changes in positions compared to the start of the previous run.
        """
        log_entry = []
        all_symbols = set(current_positions.keys()) | set(self.previous_positions.keys())
        # Only log for RAINFOREST_RESIN if desired, or keep all for general info
        symbols_to_log = ["RAINFOREST_RESIN"] # Or use sorted(list(all_symbols)) for all
        for symbol in symbols_to_log:
            if symbol not in all_symbols: continue # Skip if symbol not present in either dict
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}") # Shortened format

        if log_entry: # Only log if there were changes
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

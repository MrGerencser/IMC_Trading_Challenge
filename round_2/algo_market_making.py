import json
import statistics # Allowed package
import math # Allowed package
import jsonpickle # Allowed package for traderData serialization
from typing import Any, Dict, List # Keep basic typing

# Assuming datamodel.py defines these classes:
# Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    # ... (Logger class remains unchanged, including flush and compression methods using standard json) ...
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjusted based on typical limits

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This flush method uses standard json for the final output, as provided
        output = [
            self.compress_state(state, trader_data),
            self.compress_orders(orders),
            conversions,
        ]
        output_json = self.to_json(output) # Uses Logger's to_json

        # ... (rest of the flush logic for handling length limits remains the same) ...
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
        # Helper for flush - uses standard json
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

class Trader:
    def __init__(self):
        # Define position limits for each product
        self.position_limits = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "RAINFOREST_RESIN": 50, # Used by MM strategy
            "KELP": 50,           # Used by MM strategy
            "SQUID_INK": 50,      # Used by MR strategy
            "DJEMBES": 60
        }

        # Track previous positions for logging changes
        self.previous_positions = {}

        # --- MM Strategy Parameters ---
        self.mm_spread_threshold_resin = 4
        self.mm_base_volume_resin = 25
        self.mm_spread_threshold_kelp = 2
        self.mm_base_volume_kelp = 25

        # --- SQUID_INK Mean Reversion Parameters ---
        self.squid_hv_window = 50 # Lookback window for volatility calculation
        self.squid_entry_threshold_pct = 1.8 # Enter if deviation > 180% of HV
        self.squid_exit_threshold_pct = 1.0 # Exit if deviation < 100% of HV (and profitable)
        self.squid_warmup_ticks = 50 # Number of ticks after window full before allowing entry
        # Use standard list for price history
        self.squid_prices: List[float] = []
        self.squid_entry_price: float | None = None # Track entry price for PnL calculation
        self.squid_calc_count: int = 0 # Counter for warmup period


    def _serialize_trader_data(self) -> str:
        """Serializes strategy state using jsonpickle"""
        state_data = {
            "squid_prices": self.squid_prices,
            "squid_entry_price": self.squid_entry_price,
            "squid_calc_count": self.squid_calc_count, # Add counter to serialization
        }
        try:
            # Use jsonpickle for encoding complex Python objects if needed
            return jsonpickle.encode(state_data, unpicklable=False) # unpicklable=False for smaller string if complex objects aren't needed back
        except Exception as e:
            logger.print(f"Error serializing trader data with jsonpickle: {e}")
            return "" # Return empty string on error

    def _deserialize_trader_data(self, trader_data: str) -> None:
        """Deserializes strategy state using jsonpickle"""
        if trader_data:
            try:
                data = jsonpickle.decode(trader_data)
                # Load historical prices (should be a list)
                self.squid_prices = data.get("squid_prices", [])
                # Ensure it doesn't exceed max window size after loading
                if len(self.squid_prices) > self.squid_hv_window:
                    self.squid_prices = self.squid_prices[-self.squid_hv_window:]
                self.squid_entry_price = data.get("squid_entry_price", None)
                self.squid_calc_count = data.get("squid_calc_count", 0) # Load counter, default to 0 if missing
                # logger.print(f"Deserialized trader_data: Prices len {len(self.squid_prices)}, Entry {self.squid_entry_price}, CalcCount {self.squid_calc_count}")
            except Exception as e:
                logger.print(f"Error decoding trader data with jsonpickle: {e}. Resetting state.")
                # Reset to default if decoding fails
                self.squid_prices = []
                self.squid_entry_price = None
                self.squid_calc_count = 0 # Reset counter on error
        else:
            # logger.print("No trader data found, initializing fresh state.")
            # Ensure state is default if no data string is provided
            self.squid_prices = []
            self.squid_entry_price = None
            self.squid_calc_count = 0 # Reset counter for fresh state

    # Helper to manage the price history list
    def _update_price_history(self, price: float):
        """Adds a price to the history list and trims it to the window size."""
        self.squid_prices.append(price)
        # Keep only the last 'squid_hv_window' prices
        if len(self.squid_prices) > self.squid_hv_window:
            self.squid_prices.pop(0) # Remove the oldest price

    # Helper function to check position limits before placing an order - Kept as it's used by MM & MR
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

    # Helper to calculate Weighted Average Price (WAP) - Re-added
    def _calculate_wap(self, depth: OrderDepth) -> float | None:
        """Calculates the Weighted Average Price from order book depth."""
        if not depth.buy_orders or not depth.sell_orders:
            return None # Cannot calculate WAP without both sides

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = depth.sell_orders[best_ask]

        # Ensure total volume is not zero to avoid division by zero
        if bid_vol + ask_vol == 0:
            return (best_bid + best_ask) / 2 # Fallback to mid-price if volumes are zero

        wap = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        return wap

    # Standard market making logic (with inventory skewing) - Kept as it's the core strategy now
    def get_market_making_orders(self, symbol: Symbol, depth: OrderDepth, position: int, spread_threshold: int, base_volume: int) -> list[Order]:
        """
        Get market making orders for a single symbol, respecting position limits
        and skewing quotes based on inventory.
        Uses passed spread_threshold and base_volume.
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

        # Place orders only if spread is wide enough and we have capacity
        spread = best_ask_price - best_bid_price
        if spread >= spread_threshold: # Use the passed threshold
            # --- Inventory Skewing ---
            # Default: Place 1 tick inside spread
            buy_price = best_bid_price + 1
            sell_price = best_ask_price - 1
            # Use the passed base_volume
            buy_volume = base_volume
            sell_volume = base_volume

            # If significantly long, make buy quote passive, sell quote aggressive, adjust volume
            if position > base_volume / 2:
                buy_price = best_bid_price # Quote at best bid (passive)
                buy_volume = max(0, base_volume - position // 2) # Reduce buy volume

            # If significantly short, make sell quote passive, buy quote aggressive, adjust volume
            elif position < -base_volume / 2:
                sell_price = best_ask_price # Quote at best ask (passive)
                sell_volume = max(0, base_volume - abs(position) // 2) # Reduce sell volume

            # --- Place Orders ---
            # Place Buy Order
            final_buy_volume = min(buy_volume, buy_capacity)
            if final_buy_volume > 0:
                if buy_price < best_ask_price:
                    orders.append(Order(symbol, buy_price, final_buy_volume))

            # Place Sell Order
            final_sell_volume = min(sell_volume, sell_capacity)
            if final_sell_volume > 0:
                if sell_price > best_bid_price:
                    orders.append(Order(symbol, sell_price, -final_sell_volume)) # Negative volume for sell

        return orders


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic:
        1. Market Make on RAINFOREST_RESIN and KELP.
        2. Mean Reversion on SQUID_INK.
        """
        # --- 0. Deserialize State & Log Position Changes ---
        self._deserialize_trader_data(state.traderData)
        self.log_position_changes(state.position, state.timestamp) # Log changes from previous state

        result = {symbol: [] for symbol in state.listings.keys()} # Initialize result dict
        # Use ONE pending_pos_delta dict for ALL strategies in this timestamp
        pending_pos_delta = {symbol: 0 for symbol in state.listings.keys()}

        # --- 1. Market Making for RAINFOREST_RESIN ---
        symbol_resin = "RAINFOREST_RESIN"
        if symbol_resin in state.order_depths:
            depth_resin = state.order_depths[symbol_resin]
            current_pos_resin = state.position.get(symbol_resin, 0)
            # Use the shared pending_pos_delta for effective position calculation
            effective_pos_resin = current_pos_resin + pending_pos_delta.get(symbol_resin, 0)

            potential_mm_orders_resin = self.get_market_making_orders(
                symbol_resin,
                depth_resin,
                effective_pos_resin, # Use effective position
                self.mm_spread_threshold_resin,
                self.mm_base_volume_resin
            )

            for order in potential_mm_orders_resin:
                # Check limit against total current position + all pending orders so far
                if self._can_place_order(order.symbol, order.quantity, current_pos_resin, pending_pos_delta):
                    result[order.symbol].append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity

        # --- 2. Market Making for KELP ---
        symbol_kelp = "KELP"
        if symbol_kelp in state.order_depths:
            depth_kelp = state.order_depths[symbol_kelp]
            current_pos_kelp = state.position.get(symbol_kelp, 0)
            # Use the shared pending_pos_delta for effective position calculation
            effective_pos_kelp = current_pos_kelp + pending_pos_delta.get(symbol_kelp, 0)

            potential_mm_orders_kelp = self.get_market_making_orders(
                symbol_kelp,
                depth_kelp,
                effective_pos_kelp, # Use effective position
                self.mm_spread_threshold_kelp,
                self.mm_base_volume_kelp
            )

            for order in potential_mm_orders_kelp:
                 # Check limit against total current position + all pending orders so far
                if self._can_place_order(order.symbol, order.quantity, current_pos_kelp, pending_pos_delta):
                    result[order.symbol].append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity

        # --- 3. Mean Reversion for SQUID_INK ---
        symbol_squid = "SQUID_INK"
        if symbol_squid in state.order_depths:
            depth_squid = state.order_depths[symbol_squid]
            current_pos_squid = state.position.get(symbol_squid, 0)
            limit_squid = self.position_limits.get(symbol_squid, 0)

            # Calculate current fair price (WAP) and update history
            wap_squid = self._calculate_wap(depth_squid)
            if wap_squid is not None:
                self._update_price_history(wap_squid) # Use helper to manage list size
                # logger.print(f"SQUID WAP: {wap_squid}, History len: {len(self.squid_prices)}")

            # Check if we have enough data for calculations (list is full)
            if len(self.squid_prices) == self.squid_hv_window and wap_squid is not None:
                try:
                    # Calculate historical mean and volatility using statistics package
                    mean_price = statistics.mean(self.squid_prices)
                    # Need at least 2 data points for stdev
                    if len(self.squid_prices) >= 2:
                        hv = statistics.stdev(self.squid_prices)
                    else:
                        hv = 0 # Cannot calculate stdev with < 2 points
                    # logger.print(f"SQUID Mean: {mean_price:.2f}, HV: {hv:.2f}")

                    if hv > 0: # Avoid division by zero or acting on zero volatility
                        # Increment the calculation counter only when HV is valid
                        self.squid_calc_count += 1

                        entry_threshold_abs = self.squid_entry_threshold_pct * hv
                        exit_threshold_abs = self.squid_exit_threshold_pct * hv
                        current_deviation = abs(wap_squid - mean_price)
                        effective_pos_squid = current_pos_squid + pending_pos_delta.get(symbol_squid, 0) # Current pos + MM orders

                        # --- Exit Logic ---
                        # (Exit logic remains the same, no warmup needed for exits)
                        if effective_pos_squid != 0 and self.squid_entry_price is not None:
                            pnl_per_unit = 0
                            if effective_pos_squid > 0: # Long position
                                pnl_per_unit = wap_squid - self.squid_entry_price
                            else: # Short position
                                pnl_per_unit = self.squid_entry_price - wap_squid

                            # logger.print(f"SQUID Check Exit: Dev={current_deviation:.2f}, ExitThr={exit_threshold_abs:.2f}, PnL/unit={pnl_per_unit:.2f}")
                            if current_deviation < exit_threshold_abs and pnl_per_unit > 0:
                                # Profitable exit condition met, close position with market order
                                close_quantity = -effective_pos_squid # Target zero position
                                order_qty = 0
                                close_price = None # Initialize close_price

                                if close_quantity > 0 and depth_squid.sell_orders: # Need to buy back
                                    best_ask_exit = min(depth_squid.sell_orders.keys())
                                    available_volume = depth_squid.sell_orders[best_ask_exit]
                                    order_qty = min(close_quantity, available_volume)
                                    close_price = best_ask_exit
                                elif close_quantity < 0 and depth_squid.buy_orders: # Need to sell back
                                    best_bid_exit = max(depth_squid.buy_orders.keys())
                                    available_volume = depth_squid.buy_orders[best_bid_exit]
                                    order_qty = max(close_quantity, -available_volume) # close_quantity is negative
                                    close_price = best_bid_exit
                                # else: # No liquidity to close or quantity is zero
                                     # close_price remains None

                                if order_qty != 0 and close_price is not None and self._can_place_order(symbol_squid, order_qty, current_pos_squid, pending_pos_delta):
                                    logger.print(f"SQUID EXIT: Closing {order_qty} @ {close_price} (Dev: {current_deviation:.2f} < {exit_threshold_abs:.2f}, PnL: {pnl_per_unit:.2f}>0)")
                                    result[symbol_squid].append(Order(symbol_squid, close_price, order_qty))
                                    pending_pos_delta[symbol_squid] = pending_pos_delta.get(symbol_squid, 0) + order_qty
                                    # Only reset entry price if fully closed
                                    if current_pos_squid + pending_pos_delta.get(symbol_squid, 0) == 0:
                                        self.squid_entry_price = None

                        # --- Entry Logic ---
                        # Only enter if effective position is currently flat AND warmup period is over
                        elif effective_pos_squid == 0 and self.squid_calc_count > self.squid_warmup_ticks:
                            upper_band = mean_price + entry_threshold_abs
                            lower_band = mean_price - entry_threshold_abs
                            best_ask = min(depth_squid.sell_orders.keys()) if depth_squid.sell_orders else None
                            best_bid = max(depth_squid.buy_orders.keys()) if depth_squid.buy_orders else None

                            # logger.print(f"SQUID Check Entry (Warmup OK): WAP={wap_squid:.2f}, Lower={lower_band:.2f}, Upper={upper_band:.2f}")

                            # Check for Buy Entry (Price below lower band)
                            if best_ask is not None and best_ask < lower_band:
                                buy_volume_available = depth_squid.sell_orders[best_ask]
                                # Calculate capacity based on current pos + pending orders
                                buy_capacity = limit_squid - (current_pos_squid + pending_pos_delta.get(symbol_squid, 0))
                                order_qty = min(buy_volume_available, buy_capacity)

                                if order_qty > 0 and self._can_place_order(symbol_squid, order_qty, current_pos_squid, pending_pos_delta):
                                    logger.print(f"SQUID ENTRY BUY: {order_qty} @ {best_ask} (Ask {best_ask:.2f} < Lower {lower_band:.2f})")
                                    result[symbol_squid].append(Order(symbol_squid, best_ask, order_qty))
                                    pending_pos_delta[symbol_squid] = pending_pos_delta.get(symbol_squid, 0) + order_qty
                                    self.squid_entry_price = best_ask # Record entry price

                            # Check for Sell Entry (Price above upper band)
                            elif best_bid is not None and best_bid > upper_band:
                                sell_volume_available = depth_squid.buy_orders[best_bid]
                                # Sell capacity based on current pos + pending orders
                                sell_capacity = limit_squid + (current_pos_squid + pending_pos_delta.get(symbol_squid, 0))
                                order_qty = -min(sell_volume_available, sell_capacity) # Negative for sell order

                                if order_qty < 0 and self._can_place_order(symbol_squid, order_qty, current_pos_squid, pending_pos_delta):
                                    logger.print(f"SQUID ENTRY SELL: {order_qty} @ {best_bid} (Bid {best_bid:.2f} > Upper {upper_band:.2f})")
                                    result[symbol_squid].append(Order(symbol_squid, best_bid, order_qty))
                                    pending_pos_delta[symbol_squid] = pending_pos_delta.get(symbol_squid, 0) + order_qty
                                    self.squid_entry_price = best_bid # Record entry price
                        # else:
                            # if effective_pos_squid == 0:
                                # logger.print(f"SQUID: Entry skipped, warmup not complete ({self.squid_calc_count}/{self.squid_warmup_ticks})")


                except statistics.StatisticsError:
                    logger.print("SQUID: Not enough data or zero variance for volatility calculation.")
                except Exception as e:
                    logger.print(f"SQUID: Error during strategy logic: {e}")
            # else:
                # if len(self.squid_prices) < self.squid_hv_window:
                    # logger.print(f"SQUID: Filling price data ({len(self.squid_prices)}/{self.squid_hv_window})")
                # elif wap_squid is None:
                    # logger.print("SQUID: No WAP available.")


        # --- Final Steps ---
        conversions = 0 # No conversions logic implemented
        trader_data = self._serialize_trader_data() # Serialize state using jsonpickle

        # Log final orders placed
        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log:
             logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")

        # Update previous positions AFTER all logic for the current timestamp is done
        self.previous_positions = state.position.copy()

        # Flush the logs, orders, conversions, and trader data to standard output
        # Logger uses standard json, trader_data string is already serialized via jsonpickle
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
        # Log for all traded symbols now
        symbols_to_log = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"] # Add SQUID_INK
        for symbol in symbols_to_log:
            if symbol not in all_symbols: continue
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")

        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

    # Removed to_json from Trader class as Logger handles final output JSON
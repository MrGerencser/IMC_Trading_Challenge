import json
import statistics # Keep for potential fallback or other strategies
import math # Allowed package
import jsonpickle # Allowed package for traderData serialization
from typing import Any, Dict, List, Optional, Tuple # Keep basic typing, add Optional, Tuple

# Assuming datamodel.py defines these classes:
# Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    # ... (Logger class remains unchanged) ...
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
            "RAINFOREST_RESIN": 50, # Used by MM strategy (commented out)
            "KELP": 50,           # Used by MM strategy (commented out)
            "SQUID_INK": 50,      # Used by MR strategy (commented out)
            "DJEMBES": 60
        }

        # Track previous positions for logging changes
        self.previous_positions = {}

        # --- Basket Definitions ---
        self.basket_components = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }

        # --- MM Strategy Parameters - COMMENTED OUT ---
        # self.mm_spread_threshold_resin = 7
        # self.mm_base_volume_resin = 25
        # self.mm_spread_threshold_kelp = 2
        # self.mm_base_volume_kelp = 25

        # --- SQUID_INK Mean Reversion Parameters (EWMA based) - COMMENTED OUT ---
        # self.squid_ewma_span = 20
        # self.squid_alpha = 2 / (self.squid_ewma_span + 1)
        # self.squid_entry_threshold_pct_hv = 1.4
        # self.squid_exit_threshold_pct_hv = 0.5
        # self.squid_stop_loss_pct_hv = 4.0
        # self.squid_target_entry_volume = 50
        # self.squid_max_levels_sweep = 3
        # self.squid_limit_order_ratio = 0.5

        # --- SQUID_INK State Variables - COMMENTED OUT ---
        # self.squid_ewma: Optional[float] = None
        # self.squid_ewm_variance: Optional[float] = None
        # self.squid_entry_price: Optional[float] = None

        # --- Basket Arbitrage Parameters ---
        self.arb_ewma_span = 50 # Window for spread EWMA/EWMSD
        self.arb_alpha = 2 / (self.arb_ewma_span + 1)
        self.arb_entry_z_threshold = 3.0 # Z-score to enter
        self.arb_exit_z_threshold = 1.5  # Z-score to exit (closer to 0)
        self.arb_stop_loss_z_threshold = 10.0 # Z-score stop loss
        self.arb_min_profit_threshold = 2.0 # Minimum expected profit to enter
        self.arb_trade_size = 5 # Number of baskets to trade per signal

        # --- Basket Arbitrage State Variables ---
        # Store EWMA and EWM Variance for each basket's spread
        self.arb_spread_ewma: Dict[Symbol, float] = {}
        self.arb_spread_ewm_variance: Dict[Symbol, float] = {}
        # Track if we are in an arbitrage position for a basket
        self.arb_position_active: Dict[Symbol, bool] = {} # True if holding an arb position for this basket

    # ... (_calculate_ewma_ewmsd, _get_ewmsd methods remain the same) ...
    def _calculate_ewma_ewmsd(self, current_value: float, symbol: Symbol, ewma_dict: Dict[Symbol, float], variance_dict: Dict[Symbol, float], alpha: float):
        """Generic EWMA/EWM Variance update function."""
        if symbol not in ewma_dict or symbol not in variance_dict:
            ewma_dict[symbol] = current_value
            variance_dict[symbol] = 0
        else:
            prev_ewma = ewma_dict[symbol]
            ewma_dict[symbol] = alpha * current_value + (1 - alpha) * prev_ewma
            variance_dict[symbol] = (1 - alpha) * (variance_dict[symbol] + alpha * (current_value - prev_ewma)**2)

    def _get_ewmsd(self, symbol: Symbol, variance_dict: Dict[Symbol, float]) -> Optional[float]:
        """Generic EWMSD retrieval function."""
        variance = variance_dict.get(symbol)
        if variance is None:
            return None
        if variance < 0:
             return 0.0
        return math.sqrt(variance)

    def _serialize_trader_data(self) -> str:
        """Serializes strategy state using jsonpickle"""
        state_data = {
            # "squid_ewma": self.squid_ewma, # Commented out
            # "squid_ewm_variance": self.squid_ewm_variance, # Commented out
            # "squid_entry_price": self.squid_entry_price, # Commented out
            "arb_spread_ewma": self.arb_spread_ewma,
            "arb_spread_ewm_variance": self.arb_spread_ewm_variance,
            "arb_position_active": self.arb_position_active,
        }
        try:
            return jsonpickle.encode(state_data, unpicklable=False)
        except Exception as e:
            logger.print(f"Error serializing trader data with jsonpickle: {e}")
            return ""

    def _deserialize_trader_data(self, trader_data: str) -> None:
        """Deserializes strategy state using jsonpickle"""
        # Reset state first
        # self.squid_ewma = None # Commented out
        # self.squid_ewm_variance = None # Commented out
        # self.squid_entry_price = None # Commented out
        self.arb_spread_ewma = {}
        self.arb_spread_ewm_variance = {}
        self.arb_position_active = {}

        if trader_data:
            try:
                data = jsonpickle.decode(trader_data)
                # self.squid_ewma = data.get("squid_ewma", None) # Commented out
                # self.squid_ewm_variance = data.get("squid_ewm_variance", None) # Commented out
                # self.squid_entry_price = data.get("squid_entry_price", None) # Commented out
                self.arb_spread_ewma = data.get("arb_spread_ewma", {})
                self.arb_spread_ewm_variance = data.get("arb_spread_ewm_variance", {})
                self.arb_position_active = data.get("arb_position_active", {})
            except Exception as e:
                logger.print(f"Error decoding trader data with jsonpickle: {e}. Resetting state.")
                # Explicitly reset again on error
                self.arb_spread_ewma = {}
                self.arb_spread_ewm_variance = {}
                self.arb_position_active = {}

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
            # logger.print(f"DEBUG: Order rejected for {symbol}. Qty: {quantity}, Current: {current_pos}, Pending: {pending_delta}, Final: {final_pos}, Limit: {limit}")
            return False

    def _calculate_wap(self, depth: OrderDepth, levels: int = 1) -> Optional[float]: # Default to level 1 WAP for simplicity
        """
        Calculates the Weighted Average Price (WAP) using up to 'levels' levels
        of the order book depth.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return None
        sorted_bids = sorted(depth.buy_orders.items(), key=lambda item: item[0], reverse=True)
        sorted_asks = sorted(depth.sell_orders.items(), key=lambda item: item[0])

        best_bid_price, best_bid_vol = sorted_bids[0]
        best_ask_price, best_ask_vol = sorted_asks[0]

        if best_bid_vol + best_ask_vol == 0:
             return (best_bid_price + best_ask_price) / 2 # Mid-price fallback

        wap = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)
        return wap

    # --- get_market_making_orders Method - COMMENTED OUT ---
    # def get_market_making_orders(self, symbol: Symbol, depth: OrderDepth, position: int, spread_threshold: int, base_volume: int) -> list[Order]:
    #     """
    #     Get market making orders for a single symbol, respecting position limits
    #     and skewing quotes based on inventory.
    #     Uses passed spread_threshold and base_volume.
    #     """
    #     orders = []
    #     position_limit = self.position_limits.get(symbol)
    #     if position_limit is None:
    #          # logger.print(f"Warning: No position limit found for {symbol}. Skipping MM.") # Already logged in _can_place_order
    #          return []
    #
    #     best_bid_price = max(depth.buy_orders.keys()) if depth.buy_orders else None
    #     best_ask_price = min(depth.sell_orders.keys()) if depth.sell_orders else None
    #
    #     if best_bid_price is None or best_ask_price is None:
    #         return orders
    #
    #     buy_capacity = position_limit - position
    #     sell_capacity = position_limit + position
    #
    #     spread = best_ask_price - best_bid_price
    #     if spread >= spread_threshold:
    #         buy_price = best_bid_price + 1
    #         sell_price = best_ask_price - 1
    #         buy_volume = base_volume
    #         sell_volume = base_volume
    #
    #         # Basic inventory skewing
    #         skew_factor = position / position_limit if position_limit != 0 else 0
    #         buy_volume = int(buy_volume * (1 - skew_factor)) # Reduce buy volume if long
    #         sell_volume = int(sell_volume * (1 + skew_factor)) # Reduce sell volume if short
    #
    #         # Adjust price based on skew (more aggressive if inventory is against desired direction)
    #         if position > position_limit * 0.3: # Significantly long
    #             buy_price = best_bid_price # Make buy more passive
    #         elif position < -position_limit * 0.3: # Significantly short
    #             sell_price = best_ask_price # Make sell more passive
    #
    #         final_buy_volume = min(buy_volume, buy_capacity)
    #         if final_buy_volume > 0:
    #             if buy_price < best_ask_price: # Ensure not crossing the spread
    #                 orders.append(Order(symbol, buy_price, final_buy_volume))
    #
    #         final_sell_volume = min(sell_volume, sell_capacity)
    #         if final_sell_volume > 0:
    #             if sell_price > best_bid_price: # Ensure not crossing the spread
    #                 orders.append(Order(symbol, sell_price, -final_sell_volume))
    #
    #     return orders

    def _get_best_price(self, depth: OrderDepth, side: str) -> Optional[int]:
        """Gets the best bid or ask price."""
        if side == 'bid' and depth.buy_orders:
            return max(depth.buy_orders.keys())
        elif side == 'ask' and depth.sell_orders:
            return min(depth.sell_orders.keys())
        return None

    def _calculate_synthetic_prices(self, components: Dict[Symbol, int], order_depths: Dict[Symbol, OrderDepth]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates the synthetic bid (value if selling components) and
        synthetic ask (cost if buying components). Returns (synthetic_bid, synthetic_ask).
        Returns (None, None) if any component price is missing.
        """
        synthetic_bid = 0.0
        synthetic_ask = 0.0
        all_components_available = True

        for component, quantity in components.items():
            if component not in order_depths:
                logger.print(f"ARB: Missing order depth for component {component}")
                all_components_available = False
                break

            comp_depth = order_depths[component]
            comp_best_bid = self._get_best_price(comp_depth, 'bid')
            comp_best_ask = self._get_best_price(comp_depth, 'ask')

            if comp_best_bid is None or comp_best_ask is None:
                logger.print(f"ARB: Missing bid/ask for component {component}")
                all_components_available = False
                break

            # Synthetic Bid (selling components): Use component BIDs
            synthetic_bid += comp_best_bid * quantity
            # Synthetic Ask (buying components): Use component ASKs
            synthetic_ask += comp_best_ask * quantity

        if not all_components_available:
            return None, None
        return synthetic_bid, synthetic_ask

    def _get_arbitrage_orders(self, basket_symbol: Symbol, state: TradingState, pending_pos_delta: Dict[Symbol, int]) -> List[Order]:
        """Calculates and returns orders for the basket arbitrage strategy."""
        orders: List[Order] = []
        components = self.basket_components.get(basket_symbol)
        if not components:
            return orders # Should not happen if called correctly

        # --- 1. Get Prices ---
        if basket_symbol not in state.order_depths:
            # logger.print(f"ARB: Missing order depth for basket {basket_symbol}")
            return orders
        basket_depth = state.order_depths[basket_symbol]
        actual_basket_bid = self._get_best_price(basket_depth, 'bid')
        actual_basket_ask = self._get_best_price(basket_depth, 'ask')

        if actual_basket_bid is None or actual_basket_ask is None:
            # logger.print(f"ARB: Missing bid/ask for basket {basket_symbol}")
            return orders

        synthetic_bid, synthetic_ask = self._calculate_synthetic_prices(components, state.order_depths)

        if synthetic_bid is None or synthetic_ask is None:
            # logger.print(f"ARB: Could not calculate synthetic prices for {basket_symbol}")
            return orders

        # --- 2. Calculate Spreads ---
        # Spread 1: Profit potential from buying basket, selling components
        spread_buy_basket = synthetic_bid - actual_basket_ask
        # Spread 2: Profit potential from selling basket, buying components
        spread_sell_basket = actual_basket_bid - synthetic_ask

        # Use the average of the two potential profit spreads for Z-score calculation
        # This represents the general mispricing level
        current_spread_metric = (spread_buy_basket + spread_sell_basket) / 2

        # --- 3. Update EWMA/EWMSD and Calculate Z-Score ---
        self._calculate_ewma_ewmsd(current_spread_metric, basket_symbol, self.arb_spread_ewma, self.arb_spread_ewm_variance, self.arb_alpha)
        spread_ewma = self.arb_spread_ewma.get(basket_symbol)
        spread_ewmsd = self._get_ewmsd(basket_symbol, self.arb_spread_ewm_variance)

        z_score: Optional[float] = None
        if spread_ewma is not None and spread_ewmsd is not None and spread_ewmsd > 1e-6: # Avoid division by zero
            z_score = (current_spread_metric - spread_ewma) / spread_ewmsd
            # logger.print(f"ARB {basket_symbol}: Spread={current_spread_metric:.2f}, EWMA={spread_ewma:.2f}, EWMSD={spread_ewmsd:.4f}, Z={z_score:.2f}")
        # else:
            # logger.print(f"ARB {basket_symbol}: Spread={current_spread_metric:.2f}, EWMA/EWMSD not ready.")


        # --- 4. Trading Logic ---
        current_basket_pos = state.position.get(basket_symbol, 0)
        is_active = self.arb_position_active.get(basket_symbol, False)

        # --- Exit Logic ---
        if is_active and z_score is not None:
            exit_signal = False
            stop_loss_signal = False
            is_profitable_exit = False # Flag for profitability check

            # Check for mean reversion exit condition (Z-score based)
            z_score_exit_condition = abs(z_score) < self.arb_exit_z_threshold

            # Check for profitability based on current spreads
            if current_basket_pos > 0: # If currently long basket
                # Profitable to exit if selling basket & buying components is now favorable
                is_profitable_exit = spread_sell_basket > 0
            elif current_basket_pos < 0: # If currently short basket
                # Profitable to exit if buying basket & selling components is now favorable
                is_profitable_exit = spread_buy_basket > 0

            # Combine Z-score condition AND profitability condition for take-profit exit
            if z_score_exit_condition and is_profitable_exit:
                exit_signal = True
                logger.print(f"ARB {basket_symbol}: TAKE PROFIT signal (Z={z_score:.2f}, Profitable Spread Found)")
            # Check for stop loss (independent of profitability)
            elif (current_basket_pos > 0 and z_score > self.arb_stop_loss_z_threshold) or \
                 (current_basket_pos < 0 and z_score < -self.arb_stop_loss_z_threshold):
                 stop_loss_signal = True
                 logger.print(f"ARB {basket_symbol}: STOP LOSS signal (Z-score {z_score:.2f} vs threshold {self.arb_stop_loss_z_threshold})")


            if exit_signal or stop_loss_signal:
                logger.print(f"ARB {basket_symbol}: Attempting to close position.")
                # ... (rest of the exit order placement logic remains the same) ...
                # Determine quantity to close based on assumed position size
                trade_qty = 0
                if current_basket_pos > 0: trade_qty = self.arb_trade_size
                elif current_basket_pos < 0: trade_qty = -self.arb_trade_size

                if trade_qty == 0:
                     logger.print(f"ARB {basket_symbol}: Exit signal but current position is zero. Resetting active state.")
                     self.arb_position_active[basket_symbol] = False
                     return orders

                # Check if we can close the basket position
                if self._can_place_order(basket_symbol, -trade_qty, current_basket_pos, pending_pos_delta):
                    # Create basket closing order
                    basket_close_order = Order(basket_symbol, actual_basket_bid if trade_qty > 0 else actual_basket_ask, -trade_qty)
                    potential_orders = [basket_close_order]
                    can_close_components = True

                    # Check component limits
                    for component, quantity in components.items():
                        comp_qty_to_close = -quantity * trade_qty # Opposite sign to basket close
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty_to_close, comp_current_pos, pending_pos_delta):
                            can_close_components = False
                            logger.print(f"ARB {basket_symbol}: Cannot close component {component} due to limits. ABORTING EXIT.")
                            break # Stop trying to close

                        # Create component closing order
                        comp_depth = state.order_depths[component]
                        comp_close_price = self._get_best_price(comp_depth, 'ask') if comp_qty_to_close > 0 else self._get_best_price(comp_depth, 'bid')
                        if comp_close_price:
                            potential_orders.append(Order(component, comp_close_price, comp_qty_to_close))
                        else:
                            logger.print(f"ARB {basket_symbol}: ERROR - Missing close price for component {component} during exit.")
                            can_close_components = False # Abort if price missing
                            break

                    # If all checks passed, add orders and update state
                    if can_close_components:
                        logger.print(f"ARB {basket_symbol}: Placing exit orders.")
                        for order in potential_orders:
                            orders.append(order)
                            pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        self.arb_position_active[basket_symbol] = False
                    # Else: Aborted exit, potential_orders are discarded

                else:
                    logger.print(f"ARB {basket_symbol}: Cannot close basket position due to limits.")

        # --- Entry Logic ---
        # Only enter if not already holding an arb position for this basket
        elif not is_active and z_score is not None:
            trade_qty = self.arb_trade_size
            basket_order = None
            component_orders = []
            can_place_all = True

            # Strategy 1: Basket Underpriced (Buy Basket, Sell Components)
            if z_score < -self.arb_entry_z_threshold and spread_buy_basket > self.arb_min_profit_threshold:
                logger.print(f"ARB {basket_symbol}: ENTRY signal LONG BASKET (Z={z_score:.2f} < {-self.arb_entry_z_threshold}, Spread={spread_buy_basket:.2f})")

                # Check basket limit
                if not self._can_place_order(basket_symbol, trade_qty, current_basket_pos, pending_pos_delta):
                    can_place_all = False
                    logger.print(f"ARB {basket_symbol}: Cannot place basket buy order due to limits.")
                else:
                    basket_order = Order(basket_symbol, actual_basket_ask, trade_qty) # Buy basket aggressively

                    # Check component limits (selling components)
                    for component, quantity in components.items():
                        comp_qty = -quantity * trade_qty # Sell components
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty, comp_current_pos, pending_pos_delta):
                            can_place_all = False
                            logger.print(f"ARB {basket_symbol}: Cannot place component sell order for {component} due to limits.")
                            break
                        else:
                            comp_depth = state.order_depths[component]
                            comp_price = self._get_best_price(comp_depth, 'bid') # Sell aggressively
                            if comp_price:
                                component_orders.append(Order(component, comp_price, comp_qty))
                            else:
                                can_place_all = False
                                logger.print(f"ARB {basket_symbol}: ERROR - Missing sell price for component {component} during entry.")
                                break

            # Strategy 2: Basket Overpriced (Sell Basket, Buy Components)
            elif z_score > self.arb_entry_z_threshold and spread_sell_basket > self.arb_min_profit_threshold:
                logger.print(f"ARB {basket_symbol}: ENTRY signal SHORT BASKET (Z={z_score:.2f} > {self.arb_entry_z_threshold}, Spread={spread_sell_basket:.2f})")
                trade_qty = -self.arb_trade_size # Negative for selling basket

                # Check basket limit
                if not self._can_place_order(basket_symbol, trade_qty, current_basket_pos, pending_pos_delta):
                    can_place_all = False
                    logger.print(f"ARB {basket_symbol}: Cannot place basket sell order due to limits.")
                else:
                    basket_order = Order(basket_symbol, actual_basket_bid, trade_qty) # Sell basket aggressively

                    # Check component limits (buying components)
                    for component, quantity in components.items():
                        comp_qty = -quantity * trade_qty # Buy components (double negative)
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty, comp_current_pos, pending_pos_delta):
                            can_place_all = False
                            logger.print(f"ARB {basket_symbol}: Cannot place component buy order for {component} due to limits.")
                            break
                        else:
                            comp_depth = state.order_depths[component]
                            comp_price = self._get_best_price(comp_depth, 'ask') # Buy aggressively
                            if comp_price:
                                component_orders.append(Order(component, comp_price, comp_qty))
                            else:
                                can_place_all = False
                                logger.print(f"ARB {basket_symbol}: ERROR - Missing buy price for component {component} during entry.")
                                break

            # Place orders if all checks passed
            if can_place_all and basket_order and component_orders:
                logger.print(f"ARB {basket_symbol}: Placing entry orders.")
                orders.append(basket_order)
                pending_pos_delta[basket_symbol] = pending_pos_delta.get(basket_symbol, 0) + basket_order.quantity
                for order in component_orders:
                    orders.append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                self.arb_position_active[basket_symbol] = True # Mark position as active

        return orders


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic:
        1. Deserialize state.
        2. Basket Arbitrage on PICNIC_BASKET1 and PICNIC_BASKET2.
        3. Serialize state and return orders.
        """
        self._deserialize_trader_data(state.traderData)
        self.log_position_changes(state.position, state.timestamp)

        result = {symbol: [] for symbol in state.listings.keys()}
        # Use a single pending delta dict, shared across strategies
        pending_pos_delta = {symbol: 0 for symbol in state.listings.keys()}

        # --- Market Making Logic (COMMENTED OUT) ---
        # mm_symbols = ["RAINFOREST_RESIN", "KELP"]
        # for symbol in mm_symbols:
        #      if symbol in state.order_depths:
        #          depth = state.order_depths[symbol]
        #          current_pos = state.position.get(symbol, 0)
        #          # Pass effective position including already pending orders for THIS symbol
        #          effective_pos = current_pos + pending_pos_delta.get(symbol, 0)
        #
        #          # Determine parameters based on symbol
        #          spread_thresh = self.mm_spread_threshold_resin if symbol == "RAINFOREST_RESIN" else self.mm_spread_threshold_kelp
        #          base_vol = self.mm_base_volume_resin if symbol == "RAINFOREST_RESIN" else self.mm_base_volume_kelp
        #
        #          # Need to uncomment get_market_making_orders method if using this
        #          # potential_mm_orders = self.get_market_making_orders(
        #          #     symbol, depth, effective_pos, spread_thresh, base_vol
        #          # )
        #          potential_mm_orders = [] # Placeholder since method is commented out
        #          for order in potential_mm_orders:
        #              # Check against original current_pos and the GLOBAL pending_pos_delta
        #              if self._can_place_order(order.symbol, order.quantity, current_pos, pending_pos_delta):
        #                  result[order.symbol].append(order)
        #                  pending_pos_delta[order.symbol] += order.quantity
        # --- End Market Making Logic ---


        # --- SQUID_INK Mean Reversion Logic (COMMENTED OUT) ---
        # symbol_squid = "SQUID_INK"
        # if symbol_squid in state.order_depths:
        #     # ... (Squid logic was here) ...
        # --- End SQUID_INK Logic ---


        # --- Basket Arbitrage Logic ---
        arb_symbols = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
        for basket_symbol in arb_symbols:
            potential_arb_orders = self._get_arbitrage_orders(basket_symbol, state, pending_pos_delta)
            # The _get_arbitrage_orders function already checks limits and updates pending_pos_delta internally
            # So we just need to add the returned orders to the result
            for order in potential_arb_orders:
                 result[order.symbol].append(order)
        # --- End Basket Arbitrage Logic ---


        # --- Final Steps ---
        conversions = 0
        trader_data = self._serialize_trader_data()

        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log:
             logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")

        self.previous_positions = state.position.copy()

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data

    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        """
        Log changes in positions compared to the start of the previous run.
        """
        log_entry = []
        all_symbols = set(current_positions.keys()) | set(self.previous_positions.keys())
        # Update symbols to log to include basket components
        symbols_to_log = sorted(list(self.position_limits.keys())) # Log all traded symbols

        for symbol in symbols_to_log:
            if symbol not in all_symbols: continue
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")

        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")


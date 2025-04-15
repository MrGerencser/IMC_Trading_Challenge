import json
import statistics # Keep for potential fallback or other strategies
import math # Allowed package
import jsonpickle # Allowed package for traderData serialization
from typing import Any, Dict, List, Optional # Keep basic typing, add Optional

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
        self.mm_spread_threshold_resin = 7
        self.mm_base_volume_resin = 25
        self.mm_spread_threshold_kelp = 2
        self.mm_base_volume_kelp = 25

        # --- SQUID_INK Mean Reversion Parameters (EWMA based) ---
        self.squid_ewma_span = 20 # Span for EWMA/EWMSD calculation (similar to old window)
        self.squid_alpha = 2 / (self.squid_ewma_span + 1) # Smoothing factor for EWMA/EWMSD
        self.squid_entry_threshold_pct_hv = 1.4 # Enter if deviation > X * EWMSD (Example: Higher based on hint)
        self.squid_exit_threshold_pct_hv = 0.5  # Exit if deviation < X * EWMSD (Example: Lower based on hint)
        self.squid_stop_loss_pct_hv = 4.0   # Stop loss if loss exceeds X * EWMSD (Example: Wider based on hint)
        self.squid_target_entry_volume = 50 # Target total volume to try and fill on entry (Example)
        self.squid_max_levels_sweep = 3     # How many book levels to check for aggressive volume
        self.squid_limit_order_ratio = 0.5 # Ratio (0.0 to 1.0) of target volume to place as passive limit order (Example: 50%)

        # --- SQUID_INK State Variables ---
        self.squid_ewma: Optional[float] = None # Exponentially Weighted Moving Average
        self.squid_ewm_variance: Optional[float] = None # Exponentially Weighted Moving Variance
        self.squid_entry_price: Optional[float] = None # Track entry price for PnL and stop-losseg

    def _calculate_ewma_ewmsd(self, current_wap: float):
        """Updates EWMA and EWM Variance based on the current WAP."""
        if self.squid_ewma is None or self.squid_ewm_variance is None:
            # Initialize EWMA and Variance with the first WAP observed
            self.squid_ewma = current_wap
            self.squid_ewm_variance = 0 # Initial variance is zero
        else:
            # Store previous EWMA before updating it
            prev_ewma = self.squid_ewma
            # Update EWMA
            self.squid_ewma = self.squid_alpha * current_wap + (1 - self.squid_alpha) * prev_ewma
            # Update EWM Variance using the previous EWMA
            self.squid_ewm_variance = (1 - self.squid_alpha) * (self.squid_ewm_variance + self.squid_alpha * (current_wap - prev_ewma)**2)

    def _get_ewmsd(self) -> Optional[float]:
        """Returns the EWMSD (sqrt of variance), or None if variance is not calculated."""
        if self.squid_ewm_variance is None:
            return None
        # Avoid sqrt of negative number due to potential floating point inaccuracies
        if self.squid_ewm_variance < 0:
             return 0.0
        return math.sqrt(self.squid_ewm_variance)

    def _serialize_trader_data(self) -> str:
        """Serializes strategy state using jsonpickle"""
        state_data = {
            "squid_ewma": self.squid_ewma,
            "squid_ewm_variance": self.squid_ewm_variance,
            "squid_entry_price": self.squid_entry_price,
        }
        try:
            return jsonpickle.encode(state_data, unpicklable=False)
        except Exception as e:
            logger.print(f"Error serializing trader data with jsonpickle: {e}")
            return ""

    def _deserialize_trader_data(self, trader_data: str) -> None:
        """Deserializes strategy state using jsonpickle"""
        if trader_data:
            try:
                data = jsonpickle.decode(trader_data)
                self.squid_ewma = data.get("squid_ewma", None)
                self.squid_ewm_variance = data.get("squid_ewm_variance", None)
                self.squid_entry_price = data.get("squid_entry_price", None)
            except Exception as e:
                logger.print(f"Error decoding trader data with jsonpickle: {e}. Resetting SQUID state.")
                self.squid_ewma = None
                self.squid_ewm_variance = None
                self.squid_entry_price = None
        else:
            self.squid_ewma = None
            self.squid_ewm_variance = None
            self.squid_entry_price = None

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
            return False

    def _calculate_wap(self, depth: OrderDepth, levels: int = 3) -> Optional[float]:
        """
        Calculates the Weighted Average Price (WAP) using up to 'levels' levels
        of the order book depth.
        """
        # Ensure depth exists and has orders
        if not depth.buy_orders or not depth.sell_orders:
            return None

        # Sort buy orders descending (best bid first) and sell orders ascending (best ask first)
        # Assuming depth.buy_orders and depth.sell_orders are dicts {price: volume}
        sorted_bids = sorted(depth.buy_orders.items(), key=lambda item: item[0], reverse=True)
        sorted_asks = sorted(depth.sell_orders.items(), key=lambda item: item[0])

        total_bid_vol_price = 0.0
        total_ask_vol_price = 0.0
        total_bid_vol = 0.0
        total_ask_vol = 0.0

        # Calculate weighted price sum and total volume for top 'levels' bids
        for i in range(min(levels, len(sorted_bids))):
            price, volume = sorted_bids[i]
            total_bid_vol_price += price * volume
            total_bid_vol += volume

        # Calculate weighted price sum and total volume for top 'levels' asks
        for i in range(min(levels, len(sorted_asks))):
            price, volume = sorted_asks[i]
            total_ask_vol_price += price * volume
            total_ask_vol += volume

        # If total volume on either side (up to 'levels') is zero, we can't calculate WAP reliably
        if total_bid_vol == 0 or total_ask_vol == 0:
             # Fallback to mid-price of best bid/ask if possible
             if sorted_bids and sorted_asks:
                 best_bid = sorted_bids[0][0]
                 best_ask = sorted_asks[0][0]
                 return (best_bid + best_ask) / 2
             else:
                 return None # Cannot determine a price

        # Calculate the Weighted Average Price using volumes from the opposite side as weights
        # WAP = (AvgBidPrice * TotalAskVol + AvgAskPrice * TotalBidVol) / (TotalBidVol + TotalAskVol)
        # where AvgBidPrice = total_bid_vol_price / total_bid_vol, etc.
        # Simplified: WAP = (best_bid * total_ask_vol + best_ask * total_bid_vol) / (total_bid_vol + total_ask_vol) is for level 1 only.
        # For multi-level, a common approach is microprice:
        # Microprice = (BestBid * AskVol_L1 + BestAsk * BidVol_L1) / (BidVol_L1 + AskVol_L1) - this is the original WAP.

        # Let's try a simpler multi-level WAP: Average price weighted by *own* level volume up to N levels
        # This isn't standard WAP, but reflects average execution price for N levels
        # avg_bid_price = total_bid_vol_price / total_bid_vol
        # avg_ask_price = total_ask_vol_price / total_ask_vol
        # wap = (avg_bid_price * total_ask_vol + avg_ask_price * total_bid_vol) / (total_bid_vol + total_ask_vol)

        # Let's stick to the standard definition but use volumes from the first level only for weighting,
        # but use best_bid/ask from the actual book. This is the most common WAP definition.
        best_bid_price, best_bid_vol = sorted_bids[0]
        best_ask_price, best_ask_vol = sorted_asks[0]

        if best_bid_vol + best_ask_vol == 0:
             return (best_bid_price + best_ask_price) / 2 # Mid-price fallback

        wap = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)

        return wap

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

        if best_bid_price is None or best_ask_price is None:
            return orders

        buy_capacity = position_limit - position
        sell_capacity = position_limit + position

        spread = best_ask_price - best_bid_price
        if spread >= spread_threshold:
            buy_price = best_bid_price + 1
            sell_price = best_ask_price - 1
            buy_volume = base_volume
            sell_volume = base_volume

            if position > base_volume / 2:
                buy_price = best_bid_price
                buy_volume = max(0, base_volume - position // 2)

            elif position < -base_volume / 2:
                sell_price = best_ask_price
                sell_volume = max(0, base_volume - abs(position) // 2)

            final_buy_volume = min(buy_volume, buy_capacity)
            if final_buy_volume > 0:
                if buy_price < best_ask_price:
                    orders.append(Order(symbol, buy_price, final_buy_volume))

            final_sell_volume = min(sell_volume, sell_capacity)
            if final_sell_volume > 0:
                if sell_price > best_bid_price:
                    orders.append(Order(symbol, sell_price, -final_sell_volume))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic:
        1. Deserialize state.
        2. Market Make on RAINFOREST_RESIN and KELP.
        3. Mean Reversion (EWMA-based) on SQUID_INK.
        4. Serialize state and return orders.
        """
        self._deserialize_trader_data(state.traderData)
        self.log_position_changes(state.position, state.timestamp)

        result = {symbol: [] for symbol in state.listings.keys()}
        pending_pos_delta = {symbol: 0 for symbol in state.listings.keys()}

        # --- Market Making Logic (Commented Out) ---
        # symbol_resin = "RAINFOREST_RESIN"
        # if symbol_resin in state.order_depths:
        #     depth_resin = state.order_depths[symbol_resin]
        #     current_pos_resin = state.position.get(symbol_resin, 0)
        #     effective_pos_resin = current_pos_resin + pending_pos_delta.get(symbol_resin, 0)
        #
        #     potential_mm_orders_resin = self.get_market_making_orders(
        #         symbol_resin, depth_resin, effective_pos_resin,
        #         self.mm_spread_threshold_resin, self.mm_base_volume_resin
        #     )
        #     for order in potential_mm_orders_resin:
        #         if self._can_place_order(order.symbol, order.quantity, current_pos_resin, pending_pos_delta):
        #             result[order.symbol].append(order)
        #             pending_pos_delta[order.symbol] += order.quantity
        #
        # symbol_kelp = "KELP"
        # if symbol_kelp in state.order_depths:
        #     depth_kelp = state.order_depths[symbol_kelp]
        #     current_pos_kelp = state.position.get(symbol_kelp, 0)
        #     effective_pos_kelp = current_pos_kelp + pending_pos_delta.get(symbol_kelp, 0)
        #
        #     potential_mm_orders_kelp = self.get_market_making_orders(
        #         symbol_kelp, depth_kelp, effective_pos_kelp,
        #         self.mm_spread_threshold_kelp, self.mm_base_volume_kelp
        #     )
        #     for order in potential_mm_orders_kelp:
        #          if self._can_place_order(order.symbol, order.quantity, current_pos_kelp, pending_pos_delta):
        #             result[order.symbol].append(order)
        #             pending_pos_delta[order.symbol] += order.quantity
        # --- End Market Making Logic ---

        # --- SQUID_INK Mean Reversion Logic ---
        symbol_squid = "SQUID_INK"
        if symbol_squid in state.order_depths:
            depth_squid = state.order_depths[symbol_squid]
            current_pos_squid = state.position.get(symbol_squid, 0)
            limit_squid = self.position_limits.get(symbol_squid, 0)

            wap_squid = self._calculate_wap(depth_squid)

            if wap_squid is not None:
                self._calculate_ewma_ewmsd(wap_squid)

                ewmsd = self._get_ewmsd()

                # Add logging for SQUID state
                # logger.print(f"SQUID @ {state.timestamp}: WAP={wap_squid:.2f}, EWMA={self.squid_ewma:.2f}, EWMSD={ewmsd:.4f}, Pos={current_pos_squid}, EntryP={self.squid_entry_price}")

                if self.squid_ewma is not None and ewmsd is not None and ewmsd > 1e-6:
                    try:
                        entry_threshold_abs = self.squid_entry_threshold_pct_hv * ewmsd
                        exit_threshold_abs = self.squid_exit_threshold_pct_hv * ewmsd
                        stop_loss_threshold_abs = self.squid_stop_loss_pct_hv * ewmsd
                        current_deviation = abs(wap_squid - self.squid_ewma)
                        effective_pos_squid = current_pos_squid + pending_pos_delta.get(symbol_squid, 0)

                        # --- Exit Logic ---
                        if effective_pos_squid != 0 and self.squid_entry_price is not None:
                            pnl_per_unit = 0
                            stop_loss_hit = False
                            take_profit_hit = False # Renamed for clarity

                            if effective_pos_squid > 0: # Long position
                                pnl_per_unit = wap_squid - self.squid_entry_price
                                if wap_squid < self.squid_entry_price - stop_loss_threshold_abs:
                                    stop_loss_hit = True
                                    logger.print(f"SQUID: STOP LOSS (Long) @ {wap_squid:.2f} (Entry: {self.squid_entry_price:.2f}, Threshold: {stop_loss_threshold_abs:.2f})")
                                elif current_deviation < exit_threshold_abs and pnl_per_unit > 0:
                                    take_profit_hit = True
                                    logger.print(f"SQUID: TAKE PROFIT (Long) @ {wap_squid:.2f} (Entry: {self.squid_entry_price:.2f}, Deviation: {current_deviation:.2f} < {exit_threshold_abs:.2f})")

                            else: # Short position
                                pnl_per_unit = self.squid_entry_price - wap_squid
                                if wap_squid > self.squid_entry_price + stop_loss_threshold_abs:
                                    stop_loss_hit = True
                                    logger.print(f"SQUID: STOP LOSS (Short) @ {wap_squid:.2f} (Entry: {self.squid_entry_price:.2f}, Threshold: {stop_loss_threshold_abs:.2f})")
                                elif current_deviation < exit_threshold_abs and pnl_per_unit > 0:
                                    take_profit_hit = True
                                    logger.print(f"SQUID: TAKE PROFIT (Short) @ {wap_squid:.2f} (Entry: {self.squid_entry_price:.2f}, Deviation: {current_deviation:.2f} < {exit_threshold_abs:.2f})")

                            # Place exit order if stop loss or take profit is hit
                            if stop_loss_hit or take_profit_hit:
                                close_quantity = -effective_pos_squid # Quantity to flatten position
                                if self._can_place_order(symbol_squid, close_quantity, current_pos_squid, pending_pos_delta):
                                    # Use aggressive limit order (cross the spread by 1 tick) to ensure fill
                                    exit_price = None
                                    if close_quantity > 0 and depth_squid.sell_orders: # Need to buy back
                                        best_ask = min(depth_squid.sell_orders.keys())
                                        exit_price = best_ask # Hit the best ask
                                    elif close_quantity < 0 and depth_squid.buy_orders: # Need to sell back
                                        best_bid = max(depth_squid.buy_orders.keys())
                                        exit_price = best_bid # Hit the best bid

                                    if exit_price is not None:
                                        logger.print(f"SQUID: Placing EXIT order {close_quantity} @ {exit_price}")
                                        result[symbol_squid].append(Order(symbol_squid, exit_price, close_quantity))
                                        pending_pos_delta[symbol_squid] = pending_pos_delta.get(symbol_squid, 0) + close_quantity
                                        self.squid_entry_price = None # Reset entry price after exit
                                    else:
                                         logger.print(f"SQUID: Could not determine exit price for closing {close_quantity}")
                                else:
                                     logger.print(f"SQUID: Cannot place exit order {close_quantity} due to position limits")


                        # --- Entry Logic ---
                        elif effective_pos_squid == 0:
                            upper_band = self.squid_ewma + entry_threshold_abs
                            lower_band = self.squid_ewma - entry_threshold_abs

                            sorted_asks = sorted(depth_squid.sell_orders.items(), key=lambda item: item[0]) if depth_squid.sell_orders else []
                            sorted_bids = sorted(depth_squid.buy_orders.items(), key=lambda item: item[0], reverse=True) if depth_squid.buy_orders else []

                            best_ask = sorted_asks[0][0] if sorted_asks else None
                            best_bid = sorted_bids[0][0] if sorted_bids else None

                            # --- Buy Signal ---
                            if best_ask is not None and best_ask < lower_band:
                                logger.print(f"SQUID: Potential LONG Entry Signal: Best Ask {best_ask:.2f} < Lower Band {lower_band:.2f}")
                                buy_capacity = limit_squid - (current_pos_squid + pending_pos_delta.get(symbol_squid, 0))
                                total_target_qty = min(buy_capacity, self.squid_target_entry_volume)

                                # Calculate split
                                limit_qty_target = math.floor(total_target_qty * self.squid_limit_order_ratio)
                                aggressive_qty_target = total_target_qty - limit_qty_target

                                # 1. Place Passive Limit Order (if qty > 0)
                                if limit_qty_target > 0:
                                    limit_order_price = best_ask
                                    limit_order_qty = limit_qty_target # Place for the target amount
                                    if self._can_place_order(symbol_squid, limit_order_qty, current_pos_squid, pending_pos_delta):
                                        logger.print(f"SQUID: Placing Passive Limit Entry LONG {limit_order_qty} @ {limit_order_price}")
                                        result[symbol_squid].append(Order(symbol_squid, limit_order_price, limit_order_qty))
                                        pending_pos_delta[symbol_squid] += limit_order_qty
                                        # Don't set entry price yet, wait for aggressive order or fill confirmation (complex)
                                    else:
                                        logger.print(f"SQUID: Cannot place passive limit order {limit_order_qty} due to limits.")
                                        aggressive_qty_target = total_target_qty # If limit fails, try all aggressive

                                # 2. Place Aggressive Order (if qty > 0)
                                if aggressive_qty_target > 0:
                                    cumulative_volume_agg = 0
                                    aggressive_price = None
                                    levels_checked_agg = 0
                                    # Calculate volume and price by sweeping levels for the aggressive portion
                                    for ask_price, ask_vol in sorted_asks:
                                        if levels_checked_agg >= self.squid_max_levels_sweep:
                                            break
                                        vol_to_take = min(ask_vol, aggressive_qty_target - cumulative_volume_agg)
                                        if vol_to_take > 0:
                                            cumulative_volume_agg += vol_to_take
                                            aggressive_price = ask_price # Price gets worse as we go deeper
                                        levels_checked_agg += 1
                                        if cumulative_volume_agg >= aggressive_qty_target:
                                            break # Filled target

                                    if cumulative_volume_agg > 0 and aggressive_price is not None:
                                        final_order_qty_agg = cumulative_volume_agg
                                        final_order_price_agg = aggressive_price

                                        if self._can_place_order(symbol_squid, final_order_qty_agg, current_pos_squid, pending_pos_delta):
                                            logger.print(f"SQUID: Placing Aggressive Entry LONG {final_order_qty_agg} @ {final_order_price_agg} (Target: {aggressive_qty_target}, Levels: {levels_checked_agg})")
                                            result[symbol_squid].append(Order(symbol_squid, final_order_price_agg, final_order_qty_agg))
                                            pending_pos_delta[symbol_squid] += final_order_qty_agg
                                            # Set entry price based on the aggressive order for simplicity
                                            self.squid_entry_price = final_order_price_agg
                                        else:
                                            logger.print(f"SQUID: Cannot place aggressive entry order {final_order_qty_agg} due to limits.")
                                    else:
                                        logger.print(f"SQUID: Aggressive LONG Entry: No volume found or price unavailable for target {aggressive_qty_target}.")
                                else:
                                     logger.print(f"SQUID: No aggressive LONG portion needed (Target: {aggressive_qty_target}).")


                            # --- Sell Signal ---
                            elif best_bid is not None and best_bid > upper_band:
                                logger.print(f"SQUID: Potential SHORT Entry Signal: Best Bid {best_bid:.2f} > Upper Band {upper_band:.2f}")
                                sell_capacity = limit_squid + (current_pos_squid + pending_pos_delta.get(symbol_squid, 0))
                                total_target_qty = min(sell_capacity, self.squid_target_entry_volume) # Target is positive volume

                                # Calculate split
                                limit_qty_target = math.floor(total_target_qty * self.squid_limit_order_ratio)
                                aggressive_qty_target = total_target_qty - limit_qty_target

                                # 1. Place Passive Limit Order (if qty > 0)
                                if limit_qty_target > 0:
                                    limit_order_price = best_bid
                                    limit_order_qty = -limit_qty_target # Negative for sell
                                    if self._can_place_order(symbol_squid, limit_order_qty, current_pos_squid, pending_pos_delta):
                                        logger.print(f"SQUID: Placing Passive Limit Entry SHORT {limit_order_qty} @ {limit_order_price}")
                                        result[symbol_squid].append(Order(symbol_squid, limit_order_price, limit_order_qty))
                                        pending_pos_delta[symbol_squid] += limit_order_qty
                                        # Don't set entry price yet
                                    else:
                                        logger.print(f"SQUID: Cannot place passive limit order {limit_order_qty} due to limits.")
                                        aggressive_qty_target = total_target_qty # If limit fails, try all aggressive

                                # 2. Place Aggressive Order (if qty > 0)
                                if aggressive_qty_target > 0:
                                    cumulative_volume_agg = 0
                                    aggressive_price = None
                                    levels_checked_agg = 0
                                    # Calculate volume and price by sweeping levels for the aggressive portion
                                    for bid_price, bid_vol in sorted_bids:
                                        if levels_checked_agg >= self.squid_max_levels_sweep:
                                            break
                                        vol_to_take = min(bid_vol, aggressive_qty_target - cumulative_volume_agg)
                                        if vol_to_take > 0:
                                            cumulative_volume_agg += vol_to_take
                                            aggressive_price = bid_price # Price gets worse (lower)
                                        levels_checked_agg += 1
                                        if cumulative_volume_agg >= aggressive_qty_target:
                                            break # Filled target

                                    if cumulative_volume_agg > 0 and aggressive_price is not None:
                                        final_order_qty_agg = -cumulative_volume_agg # Negative for sell
                                        final_order_price_agg = aggressive_price

                                        if self._can_place_order(symbol_squid, final_order_qty_agg, current_pos_squid, pending_pos_delta):
                                            logger.print(f"SQUID: Placing Aggressive Entry SHORT {final_order_qty_agg} @ {final_order_price_agg} (Target: {aggressive_qty_target}, Levels: {levels_checked_agg})")
                                            result[symbol_squid].append(Order(symbol_squid, final_order_price_agg, final_order_qty_agg))
                                            pending_pos_delta[symbol_squid] += final_order_qty_agg
                                            # Set entry price based on the aggressive order for simplicity
                                            self.squid_entry_price = final_order_price_agg
                                        else:
                                            logger.print(f"SQUID: Cannot place aggressive entry order {final_order_qty_agg} due to limits.")
                                    else:
                                        logger.print(f"SQUID: Aggressive SHORT Entry: No volume found or price unavailable for target {aggressive_qty_target}.")
                                else:
                                     logger.print(f"SQUID: No aggressive SHORT portion needed (Target: {aggressive_qty_target}).")

                    except Exception as e:
                        logger.print(f"SQUID: Error during strategy logic: {e}")
                # else:
                    # logger.print(f"SQUID @ {state.timestamp}: Conditions not met for trading (EWMA/EWMSD invalid or EWMSD too small)")
            # else:
                # logger.print(f"SQUID @ {state.timestamp}: WAP calculation failed.")
        # --- End SQUID_INK Logic ---

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
        symbols_to_log = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        for symbol in symbols_to_log:
            if symbol not in all_symbols: continue
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")

        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

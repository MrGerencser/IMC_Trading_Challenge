import json
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import warnings
import pandas as pd
from collections import deque

# Assuming datamodel.py defines these classes:
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# --- Black-Scholes Functions ---
def cdf_standard_normal(x):
    """Approximate CDF of the standard normal distribution."""
    try:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    except ValueError:
        return np.nan

def vega(S, K, r, T, sigma):
    """Partial derivative of the call price wrt sigma (aka Vega)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    try:
        sigma_sqrt_T = sigma * math.sqrt(T)
        if abs(sigma_sqrt_T) < 1e-10: return 0.0
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        pdf_d1 = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)
        return S * math.sqrt(T) * pdf_d1
    except (ValueError, OverflowError):
        return 0.0

def black_scholes_call_price(S, K, r, T, sigma):
    """Computes the Black–Scholes price for a call option."""
    if T <= 1e-10:
        return max(S - K, 0.0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    try:
        sigma_sqrt_T = sigma * math.sqrt(T)
        if abs(sigma_sqrt_T) < 1e-10: return max(S - K, 0.0)
        d1_num = math.log(S/K) + (r + 0.5 * sigma**2) * T
        d1 = d1_num / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        call_val = (S * cdf_standard_normal(d1)) - (K * math.exp(-r * T) * cdf_standard_normal(d2))
        return max(call_val, 0.0)
    except (ValueError, OverflowError):
        return max(S - K, 0.0)

def implied_vol_call_price(S, K, r, T, market_price, initial_guess=0.16, tol=1e-6, max_iterations=100):
    """Numerically find implied volatility via Newton–Raphson."""
    intrinsic_value = max(S - K, 0.0)
    if market_price < intrinsic_value - tol:
        return np.nan
    if T < 1e-5:
        if abs(market_price - intrinsic_value) < tol:
             return 1e-5
        else:
             return np.nan

    sigma = initial_guess
    for i in range(max_iterations):
        try:
            price = black_scholes_call_price(S, K, r, T, sigma)
            v = vega(S, K, r, T, sigma)
        except Exception:
            return np.nan

        diff = price - market_price
        if abs(diff) < tol:
            return max(sigma, 1e-5)

        if v < 1e-8:
             if abs(diff) < tol * 10: return max(sigma, 1e-5)
             return np.nan

        sigma = sigma - diff / v
        sigma = max(1e-5, min(sigma, 10.0))

    if abs(diff) < tol * 100: return max(sigma, 1e-5)
    return np.nan
# --- End Black-Scholes Functions ---

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        output = [
            self.compress_state(state, trader_data),
            self.compress_orders(orders),
            conversions,
        ]
        output_json = self.to_json(output)

        if len(output_json) > self.max_log_length:
            base_output = [
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
            ]
            base_length = len(self.to_json(base_output))
            base_length += 24 # Approx length for keys

            available_length = self.max_log_length - base_length
            max_item_length = available_length // 2

            if max_item_length < 0: max_item_length = 0

            truncated_trader_data = self.truncate(trader_data, max_item_length)
            truncated_logs = self.truncate(self.logs, max_item_length)

            final_output = [
                self.compress_state(state, truncated_trader_data),
                self.compress_orders(orders),
                conversions,
                truncated_logs,
            ]
            print(self.to_json(final_output))

        else:
             final_output = [
                self.compress_state(state, trader_data),
                self.compress_orders(orders),
                conversions,
                self.logs,
            ]
             print(self.to_json(final_output))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data, self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length: return value
        if max_length < 3: return value[:max_length]
        return value[: max_length - 3] + "..."

logger = Logger()

# Structure for storing open volatility position details
class VolPosition:
    def __init__(self, quantity: int, entry_price: float, entry_diff: float, entry_vega: float):
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_diff = entry_diff # v_t - v_t_fit at entry
        self.entry_vega = entry_vega

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantity": self.quantity, "entry_price": self.entry_price,
            "entry_diff": self.entry_diff, "entry_vega": self.entry_vega,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VolPosition':
        return VolPosition(
            data["quantity"], data["entry_price"],
            data["entry_diff"], data["entry_vega"],
        )

class Trader:
    def __init__(self):
        # --- General Parameters ---
        self.position_limits = {
            # From MM/Arb
            "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
            "CROISSANTS": 250, "JAMS": 350,
            "RAINFOREST_RESIN": 50, "KELP": 50,
            "DJEMBES": 60,
            # From BS
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        self.previous_positions = {}
        self.trader_data = ""

        # --- Basket Definitions (from MM/Arb) ---
        self.basket_components = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }

        # --- MM Strategy Parameters (from MM/Arb) ---
        self.mm_spread_threshold_resin = 7
        self.mm_base_volume_resin = 25
        self.mm_spread_threshold_kelp = 2
        self.mm_base_volume_kelp = 25

        # --- Basket Arbitrage Parameters (from MM/Arb) ---
        self.arb_ewma_span = 25
        self.arb_alpha = 2 / (self.arb_ewma_span + 1)
        self.arb_entry_z_threshold = 1.5
        self.arb_exit_z_threshold = 0.5
        self.arb_stop_loss_z_threshold = 4.0
        self.arb_min_profit_threshold_basket1 = 160.0
        self.arb_min_profit_threshold_basket2 = 105.0
        self.arb_trade_size = 12

        # --- Volatility Strategy Parameters (from BS) ---
        self.vol_vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        self.vol_underlying = "VOLCANIC_ROCK"
        self.vol_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500.0, "VOLCANIC_ROCK_VOUCHER_9750": 9750.0,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000.0, "VOLCANIC_ROCK_VOUCHER_10250": 10250.0,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500.0
        }
        self.vol_risk_free_rate = 0.0
        self.vol_total_days = 7.0
        self.vol_days_per_year = 252.0
        self.vol_max_timestamp_per_day = 1_000_000.0
        self.underlying_price_history_len = 5
        self.vol_min_points_for_fit = 4
        self.vol_max_fit_mse = 0.0005
        self.vol_use_weighted_fit = False
        self.vol_base_diff_threshold = 0.012
        self.vol_base_tp_threshold = 0.002
        self.vol_base_sl_iv_diff = 0.04
        self.vol_sl_price_pct = 0.2
        self.vol_base_volume = 10
        self.vol_max_size_multiplier = 5.0
        self.vol_size_vega_sensitivity = 0.5
        self.vol_size_diff_sensitivity = 0.5
        self.vol_use_limit_orders_entry = True
        self.vol_max_spread_pct = 0.05
        self.vol_adjust_for_tte = True
        self.vol_min_tte_aggressiveness = 0.2
        self.vol_short_bias_factor = 2.0

        # --- State Variables ---
        # Basket Arb State
        self.arb_spread_ewma: Dict[Symbol, float] = {}
        self.arb_spread_ewm_variance: Dict[Symbol, float] = {}
        self.arb_position_active: Dict[Symbol, bool] = {}
        # Volatility State
        self.open_vol_positions: Dict[Symbol, VolPosition] = {}
        self.underlying_price_history: deque[float] = deque(maxlen=self.underlying_price_history_len)


    # --- Helper Methods ---
    def _load_trader_data(self, trader_data: str):
        """Load state from trader_data string using standard json."""
        # Reset state first
        self.arb_spread_ewma = {}
        self.arb_spread_ewm_variance = {}
        self.arb_position_active = {}
        self.open_vol_positions = {}
        # Note: underlying_price_history is not persisted

        if trader_data:
            try:
                data = json.loads(trader_data)
                self.arb_spread_ewma = data.get("arb_spread_ewma", {})
                self.arb_spread_ewm_variance = data.get("arb_spread_ewm_variance", {})
                self.arb_position_active = data.get("arb_position_active", {})
                self.open_vol_positions = {
                    symbol: VolPosition.from_dict(pos_data)
                    for symbol, pos_data in data.get("open_vol_positions", {}).items()
                }
                logger.print("Loaded trader data successfully.")
            except Exception as e:
                logger.print(f"Error loading trader data: {e}. Resetting state.")
                # Explicitly reset again on error
                self.arb_spread_ewma = {}
                self.arb_spread_ewm_variance = {}
                self.arb_position_active = {}
                self.open_vol_positions = {}
        else:
             logger.print("No trader data found, starting fresh.")


    def _save_trader_data(self) -> str:
        """Save state to trader_data string using standard json."""
        try:
            data_to_save = {
                "arb_spread_ewma": self.arb_spread_ewma,
                "arb_spread_ewm_variance": self.arb_spread_ewm_variance,
                "arb_position_active": self.arb_position_active,
                "open_vol_positions": {
                    symbol: pos.to_dict()
                    for symbol, pos in self.open_vol_positions.items()
                }
            }
            return json.dumps(data_to_save, separators=(",", ":"))
        except Exception as e:
            logger.print(f"Error saving trader data: {e}")
            return ""

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

    def _calculate_wap(self, depth: OrderDepth, levels: int = 1) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        sorted_bids = sorted(depth.buy_orders.items(), key=lambda item: item[0], reverse=True)
        sorted_asks = sorted(depth.sell_orders.items(), key=lambda item: item[0])
        best_bid_price, best_bid_vol = sorted_bids[0]
        best_ask_price, best_ask_vol = sorted_asks[0]
        if best_bid_vol + best_ask_vol == 0:
             return (best_bid_price + best_ask_price) / 2
        wap = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)
        return wap

    def _get_best_price(self, depth: OrderDepth, side: str) -> Optional[int]:
        if side == 'bid' and depth.buy_orders:
            return max(depth.buy_orders.keys())
        elif side == 'ask' and depth.sell_orders:
            return min(depth.sell_orders.keys())
        return None

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def _get_smoothed_underlying_price(self, state: TradingState) -> Optional[float]:
        mid_price = self._get_mid_price(self.vol_underlying, state)
        if mid_price is not None:
            self.underlying_price_history.append(mid_price)
        if not self.underlying_price_history:
            return None
        return np.mean(self.underlying_price_history)

    def _calculate_tte(self, timestamp: int) -> float:
        total_timestamps = self.vol_total_days * self.vol_max_timestamp_per_day
        current_timestamp = min(timestamp, total_timestamps)
        timestamps_remaining = max(0.0, total_timestamps - current_timestamp)
        T = timestamps_remaining / (self.vol_days_per_year * self.vol_max_timestamp_per_day)
        return T + 1e-9

    def _get_tte_adjustment_factor(self, T: float) -> float:
        if not self.vol_adjust_for_tte or self.vol_total_days <= 0:
            return 1.0
        initial_T = self.vol_total_days / self.vol_days_per_year
        if initial_T <= 1e-6: return 1.0
        factor = T / initial_T
        adjusted_factor = self.vol_min_tte_aggressiveness + (1.0 - self.vol_min_tte_aggressiveness) * factor
        return max(self.vol_min_tte_aggressiveness, min(1.0, adjusted_factor))

    def _calculate_dynamic_size(self, base_volume: int, iv_diff: float, option_vega: float, tte_factor: float) -> int:
        if option_vega < 1e-4: option_vega = 1e-4
        diff_scale = min(1.0, abs(iv_diff) / (self.vol_base_diff_threshold * 2))
        vega_scale = min(1.0, option_vega / 100.0) # Tune 100
        size_confidence_factor = 1.0 + (self.vol_max_size_multiplier - 1.0) * (
            self.vol_size_diff_sensitivity * diff_scale +
            self.vol_size_vega_sensitivity * vega_scale
        ) / (self.vol_size_diff_sensitivity + self.vol_size_vega_sensitivity + 1e-6)
        adjusted_size_factor = size_confidence_factor * tte_factor
        final_volume = int(round(base_volume * max(0.1, min(self.vol_max_size_multiplier, adjusted_size_factor))))
        return max(1, final_volume)

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
        if variance is None: return None
        if variance < 0: return 0.0
        return math.sqrt(variance)

    def _calculate_synthetic_prices(self, components: Dict[Symbol, int], order_depths: Dict[Symbol, OrderDepth]) -> Tuple[Optional[float], Optional[float]]:
        synthetic_bid = 0.0
        synthetic_ask = 0.0
        all_components_available = True
        for component, quantity in components.items():
            if component not in order_depths:
                logger.print(f"ARB: Missing order depth for component {component}")
                all_components_available = False; break
            comp_depth = order_depths[component]
            comp_best_bid = self._get_best_price(comp_depth, 'bid')
            comp_best_ask = self._get_best_price(comp_depth, 'ask')
            if comp_best_bid is None or comp_best_ask is None:
                logger.print(f"ARB: Missing bid/ask for component {component}")
                all_components_available = False; break
            synthetic_bid += comp_best_bid * quantity
            synthetic_ask += comp_best_ask * quantity
        if not all_components_available: return None, None
        return synthetic_bid, synthetic_ask

    # --- Strategy Methods ---

    def get_market_making_orders(self, symbol: Symbol, depth: OrderDepth, position: int, spread_threshold: int, base_volume: int) -> list[Order]:
        """Market making for a single symbol."""
        orders = []
        position_limit = self.position_limits.get(symbol)
        if position_limit is None: return []

        best_bid_price = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask_price = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_bid_price is None or best_ask_price is None: return orders

        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        spread = best_ask_price - best_bid_price

        if spread >= spread_threshold:
            buy_price = best_bid_price + 1
            sell_price = best_ask_price - 1
            buy_volume = base_volume
            sell_volume = base_volume

            skew_factor = position / position_limit if position_limit != 0 else 0
            buy_volume = int(buy_volume * (1 - skew_factor))
            sell_volume = int(sell_volume * (1 + skew_factor))

            if position > position_limit * 0.3: buy_price = best_bid_price
            elif position < -position_limit * 0.3: sell_price = best_ask_price

            final_buy_volume = min(buy_volume, buy_capacity)
            if final_buy_volume > 0 and buy_price < best_ask_price:
                orders.append(Order(symbol, buy_price, final_buy_volume))

            final_sell_volume = min(sell_volume, sell_capacity)
            if final_sell_volume > 0 and sell_price > best_bid_price:
                orders.append(Order(symbol, sell_price, -final_sell_volume))
        return orders

    def _get_arbitrage_orders(self, basket_symbol: Symbol, state: TradingState, pending_pos_delta: Dict[Symbol, int]) -> List[Order]:
        """Basket arbitrage strategy."""
        orders: List[Order] = []
        components = self.basket_components.get(basket_symbol)
        if not components: return orders

        # --- 1. Get Prices ---
        if basket_symbol not in state.order_depths: return orders
        basket_depth = state.order_depths[basket_symbol]
        actual_basket_bid = self._get_best_price(basket_depth, 'bid')
        actual_basket_ask = self._get_best_price(basket_depth, 'ask')
        if actual_basket_bid is None or actual_basket_ask is None: return orders
        synthetic_bid, synthetic_ask = self._calculate_synthetic_prices(components, state.order_depths)
        if synthetic_bid is None or synthetic_ask is None: return orders

        # --- 2. Calculate Spreads ---
        spread_buy_basket = synthetic_bid - actual_basket_ask
        spread_sell_basket = actual_basket_bid - synthetic_ask
        current_spread_metric = (spread_buy_basket + spread_sell_basket) / 2

        # --- 3. Update EWMA/EWMSD and Calculate Z-Score ---
        self._calculate_ewma_ewmsd(current_spread_metric, basket_symbol, self.arb_spread_ewma, self.arb_spread_ewm_variance, self.arb_alpha)
        spread_ewma = self.arb_spread_ewma.get(basket_symbol)
        spread_ewmsd = self._get_ewmsd(basket_symbol, self.arb_spread_ewm_variance)
        z_score: Optional[float] = None
        if spread_ewma is not None and spread_ewmsd is not None and spread_ewmsd > 1e-6:
            z_score = (current_spread_metric - spread_ewma) / spread_ewmsd

        # --- 4. Trading Logic ---
        current_basket_pos = state.position.get(basket_symbol, 0)
        is_active = self.arb_position_active.get(basket_symbol, False)

        # --- Exit Logic ---
        if is_active and z_score is not None:
            exit_signal = False; stop_loss_signal = False; is_profitable_exit = False
            z_score_exit_condition = abs(z_score) < self.arb_exit_z_threshold
            if current_basket_pos > 0: is_profitable_exit = spread_sell_basket > 0
            elif current_basket_pos < 0: is_profitable_exit = spread_buy_basket > 0

            if z_score_exit_condition and is_profitable_exit:
                exit_signal = True; logger.print(f"ARB {basket_symbol}: TAKE PROFIT signal (Z={z_score:.2f}, Profitable)")
            elif (current_basket_pos > 0 and z_score > self.arb_stop_loss_z_threshold) or \
                 (current_basket_pos < 0 and z_score < -self.arb_stop_loss_z_threshold):
                 stop_loss_signal = True; logger.print(f"ARB {basket_symbol}: STOP LOSS signal (Z={z_score:.2f})")

            if exit_signal or stop_loss_signal:
                logger.print(f"ARB {basket_symbol}: Attempting to close position.")
                trade_qty = 0
                if current_basket_pos > 0: trade_qty = self.arb_trade_size
                elif current_basket_pos < 0: trade_qty = -self.arb_trade_size
                if trade_qty == 0:
                     logger.print(f"ARB {basket_symbol}: Exit signal but pos=0. Resetting."); self.arb_position_active[basket_symbol] = False; return orders

                if self._can_place_order(basket_symbol, -trade_qty, current_basket_pos, pending_pos_delta):
                    basket_close_order = Order(basket_symbol, actual_basket_bid if trade_qty > 0 else actual_basket_ask, -trade_qty)
                    potential_orders = [basket_close_order]
                    can_close_components = True
                    for component, quantity in components.items():
                        comp_qty_to_close = -quantity * trade_qty
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty_to_close, comp_current_pos, pending_pos_delta):
                            can_close_components = False; logger.print(f"ARB {basket_symbol}: Cannot close {component} limit. ABORT EXIT."); break
                        comp_depth = state.order_depths[component]
                        comp_close_price = self._get_best_price(comp_depth, 'ask') if comp_qty_to_close > 0 else self._get_best_price(comp_depth, 'bid')
                        if comp_close_price: potential_orders.append(Order(component, comp_close_price, comp_qty_to_close))
                        else: logger.print(f"ARB {basket_symbol}: ERROR - Missing close price for {component}."); can_close_components = False; break
                    if can_close_components:
                        logger.print(f"ARB {basket_symbol}: Placing exit orders.")
                        for order in potential_orders:
                            orders.append(order)
                            pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        self.arb_position_active[basket_symbol] = False
                else: logger.print(f"ARB {basket_symbol}: Cannot close basket limit.")

        # --- Entry Logic ---
        elif not is_active and z_score is not None:
            trade_qty = self.arb_trade_size
            basket_order = None; component_orders = []; can_place_all = True
            min_profit_threshold = 0.0
            if basket_symbol == "PICNIC_BASKET1": min_profit_threshold = self.arb_min_profit_threshold_basket1
            elif basket_symbol == "PICNIC_BASKET2": min_profit_threshold = self.arb_min_profit_threshold_basket2
            else: logger.print(f"ARB WARNING: Unknown basket '{basket_symbol}'."); return orders

            # Strategy 1: Buy Basket, Sell Components
            if z_score < -self.arb_entry_z_threshold and spread_buy_basket > min_profit_threshold:
                logger.print(f"ARB {basket_symbol}: ENTRY LONG BASKET (Z={z_score:.2f}, Spread={spread_buy_basket:.2f})")
                if not self._can_place_order(basket_symbol, trade_qty, current_basket_pos, pending_pos_delta):
                    can_place_all = False; logger.print(f"ARB {basket_symbol}: Cannot place basket buy limit.")
                else:
                    basket_order = Order(basket_symbol, actual_basket_ask, trade_qty)
                    for component, quantity in components.items():
                        comp_qty = -quantity * trade_qty
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty, comp_current_pos, pending_pos_delta):
                            can_place_all = False; logger.print(f"ARB {basket_symbol}: Cannot place {component} sell limit."); break
                        else:
                            comp_depth = state.order_depths[component]
                            comp_price = self._get_best_price(comp_depth, 'bid')
                            if comp_price: component_orders.append(Order(component, comp_price, comp_qty))
                            else: can_place_all = False; logger.print(f"ARB {basket_symbol}: ERROR - Missing sell price {component}."); break

            # Strategy 2: Sell Basket, Buy Components
            elif z_score > self.arb_entry_z_threshold and spread_sell_basket > min_profit_threshold:
                logger.print(f"ARB {basket_symbol}: ENTRY SHORT BASKET (Z={z_score:.2f}, Spread={spread_sell_basket:.2f})")
                trade_qty = -self.arb_trade_size
                if not self._can_place_order(basket_symbol, trade_qty, current_basket_pos, pending_pos_delta):
                    can_place_all = False; logger.print(f"ARB {basket_symbol}: Cannot place basket sell limit.")
                else:
                    basket_order = Order(basket_symbol, actual_basket_bid, trade_qty)
                    for component, quantity in components.items():
                        comp_qty = -quantity * trade_qty
                        comp_current_pos = state.position.get(component, 0)
                        if not self._can_place_order(component, comp_qty, comp_current_pos, pending_pos_delta):
                            can_place_all = False; logger.print(f"ARB {basket_symbol}: Cannot place {component} buy limit."); break
                        else:
                            comp_depth = state.order_depths[component]
                            comp_price = self._get_best_price(comp_depth, 'ask')
                            if comp_price: component_orders.append(Order(component, comp_price, comp_qty))
                            else: can_place_all = False; logger.print(f"ARB {basket_symbol}: ERROR - Missing buy price {component}."); break

            if can_place_all and basket_order and component_orders:
                logger.print(f"ARB {basket_symbol}: Placing entry orders.")
                orders.append(basket_order)
                pending_pos_delta[basket_symbol] = pending_pos_delta.get(basket_symbol, 0) + basket_order.quantity
                for order in component_orders:
                    orders.append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                self.arb_position_active[basket_symbol] = True
        return orders

    def run_volatility_strategy(self, state: TradingState) -> Tuple[List[Order], set[Symbol], set[Symbol], Dict[Symbol, VolPosition]]:
        """Volatility curve fitting strategy."""
        potential_orders: List[Order] = []
        symbols_to_close: set[Symbol] = set()
        symbols_to_open: set[Symbol] = set()
        positions_to_update: Dict[Symbol, VolPosition] = {}
        timestamp = state.timestamp

        # 1. Calculate TTE & Adjustment Factor
        T = self._calculate_tte(timestamp)
        if T < 1e-7: return [], set(), set(), {}
        tte_factor = self._get_tte_adjustment_factor(T)
        long_bias_multiplier = 1.0 + (self.vol_short_bias_factor - 1.0) * (1.0 - tte_factor)
        short_bias_multiplier = 1.0
        safe_tte_factor = max(tte_factor, 1e-6)
        current_long_entry_threshold = (self.vol_base_diff_threshold * long_bias_multiplier) / safe_tte_factor
        current_short_entry_threshold = (self.vol_base_diff_threshold * short_bias_multiplier) / safe_tte_factor
        current_tp_threshold = self.vol_base_tp_threshold / safe_tte_factor
        current_sl_iv_diff = self.vol_base_sl_iv_diff / safe_tte_factor

        # 2. Get Smoothed Underlying Price (S)
        S = self._get_smoothed_underlying_price(state)
        if S is None: return [], set(), set(), {}

        # 3. Gather Market Data
        market_data_list: List[Dict[str, Any]] = []
        for symbol in self.vol_vouchers:
            K = self.vol_strikes.get(symbol)
            depth = state.order_depths.get(symbol)
            if K is None or not depth or not depth.buy_orders or not depth.sell_orders: continue
            best_bid = max(depth.buy_orders.keys()); best_ask = min(depth.sell_orders.keys())
            market_mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            if market_mid_price > 0 and (spread / market_mid_price) > self.vol_max_spread_pct: continue
            v_t = implied_vol_call_price(S, K, self.vol_risk_free_rate, T, market_mid_price)
            m_t = np.nan
            if S > 0 and T > 1e-10:
                try: m_t = math.log(K / S) / math.sqrt(T)
                except (ValueError, OverflowError): m_t = np.nan
            if pd.notna(v_t) and pd.notna(m_t):
                option_vega = vega(S, K, self.vol_risk_free_rate, T, v_t)
                weight = option_vega if self.vol_use_weighted_fit and pd.notna(option_vega) and option_vega > 1e-3 else 1.0
                market_data_list.append({'symbol': symbol, 'm_t': m_t, 'v_t': v_t, 'K': K, 'market_price': market_mid_price,
                                         'best_bid': best_bid, 'best_ask': best_ask, 'vega': option_vega if pd.notna(option_vega) else 0.0, 'weight': weight})

        # 4. Fit the Curve & Check Quality
        if len(market_data_list) < self.vol_min_points_for_fit: return [], set(), set(), {}
        try:
            m_t_values = np.array([d['m_t'] for d in market_data_list])
            v_t_values = np.array([d['v_t'] for d in market_data_list])
            weights = np.array([d['weight'] for d in market_data_list]) if self.vol_use_weighted_fit else None
            coeffs = np.polyfit(m_t_values, v_t_values, 2, w=weights)
            poly = np.poly1d(coeffs)
            residuals = v_t_values - poly(m_t_values)
            mse = np.mean(residuals**2)
            if mse > self.vol_max_fit_mse:
                logger.print(f"Vol Strat: Skipping trade, poor fit quality (MSE={mse:.6f} > {self.vol_max_fit_mse:.6f})")
                return [], set(), set(), {}
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.print(f"Vol Strat: Curve fitting failed: {e}")
            return [], set(), set(), {}

        # --- Generate Orders ---
        processed_symbols = set()

        # 5. Generate Closing Orders
        for symbol, open_pos in list(self.open_vol_positions.items()):
            data_point = next((d for d in market_data_list if d['symbol'] == symbol), None)
            if data_point is None: continue
            v_t = data_point['v_t']; v_t_fit = poly(data_point['m_t']); diff = v_t - v_t_fit
            best_bid = data_point['best_bid']; best_ask = data_point['best_ask']; market_price = data_point['market_price']
            close_reason = None; close_price = None; close_quantity = -open_pos.quantity

            if open_pos.quantity > 0 and diff > -current_tp_threshold: close_reason = f"TP Long"; close_price = best_bid
            elif open_pos.quantity < 0 and diff < current_tp_threshold: close_reason = f"TP Short"; close_price = best_ask
            if close_reason is None:
                if open_pos.quantity > 0 and diff < open_pos.entry_diff - current_sl_iv_diff: close_reason = f"SL IV Long"; close_price = best_bid
                elif open_pos.quantity < 0 and diff > open_pos.entry_diff + current_sl_iv_diff: close_reason = f"SL IV Short"; close_price = best_ask
            if close_reason is None and open_pos.entry_price > 0:
                price_change_pct = (market_price - open_pos.entry_price) / open_pos.entry_price
                if open_pos.quantity > 0 and price_change_pct < -self.vol_sl_price_pct: close_reason = f"SL Price Long"; close_price = best_bid
                elif open_pos.quantity < 0 and price_change_pct > self.vol_sl_price_pct: close_reason = f"SL Price Short"; close_price = best_ask

            if close_reason and close_price is not None:
                logger.print(f"Vol Close {symbol}: {close_reason}. Order: {close_quantity} @{close_price}")
                potential_orders.append(Order(symbol, close_price, close_quantity))
                symbols_to_close.add(symbol); processed_symbols.add(symbol)

        # 6. Generate Opening Orders
        for data_point in market_data_list:
            symbol = data_point['symbol']
            if symbol in processed_symbols: continue
            v_t = data_point['v_t']; v_t_fit = poly(data_point['m_t']); diff = v_t - v_t_fit
            option_vega = data_point['vega']; K = data_point['K']
            best_bid = data_point['best_bid']; best_ask = data_point['best_ask']
            order_quantity = 0; order_price = None; entry_reason = None; target_pos = None

            if diff < -current_long_entry_threshold:
                order_quantity = self._calculate_dynamic_size(self.vol_base_volume, diff, option_vega, tte_factor)
                entry_reason = f"Entry BUY CHEAP (diff={diff:.4f} < {-current_long_entry_threshold:.4f})"
                if self.vol_use_limit_orders_entry:
                    theo_price = black_scholes_call_price(S, K, self.vol_risk_free_rate, T, v_t_fit)
                    order_price = math.floor(theo_price); order_price = min(order_price, best_ask)
                else: order_price = best_ask
                target_pos = VolPosition(order_quantity, order_price, diff, option_vega)
            elif diff > current_short_entry_threshold:
                order_quantity = -self._calculate_dynamic_size(self.vol_base_volume, diff, option_vega, tte_factor)
                entry_reason = f"Entry SELL EXPENSIVE (diff={diff:.4f} > {current_short_entry_threshold:.4f})"
                if self.vol_use_limit_orders_entry:
                    theo_price = black_scholes_call_price(S, K, self.vol_risk_free_rate, T, v_t_fit)
                    order_price = math.ceil(theo_price); order_price = max(order_price, best_bid)
                else: order_price = best_bid
                target_pos = VolPosition(order_quantity, order_price, diff, option_vega)

            if entry_reason and order_price is not None and order_quantity != 0 and target_pos is not None:
                if order_price <= 0: continue
                # logger.print(f"Vol {entry_reason}. Order: {order_quantity} @{order_price}") # Reduce log noise
                potential_orders.append(Order(symbol, order_price, order_quantity))
                symbols_to_open.add(symbol); positions_to_update[symbol] = target_pos; processed_symbols.add(symbol)

        return potential_orders, symbols_to_close, symbols_to_open, positions_to_update

    # --- Main Run Loop ---
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic loop combining all strategies.
        """
        self._load_trader_data(state.traderData)
        self.log_position_changes(state.position, state.timestamp)

        result = {symbol: [] for symbol in state.listings.keys()}
        # Use a single pending delta dict, shared across strategies
        pending_pos_delta = {symbol: 0 for symbol in state.listings.keys()}

        # --- 1. Market Making Logic ---
        mm_symbols = ["RAINFOREST_RESIN", "KELP"]
        for symbol in mm_symbols:
             if symbol in state.order_depths:
                 depth = state.order_depths[symbol]
                 current_pos = state.position.get(symbol, 0)
                 # Pass effective position including already pending orders for THIS symbol
                 effective_pos = current_pos + pending_pos_delta.get(symbol, 0)
                 spread_thresh = self.mm_spread_threshold_resin if symbol == "RAINFOREST_RESIN" else self.mm_spread_threshold_kelp
                 base_vol = self.mm_base_volume_resin if symbol == "RAINFOREST_RESIN" else self.mm_base_volume_kelp

                 potential_mm_orders = self.get_market_making_orders(symbol, depth, effective_pos, spread_thresh, base_vol)
                 for order in potential_mm_orders:
                     # Check against original current_pos and the GLOBAL pending_pos_delta
                     if self._can_place_order(order.symbol, order.quantity, current_pos, pending_pos_delta):
                         result[order.symbol].append(order)
                         pending_pos_delta[order.symbol] += order.quantity
                     # else: logger.print(f"MM Order skipped for {order.symbol} due to limit.")

        # --- 2. Basket Arbitrage Logic ---
        arb_symbols = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
        for basket_symbol in arb_symbols:
            # Pass the GLOBAL pending_pos_delta, which _get_arbitrage_orders will update if it places trades
            potential_arb_orders = self._get_arbitrage_orders(basket_symbol, state, pending_pos_delta)
            # Add the returned orders (already limit-checked and delta-updated)
            for order in potential_arb_orders:
                 result[order.symbol].append(order)

        # --- 3. Volatility Strategy Logic ---
        potential_vol_orders, symbols_to_close, symbols_to_open, positions_to_update = self.run_volatility_strategy(state)
        executed_vol_trades: Dict[Symbol, int] = {} # Track executed vol trades

        for order in potential_vol_orders:
            current_pos = state.position.get(order.symbol, 0)
            # Check against original current_pos and the GLOBAL pending_pos_delta
            if self._can_place_order(order.symbol, order.quantity, current_pos, pending_pos_delta):
                result[order.symbol].append(order)
                pending_pos_delta[order.symbol] += order.quantity
                executed_vol_trades[order.symbol] = order.quantity
            # else: logger.print(f"Volatility Order skipped for {order.symbol} due to limit.")

        # Update open_vol_positions based on *executed* vol trades
        for symbol, executed_quantity in executed_vol_trades.items():
            if symbol in symbols_to_close:
                if symbol in self.open_vol_positions:
                    if executed_quantity == -self.open_vol_positions[symbol].quantity:
                        del self.open_vol_positions[symbol]; logger.print(f"Closed Vol Position: {symbol}")
                    else:
                        logger.print(f"WARNING: Partial/Mismatched close {symbol}. Exec: {executed_quantity}, Open: {self.open_vol_positions[symbol].quantity}. Adjusting.")
                        self.open_vol_positions[symbol].quantity += executed_quantity
                        if abs(self.open_vol_positions[symbol].quantity) < 1: del self.open_vol_positions[symbol]
            elif symbol in symbols_to_open:
                if symbol not in self.open_vol_positions:
                    if symbol in positions_to_update:
                        new_pos_details = positions_to_update[symbol]
                        self.open_vol_positions[symbol] = VolPosition(executed_quantity, new_pos_details.entry_price, new_pos_details.entry_diff, new_pos_details.entry_vega)
                        logger.print(f"Opened Vol Position: {symbol}, Qty: {executed_quantity} @Assumed {new_pos_details.entry_price}")
                    else: logger.print(f"ERROR: Executed open {symbol} but no details found.")
                else:
                    logger.print(f"WARNING: Adding to existing vol pos {symbol}. Exec: {executed_quantity}")
                    self.open_vol_positions[symbol].quantity += executed_quantity # Simple update

        # --- Final Steps ---
        conversions = 0
        trader_data = self._save_trader_data()

        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log: logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")
        if self.open_vol_positions: logger.print(f"Open Vol Positions: { {s: p.to_dict() for s, p in self.open_vol_positions.items()} }")
        if self.arb_position_active: logger.print(f"Active Arb Positions: {self.arb_position_active}")


        self.previous_positions = state.position.copy()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        """Log changes in positions."""
        log_entry = []
        all_symbols = set(current_positions.keys()) | set(self.previous_positions.keys())
        symbols_to_log = sorted(list(self.position_limits.keys())) # Log all traded symbols

        for symbol in symbols_to_log:
            if symbol not in all_symbols: continue
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")
        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

import json
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import warnings
import pandas as pd
from collections import deque # For price smoothing



# Assuming datamodel.py defines these classes:
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# --- Black-Scholes Functions (Unchanged) ---
def cdf_standard_normal(x):
    """Approximate CDF of the standard normal distribution."""
    try:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    except ValueError:
        return np.nan # Handle potential math errors

def vega(S, K, r, T, sigma):
    """Partial derivative of the call price wrt sigma (aka Vega)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    try:
        sigma_sqrt_T = sigma * math.sqrt(T)
        if abs(sigma_sqrt_T) < 1e-10: return 0.0 # Avoid division by zero
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
        return np.nan # Market price below intrinsic
    if T < 1e-5:
        if abs(market_price - intrinsic_value) < tol:
             return 1e-5 # Near expiry, price is intrinsic -> vol near 0
        else:
             return np.nan # Near expiry, but price differs from intrinsic -> error/arbitrage?

    sigma = initial_guess
    for i in range(max_iterations):
        try:
            price = black_scholes_call_price(S, K, r, T, sigma)
            v = vega(S, K, r, T, sigma)
        except Exception:
            return np.nan # Error during BS calculation

        diff = price - market_price
        if abs(diff) < tol:
            return max(sigma, 1e-5)

        if v < 1e-8: # Vega too small
             if abs(diff) < tol * 10: return max(sigma, 1e-5) # Close enough given low vega
             return np.nan # Failed due to low vega

        # Newton step
        sigma = sigma - diff / v
        sigma = max(1e-5, min(sigma, 10.0)) # Clamp sigma

    # Did not converge
    if abs(diff) < tol * 100: return max(sigma, 1e-5) # Return if reasonably close
    return np.nan
# --- End Black-Scholes Functions ---


class Logger:
    # ... (Logger class remains unchanged) ...
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
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
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
    # ... (VolPosition class remains unchanged) ...
    def __init__(self, quantity: int, entry_price: float, entry_diff: float, entry_vega: float):
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_diff = entry_diff # v_t - v_t_fit at entry
        self.entry_vega = entry_vega

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_diff": self.entry_diff,
            "entry_vega": self.entry_vega,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VolPosition':
        return VolPosition(
            data["quantity"],
            data["entry_price"],
            data["entry_diff"],
            data["entry_vega"],
        )

class Trader:
    def __init__(self):
        # --- General Parameters ---
        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
            # Add other product limits if needed
        }
        self.previous_positions = {}
        self.trader_data = "" # Will store JSON string of open_vol_positions

        # --- Volatility Strategy Parameters ---
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
        self.vol_total_days = 7.0 # Total duration of the option
        self.vol_days_per_year = 252.0
        self.vol_max_timestamp_per_day = 1_000_000.0

        # Underlying Price Smoothing
        self.underlying_price_history_len = 5 # Number of ticks to average for S
        self.underlying_price_history: deque[float] = deque(maxlen=self.underlying_price_history_len)

        # Curve Fitting
        self.vol_min_points_for_fit = 4
        self.vol_max_fit_mse = 0.0005 # Max Mean Squared Error for fit quality check (tune this!)
        self.vol_use_weighted_fit = False # Use Vega weighting for fit

        # Entry/Exit Thresholds
        self.vol_base_diff_threshold = 0.012 # Base entry threshold (1.2% IV diff)
        self.vol_base_tp_threshold = 0.002   # Base take profit threshold (0.2% IV diff towards zero)
        self.vol_base_sl_iv_diff = 0.04   # Stop Loss if IV diff moves further against by 1.5%
        self.vol_sl_price_pct = 0.2       # Stop Loss if price moves against entry by 1% (tune this!)

        # Dynamic Sizing
        self.vol_base_volume = 10          # Base order size
        self.vol_max_size_multiplier = 5.0 # Max multiplier for dynamic sizing
        self.vol_size_vega_sensitivity = 0.5 # How much Vega impacts size (0 to 1)
        self.vol_size_diff_sensitivity = 0.5 # How much IV diff impacts size (0 to 1)

        # Execution Logic
        self.vol_use_limit_orders_entry = True # Use limit orders for entry
        self.vol_max_spread_pct = 0.05      # Max allowed bid-ask spread (0.5%) for trading

        # Time Decay Adjustment & Short Vol Bias
        self.vol_adjust_for_tte = True       # Enable TTE adjustments
        self.vol_min_tte_aggressiveness = 0.2 # At T=0, thresholds are 1/0.2=5x higher, size is 0.2x
        self.vol_short_bias_factor = 2.0     # How much more to favor short vol near expiry (1.0 = no bias, >1 = favor short)

        # State Tracking (Loaded in run method)
        self.open_vol_positions: Dict[Symbol, VolPosition] = {}


    # --- Helper Methods (Unchanged) ---
    def _load_trader_data(self, trader_data: str):
        """Load state from trader_data string."""
        if trader_data:
            try:
                data = json.loads(trader_data)
                self.open_vol_positions = {
                    symbol: VolPosition.from_dict(pos_data)
                    for symbol, pos_data in data.get("open_vol_positions", {}).items()
                }
                logger.print("Loaded trader data successfully.")
            except Exception as e:
                logger.print(f"Error loading trader data: {e}")
                self.open_vol_positions = {}
        else:
            self.open_vol_positions = {}

    def _save_trader_data(self) -> str:
        """Save state to trader_data string."""
        try:
            data_to_save = {
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
            return False

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def _get_smoothed_underlying_price(self, state: TradingState) -> Optional[float]:
        """Get smoothed underlying price using recent history."""
        mid_price = self._get_mid_price(self.vol_underlying, state)
        if mid_price is not None:
            self.underlying_price_history.append(mid_price)
        if not self.underlying_price_history:
            return None
        return np.mean(self.underlying_price_history)

    def _calculate_tte(self, timestamp: int) -> float:
        """Calculates Time To Expiry (TTE) in years."""
        total_timestamps = self.vol_total_days * self.vol_max_timestamp_per_day
        # Ensure timestamp doesn't exceed total duration (can happen in backtests/edge cases)
        current_timestamp = min(timestamp, total_timestamps)
        timestamps_remaining = max(0.0, total_timestamps - current_timestamp)
        T = timestamps_remaining / (self.vol_days_per_year * self.vol_max_timestamp_per_day)
        # Add a tiny epsilon to avoid T=0 issues if timestamp is exactly at expiry
        return T + 1e-9

    def _get_tte_adjustment_factor(self, T: float) -> float:
        """Calculate adjustment factor based on TTE (0 near expiry, 1 far from expiry)."""
        if not self.vol_adjust_for_tte or self.vol_total_days <= 0:
            return 1.0
        initial_T = self.vol_total_days / self.vol_days_per_year
        if initial_T <= 1e-6: return 1.0
        # Linear scaling from min_aggressiveness up to 1.0
        # factor = self.vol_min_tte_aggressiveness + (1.0 - self.vol_min_tte_aggressiveness) * (T / initial_T)
        # Let's make it simpler: factor is proportion of time remaining
        factor = T / initial_T
        # Apply the min_aggressiveness floor
        adjusted_factor = self.vol_min_tte_aggressiveness + (1.0 - self.vol_min_tte_aggressiveness) * factor
        return max(self.vol_min_tte_aggressiveness, min(1.0, adjusted_factor)) # Clamp between min and 1.0

    def _calculate_dynamic_size(self, base_volume: int, iv_diff: float, option_vega: float, tte_factor: float) -> int:
        """Calculate order size based on confidence (diff) and vega, adjusted for TTE."""
        if option_vega < 1e-4: option_vega = 1e-4

        # Normalize diff and vega
        diff_scale = min(1.0, abs(iv_diff) / (self.vol_base_diff_threshold * 2))
        vega_scale = min(1.0, option_vega / 100.0) # Tune 100 based on typical Vega values

        # Combine factors
        size_confidence_factor = 1.0 + (self.vol_max_size_multiplier - 1.0) * (
            self.vol_size_diff_sensitivity * diff_scale +
            self.vol_size_vega_sensitivity * vega_scale
        ) / (self.vol_size_diff_sensitivity + self.vol_size_vega_sensitivity + 1e-6)

        # Apply TTE decay to size
        adjusted_size_factor = size_confidence_factor * tte_factor

        # Clamp and calculate final volume
        final_volume = int(round(base_volume * max(0.1, min(self.vol_max_size_multiplier, adjusted_size_factor))))
        return max(1, final_volume)

    # --- Volatility Strategy ---
    def run_volatility_strategy(self, state: TradingState) -> Tuple[List[Order], set[Symbol], set[Symbol], Dict[Symbol, VolPosition]]:
        """
        Implements the enhanced volatility curve fitting strategy with short bias near expiry.
        """
        potential_orders: List[Order] = []
        symbols_to_close: set[Symbol] = set()
        symbols_to_open: set[Symbol] = set()
        positions_to_update: Dict[Symbol, VolPosition] = {}
        timestamp = state.timestamp

        # 1. Calculate TTE & Adjustment Factor
        T = self._calculate_tte(timestamp)
        if T < 1e-7: return [], set(), set(), {} # Skip very close to expiry
        tte_factor = self._get_tte_adjustment_factor(T) # Factor from min_agg -> 1.0

        # Adjust thresholds: Make thresholds LARGER (less sensitive) near expiry (low tte_factor)
        # Apply short bias: Make long entry threshold increase MORE than short entry threshold near expiry
        long_bias_multiplier = 1.0 + (self.vol_short_bias_factor - 1.0) * (1.0 - tte_factor) # Increases from 1.0 up to short_bias_factor as tte_factor -> min_agg
        short_bias_multiplier = 1.0 # No bias adjustment for short threshold increase

        # Ensure tte_factor is not too close to zero before division
        safe_tte_factor = max(tte_factor, 1e-6)

        current_long_entry_threshold = (self.vol_base_diff_threshold * long_bias_multiplier) / safe_tte_factor
        current_short_entry_threshold = (self.vol_base_diff_threshold * short_bias_multiplier) / safe_tte_factor
        # Keep TP/SL symmetric for now, just adjusted by base tte_factor
        current_tp_threshold = self.vol_base_tp_threshold / safe_tte_factor
        current_sl_iv_diff = self.vol_base_sl_iv_diff / safe_tte_factor

        # 2. Get Smoothed Underlying Price (S)
        S = self._get_smoothed_underlying_price(state)
        if S is None: return [], set(), set(), {}

        # 3. Gather Market Data & Calculate Vega/Weights (Unchanged)
        market_data_list: List[Dict[str, Any]] = []
        for symbol in self.vol_vouchers:
            K = self.vol_strikes.get(symbol)
            depth = state.order_depths.get(symbol)
            if K is None or not depth or not depth.buy_orders or not depth.sell_orders: continue

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            market_mid_price = (best_bid + best_ask) / 2.0

            spread = best_ask - best_bid
            if market_mid_price > 0 and (spread / market_mid_price) > self.vol_max_spread_pct:
                continue

            v_t = implied_vol_call_price(S, K, self.vol_risk_free_rate, T, market_mid_price)
            m_t = np.nan
            if S > 0 and T > 1e-10:
                try: m_t = math.log(K / S) / math.sqrt(T)
                except (ValueError, OverflowError): m_t = np.nan

            if pd.notna(v_t) and pd.notna(m_t):
                option_vega = vega(S, K, self.vol_risk_free_rate, T, v_t)
                weight = option_vega if self.vol_use_weighted_fit and pd.notna(option_vega) and option_vega > 1e-3 else 1.0
                market_data_list.append({
                    'symbol': symbol, 'm_t': m_t, 'v_t': v_t, 'K': K,
                    'market_price': market_mid_price, 'best_bid': best_bid, 'best_ask': best_ask,
                    'vega': option_vega if pd.notna(option_vega) else 0.0, 'weight': weight
                })

        # 4. Fit the Curve & Check Quality (Unchanged)
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

            base_iv = poly(0)
            # logger.print(f"Vol Strat: Fitted Curve (MSE={mse:.6f}). Base IV: {base_iv:.4f}") # Reduce log noise

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.print(f"Vol Strat: Curve fitting failed: {e}")
            return [], set(), set(), {}

        # --- Generate Orders ---
        processed_symbols = set()

        # 5. Generate Closing Orders (TP and SL) (Unchanged logic, uses current_tp/sl thresholds)
        for symbol, open_pos in list(self.open_vol_positions.items()):
            data_point = next((d for d in market_data_list if d['symbol'] == symbol), None)
            if data_point is None: continue

            v_t = data_point['v_t']
            v_t_fit = poly(data_point['m_t'])
            diff = v_t - v_t_fit
            best_bid = data_point['best_bid']
            best_ask = data_point['best_ask']
            market_price = data_point['market_price']
            close_reason = None
            close_price = None
            close_quantity = -open_pos.quantity

            # Check TP
            if open_pos.quantity > 0 and diff > -current_tp_threshold:
                close_reason = f"TP Long (diff={diff:.4f} > {-current_tp_threshold:.4f})"
                close_price = best_bid
            elif open_pos.quantity < 0 and diff < current_tp_threshold:
                close_reason = f"TP Short (diff={diff:.4f} < {current_tp_threshold:.4f})"
                close_price = best_ask

            # Check SL - IV Diff
            if close_reason is None:
                if open_pos.quantity > 0 and diff < open_pos.entry_diff - current_sl_iv_diff:
                    close_reason = f"SL IV Long (diff={diff:.4f} < {open_pos.entry_diff - current_sl_iv_diff:.4f})"
                    close_price = best_bid
                elif open_pos.quantity < 0 and diff > open_pos.entry_diff + current_sl_iv_diff:
                    close_reason = f"SL IV Short (diff={diff:.4f} > {open_pos.entry_diff + current_sl_iv_diff:.4f})"
                    close_price = best_ask

            # Check SL - Price %
            if close_reason is None and open_pos.entry_price > 0:
                price_change_pct = (market_price - open_pos.entry_price) / open_pos.entry_price
                if open_pos.quantity > 0 and price_change_pct < -self.vol_sl_price_pct:
                    close_reason = f"SL Price Long ({price_change_pct*100:.2f}% < {-self.vol_sl_price_pct*100:.2f}%)"
                    close_price = best_bid
                elif open_pos.quantity < 0 and price_change_pct > self.vol_sl_price_pct:
                    close_reason = f"SL Price Short ({price_change_pct*100:.2f}% > {self.vol_sl_price_pct*100:.2f}%)"
                    close_price = best_ask

            if close_reason and close_price is not None:
                logger.print(f"Vol Close {symbol}: {close_reason}. Order: {close_quantity} @{close_price}")
                potential_orders.append(Order(symbol, close_price, close_quantity))
                symbols_to_close.add(symbol)
                processed_symbols.add(symbol)


        # 6. Generate Opening Orders (Uses asymmetric entry thresholds)
        for data_point in market_data_list:
            symbol = data_point['symbol']
            if symbol in processed_symbols: continue

            v_t = data_point['v_t']
            v_t_fit = poly(data_point['m_t'])
            diff = v_t - v_t_fit
            option_vega = data_point['vega']
            K = data_point['K']
            best_bid = data_point['best_bid']
            best_ask = data_point['best_ask']

            order_quantity = 0
            order_price = None
            entry_reason = None
            target_pos = None

            # Check Entry condition: market vol significantly CHEAPER -> BUY (Uses long threshold)
            if diff < -current_long_entry_threshold:
                order_quantity = self._calculate_dynamic_size(self.vol_base_volume, diff, option_vega, tte_factor)
                entry_reason = f"Entry BUY CHEAP (diff={diff:.4f} < {-current_long_entry_threshold:.4f})"
                if self.vol_use_limit_orders_entry:
                    theo_price = black_scholes_call_price(S, K, self.vol_risk_free_rate, T, v_t_fit)
                    order_price = math.floor(theo_price)
                    order_price = min(order_price, best_ask)
                else:
                    order_price = best_ask
                target_pos = VolPosition(order_quantity, order_price, diff, option_vega)

            # Check Entry condition: market vol significantly MORE EXPENSIVE -> SELL (Uses short threshold)
            elif diff > current_short_entry_threshold:
                order_quantity = -self._calculate_dynamic_size(self.vol_base_volume, diff, option_vega, tte_factor)
                entry_reason = f"Entry SELL EXPENSIVE (diff={diff:.4f} > {current_short_entry_threshold:.4f})"
                if self.vol_use_limit_orders_entry:
                    theo_price = black_scholes_call_price(S, K, self.vol_risk_free_rate, T, v_t_fit)
                    order_price = math.ceil(theo_price)
                    order_price = max(order_price, best_bid)
                else:
                    order_price = best_bid
                target_pos = VolPosition(order_quantity, order_price, diff, option_vega)

            if entry_reason and order_price is not None and order_quantity != 0 and target_pos is not None:
                if order_price <= 0:
                    # logger.print(f"Vol Entry {symbol}: Invalid order price {order_price}, skipping.") # Reduce log noise
                    continue

                # logger.print(f"Vol {entry_reason}. Order: {order_quantity} @{order_price}") # Reduce log noise
                potential_orders.append(Order(symbol, order_price, order_quantity))
                symbols_to_open.add(symbol)
                positions_to_update[symbol] = target_pos
                processed_symbols.add(symbol)

        return potential_orders, symbols_to_close, symbols_to_open, positions_to_update


    # --- Main Run Loop (Unchanged) ---
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic loop.
        """
        self._load_trader_data(state.traderData)

        result = {symbol: [] for symbol in state.order_depths.keys()}
        pending_pos_delta: Dict[Symbol, int] = {}
        traded_symbols_this_tick = set()

        if not self.previous_positions: self.previous_positions = state.position.copy()
        else: self.log_position_changes(state.position, state.timestamp)

        potential_vol_orders, symbols_to_close, symbols_to_open, positions_to_update = self.run_volatility_strategy(state)

        executed_vol_trades: Dict[Symbol, int] = {}

        for order in potential_vol_orders:
            if self._can_place_order(order.symbol, order.quantity, state.position.get(order.symbol, 0), pending_pos_delta):
                result[order.symbol].append(order)
                pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                traded_symbols_this_tick.add(order.symbol)
                executed_vol_trades[order.symbol] = order.quantity

        for symbol, executed_quantity in executed_vol_trades.items():
            if symbol in symbols_to_close:
                if symbol in self.open_vol_positions:
                    if executed_quantity == -self.open_vol_positions[symbol].quantity:
                        del self.open_vol_positions[symbol]
                        logger.print(f"Closed Vol Position: {symbol}")
                    else:
                        logger.print(f"WARNING: Partial/Mismatched close for {symbol}. Executed: {executed_quantity}, Open: {self.open_vol_positions[symbol].quantity}. Adjusting.")
                        self.open_vol_positions[symbol].quantity += executed_quantity
                        if abs(self.open_vol_positions[symbol].quantity) < 1:
                            del self.open_vol_positions[symbol]

            elif symbol in symbols_to_open:
                if symbol not in self.open_vol_positions:
                    if symbol in positions_to_update:
                        new_pos_details = positions_to_update[symbol]
                        self.open_vol_positions[symbol] = VolPosition(
                            executed_quantity, new_pos_details.entry_price,
                            new_pos_details.entry_diff, new_pos_details.entry_vega
                        )
                        logger.print(f"Opened Vol Position: {symbol}, Quantity: {executed_quantity} @Assumed {new_pos_details.entry_price}")
                    else:
                         logger.print(f"ERROR: Executed open for {symbol} but no details found.")
                else:
                    logger.print(f"WARNING: Adding to existing vol position {symbol}. Executed: {executed_quantity}")
                    self.open_vol_positions[symbol].quantity += executed_quantity

        conversions = 0
        self.trader_data = self._save_trader_data()

        orders_to_log = {sym: ords for sym, ords in result.items() if ords}
        if orders_to_log: logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")
        if self.open_vol_positions: logger.print(f"Open Vol Positions: { {s: p.to_dict() for s, p in self.open_vol_positions.items()} }")

        self.previous_positions = state.position.copy()
        logger.flush(state, result, conversions, self.trader_data)
        return result, conversions, self.trader_data

    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        # ... (log_position_changes remains unchanged) ...
        log_entry = []
        all_symbols = set(current_positions.keys()) | set(self.previous_positions.keys())
        for symbol in sorted(list(all_symbols)):
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")

        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")
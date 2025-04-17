import json
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
# import pandas as pd # Not strictly needed for this trader logic
import jsonpickle
from collections import deque # Good practice

# --- Data Model Handling ---
try:
    # Use the actual datamodel provided by the environment
    from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
except ImportError:
    print("Warning: datamodel.py not found. Using dummy classes for local testing.")
    # Define dummy classes if datamodel.py is not available
    class ProsperityEncoder(json.JSONEncoder):
        def default(self, o):
            try: return o.to_dict()
            except AttributeError: return json.JSONEncoder.default(self, o)
    class Symbol(str): pass
    class Order:
        def __init__(self, symbol: Symbol, price: float, quantity: int):
            self.symbol = symbol; self.price = price; self.quantity = quantity
        def __str__(self): return f"Order({self.symbol}, {self.price}, {self.quantity})"
        def __repr__(self): return f"Order({self.symbol}, {self.price}, {self.quantity})"
    class OrderDepth:
        def __init__(self): self.buy_orders: Dict[int, int] = {}; self.sell_orders: Dict[int, int] = {}
    class Trade:
        def __init__(self, symbol: Symbol, price: float, quantity: int, buyer: str = "", seller: str = "", timestamp: int = 0):
            self.symbol=symbol; self.price=price; self.quantity=quantity; self.buyer=buyer; self.seller=seller; self.timestamp=timestamp
    class Listing:
        def __init__(self, symbol: Symbol, product: str, denomination: str):
            self.symbol=symbol; self.product=product; self.denomination=denomination
    class ConversionObservation: # Dummy, as it's not used live
         bidPrice: float = 0.0; askPrice: float = 0.0; transportFees: float = 0.0; exportTariff: float = 0.0
         importTariff: float = 0.0; sugarPrice: float = 0.0; sunlightIndex: float = 0.0
    class Observation:
        def __init__(self): self.plainValueObservations: Dict[Symbol, float] = {}; self.conversionObservations: Dict[Symbol, ConversionObservation] = {}
    class TradingState:
        def __init__(self):
            self.timestamp: int = 0; self.traderData: str = ""; self.listings: Dict[Symbol, Listing] = {}
            self.order_depths: Dict[Symbol, OrderDepth] = {}; self.own_trades: Dict[Symbol, List[Trade]] = {}
            self.market_trades: Dict[Symbol, List[Trade]] = {}; self.position: Dict[Symbol, int] = {}
            self.observations: Observation = Observation()

# --- Logger Class (Use the provided one) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Use ProsperityEncoder if available, otherwise standard JSONEncoder
        encoder_cls = ProsperityEncoder if 'ProsperityEncoder' in globals() else json.JSONEncoder
        try:
            compressed_state_empty_trader = self.compress_state(state, "")
        except AttributeError as e:
             print(f"Warning: Error compressing state for base length calculation: {e}. Using estimate.")
             compressed_state_empty_trader = []
        base_payload = [compressed_state_empty_trader, self.compress_orders(orders), conversions, "", ""]
        try:
            base_json = json.dumps(base_payload, cls=encoder_cls, separators=(",", ":"))
            base_length = len(base_json)
        except Exception as e:
            print(f"Warning: Error calculating base JSON length: {e}. Using estimate.")
            base_length = 200
        available_length = self.max_log_length - base_length
        max_item_length = max(1, available_length // 3) if available_length > 0 else 1
        if available_length <= 0: print(f"Warning: Base log structure length ({base_length}) exceeds limit ({self.max_log_length}). Truncating heavily.")
        final_payload = []
        try:
            truncated_state_trader_data = self.truncate(getattr(state, 'traderData', ''), max_item_length)
            compressed_state = self.compress_state(state, truncated_state_trader_data)
            final_payload = [compressed_state, self.compress_orders(orders), conversions, self.truncate(trader_data, max_item_length), self.truncate(self.logs, max_item_length)]
            final_json = json.dumps(final_payload, cls=encoder_cls, separators=(",", ":"))
            if len(final_json) > self.max_log_length: print(json.dumps({"error": "Log message too long after truncation.", "final_len": len(final_json), "max_len": self.max_log_length}))
            else: print(final_json)
        except AttributeError as e: print(json.dumps({"error": f"AttributeError during final log generation: {e}"}))
        except Exception as e: print(json.dumps({"error": f"Exception during final log generation: {e}"}))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            getattr(state, 'timestamp', 0), trader_data,
            self.compress_listings(getattr(state, 'listings', {})),
            self.compress_order_depths(getattr(state, 'order_depths', {})),
            self.compress_trades(getattr(state, 'own_trades', {})),
            self.compress_trades(getattr(state, 'market_trades', {})),
            getattr(state, 'position', {}),
            self.compress_observations(getattr(state, 'observations', None)),
        ]
    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings:
            for listing in listings.values():
                compressed.append([getattr(listing, 'symbol', ''), getattr(listing, 'product', ''), getattr(listing, 'denomination', '')])
        return compressed
    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths:
            for symbol, order_depth in order_depths.items():
                 compressed[symbol] = [getattr(order_depth, 'buy_orders', {}), getattr(order_depth, 'sell_orders', {})]
        return compressed
    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                for trade in arr:
                     compressed.append([getattr(trade, 'symbol', ''), getattr(trade, 'price', 0), getattr(trade, 'quantity', 0),
                                        getattr(trade, 'buyer', ''), getattr(trade, 'seller', ''), getattr(trade, 'timestamp', 0)])
        return compressed
    def compress_observations(self, observations: Optional[Observation]) -> list[Any]:
        if observations is None: return [{}, {}]
        plain_obs = getattr(observations, 'plainValueObservations', {})
        conv_obs_comp = {}
        # NOTE: Even if conv_obs_data exists, it will be empty in Round 4 based on logs
        conv_obs_data = getattr(observations, 'conversionObservations', {})
        if conv_obs_data:
            for product, observation_data in conv_obs_data.items():
                if observation_data is not None:
                    # This part likely won't run in Round 4
                    obs_values = [getattr(observation_data, attr, None) for attr in ['bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sugarPrice', 'sunlightIndex']]
                    conv_obs_comp[product] = obs_values
                else: conv_obs_comp[product] = []
        return [plain_obs, conv_obs_comp]
    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders:
            for arr in orders.values():
                for order in arr:
                     compressed.append([getattr(order, 'symbol', ''), getattr(order, 'price', 0), getattr(order, 'quantity', 0)])
        return compressed
    def to_json(self, value: Any) -> str:
        cls = ProsperityEncoder if 'ProsperityEncoder' in globals() else json.JSONEncoder
        try: return json.dumps(value, cls=cls, separators=(",", ":"))
        except TypeError as e: print(f"Warning: JSON serialization error {e}. Using default."); return json.dumps(value, default=str, separators=(",", ":"))
    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str):
            try: value = str(value)
            except Exception: return ""
        max_length = max(0, max_length)
        lo, hi = 0, len(value); best_fit_prefix = ""
        if len(json.dumps("")) <= max_length: best_fit_prefix = ""
        while lo <= hi:
            mid = (lo + hi) // 2; prefix = value[:mid]; candidate = prefix
            if len(prefix) < len(value): candidate += "..."
            try: encoded_candidate = json.dumps(candidate)
            except Exception: encoded_candidate = ""; hi = mid - 1; continue
            if len(encoded_candidate) <= max_length: best_fit_prefix = candidate; lo = mid + 1
            else: hi = mid - 1
        return best_fit_prefix

logger = Logger()

# --- Model Parameters Structure ---
class ModelParameters:
    def __init__(self, intercept: float, coefficients: List[float], feature_names: List[str]):
        self.intercept = intercept
        self.coefficients = np.array(coefficients) # Ensure numpy array
        self.feature_names = feature_names
        if len(self.coefficients) != len(self.feature_names):
             raise ValueError(f"Mismatch between coefficients ({len(self.coefficients)}) and feature names ({len(self.feature_names)})")


# --- Trader Class ---
class Trader:
    def __init__(self):
        self.product = "MAGNIFICENT_MACARONS"
        self.position_limits = {self.product: 75}
        self.order_volume = 5 # Example: Max order size per step
        self.trade_threshold = 0.5 # Example: Required edge to place order

        self.model_params: Optional[ModelParameters] = None
        self.trader_data_state: Dict[str, Any] = {}
        self.previous_positions: Dict[Symbol, int] = {}

        try:
            # --- Hardcoded Model Parameters ---
            hardcoded_intercept = 671.00589830
            hardcoded_coefficients = [
                0.9986979450275897,
                0.039974734930852476,
                -0.09455548357115615,
                -0.07177402537508885,
                -0.0007342619205335009,
                -0.006340504771143118
            ]
            hardcoded_feature_names = [
                "mid_price",
                "transportFees",
                "exportTariff",
                "importTariff",
                "sugarPrice",
                "sunlightIndex"
            ]

            if len(hardcoded_coefficients) != len(hardcoded_feature_names):
                 raise ValueError("Hardcoded coefficients and feature names length mismatch.")

            self.model_params = ModelParameters(
                intercept=hardcoded_intercept,
                coefficients=hardcoded_coefficients,
                feature_names=hardcoded_feature_names
            )
            logger.print("Trader initialized successfully with hardcoded parameters.")
            logger.print(f"  Intercept: {self.model_params.intercept}")
            logger.print(f"  Features ({len(self.model_params.feature_names)}): {self.model_params.feature_names}")
            coeffs_str = ", ".join([f"{c:.4f}" for c in self.model_params.coefficients])
            logger.print(f"  Coefficients ({len(self.model_params.coefficients)}): [{coeffs_str}]")

        except Exception as e:
            logger.print(f"FATAL: Trader initialization failed processing hardcoded parameters: {e}")
            self.model_params = None

    def _load_trader_data(self, trader_data: str):
        """Load state from trader_data string using standard json."""
        self.trader_data_state = {}
        if trader_data:
            try:
                self.trader_data_state = json.loads(trader_data)
                self.previous_positions = self.trader_data_state.get("previous_positions", {})
            except json.JSONDecodeError as e:
                logger.print(f"Error loading trader data JSON: {e}. Starting fresh.")
                self.previous_positions = {}
            except Exception as e:
                 logger.print(f"Unexpected error loading trader data: {e}. Starting fresh.")
                 self.previous_positions = {}
        else:
            self.previous_positions = {}

    def _save_trader_data(self, current_positions: Dict[Symbol, int]) -> str:
        """Save state to trader_data string using standard json."""
        try:
            self.trader_data_state["previous_positions"] = current_positions
            return json.dumps(self.trader_data_state, separators=(",", ":"))
        except Exception as e:
            logger.print(f"Error saving trader data: {e}")
            return ""

    def _can_place_order(self, symbol: Symbol, quantity: int, current_pos: int, pending_pos_delta: int) -> bool:
        limit = self.position_limits.get(symbol)
        if limit is None: return False
        final_pos = current_pos + pending_pos_delta + quantity
        return abs(final_pos) <= limit

    def _get_best_price(self, depth: OrderDepth, side: str) -> Optional[int]:
        try:
            if side == 'bid': return max(depth.buy_orders.keys()) if hasattr(depth, 'buy_orders') and depth.buy_orders else None
            elif side == 'ask': return min(depth.sell_orders.keys()) if hasattr(depth, 'sell_orders') and depth.sell_orders else None
        except Exception as e: logger.print(f"Error getting best price for {side}: {e}")
        return None

    def _calculate_mid_price(self, best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        if not hasattr(self, 'previous_positions'):
             self.previous_positions = {}

        log_entry = []
        symbols_to_log = sorted([s for s in (set(current_positions.keys()) | set(self.previous_positions.keys())) if s == self.product])

        for symbol in symbols_to_log:
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}")

        if log_entry:
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        self._load_trader_data(state.traderData)

        self.log_position_changes(state.position, state.timestamp)

        result = {self.product: []}
        conversions = 0
        pending_pos_delta = 0
        trader_data_str = ""
        timestamp = state.timestamp

        try:
            if self.model_params is None: raise RuntimeError("Model parameters not loaded during init.")

            current_position = state.position.get(self.product, 0)
            order_depth = state.order_depths.get(self.product)
            if not order_depth: raise ValueError(f"No order depth data for {self.product}")

            best_bid = self._get_best_price(order_depth, 'bid')
            best_ask = self._get_best_price(order_depth, 'ask')
            current_mid_price = self._calculate_mid_price(best_bid, best_ask)

            if current_mid_price is None: raise ValueError(f"Missing best bid/ask for {self.product}")

            current_features_list = []
            missing_features = []
            feature_extraction_successful = True
            temp_feature_dict = {}

            if 'mid_price' in self.model_params.feature_names:
                 temp_feature_dict['mid_price'] = current_mid_price
            else:
                 missing_features.append('mid_price')
                 feature_extraction_successful = False

            observations = getattr(state, 'observations', None)
            conv_observations_map = getattr(observations, 'conversionObservations', {}) if observations else {}
            product_key_normalized = self.product.strip().lower()
            obs_data = None
            for key, value in conv_observations_map.items():
                if isinstance(key, str) and key.strip().lower() == product_key_normalized:
                    obs_data = value
                    break

            conv_features_needed = [f for f in self.model_params.feature_names if f != 'mid_price']

            if not conv_features_needed:
                 pass
            elif obs_data is None:
                 logger.print(f"DEBUG @ {timestamp}: ConversionObservation data for {self.product} not found. Expected in backtester.")
                 missing_features.extend(conv_features_needed)
                 feature_extraction_successful = False
            else:
                 for feature_name in conv_features_needed:
                     value = getattr(obs_data, feature_name, None)
                     if value is None:
                         missing_features.append(feature_name)
                         feature_extraction_successful = False
                     elif isinstance(value, (int, float)):
                         temp_feature_dict[feature_name] = float(value)
                     else:
                         logger.print(f"Warning: Feature '{feature_name}' has non-numeric value '{value}'. Treating as missing.")
                         missing_features.append(feature_name)
                         feature_extraction_successful = False

            if feature_extraction_successful:
                try:
                    current_features_list = [temp_feature_dict[fname] for fname in self.model_params.feature_names]
                except KeyError as e:
                     raise RuntimeError(f"Internal error: Feature '{e}' expected but not found in extracted features.")

                current_features = np.array(current_features_list)
                if len(current_features) != len(self.model_params.feature_names):
                     raise RuntimeError(f"Feature count mismatch: Got {len(current_features)}, expected {len(self.model_params.feature_names)}")

                predicted_mid_price = self.model_params.intercept + np.dot(current_features, self.model_params.coefficients)
                logger.print(f"Timestamp {timestamp}: Pos: {current_position}, CurrMid: {current_mid_price:.2f}, PredMid: {predicted_mid_price:.2f}, Bid: {best_bid}, Ask: {best_ask} (Using Full Model)")

                product_limit = self.position_limits.get(self.product, 0)
                buy_trigger_price = best_ask + self.trade_threshold
                if predicted_mid_price > buy_trigger_price:
                    available_buy_capacity = product_limit - (current_position + pending_pos_delta)
                    volume_to_buy = min(self.order_volume, available_buy_capacity); volume_to_buy = max(0, volume_to_buy)
                    if volume_to_buy > 0:
                        order_price = best_ask
                        if self._can_place_order(self.product, volume_to_buy, current_position, pending_pos_delta):
                            logger.print(f"  BUY Signal. Pred {predicted_mid_price:.2f} > Trigger {buy_trigger_price:.2f}. Order: {volume_to_buy} @ {order_price}")
                            result[self.product].append(Order(self.product, order_price, volume_to_buy))
                            pending_pos_delta += volume_to_buy
                sell_trigger_price = best_bid - self.trade_threshold
                if predicted_mid_price < sell_trigger_price:
                    available_sell_capacity = product_limit + (current_position + pending_pos_delta)
                    volume_to_sell = min(self.order_volume, available_sell_capacity); volume_to_sell = max(0, volume_to_sell)
                    if volume_to_sell > 0:
                        order_price = best_bid; order_quantity = -volume_to_sell
                        if self._can_place_order(self.product, order_quantity, current_position, pending_pos_delta):
                            logger.print(f"  SELL Signal. Pred {predicted_mid_price:.2f} < Trigger {sell_trigger_price:.2f}. Order: {order_quantity} @ {order_price}")
                            result[self.product].append(Order(self.product, order_price, order_quantity))
                            pending_pos_delta += order_quantity
            else:
                logger.print(f"DEBUG @ {timestamp}: Missing features {missing_features}. Skipping model prediction. (Expected in backtester)")
                pass

        except Exception as e:
             logger.print(f"ERROR during trading logic @ {timestamp}: {e}")

        trader_data_str = self._save_trader_data(state.position)
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str

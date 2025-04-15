import json
from typing import Any, Dict, List
import math

# Assuming datamodel.py defines these classes:
# Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
# If not, you'll need to include their definitions.
# For example:
# Symbol = str
# Product = str
# Position = int
# Time = int
# UserId = str
# class Order: ... etc.
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjusted based on typical limits

    def cdf_standard_normal(x):
     """Approximate CDF of the standard normal distribution."""
     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def vega(S, K, r, T, sigma):
        """Partial derivative of the call price wrt sigma (aka Vega)."""
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return S * math.sqrt(T) * (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)


    def black_scholes_call_price(self, S, K, r, T, sigma):
        """
        Computes the Black–Scholes price for a call option.
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration (in years)
        sigma: volatility (annualized)
        """
        if T <= 0:
            # If time to expiry is 0, the option's value is max(S-K, 0).
            return max(S - K, 0)
        # d1 and d2
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        # Call price
        call_val = (S * self.cdf_standard_normal(d1)) - (K * math.exp(-r * T) * self.cdf_standard_normal(d2))
        return call_val
    
    def implied_vol_call_price(self, S, K, r, T, market_price, initial_guess=0.2, tol=1e-6, max_iterations=100):
        """
        Numerically find implied volatility via Newton–Raphson.
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration (in years)
        market_price: the current market premium of the option
        initial_guess: starting volatility guess
        tol: tolerance for the final result
        max_iterations: maximum number of iterations
        """
        # Edge case: Deep in-the-money with very low time value
        intrinsic_value = max(S - K, 0)
        if T < 1e-5 and abs(market_price - intrinsic_value) < tol:
            return 1e-5  # minimal volatility, since price is nearly intrinsic

        sigma = initial_guess
        for i in range(max_iterations):
            price = self.black_scholes_call_price(S, K, r, T, sigma)
            diff = price - market_price  # how far off we are
            if abs(diff) < tol:
                return sigma  # found a good enough solution
            v = self.vega(S, K, r, T, sigma)
            if v < 1e-8:
                # If vega is extremely small, we risk dividing by zero or huge jumps.
                break
            # Newton step
            sigma = sigma - diff / v
            # keep sigma positive
            if sigma < 0:
                sigma = 1e-5
        # If we exit the loop without returning, we can either raise an error or return the last sigma
        return sigma
    
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


class Trader:
    def __init__(self):
        # Define position limits for each product
        self.position_limits = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "DJEMBES": 60,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500 ": 200,
            "VOLCANIC_ROCK_VOUCHER_9750 ": 200,
            "VOLCANIC_ROCK_VOUCHER_10000 ": 200,
            "VOLCANIC_ROCK_VOUCHER_10250 ": 200,
            "VOLCANIC_ROCK_VOUCHER_10500 ": 200
        }

        # Define the components of each basket
        self.basket_components = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }

        # Track previous positions for logging changes
        self.previous_positions = {}
        # Store traderData if needed between runs
        self.trader_data = "" # Can be used for more complex state tracking if needed

        # --- Strategy Parameters ---
        self.min_arbitrage_profit = 100.0 # Threshold for initiating NEW arbitrage
        self.flattening_profit_threshold = 10.0 # Lower threshold for closing existing positions
        self.mm_spread_threshold = 3
        self.mm_base_volume = 10

    # Helper function to check position limits before placing an order
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

    # Helper to calculate component costs, values, and volume limits
    def _calculate_component_prices_volumes(self, state: TradingState, components: Dict[str, int]) -> tuple[float, float, list[int], list[int], bool, bool]:
        """Calculates cost to buy components, value to sell components, and volume limits per basket."""
        component_cost = 0
        component_ask_volumes_at_best = []
        can_buy_components = True
        for component, quantity in components.items():
            if component not in state.order_depths:
                can_buy_components = False
                break
            component_depth = state.order_depths[component]
            if not component_depth.sell_orders:
                can_buy_components = False
                break
            best_component_ask = min(component_depth.sell_orders.keys())
            component_cost += best_component_ask * quantity
            component_ask_volume = abs(component_depth.sell_orders.get(best_component_ask, 0))
            if quantity > 0:
                # Volume limit per basket for this component at best ask
                component_ask_volumes_at_best.append(component_ask_volume // quantity)
            else:
                component_ask_volumes_at_best.append(float('inf'))
        if not can_buy_components:
            component_cost = float('inf')

        component_value = 0
        component_bid_volumes_at_best = []
        can_sell_components = True
        for component, quantity in components.items():
            # Need to re-check availability for selling
            if component not in state.order_depths:
                 can_sell_components = False
                 break
            component_depth = state.order_depths[component]
            if not component_depth.buy_orders:
                can_sell_components = False
                break
            best_component_bid = max(component_depth.buy_orders.keys())
            component_value += best_component_bid * quantity
            component_bid_volume = abs(component_depth.buy_orders.get(best_component_bid, 0))
            if quantity > 0:
                # Volume limit per basket for this component at best bid
                component_bid_volumes_at_best.append(component_bid_volume // quantity)
            else:
                component_bid_volumes_at_best.append(float('inf'))
        if not can_sell_components:
            component_value = 0

        return component_cost, component_value, component_ask_volumes_at_best, component_bid_volumes_at_best, can_buy_components, can_sell_components

    # Find opportunities to flatten existing arbitrage positions
    def find_flattening_opportunities(self, state: TradingState) -> dict[str, list[Order]]:
        """Checks existing positions and generates closing orders if opposite arbitrage is profitable."""
        flattening_orders = {}

        for basket, components in self.basket_components.items():
            if basket not in state.order_depths: continue
            all_components_available = all(comp in state.order_depths for comp in components)
            if not all_components_available: continue

            current_pos_basket = state.position.get(basket, 0)
            # Simplification: Assume position direction indicates potential arb position
            # A more robust check would verify component ratios, but adds complexity
            potential_strat1_pos = current_pos_basket > 0 # Potentially long basket, short components
            potential_strat2_pos = current_pos_basket < 0 # Potentially short basket, long components

            if not potential_strat1_pos and not potential_strat2_pos:
                continue # No basket position, likely no arb position to flatten

            basket_depth = state.order_depths[basket]
            best_basket_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else 0
            best_basket_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else float('inf')

            component_cost, component_value, comp_ask_vols, comp_bid_vols, can_buy_comps, can_sell_comps = self._calculate_component_prices_volumes(state, components)

            # --- Check if we have a Strat 1 position and Strat 2 (closing) is profitable ---
            if potential_strat1_pos:
                closing_profit_margin = best_basket_bid - component_cost # Profit from selling basket, buying components
                if can_buy_comps and best_basket_bid != 0 and closing_profit_margin >= self.flattening_profit_threshold:
                    max_flatten_size = current_pos_basket # Try to close the whole position

                    basket_bid_volume = abs(basket_depth.buy_orders.get(best_basket_bid, 0))
                    max_component_buy_based_on_volume = min(comp_ask_vols) if comp_ask_vols else 0
                    limit_basket = self.position_limits[basket]
                    # Capacity to sell basket (current_pos is positive)
                    position_limit_sell_basket = limit_basket + current_pos_basket

                    component_buy_limits_allow = []
                    for component, quantity in components.items():
                         if quantity > 0:
                            comp_pos = state.position.get(component, 0) # Likely negative
                            comp_limit = self.position_limits[component]
                            max_can_buy_comp = comp_limit - comp_pos # Capacity to buy (towards positive limit)
                            max_baskets_allowed = max_can_buy_comp // quantity if quantity > 0 else float('inf')
                            component_buy_limits_allow.append(max_baskets_allowed)
                         else: component_buy_limits_allow.append(float('inf'))
                    max_baskets_comp_limits_buy = min(component_buy_limits_allow) if component_buy_limits_allow else 0

                    actual_trade_size = min(
                        max_flatten_size,
                        max(0, basket_bid_volume),
                        max(0, position_limit_sell_basket),
                        max(0, max_component_buy_based_on_volume),
                        max(0, max_baskets_comp_limits_buy)
                    )

                    if actual_trade_size > 0:
                        strat_key = f"{basket}_flatten_strat1"
                        flattening_orders[strat_key] = []
                        flattening_orders[strat_key].append(Order(basket, best_basket_bid, -actual_trade_size))
                        for component, quantity in components.items():
                            comp_depth = state.order_depths[component]
                            best_comp_ask = min(comp_depth.sell_orders.keys())
                            flattening_orders[strat_key].append(Order(component, best_comp_ask, actual_trade_size * quantity))
                        # logger.print(f"Potential Flattening (Strat 1 -> 2): Sell {actual_trade_size} {basket}@{best_basket_bid}. Profit/Unit: {closing_profit_margin}")


            # --- Check if we have a Strat 2 position and Strat 1 (closing) is profitable ---
            elif potential_strat2_pos:
                closing_profit_margin = component_value - best_basket_ask # Profit from selling components, buying basket
                if can_sell_comps and best_basket_ask != float('inf') and closing_profit_margin >= self.flattening_profit_threshold:
                    max_flatten_size = abs(current_pos_basket) # Try to close the whole position

                    basket_ask_volume = abs(basket_depth.sell_orders.get(best_basket_ask, 0))
                    max_component_sell_based_on_volume = min(comp_bid_vols) if comp_bid_vols else 0
                    limit_basket = self.position_limits[basket]
                    # Capacity to buy basket (current_pos is negative)
                    position_limit_buy_basket = limit_basket - current_pos_basket

                    component_sell_limits_allow = []
                    for component, quantity in components.items():
                        if quantity > 0:
                            comp_pos = state.position.get(component, 0) # Likely positive
                            comp_limit = self.position_limits[component]
                            max_can_sell_comp = comp_pos + comp_limit # Capacity to sell (towards negative limit)
                            max_baskets_allowed = max_can_sell_comp // quantity if quantity > 0 else float('inf')
                            component_sell_limits_allow.append(max_baskets_allowed)
                        else: component_sell_limits_allow.append(float('inf'))
                    max_baskets_comp_limits_sell = min(component_sell_limits_allow) if component_sell_limits_allow else 0

                    actual_trade_size = min(
                        max_flatten_size,
                        max(0, basket_ask_volume),
                        max(0, position_limit_buy_basket),
                        max(0, max_component_sell_based_on_volume),
                        max(0, max_baskets_comp_limits_sell)
                    )

                    if actual_trade_size > 0:
                        strat_key = f"{basket}_flatten_strat2"
                        flattening_orders[strat_key] = []
                        flattening_orders[strat_key].append(Order(basket, best_basket_ask, actual_trade_size))
                        for component, quantity in components.items():
                            comp_depth = state.order_depths[component]
                            best_comp_bid = max(comp_depth.buy_orders.keys())
                            flattening_orders[strat_key].append(Order(component, best_comp_bid, -actual_trade_size * quantity))
                        # logger.print(f"Potential Flattening (Strat 2 -> 1): Buy {actual_trade_size} {basket}@{best_basket_ask}. Profit/Unit: {closing_profit_margin}")

        return flattening_orders

    # Find NEW arbitrage opportunities
    def find_arbitrage_opportunities(self, state: TradingState) -> dict[str, list[Order]]:
        """Find potential NEW arbitrage opportunities. Uses self.min_arbitrage_profit."""
        potential_orders = {}
        for basket, components in self.basket_components.items():
            if basket not in state.order_depths: continue
            all_components_available = all(comp in state.order_depths for comp in components)
            if not all_components_available: continue

            basket_depth = state.order_depths[basket]
            best_basket_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else 0
            best_basket_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else float('inf')

            component_cost, component_value, comp_ask_vols, comp_bid_vols, can_buy_comps, can_sell_comps = self._calculate_component_prices_volumes(state, components)

            # --- Strategy 1: Buy basket, sell components ---
            profit_margin_1 = component_value - best_basket_ask
            if can_sell_comps and best_basket_ask != float('inf') and profit_margin_1 >= self.min_arbitrage_profit:
                basket_ask_volume = abs(basket_depth.sell_orders.get(best_basket_ask, 0))
                current_pos_basket = state.position.get(basket, 0)
                limit_basket = self.position_limits[basket]
                position_limit_buy_basket = limit_basket - current_pos_basket
                max_component_sell_based_on_volume = min(comp_bid_vols) if comp_bid_vols else 0

                component_sell_limits_allow = []
                for component, quantity in components.items():
                    if quantity > 0:
                        comp_pos = state.position.get(component, 0)
                        comp_limit = self.position_limits[component]
                        max_can_sell_comp = comp_pos + comp_limit
                        max_baskets_allowed = max_can_sell_comp // quantity if quantity > 0 else float('inf')
                        component_sell_limits_allow.append(max_baskets_allowed)
                    else: component_sell_limits_allow.append(float('inf'))
                max_baskets_comp_limits_sell = min(component_sell_limits_allow) if component_sell_limits_allow else 0

                max_trade_size = min(
                    max(0, basket_ask_volume),
                    max(0, position_limit_buy_basket),
                    max(0, max_component_sell_based_on_volume),
                    max(0, max_baskets_comp_limits_sell)
                )

                if max_trade_size > 0:
                    strat_key = f"{basket}_strat1"
                    potential_orders[strat_key] = []
                    potential_orders[strat_key].append(Order(basket, best_basket_ask, max_trade_size))
                    for component, quantity in components.items():
                        component_depth = state.order_depths[component]
                        best_component_bid = max(component_depth.buy_orders.keys())
                        potential_orders[strat_key].append(Order(component, best_component_bid, -max_trade_size * quantity))
                    # logger.print(f"Potential Arbitrage (Strat 1): Buy {max_trade_size} {basket}@{best_basket_ask}. Profit/Unit: {profit_margin_1}")


            # --- Strategy 2: Buy components, sell basket ---
            profit_margin_2 = best_basket_bid - component_cost
            if can_buy_comps and best_basket_bid != 0 and profit_margin_2 >= self.min_arbitrage_profit:
                basket_bid_volume = abs(basket_depth.buy_orders.get(best_basket_bid, 0))
                current_pos_basket = state.position.get(basket, 0)
                limit_basket = self.position_limits[basket]
                position_limit_sell_basket = limit_basket + current_pos_basket
                max_component_buy_based_on_volume = min(comp_ask_vols) if comp_ask_vols else 0

                component_buy_limits_allow = []
                for component, quantity in components.items():
                    if quantity > 0:
                        comp_pos = state.position.get(component, 0)
                        comp_limit = self.position_limits[component]
                        max_can_buy_comp = comp_limit - comp_pos
                        max_baskets_allowed = max_can_buy_comp // quantity if quantity > 0 else float('inf')
                        component_buy_limits_allow.append(max_baskets_allowed)
                    else: component_buy_limits_allow.append(float('inf'))
                max_baskets_comp_limits_buy = min(component_buy_limits_allow) if component_buy_limits_allow else 0

                max_trade_size = min(
                    max(0, basket_bid_volume),
                    max(0, position_limit_sell_basket),
                    max(0, max_component_buy_based_on_volume),
                    max(0, max_baskets_comp_limits_buy)
                )

                if max_trade_size > 0:
                    strat_key = f"{basket}_strat2"
                    potential_orders[strat_key] = []
                    potential_orders[strat_key].append(Order(basket, best_basket_bid, -max_trade_size))
                    for component, quantity in components.items():
                        component_depth = state.order_depths[component]
                        best_component_ask = min(component_depth.sell_orders.keys())
                        potential_orders[strat_key].append(Order(component, best_component_ask, max_trade_size * quantity))
                    # logger.print(f"Potential Arbitrage (Strat 2): Sell {max_trade_size} {basket}@{best_basket_bid}. Profit/Unit: {profit_margin_2}")

        return potential_orders

    # Standard market making logic (with inventory skewing)
    def get_market_making_orders(self, symbol: Symbol, depth: OrderDepth, position: int) -> list[Order]:
        """
        Get market making orders for a single symbol, respecting position limits
        and skewing quotes based on inventory.
        Uses self.mm_spread_threshold and self.mm_base_volume.
        """
        orders = []
        position_limit = self.position_limits.get(symbol)
        if position_limit is None:
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

        return orders # Corrected typo here


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic: Flatten -> Arbitrage -> Market Make
        """
        result = {symbol: [] for symbol in state.order_depths.keys()}
        pending_pos_delta: Dict[Symbol, int] = {} # Tracks position changes *within* this timestep
        traded_symbols_this_tick = set() # Symbols involved in *any* executed arbitrage or flattening trade

        # Log position changes from the previous state
        if not self.previous_positions: # Initialize if first run
            self.previous_positions = state.position.copy()
        else: # Log changes on subsequent runs
            self.log_position_changes(state.position, state.timestamp)

        # --- 0. Check for Flattening Opportunities FIRST ---
        potential_flattening_orders = self.find_flattening_opportunities(state)
        executed_flattening_baskets = set()

        for strat_key, orders_to_flatten in potential_flattening_orders.items():
            if not orders_to_flatten: continue
            basket_order = orders_to_flatten[0]
            component_orders = orders_to_flatten[1:]
            basket_symbol = basket_order.symbol
            can_place_all_legs = True

            # Check basket leg limit
            if not self._can_place_order(basket_symbol, basket_order.quantity, state.position.get(basket_symbol, 0), pending_pos_delta):
                can_place_all_legs = False

            # Check component legs limits
            if can_place_all_legs:
                for comp_order in component_orders:
                    if not self._can_place_order(comp_order.symbol, comp_order.quantity, state.position.get(comp_order.symbol, 0), pending_pos_delta):
                        can_place_all_legs = False
                        break

            # If all legs possible, execute flattening trade
            if can_place_all_legs:
                logger.print(f"Executing Flattening Trade ({strat_key}): {orders_to_flatten}")
                executed_flattening_baskets.add(basket_symbol)
                for order in orders_to_flatten:
                    result[order.symbol].append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                    traded_symbols_this_tick.add(order.symbol) # Mark symbol as traded

        # --- 1. Find NEW Arbitrage Opportunities ---
        potential_arbitrage_orders = self.find_arbitrage_opportunities(state)

        # Process potential NEW arbitrage orders, checking limits and avoiding conflicts with flattening
        for basket, components in self.basket_components.items():
            # Skip if this basket was involved in flattening this tick
            if basket in executed_flattening_baskets:
                # logger.print(f"Skipping new arbitrage for {basket} due to flattening trade.")
                continue

            # Check Strategy 1 (Buy Basket, Sell Components)
            strat1_key = f"{basket}_strat1"
            strat1_orders = potential_arbitrage_orders.get(strat1_key, [])
            if strat1_orders:
                basket_order = strat1_orders[0]
                component_orders = strat1_orders[1:]
                can_place_all_legs = True
                if not self._can_place_order(basket_order.symbol, basket_order.quantity, state.position.get(basket, 0), pending_pos_delta): can_place_all_legs = False
                if can_place_all_legs:
                    for comp_order in component_orders:
                        if not self._can_place_order(comp_order.symbol, comp_order.quantity, state.position.get(comp_order.symbol, 0), pending_pos_delta):
                            can_place_all_legs = False; break
                if can_place_all_legs:
                    logger.print(f"Executing Arbitrage (Strat 1) for {basket}: {strat1_orders}")
                    for order in strat1_orders:
                        result[order.symbol].append(order)
                        pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        traded_symbols_this_tick.add(order.symbol)

            # Check Strategy 2 (Sell Basket, Buy Components)
            strat2_key = f"{basket}_strat2"
            strat2_orders = potential_arbitrage_orders.get(strat2_key, [])
            if strat2_orders:
                basket_order = strat2_orders[0]
                component_orders = strat2_orders[1:]
                can_place_all_legs = True
                if not self._can_place_order(basket_order.symbol, basket_order.quantity, state.position.get(basket, 0), pending_pos_delta): can_place_all_legs = False
                if can_place_all_legs:
                    for comp_order in component_orders:
                        if not self._can_place_order(comp_order.symbol, comp_order.quantity, state.position.get(comp_order.symbol, 0), pending_pos_delta):
                            can_place_all_legs = False; break
                if can_place_all_legs:
                    logger.print(f"Executing Arbitrage (Strat 2) for {basket}: {strat2_orders}")
                    for order in strat2_orders:
                        result[order.symbol].append(order)
                        pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        traded_symbols_this_tick.add(order.symbol)


        # --- 2. Market Making ---
        # Skip MM for components always involved in arbs, SQUID_INK, and anything touched by arb/flattening THIS tick
        # Also skip baskets themselves if they were traded.
        symbols_to_skip_mm = traded_symbols_this_tick | {"SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES"}
        # Add baskets explicitly if they were traded
        if "PICNIC_BASKET1" in traded_symbols_this_tick: symbols_to_skip_mm.add("PICNIC_BASKET1")
        if "PICNIC_BASKET2" in traded_symbols_this_tick: symbols_to_skip_mm.add("PICNIC_BASKET2")


        for symbol in state.order_depths:
            if symbol in symbols_to_skip_mm:
                # logger.print(f"Skipping MM for {symbol} (Arbitrage/Flattening or Explicit Skip)")
                continue

            # Use the current position + any delta from executed arbitrage/flattening trades
            effective_position = state.position.get(symbol, 0) + pending_pos_delta.get(symbol, 0)
            potential_mm_orders = self.get_market_making_orders(symbol, state.order_depths[symbol], effective_position)

            for order in potential_mm_orders:
                # Check limit again before adding MM order, using the *original* state position
                # but considering the cumulative pending delta
                if self._can_place_order(order.symbol, order.quantity, state.position.get(symbol, 0), pending_pos_delta):
                    result[order.symbol].append(order)
                    pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                # else:
                     # logger.print(f"MM Order skipped for {symbol} due to limit.")


        # --- Final Steps ---
        conversions = 0 # No conversions logic implemented yet

        # Log final orders placed
        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log:
             logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")
             # logger.print(f"Timestamp: {state.timestamp}, Pending Deltas: {pending_pos_delta}") # Can be noisy
        # else: # Reduce logging noise
             # logger.print(f"Timestamp: {state.timestamp}, No orders placed.")


        # Update trader data if necessary (not used in this version)
        # self.trader_data = "..."

        # Update previous positions AFTER all logic for the current timestamp is done
        self.previous_positions = state.position.copy()

        # Flush the logs, orders, conversions, and trader data to standard output
        logger.flush(state, result, conversions, self.trader_data)

        # Return the orders, conversions, and trader data
        return result, conversions, self.trader_data

    def log_position_changes(self, current_positions: dict[Symbol, int], timestamp: int) -> None:
        """
        Log changes in positions compared to the start of the previous run.
        """
        log_entry = []
        all_symbols = set(current_positions.keys()) | set(self.previous_positions.keys())
        for symbol in sorted(list(all_symbols)):
            prev_pos = self.previous_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            if prev_pos != curr_pos:
                log_entry.append(f"{symbol}: {prev_pos}->{curr_pos}") # Shortened format

        if log_entry: # Only log if there were changes
            logger.print(f"Pos Changes @ {timestamp}: {', '.join(log_entry)}")

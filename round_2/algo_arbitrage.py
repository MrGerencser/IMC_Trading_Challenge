import json
from typing import Any, Dict, List

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
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This flush method is provided for context and remains unchanged.
        # It handles compressing the state and logs to fit within the output limits.
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Helper for flush - unchanged
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
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        # Helper for flush - unchanged
        if len(value) <= max_length:
            return value
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
            "DJEMBES": 60
        }

        # Define the components of each basket
        self.basket_components = {
            "PICNIC_BASKET1": {"CROISSANTS": 1, "JAMS": 1},
            "PICNIC_BASKET2": {"RAINFOREST_RESIN": 1, "KELP": 1, "SQUID_INK": 1}
        }

        # Track previous positions for logging changes
        self.previous_positions = {}
        # Store traderData if needed between runs
        self.trader_data = ""

        # --- Strategy Parameters ---
        # Minimum profit per basket for arbitrage trades
        self.min_arbitrage_profit = 2.0 # Increased from 1.0
        # Minimum spread for market making
        self.mm_spread_threshold = 3 # Increased from > 1
        # Base volume for market making orders
        self.mm_base_volume = 4 # Reduced from 5

    # Helper function to check and update positions before adding order
    def _can_place_order(self, symbol: Symbol, quantity: int, current_pos: int, pending_pos_delta: Dict[Symbol, int]) -> bool:
        limit = self.position_limits.get(symbol)
        if limit is None:
            logger.print(f"Warning: No position limit found for {symbol}. Order rejected.")
            return False # Should not happen if limits are defined for all traded symbols

        pending_delta = pending_pos_delta.get(symbol, 0)
        final_pos = current_pos + pending_delta + quantity

        if abs(final_pos) <= limit:
            return True
        else:
            # logger.print(f"Order rejected for {symbol}: Quantity {quantity} would exceed limit {limit}. Current: {current_pos}, Pending Delta: {pending_delta}, Resulting: {final_pos}")
            return False

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main trading logic
        """
        result = {symbol: [] for symbol in state.order_depths.keys()}
        pending_pos_delta: Dict[Symbol, int] = {} # Tracks position changes within this timestep

        # Log position changes from the previous state
        if not self.previous_positions: # Initialize if first run
            self.previous_positions = state.position.copy()
        else: # Log changes on subsequent runs
            self.log_position_changes(state.position, state.timestamp)


        # 1. Find potential arbitrage opportunities
        potential_arbitrage_orders = self.find_arbitrage_opportunities(state)
        arbitrage_symbols_traded = set() # Keep track of symbols involved in *executed* arbitrage

        # Process potential arbitrage orders, checking limits for each leg
        for basket, components in self.basket_components.items():
            # Check Strategy 1 (Buy Basket, Sell Components) potential orders
            strat1_orders = potential_arbitrage_orders.get(f"{basket}_strat1", [])
            if strat1_orders:
                basket_order = strat1_orders[0]
                component_orders = strat1_orders[1:]
                can_place_all_legs = True

                # Check basket leg
                if not self._can_place_order(basket_order.symbol, basket_order.quantity, state.position.get(basket, 0), pending_pos_delta):
                    can_place_all_legs = False

                # Check component legs
                if can_place_all_legs:
                    for comp_order in component_orders:
                        if not self._can_place_order(comp_order.symbol, comp_order.quantity, state.position.get(comp_order.symbol, 0), pending_pos_delta):
                            can_place_all_legs = False
                            break

                # If all legs are possible, add them and update pending positions
                if can_place_all_legs:
                    logger.print(f"Executing Arbitrage (Strat 1) for {basket}: {strat1_orders}")
                    for order in strat1_orders:
                        result[order.symbol].append(order)
                        pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        arbitrage_symbols_traded.add(order.symbol)
                    arbitrage_symbols_traded.add(basket) # Ensure basket is marked

            # Check Strategy 2 (Sell Basket, Buy Components) potential orders
            strat2_orders = potential_arbitrage_orders.get(f"{basket}_strat2", [])
            if strat2_orders:
                basket_order = strat2_orders[0]
                component_orders = strat2_orders[1:]
                can_place_all_legs = True

                # Check basket leg
                if not self._can_place_order(basket_order.symbol, basket_order.quantity, state.position.get(basket, 0), pending_pos_delta):
                    can_place_all_legs = False

                # Check component legs
                if can_place_all_legs:
                    for comp_order in component_orders:
                        if not self._can_place_order(comp_order.symbol, comp_order.quantity, state.position.get(comp_order.symbol, 0), pending_pos_delta):
                            can_place_all_legs = False
                            break

                # If all legs are possible, add them and update pending positions
                if can_place_all_legs:
                    logger.print(f"Executing Arbitrage (Strat 2) for {basket}: {strat2_orders}")
                    for order in strat2_orders:
                        result[order.symbol].append(order)
                        pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                        arbitrage_symbols_traded.add(order.symbol)
                    arbitrage_symbols_traded.add(basket) # Ensure basket is marked


        # 2. Add market making orders ONLY for symbols NOT involved in executed arbitrage this tick
        for symbol in state.order_depths:
            if symbol not in arbitrage_symbols_traded:
                # Use the current position + any delta from executed arbitrage trades
                effective_position = state.position.get(symbol, 0) + pending_pos_delta.get(symbol, 0)
                potential_mm_orders = self.get_market_making_orders(symbol, state.order_depths[symbol], effective_position)

                for order in potential_mm_orders:
                    # Check limit again before adding MM order
                    if self._can_place_order(order.symbol, order.quantity, state.position.get(symbol, 0), pending_pos_delta):
                        result[order.symbol].append(order)
                        pending_pos_delta[order.symbol] = pending_pos_delta.get(order.symbol, 0) + order.quantity
                    # else:
                        # logger.print(f"MM Order skipped for {symbol} due to limit.")


        # Determine number of conversions (0 for now)
        conversions = 0

        # Log final orders placed
        orders_to_log = {symbol: orders for symbol, orders in result.items() if orders}
        if orders_to_log:
             logger.print(f"Timestamp: {state.timestamp}, Final Orders: {orders_to_log}")
             # logger.print(f"Timestamp: {state.timestamp}, Pending Deltas: {pending_pos_delta}") # Can be noisy
        # else: # Reduce logging noise
             # logger.print(f"Timestamp: {state.timestamp}, No orders placed.")


        # Update trader data if necessary
        # self.trader_data = "..." # Example: Storing some state

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
                log_entry.append(f"{symbol}: {prev_pos} -> {curr_pos}")

        if log_entry: # Only log if there were changes
            logger.print(f"Position Changes @ {timestamp}: {', '.join(log_entry)}")


    def find_arbitrage_opportunities(self, state: TradingState) -> dict[str, list[Order]]:
        """
        Find potential arbitrage opportunities. Returns a dictionary where keys
        are like 'BASKET_strat1' or 'BASKET_strat2' and values are lists of
        potential orders for that strategy leg. Uses self.min_arbitrage_profit.
        """
        potential_orders = {} # Use unique keys for each potential strategy execution

        for basket, components in self.basket_components.items():
            if basket not in state.order_depths:
                continue

            all_components_available = True
            for component in components:
                if component not in state.order_depths:
                    all_components_available = False
                    break
            if not all_components_available:
                continue

            basket_depth = state.order_depths[basket]
            best_basket_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else 0
            best_basket_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else float('inf')

            # --- Calculate cost to BUY components (using best asks) ---
            component_cost = 0
            component_ask_volumes_at_best = []
            can_buy_components = True
            for component, quantity in components.items():
                component_depth = state.order_depths[component]
                if not component_depth.sell_orders:
                    can_buy_components = False
                    break
                best_component_ask = min(component_depth.sell_orders.keys())
                component_cost += best_component_ask * quantity
                component_ask_volume = abs(component_depth.sell_orders.get(best_component_ask, 0))
                if quantity > 0:
                    # Calculate how many *baskets* worth of this component we could buy at best ask
                    component_ask_volumes_at_best.append(component_ask_volume // quantity)
                else: # Should not happen for these baskets, but good practice
                    component_ask_volumes_at_best.append(float('inf'))

            if not can_buy_components:
                component_cost = float('inf')

            # --- Calculate value to SELL components (using best bids) ---
            component_value = 0
            component_bid_volumes_at_best = []
            can_sell_components = True
            for component, quantity in components.items():
                component_depth = state.order_depths[component]
                if not component_depth.buy_orders:
                    can_sell_components = False
                    break
                best_component_bid = max(component_depth.buy_orders.keys())
                component_value += best_component_bid * quantity
                component_bid_volume = abs(component_depth.buy_orders.get(best_component_bid, 0))
                if quantity > 0:
                     # Calculate how many *baskets* worth of this component we could sell at best bid
                    component_bid_volumes_at_best.append(component_bid_volume // quantity)
                else:
                    component_bid_volumes_at_best.append(float('inf'))

            if not can_sell_components:
                component_value = 0


            # --- Strategy 1: Buy basket, sell components ---
            profit_margin_1 = component_value - best_basket_ask
            # Use the minimum profit threshold
            if profit_margin_1 >= self.min_arbitrage_profit and component_value > 0 and best_basket_ask != float('inf'):
                basket_ask_volume = abs(basket_depth.sell_orders.get(best_basket_ask, 0))
                current_pos_basket = state.position.get(basket, 0)
                limit_basket = self.position_limits[basket]
                # Max baskets we can buy based on basket position limit
                position_limit_buy_basket = limit_basket - current_pos_basket
                # Max baskets we can make based on component sell volume available at best bid
                max_component_sell_based_on_volume = min(component_bid_volumes_at_best) if component_bid_volumes_at_best else 0

                # Max baskets we can make based on component position limits (ability to sell)
                component_sell_limits_allow = []
                for component, quantity in components.items():
                    if quantity > 0:
                        comp_pos = state.position.get(component, 0)
                        comp_limit = self.position_limits[component]
                        # Max we can sell = current position + limit (e.g., pos 10, limit 50 -> can sell 60 -> pos -50)
                        max_can_sell_comp = comp_pos + comp_limit
                        # How many baskets does this allow?
                        max_baskets_allowed = max_can_sell_comp // quantity if quantity > 0 else float('inf')
                        component_sell_limits_allow.append(max_baskets_allowed)
                    else:
                        component_sell_limits_allow.append(float('inf'))
                max_baskets_comp_limits_sell = min(component_sell_limits_allow) if component_sell_limits_allow else 0

                max_trade_size = min(
                    max(0, basket_ask_volume),             # Limit by basket ask volume
                    max(0, position_limit_buy_basket),     # Limit by basket buy position capacity
                    max(0, max_component_sell_based_on_volume), # Limit by component bid volume
                    max(0, max_baskets_comp_limits_sell)   # Limit by component sell position capacity
                )

                if max_trade_size > 0:
                    strat_key = f"{basket}_strat1"
                    potential_orders[strat_key] = []
                    potential_orders[strat_key].append(Order(basket, best_basket_ask, max_trade_size))
                    for component, quantity in components.items():
                        component_depth = state.order_depths[component]
                        best_component_bid = max(component_depth.buy_orders.keys()) # Price to sell component
                        component_sell_volume = -max_trade_size * quantity
                        potential_orders[strat_key].append(Order(component, best_component_bid, component_sell_volume))
                    # logger.print(f"Potential Arbitrage (Strat 1): Buy {max_trade_size} {basket}@{best_basket_ask}, Sell Components (Value: {component_value}). Profit/Unit: {profit_margin_1}")


            # --- Strategy 2: Buy components, sell basket ---
            profit_margin_2 = best_basket_bid - component_cost
            # Use the minimum profit threshold
            if profit_margin_2 >= self.min_arbitrage_profit and component_cost != float('inf') and best_basket_bid != 0:
                basket_bid_volume = abs(basket_depth.buy_orders.get(best_basket_bid, 0))
                current_pos_basket = state.position.get(basket, 0)
                limit_basket = self.position_limits[basket]
                # Max baskets we can sell based on basket position limit
                position_limit_sell_basket = limit_basket + current_pos_basket
                # Max baskets we can make based on component ask volume available at best ask
                max_component_buy_based_on_volume = min(component_ask_volumes_at_best) if component_ask_volumes_at_best else 0

                # Max baskets we can make based on component position limits (ability to buy)
                component_buy_limits_allow = []
                for component, quantity in components.items():
                    if quantity > 0:
                        comp_pos = state.position.get(component, 0)
                        comp_limit = self.position_limits[component]
                        # Max we can buy = limit - current position (e.g., pos 10, limit 50 -> can buy 40 -> pos 50)
                        max_can_buy_comp = comp_limit - comp_pos
                        # How many baskets does this allow?
                        max_baskets_allowed = max_can_buy_comp // quantity if quantity > 0 else float('inf')
                        component_buy_limits_allow.append(max_baskets_allowed)
                    else:
                        component_buy_limits_allow.append(float('inf'))
                max_baskets_comp_limits_buy = min(component_buy_limits_allow) if component_buy_limits_allow else 0

                max_trade_size = min(
                    max(0, basket_bid_volume),              # Limit by basket bid volume
                    max(0, position_limit_sell_basket),     # Limit by basket sell position capacity
                    max(0, max_component_buy_based_on_volume), # Limit by component ask volume
                    max(0, max_baskets_comp_limits_buy)    # Limit by component buy position capacity
                )

                if max_trade_size > 0:
                    strat_key = f"{basket}_strat2"
                    potential_orders[strat_key] = []
                    potential_orders[strat_key].append(Order(basket, best_basket_bid, -max_trade_size))
                    for component, quantity in components.items():
                        component_depth = state.order_depths[component]
                        best_component_ask = min(component_depth.sell_orders.keys()) # Price to buy component
                        component_buy_volume = max_trade_size * quantity
                        potential_orders[strat_key].append(Order(component, best_component_ask, component_buy_volume))
                    # logger.print(f"Potential Arbitrage (Strat 2): Sell {max_trade_size} {basket}@{best_basket_bid}, Buy Components (Cost: {component_cost}). Profit/Unit: {profit_margin_2}")

        return potential_orders

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

            # If significantly long, make buy quote passive, sell quote aggressive
            if position > base_volume / 2: # Example threshold: long more than half base volume
                buy_price = best_bid_price # Quote at best bid (passive)
                # sell_price remains best_ask - 1 (aggressive)
                # logger.print(f"MM Skew LONG for {symbol}: Buy@Bid, Sell@Ask-1")

            # If significantly short, make sell quote passive, buy quote aggressive
            elif position < -base_volume / 2: # Example threshold: short more than half base volume
                sell_price = best_ask_price # Quote at best ask (passive)
                # buy_price remains best_bid + 1 (aggressive)
                # logger.print(f"MM Skew SHORT for {symbol}: Buy@Bid+1, Sell@Ask")

            # else: # Near flat, quote aggressively on both sides
                # logger.print(f"MM Skew FLAT for {symbol}: Buy@Bid+1, Sell@Ask-1")


            # --- Place Orders ---
            # Place Buy Order
            if buy_capacity > 0:
                # Volume is minimum of base and capacity
                buy_volume = min(base_volume, buy_capacity)
                if buy_volume > 0:
                    # Ensure our buy price is still below the best ask
                    if buy_price < best_ask_price:
                        orders.append(Order(symbol, buy_price, buy_volume))
                        # logger.print(f"Potential MM Buy Order: {buy_volume} {symbol}@{buy_price}")
                    # else:
                        # logger.print(f"MM Buy price {buy_price} crossed ask {best_ask_price} for {symbol}, skipped.")


            # Place Sell Order
            if sell_capacity > 0:
                # Volume is minimum of base and capacity
                sell_volume = min(base_volume, sell_capacity)
                if sell_volume > 0:
                     # Ensure our sell price is still above the best bid
                    if sell_price > best_bid_price:
                        orders.append(Order(symbol, sell_price, -sell_volume)) # Negative volume for sell
                        # logger.print(f"Potential MM Sell Order: {-sell_volume} {symbol}@{sell_price}")
                    # else:
                        # logger.print(f"MM Sell price {sell_price} crossed bid {best_bid_price} for {symbol}, skipped.")

        return orders

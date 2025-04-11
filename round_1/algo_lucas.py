from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
import jsonpickle  # Add this import

class Trader:
    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50
    }

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # Load persistent state using jsonpickle
        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {"cycle_count": 0, "products": {}}
        
        # Increment cycle counter
        memory["cycle_count"] = memory.get("cycle_count", 0) + 1
        
        # Only trade every N cycles to avoid accumulating too many orders
        should_trade = memory["cycle_count"] % 5 == 0
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]
            
            # Initialize product memory if needed
            if product not in memory["products"]:
                memory["products"][product] = {"mid_prices": []}
            
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Update price history
            memory["products"][product]["mid_prices"].append(mid_price)
            if len(memory["products"][product]["mid_prices"]) > 20:
                memory["products"][product]["mid_prices"].pop(0)
                
            # Skip trading if it's not a trade cycle
            if not should_trade:
                result[product] = []
                continue

            if product in ["RAINFOREST_RESIN", "KELP"]:
                if spread > 2:
                    bid_price = int(mid_price - 1)
                    ask_price = int(mid_price + 1)
                    
                    # Use smaller volumes
                    buy_volume = min(1, limit - position)
                    sell_volume = min(1, limit + position)

                    if buy_volume > 0:
                        orders.append(Order(product, bid_price, buy_volume))
                    if sell_volume > 0:
                        orders.append(Order(product, ask_price, -sell_volume))

            elif product == "SQUID_INK":
                price_memory = memory["products"][product]["mid_prices"]
                
                # Calculate simple EMA like in algo_lucas.py
                alpha = 0.17  # Smoothing factor
                
                prev_ema = memory["products"][product].get("ema")
                if prev_ema is None:
                    ema = mid_price  # Initialize
                else:
                    ema = alpha * mid_price + (1 - alpha) * prev_ema
                
                memory["products"][product]["ema"] = ema
                
                # Use EMA as moving average
                threshold = 3
                short_term_ma = ema
                
                # Smaller volumes to reduce risk
                if mid_price < short_term_ma - threshold:
                    buy_volume = min(1, limit - position)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        
                elif mid_price > short_term_ma + threshold:
                    sell_volume = min(1, limit + position)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

            result[product] = orders

        # Serialize memory using jsonpickle
        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData
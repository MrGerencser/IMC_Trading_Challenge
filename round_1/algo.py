from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import matplotlib.pyplot as plt
import pandas as pd
import os

class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # Load persistent state
        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {"mid_prices": [], "ema": None}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # === STRATEGY PER PRODUCT ===
            if product in ["RAINFOREST_RESIN", "KELP"]:
                if spread > 2:
                    bid_price = int(mid_price - 1)
                    ask_price = int(mid_price + 1)

                    orders.append(Order(product, bid_price, 2))
                    orders.append(Order(product, ask_price, -2))

            elif product == "SQUID_INK":
                # EMA update
                alpha = 0.17  # Smoothing factor; can be tuned

                prev_ema = memory.get("ema")
                if prev_ema is None:
                    ema = mid_price  # Initialize
                else:
                    ema = alpha * mid_price + (1 - alpha) * prev_ema

                memory["ema"] = ema
                memory.setdefault("mid_prices", []).append(mid_price)

                threshold = 40
                short_term_ma = ema  # Use EMA as moving average

                if mid_price < short_term_ma - threshold:
                    orders.append(Order(product, best_ask, 3))

                elif mid_price > short_term_ma + threshold:
                    orders.append(Order(product, best_bid, -3))

            result[product] = orders

        # Save persistent memory
        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData





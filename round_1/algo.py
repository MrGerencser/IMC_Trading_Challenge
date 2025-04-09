from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import matplotlib.pyplot as plt
import pandas as pd
import os

class Trader:

    def run(self, state: TradingState):
        result = {}
        traderData = state.traderData
        conversions = 0  # No FX involved for now

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
                # Market making strategy
                if spread > 2:
                    bid_price = int(mid_price - 1)
                    ask_price = int(mid_price + 1)

                    # Small volume to reduce risk exposure
                    orders.append(Order(product, bid_price, 2))   # Buy
                    orders.append(Order(product, ask_price, -2))  # Sell

            elif product == "SQUID_INK":
                # Mean reversion strategy
                short_term_ma = 1970  # Can be dynamically updated with traderData
                threshold = 40        # Threshold to trigger reversal trades

                if mid_price < short_term_ma - threshold:
                    # Price is undervalued → Buy expecting mean reversion
                    orders.append(Order(product, best_ask, 3))  # Buy 3 at ask

                elif mid_price > short_term_ma + threshold:
                    # Price is overvalued → Sell expecting reversion
                    orders.append(Order(product, best_bid, -3))  # Sell 3 at bid

            result[product] = orders

        return result, conversions, traderData






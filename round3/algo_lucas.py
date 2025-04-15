from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math



class Trader:

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
        sigma = initial_guess
        for i in range(max_iterations):
            price = self.black_scholes_call_price(S, K, r, T, sigma)
            diff = price - market_price  # how far off we are
            if abs(diff) < tol:
                return sigma  # found a good enough solution
            v = self.vega(S, K, r, T, sigma)
            if v < 1e-8:
                # If vega is extremely small, we risk dividing by zero or huge jumps.
                # Could switch to a fallback method (bisection) or stop.
                break
            # Newton step
            sigma = sigma - diff / v
            # keep sigma positive
            if sigma < 0:
                sigma = 1e-5
        # If we exit the loop without returning, we can either raise an error or return the last sigma
        return sigma
        # Edge case: if option is deep in-the-money and market_price ~ (S-K),
        # you might want to set a floor or do a quick check here.
        
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
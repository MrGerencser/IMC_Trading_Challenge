import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error


def fit_bayes_model(training_data, plot = True):
    models = []
    """
    training_data: DataFrame containing the training data for a training day
    """
    products = training_data['product'].unique()

    # Plot results for each product
    for product in products:
        product_data = training_data[training_data['product'] == product].copy()
        
        # Create lagged features (for time k and k+1 prediction)
        product_data['mid_price_lagged'] = product_data['mid_price'].shift(-1)
        
        # Drop rows where we don't have a target for prediction (last row will be NaN)
        product_data = product_data.dropna(subset=['mid_price_lagged'])
        
        # Prepare features (all except day, timestamp, mid_price, and product)
        features = ['bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
                    'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1',
                    'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3']
        
        X = product_data[features]
        y = product_data['mid_price_lagged']
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Bayesian Linear Regression
        model = BayesianRidge()
        model.fit(X_train, y_train)
        models.append(model)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
    
        # Calculate mean squared error and standard deviation
        mse = mean_squared_error(y_test, y_pred)
        std_dev = np.std(y_pred - y_test)

        # Ensure that y_test.index is sorted in ascending order for correct plotting
        sorted_index = np.argsort(y_test.index)
        y_test_sorted = y_test.iloc[sorted_index]
        y_pred_sorted = y_pred[sorted_index]
        
        # Plot results (sorted by time index)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_sorted.index, y_test_sorted.values, label="Actual Mid Price", marker='o', linestyle='-', color='b')
            plt.plot(y_test_sorted.index, y_pred_sorted, label="Predicted Mid Price", linestyle='dashed', color='r')
            plt.fill_between(y_test_sorted.index, y_pred_sorted - std_dev, y_pred_sorted + std_dev, alpha=0.2, label="Prediction Std Dev")
            plt.title(f'{product} - Bayesian Linear Regression Prediction vs Actual')
            plt.xlabel('Time Index')
            plt.ylabel('Mid Price')
            plt.legend()
            plt.show()
        
        # Extract and print the regression parameters (coefficients and intercept)
        print(f"\nParameters for {product}:")
        print("Intercept:", model.intercept_)
        print("Coefficients (weights for features):")
        for feature, coef in zip(features, model.coef_):
            print(f"{feature}: {coef}")
    return models

def test_bayes_model(models, test_data):
    """
    test_data: DataFrame containing the test data for a training day
    """
    products = test_data['product'].unique()
    # Plot results for each product
    for product, model in zip(products, models):
        product_data = test_data[test_data['product'] == product].copy()
        
        # Create lagged features (for time k and k+1 prediction)
        product_data['mid_price_lagged'] = product_data['mid_price'].shift(-1)
        
        # Drop rows where we don't have a target for prediction (last row will be NaN)
        product_data = product_data.dropna(subset=['mid_price_lagged'])
        
        # Prepare features (all except day, timestamp, mid_price, and product)
        features = ['bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
                    'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1',
                    'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3']
        
        X_test = product_data[features]
        y_test = product_data['mid_price_lagged']
        # Predict on the test set
        y_pred = model.predict(X_test)
    
        # Calculate mean squared error and standard deviation
        mse = mean_squared_error(y_test, y_pred)
        std_dev = np.std(y_pred - y_test)
        # Ensure that y_test.index is sorted in ascending order for correct plotting
        sorted_index = np.argsort(y_test.index)
        y_test_sorted = y_test.iloc[sorted_index]
        y_pred_sorted = y_pred[sorted_index]
        
        # Plot results (sorted by time index)
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_sorted.index, y_test_sorted.values, label="Actual Mid Price", marker='o', linestyle='-', color='b')
        plt.plot(y_test_sorted.index, y_pred_sorted, label="Predicted Mid Price", linestyle='dashed', color='r')
        plt.fill_between(y_test_sorted.index, y_pred_sorted - std_dev, y_pred_sorted + std_dev, alpha=0.2, label="Prediction Std Dev")
        plt.title(f'{product} - Bayesian Linear Regression Prediction vs Actual on the TEST SET')
        plt.xlabel('Time Index')
        plt.ylabel('Mid Price')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Load CSV (you can modify the path accordingly)
    training_data = pd.read_csv('sample_data/round-2-island-data-bottle/prices_round_2_day_-1.csv', delimiter=';')
    # Fill NaN values with 0 for the entire dataset
    training_data = training_data.fillna(0)

    models = fit_bayes_model(training_data, plot=False)

    # Load CSV (you can modify the path accordingly)
    test_data = pd.read_csv('sample_data/round-2-island-data-bottle/prices_round_2_day_0.csv', delimiter=';')
    # Fill NaN values with 0 for the entire dataset
    test_data = test_data.fillna(0)

    test_bayes_model(models, test_data)
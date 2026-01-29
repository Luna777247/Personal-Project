#!/usr/bin/env python3
"""
Gold Price Forecasting with Real-Time Data Integration
======================================================

This script integrates real-time gold price data from multiple reputable sources:
- Yahoo Finance (yfinance)
- Alpha Vantage API
- Quandl (Nasdaq Data Link)

Features:
- Real-time data fetching
- Multiple data sources for reliability
- Data validation and cleaning
- Enhanced forecasting models
- Historical data storage

Author: AI Assistant
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
import quandl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys (set these as environment variables)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "demo")

class GoldPriceDataManager:
    """
    Manages gold price data from multiple sources
    """

    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def fetch_yahoo_finance_data(self, start_date="2020-01-01", end_date=None):
        """
        Fetch gold price data from Yahoo Finance
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("Fetching gold price data from Yahoo Finance...")

        try:
            # Gold futures ticker
            gold = yf.Ticker("GC=F")
            df = gold.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning("No data received from Yahoo Finance")
                return None

            # Clean and format data
            df = df[['Close']].rename(columns={'Close': 'price'})
            df.index = df.index.date
            df.index.name = 'date'

            logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
            return None

    def fetch_alpha_vantage_data(self, symbol="GLD"):
        """
        Fetch gold ETF data from Alpha Vantage
        """
        logger.info("Fetching gold price data from Alpha Vantage...")

        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

            if symbol == "GLD":
                # SPDR Gold Shares ETF
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
                df = data[['4. close']].rename(columns={'4. close': 'price'})
            else:
                # Try commodity data
                data, meta_data = ts.get_daily(symbol="GLD", outputsize='full')
                df = data[['4. close']].rename(columns={'4. close': 'price'})

            df.index = pd.to_datetime(df.index).date
            df.index.name = 'date'

            logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage")
            return df

        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")
            return None

    def fetch_quandl_data(self, start_date="2020-01-01"):
        """
        Fetch gold price data from Quandl (Nasdaq Data Link)
        """
        logger.info("Fetching gold price data from Quandl...")

        try:
            quandl.ApiConfig.api_key = QUANDL_API_KEY

            # London Bullion Market Association gold price
            df = quandl.get("LBMA/GOLD", start_date=start_date)

            if df.empty:
                logger.warning("No data received from Quandl")
                return None

            # Clean and format data
            df = df[['USD (AM)']].rename(columns={'USD (AM)': 'price'})
            df.index = pd.to_datetime(df.index).date
            df.index.name = 'date'
            df = df.dropna()

            logger.info(f"Successfully fetched {len(df)} records from Quandl")
            return df

        except Exception as e:
            logger.error(f"Error fetching from Quandl: {e}")
            return None

    def combine_data_sources(self, yahoo_df=None, alpha_df=None, quandl_df=None):
        """
        Combine data from multiple sources with validation
        """
        logger.info("Combining data from multiple sources...")

        dataframes = []
        sources = []

        if yahoo_df is not None:
            dataframes.append(yahoo_df)
            sources.append('yahoo')

        if alpha_df is not None:
            dataframes.append(alpha_df)
            sources.append('alpha_vantage')

        if quandl_df is not None:
            dataframes.append(quandl_df)
            sources.append('quandl')

        if not dataframes:
            logger.error("No data sources available")
            return None

        # Combine all dataframes
        combined_df = pd.concat(dataframes, keys=sources, names=['source', 'date'])

        # Reset index to work with date column
        combined_df = combined_df.reset_index()

        # Group by date and calculate mean price
        final_df = combined_df.groupby('date')['price'].agg(['mean', 'std', 'count']).round(2)

        # Rename columns
        final_df.columns = ['price', 'price_std', 'source_count']

        # Sort by date
        final_df = final_df.sort_index()

        logger.info(f"Combined data: {len(final_df)} records from {len(sources)} sources")
        return final_df

    def save_data(self, df, filename="gold_prices_realtime.csv"):
        """
        Save data to CSV file
        """
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename="gold_prices_realtime.csv"):
        """
        Load data from CSV file
        """
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            logger.info(f"Data loaded from {filepath}")
            return df
        else:
            logger.warning(f"File {filepath} not found")
            return None

class GoldPriceForecaster:
    """
    Enhanced forecasting with multiple models
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.models = {}

    def prepare_data(self, df):
        """
        Prepare data for forecasting
        """
        # Use only price column
        price_data = df['price'].dropna()

        # Ensure we have enough data
        if len(price_data) < 100:
            logger.warning("Insufficient data for reliable forecasting")
            return None

        return price_data

    def fit_arima_model(self, data, order=(5,1,0)):
        """
        Fit ARIMA model
        """
        logger.info("Fitting ARIMA model...")

        try:
            model = ARIMA(data, order=order)
            fitted_model = model.fit()

            self.models['arima'] = fitted_model
            logger.info("ARIMA model fitted successfully")
            return fitted_model

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            return None

    def fit_sarimax_model(self, data, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Fit SARIMAX model for seasonal data
        """
        logger.info("Fitting SARIMAX model...")

        try:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)

            self.models['sarimax'] = fitted_model
            logger.info("SARIMAX model fitted successfully")
            return fitted_model

        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {e}")
            return None

    def forecast(self, model_name, steps=30):
        """
        Generate forecasts
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None

        model = self.models[model_name]

        try:
            forecast = model.forecast(steps=steps)
            logger.info(f"Generated {steps} step forecast using {model_name}")
            return forecast

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None

    def evaluate_model(self, model_name, train_data, test_data):
        """
        Evaluate model performance
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None

        model = self.models[model_name]

        try:
            # Generate predictions for test period
            predictions = model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)

            metrics = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse
            }

            logger.info(f"Model evaluation - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

def main():
    """
    Main function to demonstrate real-time gold price data integration
    """
    print("=" * 70)
    print("🏆 Gold Price Forecasting with Real-Time Data Integration")
    print("=" * 70)

    # Initialize data manager
    data_manager = GoldPriceDataManager()

    # Fetch data from multiple sources
    print("\n📊 Fetching data from multiple sources...")

    yahoo_data = data_manager.fetch_yahoo_finance_data()
    alpha_data = data_manager.fetch_alpha_vantage_data()
    quandl_data = data_manager.fetch_quandl_data()

    # Combine data
    combined_data = data_manager.combine_data_sources(
        yahoo_df=yahoo_data,
        alpha_df=alpha_data,
        quandl_df=quandl_data
    )

    if combined_data is None or combined_data.empty:
        print("❌ No data available. Please check API keys and internet connection.")
        return

    print(f"✅ Successfully combined data: {len(combined_data)} records")
    print(f"📈 Data range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"💰 Current price: ${combined_data['price'].iloc[-1]:.2f}")

    # Save data
    data_manager.save_data(combined_data)

    # Initialize forecaster
    forecaster = GoldPriceForecaster(data_manager)

    # Prepare data for forecasting
    price_data = forecaster.prepare_data(combined_data)

    if price_data is None:
        print("❌ Insufficient data for forecasting")
        return

    # Fit models
    print("\n🤖 Training forecasting models...")

    arima_model = forecaster.fit_arima_model(price_data)
    sarimax_model = forecaster.fit_sarimax_model(price_data)

    # Generate forecasts
    print("\n🔮 Generating forecasts...")

    if arima_model:
        arima_forecast = forecaster.forecast('arima', steps=30)
        if arima_forecast is not None:
            print("📊 ARIMA Forecast (next 30 days):")
            print(arima_forecast.tail())

    if sarimax_model:
        sarimax_forecast = forecaster.forecast('sarimax', steps=30)
        if sarimax_forecast is not None:
            print("\n📊 SARIMAX Forecast (next 30 days):")
            print(sarimax_forecast.tail())

    # Visualize results
    print("\n📈 Creating visualization...")

    plt.figure(figsize=(15, 10))

    # Plot 1: Historical data
    plt.subplot(2, 2, 1)
    plt.plot(combined_data.index, combined_data['price'], label='Historical Price', color='blue')
    plt.title('Gold Price History (Multi-Source)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Price with confidence intervals
    plt.subplot(2, 2, 2)
    plt.plot(combined_data.index, combined_data['price'], label='Price', color='blue', alpha=0.7)
    plt.fill_between(combined_data.index,
                    combined_data['price'] - combined_data['price_std'],
                    combined_data['price'] + combined_data['price_std'],
                    alpha=0.3, color='blue', label='±1 Std Dev')
    plt.title('Gold Price with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Forecast comparison
    plt.subplot(2, 2, 3)
    last_date = combined_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=31)[1:]

    plt.plot(combined_data.index[-90:], combined_data['price'].tail(90),
            label='Historical', color='blue')

    if arima_model and arima_forecast is not None:
        plt.plot(future_dates, arima_forecast, label='ARIMA Forecast',
                color='red', linestyle='--')

    if sarimax_model and sarimax_forecast is not None:
        plt.plot(future_dates, sarimax_forecast, label='SARIMAX Forecast',
                color='green', linestyle='--')

    plt.title('Gold Price Forecast (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Plot 4: Data source reliability
    plt.subplot(2, 2, 4)
    plt.bar(range(len(combined_data)), combined_data['source_count'],
           color='orange', alpha=0.7)
    plt.title('Data Source Reliability')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Sources')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gold_price_analysis_realtime.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Analysis complete! Results saved to 'gold_price_analysis_realtime.png'")

    # Summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Total records: {len(combined_data)}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"Current price: ${combined_data['price'].iloc[-1]:.2f}")
    print(f"Price range: ${combined_data['price'].min():.2f} - ${combined_data['price'].max():.2f}")
    print(f"Average price: ${combined_data['price'].mean():.2f}")
    print(f"Average sources per data point: {combined_data['source_count'].mean():.1f}")

    print("\n🎯 Next Steps:")
    print("1. Set API keys for Alpha Vantage and Quandl for enhanced data")
    print("2. Implement automated daily data updates")
    print("3. Add more sophisticated forecasting models")
    print("4. Create web dashboard for real-time monitoring")

if __name__ == "__main__":
    main()
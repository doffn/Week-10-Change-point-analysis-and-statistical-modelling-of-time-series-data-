import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML


class DataVisualizer:
    def __init__(self, data: pd.DataFrame, logger: logging.Logger):
        """
        Initialize the DataVisualizer with data and a logger.

        Parameters:
            data (pd.DataFrame): DataFrame with 'Date' as index and 'Price' column.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.data = data
        self.logger = logger
        self.logger.info("DataVisualizer initialized.")

    def _display_error_message(self, method_name: str):
        """
        Display an error message with a clickable log file link in a notebook.

        Parameters:
            method_name (str): The name of the method where the error occurred.
        """
        log_link = '<a href="../logs/notebooks.log" target="_blank">Check the log file for details</a>'
        html = f"<p style='color:red;'>An error occurred in <strong>{method_name}</strong>. {log_link}</p>"
        display(HTML(html))

    def plot_box(self):
        """Display a box plot of Brent oil prices."""
        try:
            plt.figure(figsize=(8, 4))
            sns.boxplot(y='Price', data=self.data)
            plt.title('Box Plot of Brent Oil Prices')
            plt.ylabel('Price (USD per barrel)')
            plt.tight_layout()
            plt.show()
            self.logger.info("Box plot displayed.")
        except Exception as e:
            self.logger.error(f"plot_box error: {e}")
            self._display_error_message("plot_box")

    def plot_price_over_time(self):
        """Plot Brent oil prices over time."""
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data.index, self.data['Price'], label='Brent Oil Price', color='blue')
            plt.title('Brent Oil Prices Over Time')
            plt.xlabel('Date')
            plt.ylabel('Price (USD per barrel)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            self.logger.info("Time series plot displayed.")
        except Exception as e:
            self.logger.error(f"plot_price_over_time error: {e}")
            self._display_error_message("plot_price_over_time")

    def plot_price_distribution(self):
        """Plot the distribution of Brent oil prices."""
        try:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.data['Price'], bins=30, kde=True)
            plt.title('Distribution of Brent Oil Prices')
            plt.xlabel('Price (USD per barrel)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            self.logger.info("Price distribution plot displayed.")
        except Exception as e:
            self.logger.error(f"plot_price_distribution error: {e}")
            self._display_error_message("plot_price_distribution")

    def plot_yearly_average(self):
        """Plot the average Brent oil price per year."""
        try:
            df = self.data.reset_index()
            df['Year'] = df['Date'].dt.year
            yearly_avg = df.groupby('Year')['Price'].mean().reset_index()

            plt.figure(figsize=(12, 6))
            sns.barplot(data=yearly_avg, x='Year', y='Price', palette='viridis')
            plt.title('Average Yearly Brent Oil Prices')
            plt.xlabel('Year')
            plt.ylabel('Average Price (USD per barrel)')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
            self.logger.info("Yearly average price plot displayed.")
        except Exception as e:
            self.logger.error(f"plot_yearly_average error: {e}")
            self._display_error_message("plot_yearly_average")

    def plot_rolling_volatility(self, window: int = 30):
        """
        Plot the rolling volatility (standard deviation) of oil prices.

        Parameters:
            window (int): Number of days to calculate rolling standard deviation.
        """
        try:
            self.data['Rolling_Volatility'] = self.data['Price'].rolling(window).std()

            plt.figure(figsize=(10, 4))
            plt.plot(self.data.index, self.data['Rolling_Volatility'],
                     color='orange', label=f'{window}-Day Rolling Volatility')
            plt.title(f'{window}-Day Rolling Volatility of Brent Oil Prices')
            plt.xlabel('Date')
            plt.ylabel('Volatility (Rolling Std Dev)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            self.logger.info(f"{window}-day rolling volatility plot displayed.")
        except Exception as e:
            self.logger.error(f"plot_rolling_volatility error: {e}")
            self._display_error_message("plot_rolling_volatility")
 
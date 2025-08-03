import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import pymc as pm
import arviz as az
from datetime import timedelta
from scipy import stats
import logging


class EventChangeAnalyzer:
    """
    Analyze event-driven changes in Brent oil prices using CUSUM, Bayesian change point detection,
    and statistical analysis of prices before and after significant events.

    Parameters:
        price_data (pd.DataFrame): DataFrame with 'Date' as index and 'Price' column.
        logger (logging.Logger, optional): Logger for logging activities.
    """

    def __init__(self, price_data: pd.DataFrame, logger: logging.Logger = None):
        self.price_data = price_data
        self.logger = logger or logging.getLogger(__name__)
        self.mean_price = self.price_data['Price'].mean()

    def calculate_cusum(self):
        """Plot the CUSUM of deviations from the mean price."""
        try:
            cusum = (self.price_data['Price'] - self.mean_price).cumsum()
            plt.figure(figsize=(10, 4))
            plt.plot(self.price_data.index, cusum, color='orange', label='CUSUM')
            plt.title('CUSUM of Brent Oil Price Deviations')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Deviation (USD)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            self.logger.info("CUSUM plot generated.")
        except Exception as e:
            self.logger.error(f"Error calculating CUSUM: {e}")

    def detect_change_point(self, n_bkps: int = 5):
        """Detect change points in oil prices using the Binary Segmentation algorithm."""
        try:
            df = self.price_data.reset_index()
            prices = df['Price'].values

            algo = rpt.Binseg(model="rbf").fit(prices)
            change_points = algo.predict(n_bkps=n_bkps)
            change_years = [df['Date'].iloc[cp].year for cp in change_points[:-1]]
            print("Detected change point years:", change_years)

            plt.figure(figsize=(12, 5))
            plt.plot(df['Date'], df['Price'], label='Price', color='blue')
            for cp in change_points[:-1]:
                plt.axvline(df['Date'].iloc[cp], color='red', linestyle='--')
                plt.text(df['Date'].iloc[cp], df['Price'].iloc[cp], str(df['Date'].iloc[cp].year), fontsize=9, color='red')

            plt.title('Change Points in Brent Oil Prices')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error detecting change points: {e}")

    def bayesian_change_point_detection(self):
        """Estimate change point using Bayesian inference with PyMC."""
        try:
            data = self.price_data['Price'].values
            prior_mu = np.mean(data)

            with pm.Model() as model:
                cp = pm.DiscreteUniform('change_point', lower=0, upper=len(data) - 1)
                mu1 = pm.Normal('mu1', mu=prior_mu, sigma=5)
                mu2 = pm.Normal('mu2', mu=prior_mu, sigma=5)
                sigma1 = pm.HalfNormal('sigma1', sigma=5)
                sigma2 = pm.HalfNormal('sigma2', sigma=5)

                likelihood = pm.Normal(
                    'obs',
                    mu=pm.math.switch(cp >= np.arange(len(data)), mu1, mu2),
                    sigma=pm.math.switch(cp >= np.arange(len(data)), sigma1, sigma2),
                    observed=data
                )

                trace = pm.sample(4000, tune=2000, chains=4, random_seed=42, progressbar=True)
                az.plot_trace(trace)
                plt.tight_layout()
                plt.show()

                cp_est = int(np.median(trace.posterior['change_point'].values.flatten()))
                cp_date = self.price_data.index[cp_est]
                print(f"Estimated Change Point Date: {cp_date}")
                self.logger.info(f"Bayesian change point detected at {cp_date}.")
                return cp_date

        except Exception as e:
            self.logger.error(f"Bayesian change point detection failed: {e}")

    def _get_prices_around_event(self, event_date: pd.Timestamp, days_before=30, days_after=30):
        before = event_date - timedelta(days=days_before)
        after = event_date + timedelta(days=days_after)
        return self.price_data[(self.price_data.index >= before) & (self.price_data.index <= after)]

    def _calculate_percentage_change(self, event_date: pd.Timestamp, days: int):
        try:
            before = self.price_data.loc[event_date - timedelta(days=days), 'Price']
            after = self.price_data.loc[event_date + timedelta(days=days), 'Price']
            return ((after - before) / before) * 100
        except KeyError:
            return None

    def _plot_price_trends_around_events(self, key_events, days_before=180, days_after=180):
        plt.figure(figsize=(14, 8))
        for event, date_str in key_events.items():
            event_date = pd.to_datetime(date_str)
            prices = self._get_prices_around_event(event_date, days_before, days_after)
            plt.plot(prices.index, prices['Price'], label=f"{event} ({event_date.date()})")
            plt.axvline(event_date, color='red', linestyle='--')

        plt.title("Price Trends Around Events")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_percentage_changes_and_cumulative_returns(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Percentage change bars
        sns.barplot(data=df.melt(id_vars=["Event", "Date"],
                                 value_vars=["Change_1M", "Change_3M", "Change_6M"]),
                    x="Event", y="value", hue="variable", ax=axes[0])
        axes[0].set_title("Percentage Change in Price")
        axes[0].set_ylabel("% Change")
        axes[0].legend(title="Period")

        # Cumulative returns bars
        sns.barplot(data=df.melt(id_vars=["Event", "Date"],
                                 value_vars=["Cumulative Return Before", "Cumulative Return After"]),
                    x="Event", y="value", hue="variable", ax=axes[1])
        axes[1].set_title("Cumulative Returns Before and After Events")
        axes[1].set_ylabel("Return")
        axes[1].legend(title="Return Type")

        plt.tight_layout()
        plt.show()

    def _perform_statistical_analysis(self, key_events):
        results = {}
        for event, date_str in key_events.items():
            try:
                event_date = pd.to_datetime(date_str)
                before = self._get_prices_around_event(event_date, days_before=180).loc[:event_date, 'Price']
                after = self._get_prices_around_event(event_date, days_after=180).loc[event_date:, 'Price']
                t_stat, p_val = stats.ttest_ind(before, after, nan_policy='omit')
                results[event] = {"t-statistic": t_stat, "p-value": p_val}
            except KeyError:
                self.logger.warning(f"Skipping event {event} at {date_str}, insufficient data.")

        df = pd.DataFrame(results).T
        print(df)
        return df

    def analyze_price_changes_around_events(self, key_events: dict):
        """Analyze price changes and volatility around a set of key events."""
        results = []

        for event, date_str in key_events.items():
            try:
                event_date = pd.to_datetime(date_str)
                prices = self._get_prices_around_event(event_date, 180, 180)

                pct_1m = self._calculate_percentage_change(event_date, 30)
                pct_3m = self._calculate_percentage_change(event_date, 90)
                pct_6m = self._calculate_percentage_change(event_date, 180)

                cum_before = prices.loc[:event_date, 'Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                cum_after = prices.loc[event_date:, 'Price'].pct_change().add(1).cumprod().iloc[-1] - 1

                results.append({
                    "Event": event,
                    "Date": date_str,
                    "Change_1M": pct_1m,
                    "Change_3M": pct_3m,
                    "Change_6M": pct_6m,
                    "Cumulative Return Before": cum_before,
                    "Cumulative Return After": cum_after
                })

            except Exception as e:
                self.logger.warning(f"Could not process event {event}: {e}")

        event_df = pd.DataFrame(results)
        self._plot_price_trends_around_events(key_events)
        self._plot_percentage_changes_and_cumulative_returns(event_df)
        t_test_df = self._perform_statistical_analysis(key_events)

        return event_df, t_test_df

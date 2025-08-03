import os
import logging
import pandas as pd
import gdown
from IPython.display import display


class DataPreprocessor:
    def __init__(
        self,
        drive_link: str,
        output_dir: str = "../data/",
        output_filename: str = "data.csv",
        logger: logging.Logger = None,
    ):
        """
        Initialize the DataPreprocessor with a Google Drive link and file settings.

        Parameters:
            drive_link (str): Shareable Google Drive link to the dataset.
            output_dir (str): Directory to save the downloaded file.
            output_filename (str): Name of the saved CSV file.
            logger (logging.Logger, optional): Custom logger for logging.
        """
        self.drive_link = drive_link
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, output_filename)
        self.logger = logger or logging.getLogger(__name__)
        self.data: pd.DataFrame = None

    def _generate_download_url(self) -> str:
        """Convert Google Drive share link to a direct download URL."""
        try:
            file_id = self.drive_link.split("/")[5]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except IndexError:
            self.logger.error("Invalid Google Drive link format.")
            raise ValueError("Invalid Google Drive link.")

    def load_data(self) -> pd.DataFrame:
        """
        Download the CSV data from Google Drive and load it into a DataFrame.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {self.output_dir}")

            url = self._generate_download_url()
            self.logger.info("Initiating file download...")
            gdown.download(url, self.output_file, quiet=False)
            self.logger.info(f"Downloaded file to: {self.output_file}")

            self.data = pd.read_csv(self.output_file)
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='mixed', errors='coerce')
            self.data.set_index('Date', inplace=True)

            self.logger.info("Data loaded and indexed successfully.")
            return self.data

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform a comprehensive inspection of the DataFrame.

        Parameters:
            df (pd.DataFrame): The dataset to inspect.

        Returns:
            pd.DataFrame: Summary statistics for numeric columns.
        """
        if df.empty:
            raise ValueError("Provided DataFrame is empty.")

        try:
            df = df.reset_index()

            print(f"Shape: {df.shape}")
            print("\nData Types:\n", df.dtypes)

            missing = df.isnull().sum()
            if missing.any():
                print("\nMissing Values:\n", missing[missing > 0])
                self.logger.warning("Dataset contains missing values.")
            else:
                print("\nNo missing values detected.")
                self.logger.info("No missing values found.")

            print("\nUnique Values per Column:\n", df.nunique())

            duplicates = df[df.duplicated()]
            print(f"\nDuplicate Rows: {len(duplicates)}")
            if not duplicates.empty:
                print(duplicates.head())

            summary = df.describe(include='number')
            print("\nSummary Statistics:")
            display(summary)

            return summary

        except Exception as e:
            self.logger.error(f"Inspection error: {e}")
            raise

import unittest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from scripts.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Set up a logger for the tests
        self.logger = logging.getLogger('TestLogger')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)

        # Create an instance of DataPreprocessor with a dummy link
        self.drive_link = "https://drive.google.com/file/d/1abcdefg/view?usp=sharing"
        self.preprocessor = DataPreprocessor(self.drive_link, logger=self.logger)

    @patch('gdown.download')
    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv, mock_gdown):
        # Mock the download and read_csv functions
        mock_gdown.return_value = None  # Simulate successful download
        # Ensure DataFrame has both 'Date' and 'Price' columns
        mock_read_csv.return_value = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Price': [80, 82]
        })

        # Call load_data
        data = self.preprocessor.load_data()

        # Assertions
        mock_gdown.assert_called_once()
        mock_read_csv.assert_called_once_with(self.preprocessor.output_file)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (2, 1))  # Expect shape to be (2, 1) due to 'Date' index

    @patch('gdown.download')
    def test_load_data_failure(self, mock_gdown):
        # Simulate a download failure
        mock_gdown.side_effect = Exception("Download error")

        # Assert that loading data raises an error
        with self.assertRaises(Exception):
            self.preprocessor.load_data()

    def test_inspect(self):
        # Test with a valid DataFrame
        valid_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Price': [80, 82, None]
        }
        valid_df = pd.DataFrame(valid_data)
        valid_df.set_index('Date', inplace=True)

        # Call inspect method for a valid DataFrame
        try:
            self.preprocessor.inspect(valid_df)  # Should not raise an error
        except Exception as e:
            self.fail(f"inspect method raised an exception unexpectedly with valid input: {e}")

        # Test with an invalid DataFrame (e.g., empty DataFrame)
        invalid_df = pd.DataFrame()  # Empty DataFrame to trigger an error

        # Check that calling inspect raises a ValueError for the invalid DataFrame
        with self.assertRaises(ValueError) as context:
            self.preprocessor.inspect(invalid_df)

        # Verify the exception message
        self.assertEqual(str(context.exception), "The DataFrame is empty.")

if __name__ == '__main__':
    unittest.main()

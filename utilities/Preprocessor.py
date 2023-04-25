from typing import List, Union

import pandas as pd


class Preprocessor:
    def __init__(self):
        self.data = None
        self.response = None
        self.normalized_data = None

    def load_data(self, file_path: str, delimiter: str = ",", column_names: List[str] = None, response_name: str = None,
                  na_values: List[str] = None):
        # Try and parse the CSV data
        self.data = pd.read_csv(file_path, delimiter=delimiter, header=None, names=column_names, na_values=na_values)

        # If the response column is provided, separate it into its own instance field
        if response_name:
            self.response = self.data[response_name]
            self.data = self.data.drop(columns=response_name)

    def drop_columns(self, columns_to_drop: Union[str, List[str]]):
        # Drop specified columns
        self.data = self.data.drop(columns=columns_to_drop)

    def drop_na(self):
        # If the response variable has been separated from the predictors, drop corresponding rows from response as well
        try:
            self.response = self.response[~self.data.isna().any(axis=1)]
        except TypeError:
            raise ValueError("Response variable has not been defined")

        # Drop rows that are missing values
        self.data = self.data.dropna(axis=0)

    def normalize(self):
        pass

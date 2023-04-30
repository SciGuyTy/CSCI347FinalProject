from typing import List, Union, Any, Dict, Callable

import pandas as pd
from utilities.Utilities import Utilities


class Preprocessor:
    def __init__(self):
        """
        A data loader and preprocessor
        """

        self.data: Union[pd.DataFrame, None] = None
        self.response: Union[pd.Series, None] = None

    def load_data(self, file_path: str, delimiter: str = ",", column_names: List[str] = None, response_name: str = None,
                  na_values: List[str] = None, converters: Any = None):
        """
        Load data from a csv file

        Parameters
        ----------
        file_path: str
            The filepath to the csv file

        delimiter: str
            The delimiter character for the data samples (defaults to ',')

        column_names: List[str]
            A list of column names for the data (defaults to None)

        response_name: str
            The name of the response column (defaults to None)

        na_values: List[str]
            A list of strings that define NA values in the data

        converters: Dict[str, Callable]
            A converter dictionary that applies a function to specified columns
        """

        # Try and parse the CSV data
        self.data = pd.read_csv(file_path, delimiter=delimiter, header=None, names=column_names, na_values=na_values,
                                converters=converters)

        # If the response column is provided, separate it into its own instance field
        if response_name:
            self.response = self.data[response_name]
            self.data = self.data.drop(columns=response_name)

    def drop_columns(self, columns_to_drop: Union[str, List[str]]):
        """
        Drop columns from the data

        Parameters
        ----------
        columns_to_drop: Union[str, List[str]]
            The name(s) of the column(s) to drop
        """

        # Drop specified columns
        self.data = self.data.drop(columns=columns_to_drop)

    def drop_na(self):
        """
        Drop missing values from the data
        """

        # If the response variable has been separated from the predictors, drop corresponding rows from response as well
        try:
            self.response = self.response[~self.data.isna().any(axis=1)]
        except TypeError:
            raise ValueError("Response variable has not been defined")

        # Drop rows that are missing values
        self.data = self.data.dropna(axis=0)

    def normalize(self):
        """
        Standard normalize the data
        """

        for column_name, column in self.data.items():
            # Compute column mean
            column_mean = column.mean()

            # Compute column standard deviation
            column_sd = Utilities.standard_deviation(column)

            # Standard normalize the column
            column = column.apply(lambda element: (element - column_mean) / column_sd)

            # Update column in data frame
            self.data[column_name] = column

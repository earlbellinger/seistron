"""defines a standardized data class for observed stellar data"""

import pandas as pd

class StellarData:
    def __init__(self, dataframe: pd.DataFrame, metadata: dict = None):
        # store the dataframe with observed data
        self.dataframe = dataframe
        # metadata contains column-specific info: labels, error types, units, etc.
        self.metadata = metadata or {}

    def get_column(self, column: str):
        # return the data column
        return self.dataframe[column]

    def get_label(self, column: str):
        # return the full label for a column if available
        return self.metadata.get(column, {}).get("label", column)

    def get_short_label(self, column: str):
        # return the short label for a column if available
        return self.metadata.get(column, {}).get("short_label", column)

    def __repr__(self):
        return f"StellarData({self.dataframe.shape[0]} rows, {len(self.dataframe.columns)} columns)"

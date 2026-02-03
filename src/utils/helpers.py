"""
File: helpers.py
Description: Utility functions for data processing and common operations.
Dependencies: pandas, datetime
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""

import pandas as pd
from datetime import datetime
from typing import Union, List


def convert_unix_timestamp(timestamp: int) -> datetime:
    """
    Converts a UNIX timestamp to a datetime object.
    
    All timestamp fields in the dataset are UNIX integers.
    Use this function to convert them to readable dates.
    
    Args:
        timestamp (int): UNIX timestamp (seconds since epoch).
    
    Returns:
        datetime: Converted datetime object.
    
    Example:
        >>> convert_unix_timestamp(1609459200)
        datetime.datetime(2021, 1, 1, 0, 0)
    """
    return datetime.fromtimestamp(timestamp)


def convert_timestamp_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Converts a timestamp column from UNIX to datetime in a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp column.
        column_name (str): Name of the column to convert.
    
    Returns:
        pd.DataFrame: DataFrame with converted timestamp column.
    """
    df = df.copy()
    df[column_name] = pd.to_datetime(df[column_name], unit='s')
    return df


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculates the percentage change between two values.
    
    Args:
        old_value (float): The original value.
        new_value (float): The new value.
    
    Returns:
        float: Percentage change (positive for increase, negative for decrease).
    
    Raises:
        ValueError: If old_value is zero.
    """
    if old_value == 0:
        raise ValueError("Cannot calculate percentage change when old value is zero")
    
    return ((new_value - old_value) / old_value) * 100


def filter_by_date_range(df: pd.DataFrame, date_column: str, 
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters a DataFrame by a date range.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        date_column (str): Name of the date column.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask]


def categorize_performance(value: float, thresholds: dict) -> str:
    """
    Categorizes a performance metric based on defined thresholds.
    
    Args:
        value (float): The value to categorize.
        thresholds (dict): Dictionary with 'low', 'medium', 'high' threshold values.
    
    Returns:
        str: Performance category ('poor', 'fair', 'good', 'excellent').
    
    Example:
        >>> categorize_performance(75, {'low': 50, 'medium': 70, 'high': 90})
        'good'
    """
    if value < thresholds['low']:
        return 'poor'
    elif value < thresholds['medium']:
        return 'fair'
    elif value < thresholds['high']:
        return 'good'
    else:
        return 'excellent'


def aggregate_by_period(df: pd.DataFrame, date_column: str, 
                       value_column: str, period: str = 'D') -> pd.DataFrame:
    """
    Aggregates data by time period (daily, weekly, monthly).
    
    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        date_column (str): Name of the date column.
        value_column (str): Name of the value column to aggregate.
        period (str): Period for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly).
    
    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    aggregated = df[value_column].resample(period).sum().reset_index()
    
    return aggregated


def format_currency(amount: float, currency: str = 'DKK') -> str:
    """
    Formats a monetary value with currency symbol.
    
    All monetary values in the dataset are in DKK (Danish Krone).
    
    Args:
        amount (float): The monetary amount.
        currency (str): Currency code (default: 'DKK').
    
    Returns:
        str: Formatted currency string.
    
    Example:
        >>> format_currency(1250.50)
        'DKK 1,250.50'
    """
    return f"{currency} {amount:,.2f}"

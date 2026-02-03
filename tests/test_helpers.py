"""
File: test_helpers.py
Description: Unit tests for utility helper functions.
Dependencies: pytest, pandas
Author: Sample Team

This is a sample file demonstrating proper test structure and documentation.
Students should replace this with their actual tests.

Run tests with: pytest tests/
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import (
    convert_unix_timestamp,
    calculate_percentage_change,
    categorize_performance,
    format_currency
)


class TestHelpers:
    """Test suite for helper utility functions."""
    
    def test_convert_unix_timestamp(self):
        """Test UNIX timestamp conversion to datetime."""
        # Test known timestamp (2021-01-01 00:00:00 UTC)
        timestamp = 1609459200
        result = convert_unix_timestamp(timestamp)
        
        assert isinstance(result, datetime)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1
    
    def test_calculate_percentage_change_increase(self):
        """Test percentage change calculation for increase."""
        old_value = 100
        new_value = 150
        result = calculate_percentage_change(old_value, new_value)
        
        assert result == 50.0
    
    def test_calculate_percentage_change_decrease(self):
        """Test percentage change calculation for decrease."""
        old_value = 200
        new_value = 150
        result = calculate_percentage_change(old_value, new_value)
        
        assert result == -25.0
    
    def test_calculate_percentage_change_zero_old_value(self):
        """Test that zero old value raises ValueError."""
        with pytest.raises(ValueError):
            calculate_percentage_change(0, 100)
    
    def test_categorize_performance_poor(self):
        """Test performance categorization for poor performance."""
        thresholds = {'low': 50, 'medium': 70, 'high': 90}
        result = categorize_performance(40, thresholds)
        
        assert result == 'poor'
    
    def test_categorize_performance_fair(self):
        """Test performance categorization for fair performance."""
        thresholds = {'low': 50, 'medium': 70, 'high': 90}
        result = categorize_performance(60, thresholds)
        
        assert result == 'fair'
    
    def test_categorize_performance_good(self):
        """Test performance categorization for good performance."""
        thresholds = {'low': 50, 'medium': 70, 'high': 90}
        result = categorize_performance(80, thresholds)
        
        assert result == 'good'
    
    def test_categorize_performance_excellent(self):
        """Test performance categorization for excellent performance."""
        thresholds = {'low': 50, 'medium': 70, 'high': 90}
        result = categorize_performance(95, thresholds)
        
        assert result == 'excellent'
    
    def test_format_currency_default(self):
        """Test currency formatting with default DKK."""
        result = format_currency(1250.50)
        
        assert result == 'DKK 1,250.50'
    
    def test_format_currency_custom(self):
        """Test currency formatting with custom currency."""
        result = format_currency(1250.50, 'USD')
        
        assert result == 'USD 1,250.50'


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

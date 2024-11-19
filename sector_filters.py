# sector_filters.py

# Sector-specific thresholds for filtering based on Alpha Vantage data
sector_thresholds = {
    'Technology': {
        'MarketCapitalization': 1e9,  # Lower threshold to capture emerging tech companies
        'EBITDA': 100e6,  # Lower EBITDA threshold to include growth-focused companies
        'PERatio': [2.4, 75],  # Wider P/E range to include high-growth stocks
        'EPS': 0.145,  # Lower EPS threshold to include companies reinvesting in growth
        'Beta': 2.5,  # Higher beta to account for volatility in high-growth tech stocks
    },
    'Healthcare': {
        'MarketCapitalization': 2.5e9,  # Lower threshold for emerging healthcare companies
        'EBITDA': 100e6,  # Lower EBITDA to include companies in R&D phases
        'PERatio': [10, 50],  # Wider P/E range to include speculative investments
        'EPS': 0.2,  # Lower EPS threshold for companies with high R&D expenses
        'Beta': 2,  # Higher beta to reflect volatility in biotech and pharmaceutical sectors
    },
    'Financial': {
        'MarketCapitalization': 10e9,  # Lower threshold to include smaller financial institutions
        'EBITDA': 500e6,  # Lower EBITDA to capture smaller but growing financial firms
        'PERatio': [5, 25],  # Slightly wider P/E range to include growth-oriented financials
        'EPS': 1,  # Lower EPS threshold for smaller financial institutions
        'Beta': 2,  # Higher beta due to increased market sensitivity
    },
    # Default values for sectors not explicitly defined
    'default': {
        'MarketCapitalization': 1e9,  # Lower threshold for broader market inclusion
        'EBITDA': 50e6,  # Lower EBITDA to include more companies
        'PERatio': [5, 40],  # Wider P/E range for speculative investments
        'EPS': 0.5,  # Lower EPS threshold to capture growth-oriented stocks
        'Beta': 2,  # Higher beta to account for increased volatility
    }
}

market_condition_adjustments = {
    'PERatio': -5,  # Tighten P/E ratio in overvalued markets
    'Beta': 0.1,  # Slightly increase beta tolerance in stable markets
}

# Exporting the settings
def get_sector_thresholds():
    return sector_thresholds

def get_market_condition_adjustments():
    return market_condition_adjustments

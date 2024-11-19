import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import logging
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlpacaDataProcessor:
    """
    A class to process and analyze Alpaca trading account data.

    Attributes:
        api (tradeapi.REST): Alpaca API client
        account (Dict): Account information
    """

    def __init__(self):
        """Initialize the AlpacaDataProcessor with API credentials."""
        load_dotenv()

        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

        # Get account information
        try:
            self.account = self.api.get_account()
            logger.info("Successfully connected to Alpaca API")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            raise

    def get_positions_df(self) -> pd.DataFrame:
        """
        Fetch current positions and convert to DataFrame with calculated metrics.

        Returns:
            pd.DataFrame: DataFrame containing position information and metrics
        """
        try:
            positions = self.api.list_positions()

            if not positions:
                logger.info("No open positions found")
                return pd.DataFrame()

            # Extract position data with comprehensive metrics
            position_data = []
            for pos in positions:
                # Use default values if any field is None
                entry_price = float(pos.avg_entry_price) if pos.avg_entry_price is not None else 0.0
                current_price = float(pos.current_price) if pos.current_price is not None else 0.0
                qty = float(pos.qty) if pos.qty is not None else 0.0

                position_dict = {
                    'symbol': pos.symbol,
                    'quantity': qty,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'market_value': float(pos.market_value) if pos.market_value is not None else 0.0,
                    'cost_basis': float(pos.cost_basis) if pos.cost_basis is not None else 0.0,
                    'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl is not None else 0.0,
                    'unrealized_plpc': (float(pos.unrealized_plpc) * 100) if pos.unrealized_plpc is not None else 0.0,
                    'asset_class': 'crypto' if pos.symbol.endswith('USD') else 'stock',
                    'side': 'long' if qty > 0 else 'short',
                    'intraday_price_change': float(
                        pos.unrealized_intraday_pl) if pos.unrealized_intraday_pl is not None else 0.0,
                    'intraday_price_change_pct': (float(
                        pos.unrealized_intraday_plpc) * 100) if pos.unrealized_intraday_plpc is not None else 0.0,
                    'position_value': abs(qty * current_price),
                    'position_weight': (abs(qty * current_price) / float(
                        self.account.portfolio_value) * 100) if self.account.portfolio_value else 0.0
                }
                position_data.append(position_dict)

            # Create DataFrame and set display formatting
            df = pd.DataFrame(position_data)
            if not df.empty:
                # Sort by position value descending
                df = df.sort_values('position_value', ascending=False)

                # Format numeric columns
                for col in ['unrealized_plpc', 'intraday_price_change_pct', 'position_weight']:
                    df[col] = df[col].round(2)
                for col in ['entry_price', 'current_price', 'market_value', 'unrealized_pl', 'position_value']:
                    df[col] = df[col].round(2)

            return df

        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            raise

    def get_trades_df(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical trades and convert to DataFrame with performance metrics.

        Args:
            start_date (str, optional): Start date for trade history (YYYY-MM-DD)
            end_date (str, optional): End date for trade history (YYYY-MM-DD)

        Returns:
            pd.DataFrame: DataFrame containing trade history and performance metrics
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

            if not start_date:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')

            # Fetch trades with ISO format including 'Z' for UTC
            trades = self.api.list_orders(
                status='closed',
                after=start_date.isoformat() + 'Z',
                until=end_date.isoformat() + 'Z'
            )

            if not trades:
                logger.info("No trades found for the specified period")
                return pd.DataFrame()

            # Extract trade data with performance metrics
            trade_data = []
            for trade in trades:
                if trade.filled_at is None:
                    continue

                filled_qty = float(trade.filled_qty)
                filled_price = float(trade.filled_avg_price)

                trade_dict = {
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'filled_qty': filled_qty,
                    'filled_price': filled_price,
                    'order_type': trade.type,
                    'trade_date': pd.to_datetime(trade.filled_at),
                    'trade_value': abs(filled_qty * filled_price),
                    'commission': float(trade.commission) if hasattr(trade, 'commission') else 0,
                    'order_id': trade.id,
                    'asset_class': 'crypto' if trade.symbol.endswith('USD') else 'stock'
                }
                trade_data.append(trade_dict)

            # Create DataFrame and add calculated metrics
            df = pd.DataFrame(trade_data)
            if not df.empty:
                # Sort by date descending
                df = df.sort_values('trade_date', ascending=False)

                # Calculate additional metrics
                df['trade_value'] = df['trade_value'].round(2)
                df['filled_price'] = df['filled_price'].round(2)

                # Add trade sequence number
                df['trade_sequence'] = range(1, len(df) + 1)

                # Format dates
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            return df

        except Exception as e:
            logger.error(f"Error fetching trades: {str(e)}")
            raise

    def get_combined_summary(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Get comprehensive summary of positions and trades with portfolio metrics.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: Positions DataFrame, Trades DataFrame, and Portfolio Metrics
        """
        try:
            positions_df = self.get_positions_df()
            trades_df = self.get_trades_df()

            # Calculate portfolio metrics
            portfolio_metrics = {
                'total_equity': float(self.account.equity),
                'cash': float(self.account.cash),
                'buying_power': float(self.account.buying_power),
                'portfolio_value': float(self.account.portfolio_value),
                'total_positions': len(positions_df) if not positions_df.empty else 0,
                'total_pl': float(self.account.equity) - float(self.account.last_equity),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Add position concentration metrics if positions exist
            if not positions_df.empty:
                portfolio_metrics.update({
                    'largest_position_symbol': positions_df.iloc[0]['symbol'],
                    'largest_position_weight': positions_df.iloc[0]['position_weight'],
                    'crypto_exposure': positions_df[positions_df['asset_class'] == 'crypto']['position_value'].sum(),
                    'stock_exposure': positions_df[positions_df['asset_class'] == 'stock']['position_value'].sum()
                })

            return positions_df, trades_df, portfolio_metrics

        except Exception as e:
            logger.error(f"Error generating combined summary: {str(e)}")
            raise


# Example usage:
if __name__ == "__main__":
    processor = AlpacaDataProcessor()
    positions_df, trades_df, portfolio_metrics = processor.get_combined_summary()

    # Print portfolio metrics
    print("\nPortfolio Metrics:")
    for key, value in portfolio_metrics.items():
        print(f"{key}: {value}")

    # Export DataFrames to CSV files if not empty
    if not positions_df.empty:
        positions_df.to_csv('positions_data.csv', index=False)
        print("\nCurrent Positions saved to 'positions_data.csv'")

    if not trades_df.empty:
        trades_df.to_csv('trades_data.csv', index=False)
        print("\nRecent Trades saved to 'trades_data.csv'")

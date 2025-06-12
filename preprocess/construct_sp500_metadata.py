import os
import pandas as pd

def load_sp500_components(input_csv: str) -> pd.DataFrame:
    """
    Load the S&P 500 constituent history file, which should have columns like:
    ticker, company_name, date_added, date_removed.
    """
    df = pd.read_csv(input_csv, parse_dates=['date_added','date_removed'])
    return df

def expand_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    For each ticker, expand its membership over each quarter (or other freq)
    between start and end, so you get one row per (ticker, period).
    """
    periods = pd.date_range(start, end, freq='Q')
    records = []
    for _, row in df.iterrows():
        for period in periods:
            if row['date_added'] <= period and (pd.isna(row['date_removed']) or period <= row['date_removed']):
                records.append({
                    'ticker': row['ticker'],
                    'period_end': period
                })
    return pd.DataFrame(records)

def save_sp500(df: pd.DataFrame, out_csv: str):
    """Write expanded S&P 500 membership to CSV."""
    df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Construct S&P 500 membership metadata")
    parser.add_argument('--input-csv', required=True,
                        help='CSV of raw S&P 500 additions/removals')
    parser.add_argument('--start', required=True,
                        help='Start date (YYYY-MM-DD) for expansion')
    parser.add_argument('--end', required=True,
                        help='End date (YYYY-MM-DD) for expansion')
    parser.add_argument('--output-csv', required=True,
                        help='Path to write sp500_expanded.csv')
    args = parser.parse_args()

    base = load_sp500_components(args.input_csv)
    expanded = expand_dates(base, args.start, args.end)
    save_sp500(expanded, args.output_csv)
"""
Helper functions for loading and cleaning various financial datasets (SAS/CSV).
"""
import pandas as pd
from datetime import datetime
dateutil_parser = pd.to_datetime


def decode_bytes_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode any byte-string columns in the DataFrame to UTF-8 strings.
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x
            )
    return df


def load_sas_dataset(path: str, usecols: list = None) -> pd.DataFrame:
    """
    Load a SAS dataset (.sas7bdat) into a pandas DataFrame, with optional column selection.
    """
    df = pd.read_sas(path, format='sas7bdat', encoding='latin1')
    if usecols:
        df = df[usecols]
    df = decode_bytes_cols(df)
    return df


def load_gvkey(path: str) -> pd.DataFrame:
    """
    Load CapitalIQ GVKEY mapping, ensuring COMPANYID as int and GVKEY as str.
    """
    df = load_sas_dataset(path, usecols=['COMPANYID','GVKEY'])
    df['COMPANYID'] = df['COMPANYID'].astype(int)
    df['GVKEY'] = df['GVKEY'].astype(str)
    return df


def load_surprise(path: str) -> pd.DataFrame:
    """
    Load SUER score dataset and filter out extreme 1st and 99th percentiles.
    """
    df = load_sas_dataset(path)
    # compute quantile bounds
    low = df['SUESCORE'].quantile(0.01)
    high = df['SUESCORE'].quantile(0.99)
    # filter
    df = df[(df['SUESCORE'] > low) & (df['SUESCORE'] < high)]
    return df


def load_compustat_link(path: str) -> pd.DataFrame:
    """
    Load Compustat-CRSP link table with key fields.
    """
    cols = ['GVKEY','TIC','LIID','LINKDT','LINKENDDT','LINKPRIM','LINKTYPE','LPERMCO','LPERMNO','CONM','CUSIP']
    df = load_sas_dataset(path, usecols=cols)
    return df


def load_ibes(path: str) -> pd.DataFrame:
    """
    Load IBES dataset and decode CUSIP.
    """
    df = load_sas_dataset(path)
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper for loading financial SAS datasets')
    parser.add_argument('--gvkey', help='Path to capitaliq_gvkey.sas7bdat')
    parser.add_argument('--surprise', help='Path to surpsumu1.sas7bdat')
    parser.add_argument('--compustat', help='Path to compustat link .sas7bdat')
    parser.add_argument('--ibes', help='Path to ibes .sas7bdat')
    args = parser.parse_args()

    if args.gvkey:
        df = load_gvkey(args.gvkey)
        print(f'Loaded GVKEY: {len(df)} rows')
    if args.surprise:
        df = load_surprise(args.surprise)
        print(f'Filtered surprise to {len(df)} rows')
    if args.compustat:
        df = load_compustat_link(args.compustat)
        print(f'Loaded compustat link: {len(df)} rows')
    if args.ibes:
        df = load_ibes(args.ibes)
        print(f'Loaded IBES: {len(df)} rows')

import requests
import pandas as pd
import io
from time import sleep
import os


def fetch_wpdx_kenya(file: str="../data/raw/wpdx_kenya.csv") -> pd.DataFrame:
    """
    Fetches the WPDX Kenya dataset with 22,000 water points.
    Returns a DataFrame containing the water points.
    """

    base = "https://data.waterpointdata.org/resource/eqje-vguj.csv"
    all_chunks = []

    for offset in range(0, 21000, 1000):
        print(f"🔃 || Fetching rows {offset} to {offset + 1000}...")            
        
        params = {
            "$select": "*",
            "$where": "clean_country_name='Kenya'",
            "$order": "report_date DESC",
            "$limit": 1000,
            "$offset": offset
        }
        
        resp = requests.get(base, params=params)
        resp.raise_for_status()
        if resp.status_code != 200:
            print(f"❌ || Error fetching data: {resp.status_code} - {resp.text}")
            break
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            print("⚠️ || No more rows, stopping early.")
            break
        all_chunks.append(df)
        sleep(0.5)

    df_wpdx = pd.concat(all_chunks, ignore_index=True)
    print(f"✅ || Pulled {len(df_wpdx)} rows with {df_wpdx.shape[1]} columns from WPDX+ dataset.")
    df_wpdx.to_csv(file, index=False)
    return df_wpdx

def get_wpdx_kenya(file: str="../data/raw/wpdx_kenya.csv") -> pd.DataFrame:
    """
    Checks if the dataset already exists and has the expected number of rows.   
    If not, it fetches the data from the source.
    Returns a DataFrame containing the water points.
    """

    try:
        with open(file, "r") as f:
            df_wpdx = pd.read_csv(f,low_memory=False)
            
            len_wpdx = len(df_wpdx)
            if len_wpdx < 21000:
                print(f"⚠️ || Expected 21,000 rows, but found {len_wpdx} rows in wpdx_kenya.csv. Deleting and refetching...")
                try:
                    os.remove(file)
                except PermissionError:
                    print(f"⚠️ || Could not delete file (in use). Will overwrite when saving.")
                return fetch_wpdx_kenya(file)
            else:
                print(f"✅ || wpdx_kenya.csv already exists with {len_wpdx} rows, skipping fetch.")
                return df_wpdx
    except FileNotFoundError:
        print("⚠️ || wpdx_kenya.csv not found, fetching data...")
    except PermissionError:
            print(f"🚨 || Permission denied reading file. Retrying fetch...")
    return fetch_wpdx_kenya(file)

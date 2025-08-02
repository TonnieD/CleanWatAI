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

def collapse_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Collapse CSV with dissolved and total measurements into single rows.
    Each timestamp record will have both Value.Dissolved and Value.Total columns.
    """
    
    
    # unqiue identification for record
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')
    df['Group.Key'] = df['GEMS.Station.Number'] + '_' + df['Sample.Date'] + '_' + df['Sample.Time'] + '_' + df['Depth'].astype(str)
    df_dis = df[df['Parameter.Code'].str.endswith('-Dis')].copy()
    df_tot = df[df['Parameter.Code'].str.endswith('-Tot')].copy()
    df_dis = df_dis.rename(columns={'Value': 'Value.Dissolved'})
    df_tot = df_tot.rename(columns={'Value': 'Value.Total'})
    common_cols = ['GEMS.Station.Number', 'Sample.Date', 'Sample.Time', 'Depth', 
                   'Analysis.Method.Code', 'Value.Flags', 'Unit', 'Data.Quality', 'Group.Key']
    
    # Merge on Group.Key
    result = df_dis[common_cols + ['Value.Dissolved']].merge(
        df_tot[common_cols + ['Value.Total']], 
        on='Group.Key', 
        how='outer',
        suffixes=('', '_tot')
    )
    
    for col in common_cols[:-1]: 
        if col + '_tot' in result.columns:
            result[col] = result[col].fillna(result[col + '_tot'])
            result = result.drop(columns=[col + '_tot'])

    result = result.drop(columns=['Group.Key'])
    other_cols = [col for col in result.columns if col not in ['Value.Dissolved', 'Value.Total']]
    result = result[other_cols + ['Value.Dissolved', 'Value.Total']]
    result.to_csv(output_file, index=False, sep=';')
    print(f"✅ || Collapsed {len(df)} rows to {len(result)} rows and saved to {output_file}")
    
    return result

def collapse_zinc_csv(input_file: str="../data/raw/zinc.csv", output_file: str="../data/processed/zinc_collapsed.csv") -> pd.DataFrame:
    return collapse_csv(input_file, output_file)

def collapse_mercury_csv(input_file: str="../data/raw/mercury.csv", output_file: str="../data/processed/mercury_collapsed.csv") -> pd.DataFrame:
    return collapse_csv(input_file, output_file)

def merge_mercury_zinc(output_file: str="../data/processed/mercury_zinc.csv") -> pd.DataFrame:
    """
    Merges the mercury and zinc data into the WPDX dataset.
    Returns a DataFrame with the merged data.
    """
    
    try:
        df_mercury = pd.read_csv("../data/processed/mercury_collapsed.csv", sep=';')
        df_zinc = pd.read_csv("../data/processed/Zinc_collapsed.csv", sep=';')

        # Merge on station, date, time - this would need proper implementation
        df_mercury_zinc = pd.concat([df_mercury, df_zinc], ignore_index=True)
        df_mercury_zinc.to_csv(output_file, index=False, sep=';')
        
        return df_mercury_zinc
    except FileNotFoundError:
        print("⚠️ || Collapsed CSV files not found, please ensure they exist in the specified path.")
        return pd.DataFrame()
  
    
def get_gems(file: str="../data/processed/gems.csv") -> pd.DataFrame:
    """
    Fetches the GEMS dataset from data folder
    Returns a DataFrame containing the GEMS data.
    """
    df_gems = pd.read_csv(file)
    return df_gems
"""
Target: GEMS.Station.Number, Sample.Date, pH, NO2N, NO3N, TP, O2-Dis, NH4N, TEMP, EC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_combined_key(df):
    df['GEMS.Station.Number_Sample.Date'] = df['GEMS.Station.Number'].astype(str) + '_' + df['Sample.Date'].astype(str)
    return df.drop(columns=['GEMS.Station.Number', 'Sample.Date'])

def process_chemical_data(file_path, parameter_code, output_column, value_filter=None):
    """
    Generic function to process chemical data from GEMS CSV files
    """
    df = pd.read_csv(file_path, sep=';')
    
    df = df[df['Parameter.Code'] == parameter_code].copy()
    df['Sample.Date'] = pd.to_datetime(df['Sample.Date'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['GEMS.Station.Number', 'Sample.Date', 'Value'])
    df = df[df['Value'] >= 0]
    
    if value_filter:
        df = df[value_filter(df)]
    
    df = create_combined_key(df)
    
    clean_df = df.groupby(['GEMS.Station.Number_Sample.Date']).agg({'Value': 'mean'}).reset_index()
    clean_df.rename(columns={'Value': output_column}, inplace=True)
    return clean_df

def clean_ph_data():
    """
    Process pH data from raw pH.csv
    """
    print("🧪 Processing pH data...")
    
    ph_clean = process_chemical_data(
        file_path=Path('data/raw/pH.csv'),
        parameter_code='pH',
        output_column='pH',
        value_filter=lambda df: (df['Value'] >= 0) & (df['Value'] <= 14)
    )
    
    print(f"   pH records: {len(ph_clean):,}")
    return ph_clean

def clean_nitrogen_data():
    """
    Process nitrogen compounds from Oxidized_Nitrogen.csv and Other_Nitrogen.csv
    """
    print("🧪 Processing nitrogen compounds...")
    
    no2n_clean = process_chemical_data(
        file_path=Path('data/raw/Oxidized_Nitrogen.csv'),
        parameter_code='NO2N',
        output_column='NO2N'
    )
    
    no3n_clean = process_chemical_data(
        file_path=Path('data/raw/Oxidized_Nitrogen.csv'),
        parameter_code='NO3N',
        output_column='NO3N'
    )
    
    nh4n_clean = process_chemical_data(
        file_path=Path('data/raw/Other_Nitrogen.csv'),
        parameter_code='NH4N',
        output_column='NH4N'
    )
    
    final_nitrogen = no2n_clean.merge(
        no3n_clean,
        on='GEMS.Station.Number_Sample.Date',
        how='outer'
    ).merge(
        nh4n_clean,
        on='GEMS.Station.Number_Sample.Date',
        how='outer'
    )
    
    print(f"   NO2N records done: {final_nitrogen['NO2N'].notna().sum():,}")
    print(f"   NO3N records done: {final_nitrogen['NO3N'].notna().sum():,}")
    print(f"   NH4N records done: {final_nitrogen['NH4N'].notna().sum():,}")
    return final_nitrogen

def clean_phosphorus_data():
    """
    Process Total Phosphorus (TP) from Phosphorus.csv
    """
    print("🧪 Processing Total Phosphorus (TP)...")
    
    tp_clean = process_chemical_data(
        file_path=Path('data/raw/Phosphorus.csv'),
        parameter_code='TP',
        output_column='TP'
    )
    
    print(f"   TP records done: {len(tp_clean):,}")
    return tp_clean

def clean_oxygen_data():
    """
    Process Dissolved Oxygen (O2-Dis) from Dissolved_Gas.csv
    """
    print("🧪 Processing Dissolved Oxygen (O2-Dis)...")
    
    o2_clean = process_chemical_data(
        file_path=Path('data/raw/Dissolved_Gas.csv'),
        parameter_code='O2-Dis',
        output_column='O2-Dis'
    )
    
    print(f"   O2-Dis records done: {len(o2_clean):,}")
    return o2_clean

def clean_temperature_data():
    """
    Process Temperature from Temperature.csv
    """
    print("🧪 Processing Temperature...")
    
    temp_clean = process_chemical_data(
        file_path=Path('data/raw/Temperature.csv'),
        parameter_code='TEMP',
        output_column='TEMP'
    )
    
    print(f"   TEMP records done: {len(temp_clean):,}")
    return temp_clean

def clean_electrical_conductance_data():
    """
    Process Electrical Conductivity from Electrical_Conductance.csv
    """
    print("🧪 Processing Electrical Conductance...")
    
    ec_clean = process_chemical_data(
        file_path=Path('data/raw/Electrical_Conductance.csv'),
        parameter_code='EC',
        output_column='EC'
    )
    
    print(f"   EC records done: {len(ec_clean):,}")
    return ec_clean

def merge_chemical_datasets(ph_data, nitrogen_data, phosphorus_data, oxygen_data, temperature_data, ec_data):
    """
    Strategic merging of chemical datasets to maximize data retention
    - Start with pH bc most common parameter)
    - Use inner joins for critical nutrients (N, P) - no missing values allowed
    - Use left join for O2-Dis to preserve chemical data, interpolate for missing vals later
    - Use left join for Temperature to preserve chemical data
    - Ensures scientifically meaningful water quality profiles
    """
    print("🔗 Merging chemical datasets...")
    
    master_df = ph_data.copy()
    print(f"   Starting with pH: {len(master_df):,} records")
    
    # nitrogen data 
    master_df = master_df.merge(
        nitrogen_data,
        on='GEMS.Station.Number_Sample.Date',
        how='inner'
    )
    print(f"   After nitrogen merge: {len(master_df):,} records")
    
    # phosphorus data 
    master_df = master_df.merge(
        phosphorus_data,
        on='GEMS.Station.Number_Sample.Date',
        how='inner'
    )
    print(f"   After phosphorus merge: {len(master_df):,} records")
    
    # oxygen data 
    master_df = master_df.merge(
        oxygen_data,
        on='GEMS.Station.Number_Sample.Date',
        how='left'
    )
    print(f"   After oxygen merge: {len(master_df):,} records")
    
    # temperature data
    master_df = master_df.merge(
        temperature_data,
        on='GEMS.Station.Number_Sample.Date',
        how='left'
    )
    print(f"   After temperature merge: {len(master_df):,} records")
    
    # electrical conductance data
    master_df = master_df.merge(
        ec_data,
        on='GEMS.Station.Number_Sample.Date',
        how='left'
    )
    print(f"   After electrical conductance merge: {len(master_df):,} records")
    
    
    return master_df

def interpolate_oxygen_by_station(df):
    """
    Scientifically accurate time-based interpolation of O2-Dis missing values by station    
    """
    
    print("⏰ Implementing time-based O2-Dis interpolation by station...")
    
    df['GEMS.Station.Number'] = df['GEMS.Station.Number_Sample.Date'].str.split('_').str[0]
    df['Sample.Date'] = pd.to_datetime(df['GEMS.Station.Number_Sample.Date'].str.split('_').str[1])
    
    stations = df['GEMS.Station.Number'].unique()
    interpolated_parts = []
    
    for i, station in enumerate(stations, 1):
        station_data = df[df['GEMS.Station.Number'] == station].copy()
        
        # sort by date at station
        station_data = station_data.sort_values('Sample.Date')
        non_null_o2 = station_data['O2-Dis'].notna().sum()
        if non_null_o2 >= 2:
            station_data.set_index('Sample.Date', inplace=True)
            
            # scientifcally correct time-based interpolation 
            station_data['O2-Dis'] = station_data['O2-Dis'].interpolate(
                method='time',
                limit_direction='both',
                limit=30
            )
            station_data.reset_index(inplace=True)
            
        interpolated_parts.append(station_data)
        
        if i % 100 == 0:
            print(f"   Processed {i:,} stations out of {len(stations):,}")
    
    result_df = pd.concat(interpolated_parts, ignore_index=True)
    result_df = result_df.drop(columns=['GEMS.Station.Number', 'Sample.Date'])
    o2_before = df['O2-Dis'].notna().sum()
    o2_after = result_df['O2-Dis'].notna().sum()
    
    print(f"   O2-Dis before interpolation: {o2_before:,}")
    print(f"   O2-Dis after interpolation: {o2_after:,}")
    print(f"   Values filled by interpolation: {o2_after - o2_before:,}")
    return result_df

def create_final_clean_dataset(df):
    """
    Create final dataset with complete records only
    """
    print("🧹 Creating final clean dataset...")
    
    target_columns = ['GEMS.Station.Number_Sample.Date', 'pH', 'NO2N', 'NO3N', 'TP', 'O2-Dis', 'NH4N', 'TEMP', 'EC']
    df = df[target_columns]
    
    before_clean = len(df)
    clean_df = df.dropna()
    after_clean = len(clean_df)
    
    print(f"   Records before cleaning: {before_clean:,}")
    print(f"   Records after cleaning: {after_clean:,}")
    print(f"   Data retention: {(after_clean / before_clean) * 100:.1f}%")
    
    # Validate 
    missing_check = clean_df.isnull().sum()
    print(f"   Final dataset shape: {clean_df.shape}")
    print(f"   Missing values check: {missing_check.sum()} (should be 0)")    
    return clean_df

def main():
    """
    Main execution pipeline for water quality dataset creation
    """
    
    print("=" * 80)
    print("Columns: GEMS.Station.Number_Sample.Date, pH, NO2N, NO3N, TP, O2-Dis, NH4N, TEMP, EC")
    print("=" * 80)
    
    # Step 1: Individual chemical datasets
    ph_data = clean_ph_data()
    nitrogen_data = clean_nitrogen_data()
    phosphorus_data = clean_phosphorus_data()
    oxygen_data = clean_oxygen_data()
    temperature_data = clean_temperature_data()
    ec_data = clean_electrical_conductance_data()
    
    # Step 2: Merging
    merged_df = merge_chemical_datasets(ph_data, nitrogen_data, phosphorus_data, oxygen_data, temperature_data, ec_data)
    
    # Step 3: O2-Dis feature engineering
    interpolated_df = interpolate_oxygen_by_station(merged_df)
    
    # Step 4: Clean 
    final_df = create_final_clean_dataset(interpolated_df)
    
    # Step 5: Save
    output_path = Path('data/processed/gems.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ WATER QUALITY DATASET CREATION COMPLETE!")
    print(f"📁 gems.csv file: {output_path}")
    print(f"📊 Final records: {len(final_df):,}")
    
    return final_df

if __name__ == "__main__":
    final_dataset = main()

"""
Data cleaning and preprocessing module.
Handles missing values, standardizes units, calculates per-capita metrics,
and filters data from 2000 onward.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data(data_dir=None):
    """Load raw dataset files from data directory."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    # Common file patterns to look for
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Try to identify the main dataset file
    # Common names: crime, incarceration, prisoners, etc.
    main_file = None
    for file in csv_files:
        name_lower = file.name.lower()
        if any(keyword in name_lower for keyword in ['crime', 'incarceration', 'prisoner', 'state']):
            main_file = file
            break
    
    # If no specific file found, use the first CSV
    if main_file is None:
        main_file = csv_files[0]
    
    print(f"Loading data from: {main_file.name}")
    df = pd.read_csv(main_file)
    
    return df, data_dir

def standardize_column_names(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def identify_key_columns(df):
    """Identify key columns in the dataset."""
    # Common column name patterns
    state_col = None
    year_col = None
    population_col = None
    crime_col = None
    incarceration_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if not state_col and any(term in col_lower for term in ['state', 'jurisdiction', 'location']):
            state_col = col
        if not year_col and any(term in col_lower for term in ['year', 'date', 'time']):
            year_col = col
        if not population_col and any(term in col_lower for term in ['population', 'pop', 'total_pop']):
            population_col = col
        # For crime, prefer columns with 'total' and avoid metadata columns
        # But we'll calculate total from violent + property, so don't set crime_col here
        # (We'll handle it after checking for violent and property crime)
        # For incarceration, prefer prisoner_count over includes_jails
        if any(term in col_lower for term in ['prisoner', 'incarceration', 'inmate']):
            if 'count' in col_lower or 'number' in col_lower:
                incarceration_col = col  # Prefer count columns
            elif not incarceration_col and 'jail' not in col_lower:
                incarceration_col = col
    
    # If we have violent and property crime, calculate total crime
    violent_col = None
    property_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'violent' in col_lower and 'total' in col_lower:
            violent_col = col
        if 'property' in col_lower and 'total' in col_lower:
            property_col = col
    
    # Always calculate total crime from violent + property if both are available
    if violent_col and property_col:
        crime_col = 'total_crime_calculated'
    elif violent_col:
        crime_col = violent_col
    elif property_col:
        crime_col = property_col
    
    return {
        'state': state_col,
        'year': year_col,
        'population': population_col,
        'crime': crime_col,
        'incarceration': incarceration_col,
        'violent_crime': violent_col,
        'property_crime': property_col
    }

def calculate_per_capita_metrics(df, key_cols):
    """Calculate per-capita metrics for crime and incarceration."""
    df = df.copy()
    
    # Calculate total crime if needed (violent + property)
    if key_cols['crime'] == 'total_crime_calculated':
        if key_cols['violent_crime'] and key_cols['property_crime']:
            df['total_crime'] = (
                df[key_cols['violent_crime']].fillna(0) + 
                df[key_cols['property_crime']].fillna(0)
            )
            crime_col = 'total_crime'
        elif key_cols['violent_crime']:
            crime_col = key_cols['violent_crime']
        else:
            crime_col = None
    else:
        crime_col = key_cols['crime']
    
    if key_cols['population'] and crime_col and crime_col in df.columns:
        # Calculate crime rate per 100,000
        df['crime_rate_per_100k'] = (df[crime_col] / df[key_cols['population']]) * 100000
    
    if key_cols['population'] and key_cols['incarceration']:
        # Calculate incarceration rate per 100,000
        df['incarceration_rate_per_100k'] = (df[key_cols['incarceration']] / df[key_cols['population']]) * 100000
    
    # Calculate ratio of incarceration to crime
    if 'crime_rate_per_100k' in df.columns and 'incarceration_rate_per_100k' in df.columns:
        df['incarceration_to_crime_ratio'] = (
            df['incarceration_rate_per_100k'] / df['crime_rate_per_100k'].replace(0, np.nan)
        )
    
    return df

def clean_data(df, key_cols):
    """Main data cleaning function."""
    df = df.copy()
    
    # Filter data from 2000 onward
    if key_cols['year']:
        df[key_cols['year']] = pd.to_numeric(df[key_cols['year']], errors='coerce')
        df = df[df[key_cols['year']] >= 2000].copy()
    
    # Handle missing values in key columns
    # Drop rows where state or year is missing
    if key_cols['state']:
        df = df.dropna(subset=[key_cols['state']])
    if key_cols['year']:
        df = df.dropna(subset=[key_cols['year']])
    
    # For numeric columns, replace negative values with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].where(df[col] >= 0, np.nan)
    
    # Handle missing population - drop rows if population is missing (needed for per-capita)
    if key_cols['population']:
        df = df.dropna(subset=[key_cols['population']])
        # Remove rows with zero or negative population
        df = df[df[key_cols['population']] > 0]
    
    # Standardize state names (remove extra whitespace, capitalize properly)
    if key_cols['state']:
        df[key_cols['state']] = df[key_cols['state']].str.strip().str.title()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df

def process_dataset(data_dir=None, output_file='cleaned_data.csv'):
    """Main function to process the entire dataset."""
    print("Loading raw data...")
    df, data_dir = load_raw_data(data_dir)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Identify key columns
    key_cols = identify_key_columns(df)
    print(f"\nIdentified key columns:")
    for key, value in key_cols.items():
        print(f"  {key}: {value}")
    
    # Clean data
    print("\nCleaning data...")
    df_clean = clean_data(df, key_cols)
    
    # Calculate per-capita metrics
    print("Calculating per-capita metrics...")
    df_clean = calculate_per_capita_metrics(df_clean, key_cols)
    
    # Save cleaned data
    output_path = data_dir / output_file
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(df_clean.describe())
    
    return df_clean, key_cols

if __name__ == "__main__":
    process_dataset()


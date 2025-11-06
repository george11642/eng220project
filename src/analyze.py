"""
Statistical analysis module.
Computes descriptive statistics, correlations, and trend analysis
with special focus on New Mexico.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def load_cleaned_data(data_dir=None, filename='cleaned_data.csv'):
    """Load the cleaned dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def identify_metric_columns(df):
    """Identify metric columns in the dataset."""
    metrics = {}
    
    # Look for per-capita rate columns
    for col in df.columns:
        col_lower = col.lower()
        if 'crime_rate' in col_lower or 'crime_per' in col_lower:
            metrics['crime_rate'] = col
        elif 'incarceration_rate' in col_lower or 'incarceration_per' in col_lower:
            metrics['incarceration_rate'] = col
        elif 'ratio' in col_lower and ('incarceration' in col_lower or 'crime' in col_lower):
            metrics['ratio'] = col
    
    # Also look for raw counts if rates not found
    if 'crime_rate' not in metrics:
        for col in df.columns:
            if any(term in col.lower() for term in ['crime', 'offense', 'arrest']):
                metrics['crime_count'] = col
                break
    
    if 'incarceration_rate' not in metrics:
        for col in df.columns:
            if any(term in col.lower() for term in ['prisoner', 'incarceration', 'inmate']):
                metrics['incarceration_count'] = col
                break
    
    return metrics

def get_state_year_columns(df):
    """Identify state and year columns."""
    state_col = None
    year_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if not state_col and any(term in col_lower for term in ['state', 'jurisdiction']):
            state_col = col
        if not year_col and 'year' in col_lower:
            year_col = col
    
    return state_col, year_col

def descriptive_statistics(df, metrics, state_col, year_col):
    """Compute descriptive statistics by state and year."""
    stats_dict = {}
    
    # Overall statistics
    stats_dict['overall'] = {}
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            stats_dict['overall'][metric_name] = {
                'mean': df[metric_col].mean(),
                'median': df[metric_col].median(),
                'std': df[metric_col].std(),
                'min': df[metric_col].min(),
                'max': df[metric_col].max(),
                'count': df[metric_col].notna().sum()
            }
    
    # Statistics by state
    if state_col:
        stats_dict['by_state'] = {}
        for state in df[state_col].unique():
            state_df = df[df[state_col] == state]
            stats_dict['by_state'][state] = {}
            for metric_name, metric_col in metrics.items():
                if metric_col in df.columns:
                    stats_dict['by_state'][state][metric_name] = {
                        'mean': state_df[metric_col].mean(),
                        'median': state_df[metric_col].median(),
                        'std': state_df[metric_col].std(),
                        'min': state_df[metric_col].min(),
                        'max': state_df[metric_col].max(),
                        'count': state_df[metric_col].notna().sum()
                    }
    
    # Statistics by year
    if year_col:
        stats_dict['by_year'] = {}
        for year in sorted(df[year_col].unique()):
            year_df = df[df[year_col] == year]
            stats_dict['by_year'][year] = {}
            for metric_name, metric_col in metrics.items():
                if metric_col in df.columns:
                    stats_dict['by_year'][year][metric_name] = {
                        'mean': year_df[metric_col].mean(),
                        'median': year_df[metric_col].median(),
                        'std': year_df[metric_col].std(),
                        'min': year_df[metric_col].min(),
                        'max': year_df[metric_col].max(),
                        'count': year_df[metric_col].notna().sum()
                    }
    
    return stats_dict

def compute_correlations(df, metrics, state_col):
    """Compute correlations between crime and incarceration rates."""
    correlations = {}
    
    # Overall correlation
    if 'crime_rate' in metrics and 'incarceration_rate' in metrics:
        crime_col = metrics['crime_rate']
        incarc_col = metrics['incarceration_rate']
        
        if crime_col in df.columns and incarc_col in df.columns:
            # Overall correlation
            valid_data = df[[crime_col, incarc_col]].dropna()
            if len(valid_data) > 1:
                corr = valid_data[crime_col].corr(valid_data[incarc_col])
                correlations['overall'] = corr
                
                # Correlation by state
                if state_col:
                    correlations['by_state'] = {}
                    for state in df[state_col].unique():
                        state_df = df[df[state_col] == state]
                        state_valid = state_df[[crime_col, incarc_col]].dropna()
                        if len(state_valid) > 1:
                            state_corr = state_valid[crime_col].corr(state_valid[incarc_col])
                            correlations['by_state'][state] = state_corr
    
    return correlations

def analyze_trends(df, metrics, state_col, year_col, focus_state='New Mexico'):
    """Analyze trends over time, with focus on New Mexico."""
    trends = {}
    
    if not year_col or not state_col:
        return trends
    
    # Overall trend
    if year_col in df.columns:
        for metric_name, metric_col in metrics.items():
            if metric_col in df.columns:
                yearly_avg = df.groupby(year_col)[metric_col].mean()
                trends[f'{metric_name}_overall'] = {
                    'slope': np.polyfit(yearly_avg.index, yearly_avg.values, 1)[0] if len(yearly_avg) > 1 else 0,
                    'data': yearly_avg.to_dict()
                }
    
    # New Mexico specific trends
    if focus_state and state_col:
        nm_df = df[df[state_col].str.contains(focus_state, case=False, na=False)]
        if len(nm_df) > 0:
            trends['new_mexico'] = {}
            for metric_name, metric_col in metrics.items():
                if metric_col in df.columns:
                    nm_yearly = nm_df.groupby(year_col)[metric_col].mean()
                    if len(nm_yearly) > 1:
                        trends['new_mexico'][metric_name] = {
                            'slope': np.polyfit(nm_yearly.index, nm_yearly.values, 1)[0],
                            'data': nm_yearly.to_dict()
                        }
    
    return trends

def compare_new_mexico(df, metrics, state_col, year_col, focus_state='New Mexico'):
    """Compare New Mexico to other states."""
    comparisons = {}
    
    if not state_col:
        return comparisons
    
    # Find New Mexico data
    nm_mask = df[state_col].str.contains(focus_state, case=False, na=False)
    nm_df = df[nm_mask]
    other_df = df[~nm_mask]
    
    if len(nm_df) == 0:
        print(f"Warning: No data found for {focus_state}")
        return comparisons
    
    comparisons['new_mexico'] = {}
    comparisons['other_states'] = {}
    comparisons['comparison'] = {}
    
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            # New Mexico statistics
            nm_values = nm_df[metric_col].dropna()
            if len(nm_values) > 0:
                comparisons['new_mexico'][metric_name] = {
                    'mean': nm_values.mean(),
                    'median': nm_values.median(),
                    'std': nm_values.std()
                }
            
            # Other states statistics
            other_values = other_df[metric_col].dropna()
            if len(other_values) > 0:
                comparisons['other_states'][metric_name] = {
                    'mean': other_values.mean(),
                    'median': other_values.median(),
                    'std': other_values.std()
                }
            
            # Statistical comparison (t-test if possible)
            if len(nm_values) > 0 and len(other_values) > 0:
                try:
                    t_stat, p_value = stats.ttest_ind(nm_values, other_values)
                    comparisons['comparison'][metric_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'nm_mean': nm_values.mean(),
                        'other_mean': other_values.mean(),
                        'difference': nm_values.mean() - other_values.mean()
                    }
                except:
                    pass
    
    return comparisons

def run_analysis(data_dir=None, focus_state='New Mexico'):
    """Run complete analysis pipeline."""
    print("Loading cleaned data...")
    df = load_cleaned_data(data_dir)
    
    print(f"Dataset shape: {df.shape}")
    
    # Identify columns
    metrics = identify_metric_columns(df)
    state_col, year_col = get_state_year_columns(df)
    
    print(f"\nState column: {state_col}")
    print(f"Year column: {year_col}")
    print(f"Metrics: {metrics}")
    
    # Run analyses
    print("\nComputing descriptive statistics...")
    desc_stats = descriptive_statistics(df, metrics, state_col, year_col)
    
    print("Computing correlations...")
    correlations = compute_correlations(df, metrics, state_col)
    
    print("Analyzing trends...")
    trends = analyze_trends(df, metrics, state_col, year_col, focus_state)
    
    print(f"Comparing {focus_state} to other states...")
    comparisons = compare_new_mexico(df, metrics, state_col, year_col, focus_state)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    if 'overall' in correlations:
        print(f"\nOverall correlation (crime vs incarceration): {correlations['overall']:.3f}")
    
    if 'new_mexico' in trends:
        print(f"\n{focus_state} Trends:")
        for metric, trend_data in trends['new_mexico'].items():
            print(f"  {metric}: slope = {trend_data['slope']:.3f}")
    
    if 'comparison' in comparisons:
        print(f"\n{focus_state} vs Other States:")
        for metric, comp_data in comparisons['comparison'].items():
            print(f"  {metric}:")
            print(f"    {focus_state} mean: {comp_data['nm_mean']:.2f}")
            print(f"    Other states mean: {comp_data['other_mean']:.2f}")
            print(f"    Difference: {comp_data['difference']:.2f}")
            print(f"    p-value: {comp_data['p_value']:.4f}")
    
    return {
        'data': df,
        'metrics': metrics,
        'state_col': state_col,
        'year_col': year_col,
        'descriptive_stats': desc_stats,
        'correlations': correlations,
        'trends': trends,
        'comparisons': comparisons
    }

if __name__ == "__main__":
    results = run_analysis()


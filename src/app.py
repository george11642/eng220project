"""
Streamlit application for Crime and Incarceration Analysis.
Interactive dashboard with filters and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from clean_data import load_raw_data, identify_key_columns, standardize_column_names
from analyze import identify_metric_columns, get_state_year_columns
import visualizations as viz

# Page configuration
st.set_page_config(
    page_title="Crime & Incarceration Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data."""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Try to load cleaned data first
    cleaned_file = data_dir / "cleaned_data.csv"
    if cleaned_file.exists():
        df = pd.read_csv(cleaned_file)
        # Identify columns
        metrics = identify_metric_columns(df)
        state_col, year_col = get_state_year_columns(df)
        return df, metrics, state_col, year_col, True
    
    # If no cleaned data, try to load and process raw data
    try:
        df, _ = load_raw_data(data_dir)
        df = standardize_column_names(df)
        key_cols = identify_key_columns(df)
        metrics = identify_metric_columns(df)
        state_col = key_cols.get('state')
        year_col = key_cols.get('year')
        return df, metrics, state_col, year_col, False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run the data download and cleaning scripts first.")
        return None, {}, None, None, False

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">Crime and Incarceration Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("### Analyzing crime rates and incarceration levels across U.S. states (2000+)")
    
    # Load data
    data_result = load_data()
    if data_result[0] is None:
        st.stop()
    
    df, metrics, state_col, year_col, is_cleaned = data_result
    
    if df is None or len(df) == 0:
        st.warning("No data available. Please run the data download and cleaning scripts.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Year filter
    if year_col and year_col in df.columns:
        years = sorted(df[year_col].dropna().unique())
        if len(years) > 0:
            min_year, max_year = int(years[0]), int(years[-1])
            year_range = st.sidebar.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1
            )
            df = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]
    else:
        year_range = None
    
    # State filter
    if state_col and state_col in df.columns:
        all_states = sorted(df[state_col].dropna().unique())
        default_states = ['New Mexico'] if 'New Mexico' in all_states else [all_states[0]] if len(all_states) > 0 else []
        
        selected_states = st.sidebar.multiselect(
            "Select States",
            options=all_states,
            default=default_states if len(default_states) > 0 else all_states[:10],
            help="Select one or more states to analyze"
        )
        
        if len(selected_states) > 0:
            df = df[df[state_col].isin(selected_states)]
    else:
        selected_states = None
    
    # Metric selector
    available_metrics = {}
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            available_metrics[metric_name.replace("_", " ").title()] = metric_col
    
    if len(available_metrics) > 0:
        selected_metric_name = st.sidebar.selectbox(
            "Select Metric",
            options=list(available_metrics.keys()),
            index=0
        )
        selected_metric_col = available_metrics[selected_metric_name]
    else:
        st.sidebar.warning("No metrics available. Data may need cleaning.")
        selected_metric_col = None
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Trends Over Time", 
        "State Comparisons", 
        "Relationships",
        "New Mexico Focus"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Overview & Summary Statistics")
        
        if selected_metric_col:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{df[selected_metric_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_metric_col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_metric_col].std():.2f}")
            with col4:
                st.metric("Count", f"{df[selected_metric_col].notna().sum()}")
        
        # Data table
        st.subheader("Data Table")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Tab 2: Trends Over Time
    with tab2:
        st.header("Trends Over Time")
        
        if selected_metric_col and state_col and year_col:
            fig = viz.plot_trends_over_time(
                df, selected_metric_col, state_col, year_col,
                selected_states=selected_states if selected_states else None,
                highlight_state='New Mexico'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to create trend chart. Check data availability.")
        else:
            st.warning("Required columns (metric, state, year) not available.")
    
    # Tab 3: State Comparisons
    with tab3:
        st.header("State Rankings & Comparisons")
        
        if selected_metric_col and state_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top States Ranking")
                if year_col:
                    selected_year = st.selectbox(
                        "Select Year for Ranking",
                        options=sorted(df[year_col].dropna().unique()),
                        index=len(sorted(df[year_col].dropna().unique())) - 1 if len(sorted(df[year_col].dropna().unique())) > 0 else 0
                    )
                else:
                    selected_year = None
                
                fig_rankings = viz.plot_state_rankings(
                    df, selected_metric_col, state_col, year=selected_year, top_n=20
                )
                if fig_rankings:
                    st.plotly_chart(fig_rankings, use_container_width=True)
            
            with col2:
                st.subheader("Distribution")
                fig_dist = viz.plot_state_distribution(
                    df, selected_metric_col, state_col, year=selected_year if year_col else None
                )
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Required columns (metric, state) not available.")
    
    # Tab 4: Relationships
    with tab4:
        st.header("Relationship Analysis")
        
        if len(available_metrics) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "X-Axis Metric",
                    options=list(available_metrics.keys()),
                    index=0
                )
                x_col = available_metrics[x_metric]
            
            with col2:
                y_metric = st.selectbox(
                    "Y-Axis Metric",
                    options=list(available_metrics.keys()),
                    index=min(1, len(available_metrics) - 1)
                )
                y_col = available_metrics[y_metric]
            
            # Scatter plot
            fig_scatter = viz.plot_scatter_relationship(
                df, x_col, y_col, state_col, highlight_state='New Mexico', year_col=year_col
            )
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            metric_cols_list = list(available_metrics.values())
            fig_heatmap = viz.plot_correlation_heatmap(df, metric_cols_list, state_col, year_col)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Need at least 2 metrics for relationship analysis.")
    
    # Tab 5: New Mexico Focus
    with tab5:
        st.header("New Mexico Analysis")
        
        if state_col and 'New Mexico' in str(df[state_col].values):
            nm_df = df[df[state_col].str.contains('New Mexico', case=False, na=False)]
            
            if len(nm_df) > 0:
                st.subheader("New Mexico vs Other States")
                
                if selected_metric_col and year_col:
                    fig_comparison = viz.plot_new_mexico_comparison(
                        df, selected_metric_col, state_col, year_col, focus_state='New Mexico'
                    )
                    if fig_comparison:
                        st.plotly_chart(fig_comparison, use_container_width=True)
                
                # New Mexico statistics
                st.subheader("New Mexico Statistics")
                if selected_metric_col:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("New Mexico Mean", f"{nm_df[selected_metric_col].mean():.2f}")
                    with col2:
                        other_df = df[~df[state_col].str.contains('New Mexico', case=False, na=False)]
                        st.metric("Other States Mean", f"{other_df[selected_metric_col].mean():.2f}")
                    with col3:
                        diff = nm_df[selected_metric_col].mean() - other_df[selected_metric_col].mean()
                        st.metric("Difference", f"{diff:.2f}")
                
                # New Mexico data table
                st.subheader("New Mexico Data")
                st.dataframe(nm_df, use_container_width=True)
            else:
                st.warning("No New Mexico data found in the filtered dataset.")
        else:
            st.warning("New Mexico data not available. Check state column and data filters.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app analyzes crime and incarceration data across U.S. states "
        "from 2000 onward, with special focus on New Mexico."
    )

if __name__ == "__main__":
    main()


"""
Visualization functions using plotly for Streamlit compatibility.
Creates line graphs, bar charts, scatter plots, and heatmaps.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_trends_over_time(df, metric_col, state_col, year_col, 
                          selected_states=None, highlight_state='New Mexico'):
    """
    Create line graph showing trends over time by state.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    metric_col : str
        Column name for the metric to plot
    state_col : str
        Column name for states
    year_col : str
        Column name for years
    selected_states : list, optional
        List of states to include. If None, includes all states.
    highlight_state : str
        State to highlight (default: 'New Mexico')
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if metric_col not in df.columns or state_col not in df.columns or year_col not in df.columns:
        return None
    
    # Filter data
    plot_df = df[[state_col, year_col, metric_col]].dropna()
    
    if selected_states:
        plot_df = plot_df[plot_df[state_col].isin(selected_states)]
    
    if len(plot_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Get unique states
    states = plot_df[state_col].unique()
    
    # Plot each state
    for state in states:
        state_data = plot_df[plot_df[state_col] == state].sort_values(year_col)
        
        # Highlight New Mexico or specified state
        if highlight_state and state.lower() == highlight_state.lower():
            fig.add_trace(go.Scatter(
                x=state_data[year_col],
                y=state_data[metric_col],
                mode='lines+markers',
                name=state,
                line=dict(width=4, color='red'),
                marker=dict(size=8)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=state_data[year_col],
                y=state_data[metric_col],
                mode='lines+markers',
                name=state,
                line=dict(width=2, color='lightgray'),
                marker=dict(size=4),
                opacity=0.6
            ))
    
    fig.update_layout(
        title=f'Trends Over Time: {metric_col.replace("_", " ").title()}',
        xaxis_title='Year',
        yaxis_title=metric_col.replace("_", " ").title(),
        hovermode='closest',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_state_rankings(df, metric_col, state_col, year=None, top_n=20):
    """
    Create bar chart showing state rankings.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    metric_col : str
        Column name for the metric to rank
    state_col : str
        Column name for states
    year : int, optional
        Specific year to rank. If None, uses average across all years.
    top_n : int
        Number of top states to show
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if metric_col not in df.columns or state_col not in df.columns:
        return None
    
    # Filter data
    plot_df = df[[state_col, metric_col]].dropna()
    
    if year is not None:
        year_col = None
        for col in df.columns:
            if 'year' in col.lower():
                year_col = col
                break
        if year_col:
            plot_df = df[df[year_col] == year][[state_col, metric_col]].dropna()
    
    # Calculate average by state
    state_avg = plot_df.groupby(state_col)[metric_col].mean().sort_values(ascending=False)
    
    # Take top N
    top_states = state_avg.head(top_n)
    
    # Color bars - highlight New Mexico
    colors = ['red' if 'New Mexico' in state or 'new mexico' in state.lower() 
              else 'steelblue' for state in top_states.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_states.values,
            y=top_states.index,
            orientation='h',
            marker_color=colors,
            text=[f'{val:.2f}' for val in top_states.values],
            textposition='auto'
        )
    ])
    
    year_text = f' ({year})' if year else ' (Average)'
    fig.update_layout(
        title=f'Top {top_n} States by {metric_col.replace("_", " ").title()}{year_text}',
        xaxis_title=metric_col.replace("_", " ").title(),
        yaxis_title='State',
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_scatter_relationship(df, x_col, y_col, state_col=None, 
                             highlight_state='New Mexico', year_col=None):
    """
    Create scatter plot showing relationship between two metrics.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    state_col : str, optional
        Column name for states (for coloring/highlighting)
    highlight_state : str
        State to highlight
    year_col : str, optional
        Column name for years (for animation)
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    plot_df = df[[x_col, y_col]].dropna()
    
    if state_col and state_col in df.columns:
        plot_df = df[[x_col, y_col, state_col]].dropna()
        
        # Create figure with color coding
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=state_col,
            hover_data=[state_col],
            title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}',
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title()
            }
        )
        
        # Highlight New Mexico
        if highlight_state:
            nm_data = plot_df[plot_df[state_col].str.contains(highlight_state, case=False, na=False)]
            if len(nm_data) > 0:
                fig.add_trace(go.Scatter(
                    x=nm_data[x_col],
                    y=nm_data[y_col],
                    mode='markers',
                    name=highlight_state,
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=True
                ))
    else:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}',
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title()
            }
        )
    
    # Add trend line
    if len(plot_df) > 1:
        z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash'),
            showlegend=True
        ))
    
    fig.update_layout(height=500)
    
    return fig

def plot_correlation_heatmap(df, metric_cols, state_col=None, year_col=None):
    """
    Create heatmap showing correlations between metrics.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    metric_cols : list
        List of column names for metrics to correlate
    state_col : str, optional
        If provided, compute correlations by state
    year_col : str, optional
        If provided, compute correlations by year
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Filter to only metric columns that exist
    available_cols = [col for col in metric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    # Compute correlation matrix
    corr_df = df[available_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=500,
        xaxis_title='',
        yaxis_title=''
    )
    
    return fig

def plot_new_mexico_comparison(df, metric_col, state_col, year_col, 
                               focus_state='New Mexico'):
    """
    Create comparison chart showing New Mexico vs other states over time.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    metric_col : str
        Column name for the metric
    state_col : str
        Column name for states
    year_col : str
        Column name for years
    focus_state : str
        State to focus on (default: 'New Mexico')
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if metric_col not in df.columns or state_col not in df.columns or year_col not in df.columns:
        return None
    
    plot_df = df[[state_col, year_col, metric_col]].dropna()
    
    # Separate New Mexico and other states
    nm_mask = plot_df[state_col].str.contains(focus_state, case=False, na=False)
    nm_df = plot_df[nm_mask]
    other_df = plot_df[~nm_mask]
    
    fig = go.Figure()
    
    # Plot New Mexico
    if len(nm_df) > 0:
        nm_yearly = nm_df.groupby(year_col)[metric_col].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=nm_yearly.index,
            y=nm_yearly.values,
            mode='lines+markers',
            name=focus_state,
            line=dict(width=4, color='red'),
            marker=dict(size=10)
        ))
    
    # Plot average of other states
    if len(other_df) > 0:
        other_yearly = other_df.groupby(year_col)[metric_col].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=other_yearly.index,
            y=other_yearly.values,
            mode='lines+markers',
            name='Other States (Average)',
            line=dict(width=3, color='blue', dash='dash'),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f'{focus_state} vs Other States: {metric_col.replace("_", " ").title()}',
        xaxis_title='Year',
        yaxis_title=metric_col.replace("_", " ").title(),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_state_distribution(df, metric_col, state_col, year=None):
    """
    Create histogram/distribution plot for a metric across states.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset
    metric_col : str
        Column name for the metric
    state_col : str
        Column name for states
    year : int, optional
        Specific year to analyze
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if metric_col not in df.columns:
        return None
    
    plot_df = df[[metric_col]].dropna()
    
    if year is not None:
        year_col = None
        for col in df.columns:
            if 'year' in col.lower():
                year_col = col
                break
        if year_col:
            plot_df = df[df[year_col] == year][[metric_col]].dropna()
    
    fig = go.Figure(data=[
        go.Histogram(
            x=plot_df[metric_col],
            nbinsx=30,
            marker_color='steelblue',
            opacity=0.7
        )
    ])
    
    # Add vertical line for New Mexico if state_col available
    if state_col and state_col in df.columns:
        nm_data = df[df[state_col].str.contains('New Mexico', case=False, na=False)]
        if len(nm_data) > 0 and metric_col in nm_data.columns:
            nm_value = nm_data[metric_col].mean()
            fig.add_vline(
                x=nm_value,
                line_dash="dash",
                line_color="red",
                annotation_text="New Mexico Average",
                annotation_position="top"
            )
    
    year_text = f' ({year})' if year else ''
    fig.update_layout(
        title=f'Distribution of {metric_col.replace("_", " ").title()}{year_text}',
        xaxis_title=metric_col.replace("_", " ").title(),
        yaxis_title='Frequency',
        height=400
    )
    
    return fig


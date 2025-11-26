import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import plotly.express as px


def create_log_heatmap(df, x_col, y_col, value_col, title, use_std=False, colorscale='Viridis', colorbar_x=0, colorbar_y=1.1, log=True):
    # Group by x and y columns, then calculate the average or standard deviation
    if use_std:
        agg_data = df.groupby([x_col, y_col])[value_col].std().reset_index()
        metric_label = 'Std Dev'
    else:
        agg_data = df.groupby([x_col, y_col])[value_col].mean().reset_index()
        metric_label = 'Avg'
    
    # Create a pivot table for aggregated values
    pivot_data = agg_data.pivot(index=y_col, columns=x_col, values=value_col)


    # Apply log10 to the values for color scaling, handling zeros and negative values
    log_values = np.log10(pivot_data.values, where=(pivot_data.values > 0)) if log else pivot_data.values
    log_values[np.isinf(log_values)] = np.nan    # Replace inf (log of 0) with nan

    # Create the heatmap with log values for color scaling, but show aggregated values
    heatmap = go.Heatmap(
        z=log_values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        colorbar=dict(
            title=f'{"Log10" if log else ""} {metric_label} Time (s)',
            tickprefix="10^" if log else "",
            len=0.4,
            yanchor="bottom",
            y= colorbar_y,
            x= colorbar_x,
            xanchor="right",
            orientation="h"
        ),
        hovertemplate=(f'{y_col}: %{{y}}<br>{x_col}: %{{x}}<br>' +
                       f'{metric_label} Time: %{{text}} s<br>' +
                       f'{"Log10" if log else ""} {metric_label} Time: %{{z:.2f}}<extra></extra>'),
        text=[[f'{val:.2e}' for val in row] for row in pivot_data.values],
        texttemplate="%{text}",
        zmin=np.nanmin(log_values), 
        zmax=np.nanmax(log_values), 
        zauto= False,
        colorbar_tickformat=".1f" if log else ".1e",
    )

    return heatmap

def create_log_heatmaps(df, x_col, y_col, value_col, title, height=600, width=1000, colorbar_x_left=0, colorbar_x_right=0, colorbar_y_left=0, colorbar_y_right=0, log=True):

    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=(f"Average {title} (Log10 Scale)", 
                                    f"Standard Deviation of {title} (Log10 Scale)") if log else (f"Average {title}", f"Standard Deviation of {title}"),
                    shared_yaxes=True,
                    horizontal_spacing=0.1)

    # Create and add heatmaps
    avg_heatmap = create_log_heatmap(df, x_col, y_col, value_col, 'Average', use_std=False, colorscale='Viridis', colorbar_x=colorbar_x_left, colorbar_y=colorbar_y_left, log=log)
    std_heatmap = create_log_heatmap(df, x_col, y_col, value_col, 'Std Dev', use_std=True, colorscale='Plasma', colorbar_x=colorbar_x_right, colorbar_y=colorbar_y_right, log=log)

    fig.add_trace(avg_heatmap, row=1, col=1)
    fig.add_trace(std_heatmap, row=1, col=2)

    # Update layout
    fig.update_layout(
        height=height,  # Increased height to accommodate colorbars below
        width=width,
        title_text=f"{title}: Average and Standard Deviation",
        xaxis=dict(title=x_col, dtick=1),
        xaxis2=dict(title=x_col, dtick=1),
        yaxis=dict(title=y_col, dtick=1)
    )

    # Update traces to ensure text is visible
    fig.update_traces(textfont_color='black', textfont_size=9)

    # Show the figure
    fig.show()

def missing_values(df, nested=None, independent=None, process_number=None, dimensions=None, 
                   modes=["random", "bagging_divide", "bagging_remove","bagging_remove_divide","bagging_remove_reverse", "bagging_remove_reverse_divide"]):
    for i in tqdm(range(1, nested + 1 if nested else max(df['nested'])+1)):
        for j in range(1, independent + 1 if independent else max(df['independent'])+1):
            for l in range(1, process_number + 1 if process_number else max(df['process_number'])+1):
                for d in range(1, dimensions + 1 if dimensions else max(df['dimensions'])+1):  
                    for m in  modes:
                        if not ((df['nested'] == i) & (df['independent'] == j) & (df['process_number'] == l) & (df['dimensions'] == d) & (df['mode'] == m)).any():
                            print(f"Missing values for nested={i}, independent={j}, process_number={l}, dimensions={d}, mode={m}")



def plot_multi_dimension_mode_surfaces(df, dimensions=None, modes=None, nested_col='nested', independent_col='independent', 
                                       mode_col='mode', dimensions_col='dimensions', pareto_col='pareto_time', 
                                       use_log_scale=True):
    """
    Create multiple 3D surface plots for specified modes across specified dimensions.
    If dimensions or modes are None, use all unique values from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    dimensions (list or None): List of dimensions to plot. If None, use all unique dimensions.
    modes (list or None): List of modes to include in the plots. If None, use all unique modes.
    nested_col (str): Name of the column for nested variable.
    independent_col (str): Name of the column for independent variable.
    mode_col (str): Name of the column for mode.
    dimensions_col (str): Name of the column for dimensions.
    pareto_col (str): Name of the column for pareto time.
    use_log_scale (bool): Whether to use logarithmic scale for z-axis. Default is True.
    
    Returns:
    plotly.graph_objs._figure.Figure: The resulting Plotly figure.
    """
    
    # Check if all required columns are present in the DataFrame
    required_columns = [nested_col, independent_col, mode_col, dimensions_col, pareto_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

    # Use all unique dimensions and modes if None is provided
    if dimensions is None:
        dimensions = sorted(df[dimensions_col].unique())
    if modes is None:
        modes = sorted(df[mode_col].unique())

    # Set up the subplot grid
    n_dims = len(dimensions)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'surface'}] * n_cols] * n_rows,
        subplot_titles=[f'Dimension {dim}' for dim in dimensions],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    # Color map for modes
    color_scale = px.colors.qualitative.Plotly
    mode_colors = {mode: color_scale[i % len(color_scale)] for i, mode in enumerate(modes)}

    # Create a surface plot for each dimension and mode
    for i, dim in enumerate(dimensions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        dim_data = df[df[dimensions_col] == dim]
        
        for mode in modes:
            mode_data = dim_data[dim_data[mode_col] == mode]
            if mode_data.empty:
                continue  # Skip if no data for this mode-dimension combination
            
            pivot = mode_data.pivot_table(values=pareto_col, index=nested_col, columns=independent_col, aggfunc='mean')
            
            if not pivot.empty:
                z_values = np.log10(pivot.values) if use_log_scale else pivot.values
                z_values[~np.isfinite(z_values)] = np.nan

                fig.add_trace(
                    go.Surface(
                        z=z_values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale=[[0, mode_colors[mode]], [1, mode_colors[mode]]],
                        name=f'Mode {mode}',
                        showscale=False,
                        opacity=0.7,
                        showlegend=(i == 0),  # Only show legend for the first subplot
                        legendgroup=f'Mode {mode}',  # Group legend items
                        hoverinfo='name+x+y+z'
                    ),
                    row=row, col=col
                )

    # Update layout
    z_axis_title = f'{"Log10 of " if use_log_scale else ""}Average {pareto_col}'
    fig.update_layout(
        height=400 * n_rows,
        width=1200 + 150,  # Add extra width for the side legend
        title=dict(
            text=f'{z_axis_title} vs {nested_col} and {independent_col} for Each Dimension',
            y=0.98,  # Move title up
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title=independent_col,
            yaxis_title=nested_col,
            zaxis_title=z_axis_title
        ),
        legend=dict(
            title='Modes',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
        ),
        margin=dict(r=150, t=100, b=40, l=40)  # Adjust right margin for legend
    )

    # Update all subplots to have the same axis labels
    for i in range(1, n_dims + 1):
        fig.update_scenes(
            xaxis_title=independent_col,
            yaxis_title=nested_col,
            zaxis_title=z_axis_title,
            row=(i-1)//n_cols+1, col=(i-1)%n_cols+1
        )

    return fig

def plot_multi_dimension_stat_plots(df, dimensions=None, modes=None, nested_col='nested', independent_col='independent', 
                                    mode_col='mode', dimensions_col='dimensions', pareto_col='pareto_time', 
                                    use_log_scale=True, stat_measure='std'):
    """
    Create multiple 2D line plots for mean or standard deviation of pareto time across specified dimensions and modes.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    dimensions (list or None): List of dimensions to plot. If None, use all unique dimensions.
    modes (list or None): List of modes to include in the plots. If None, use all unique modes.
    nested_col (str): Name of the column for nested variable.
    independent_col (str): Name of the column for independent variable.
    mode_col (str): Name of the column for mode.
    dimensions_col (str): Name of the column for dimensions.
    pareto_col (str): Name of the column for pareto time.
    use_log_scale (bool): Whether to use logarithmic scale for y-axis. Default is True.
    stat_measure (str): Statistical measure to plot. Either 'mean' or 'std'. Default is 'std'.
    
    Returns:
    plotly.graph_objs._figure.Figure: The resulting Plotly figure.
    """
    
    if stat_measure not in ['mean', 'std']:
        raise ValueError("stat_measure must be either 'mean' or 'std'")

    # Calculate the product of nested and independent
    df['nested_independent_product'] = df[nested_col] * df[independent_col]

    # Get unique dimensions and modes
    if dimensions is None:
        dimensions = sorted(df[dimensions_col].unique())
    if modes is None:
        modes = sorted(df[mode_col].unique())

    # Set up the subplot grid
    n_dims = len(dimensions)
    n_cols = 4  # 4 graphs per row
    n_rows = (n_dims + n_cols - 1) // n_cols
    last_row_cols = n_dims % n_cols if n_dims % n_cols != 0 else n_cols

    # Calculate the starting column for centering the last row
    start_col = (n_cols - last_row_cols) // 2 + 1 if last_row_cols < n_cols else 1

    # Create subplot titles with correct alignment
    subplot_titles = []
    for i in range(n_rows):
        row_titles = [f'Dimension {dim}' for dim in dimensions[i*n_cols:(i+1)*n_cols]]
        if i == n_rows - 1 and last_row_cols < n_cols:
            # Center the titles in the last row
            row_titles = [''] * (start_col - 1) + row_titles + [''] * (n_cols - last_row_cols - (start_col - 1))
        subplot_titles.extend(row_titles)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title='Product of Nested and Independent',
        y_title=f'{"Standard Deviation" if stat_measure == "std" else "Mean"} of {pareto_col} {"(Log Scale)" if use_log_scale else ""}'
    )

    # Color map for modes
    color_map = px.colors.qualitative.Plotly

    # Create a plot for each dimension
    for i, dim in enumerate(dimensions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Adjust column for the last row to center the plots
        if row == n_rows and last_row_cols < n_cols:
            col = (i % n_cols) + start_col

        dim_data = df[df[dimensions_col] == dim]
        
        for j, mode in enumerate(modes):
            mode_data = dim_data[dim_data[mode_col] == mode]
            if stat_measure == 'std':
                grouped = mode_data.groupby('nested_independent_product')[pareto_col].std().reset_index()
            else:  # mean
                grouped = mode_data.groupby('nested_independent_product')[pareto_col].mean().reset_index()
            
            # Filter out any zero or NaN values
            grouped = grouped[grouped[pareto_col] > 0]
            
            if not grouped.empty:
                fig.add_trace(
                    go.Scatter(
                        x=grouped['nested_independent_product'],
                        y=grouped[pareto_col],
                        mode='lines+markers',
                        name=f'Mode {mode}',
                        legendgroup=f'Mode {mode}',
                        showlegend=(i == 0),  # Show legend only for the first subplot
                        marker=dict(color=color_map[j % len(color_map)]),
                        line=dict(color=color_map[j % len(color_map)]),
                        text=[f'{"Std Dev" if stat_measure == "std" else "Mean"}: {y:.2f}' for y in grouped[pareto_col]],
                        hoverinfo='text+x+name'
                    ),
                    row=row, col=col
                )

        # Update y-axis to log scale for this subplot if specified
        if use_log_scale:
            fig.update_yaxes(type="log", row=row, col=col)
        
        # Ensure x and y axes are visible for all plots
        fig.update_xaxes(showticklabels=True, row=row, col=col)
        fig.update_yaxes(showticklabels=True, row=row, col=col)

    # Update layout
    fig.update_layout(
        height=300 * n_rows + 50,  # Added extra height for the two-row legend
        width=1200,
        title_text=f'{"Standard Deviation" if stat_measure == "std" else "Mean"} of {pareto_col} vs Product of {nested_col} and {independent_col} for Each Dimension',
        legend_title='Modes',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Moved further down to accommodate two rows
            xanchor="center",
            x=0.5,
            traceorder="grouped",
            tracegroupgap=10  # Add gap between legend groups for better separation into two rows
        )
    )

    return fig
import numpy as np
from generate_impacts import generate_vectors
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

def custom_cosine_distance(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    else:
        return 1.0 - (dot_product / (norm_u * norm_v))

def compute_cosine_distances(vectors):
    n = len(vectors)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distance = custom_cosine_distance(vectors[i], vectors[j])
            distances.append(distance)
    return np.array(distances)

def generate_and_analyze_vectors(num_vectors=100, max_dim=10):
    modes = ["random", "bagging_divide", "bagging_remove", "bagging_remove_divide", "bagging_remove_reverse", "bagging_remove_reverse_divide"]
    results = []

    for mode in tqdm(modes, desc="Analyzing modes"):
        for dim in range(1, max_dim + 1):
            vectors = generate_vectors(num_vectors, dim, mode=mode)
            distances = compute_cosine_distances(vectors)
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            results.append({
                "mode": mode,
                "dimension": dim,
                "avg_distance": avg_distance,
                "std_distance": std_distance
            })

    return results, modes

def plot_mean_std_intervals_by_dimension(results, modes):
    color_scale = px.colors.qualitative.Plotly
    num_modes = len(modes)

    fig = go.Figure()

    for i, mode in enumerate(modes):
        mode_results = [r for r in results if r["mode"] == mode]
        x = [r["dimension"] + (i - num_modes/2 + 0.5) * 0.1 for r in mode_results]
        y = [r["avg_distance"] for r in mode_results]
        error_y_plus = [min(1 - avg, std) for avg, std in zip(y, [r["std_distance"] for r in mode_results])]
        error_y_minus = [min(avg, std) for avg, std in zip(y, [r["std_distance"] for r in mode_results])]
        
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y,
                error_y=dict(
                    type='data',
                    array=error_y_plus,
                    arrayminus=error_y_minus,
                    visible=True
                ),
                mode='markers',
                marker=dict(
                    color=color_scale[i % len(color_scale)],
                    size=8,
                    symbol=i
                ),
                name=mode,
                hovertemplate=(
                    "Dimension: %{text}<br>"
                    "Avg Distance: %{y:.4f}<br>"
                    "Std Dev: %{customdata[0]:.4f}<br>"
                    "Lower Bound: %{customdata[1]:.4f}<br>"
                    "Upper Bound: %{customdata[2]:.4f}<br>"
                    "Mode: " + mode
                ),
                text=[int(round(x_val)) for x_val in x],
                customdata=list(zip(
                    [r["std_distance"] for r in mode_results],
                    [max(0, avg - std) for avg, std in zip(y, [r["std_distance"] for r in mode_results])],
                    [min(1, avg + std) for avg, std in zip(y, [r["std_distance"] for r in mode_results])]
                ))
            )
        )

    fig.update_layout(
        title="Vector Analysis: Average Cosine Distance vs Dimension (with Clamped Standard Deviation)",
        xaxis_title="Dimension",
        yaxis_title="Average Cosine Distance",
        legend_title="Generation Mode",
        hovermode="closest",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()

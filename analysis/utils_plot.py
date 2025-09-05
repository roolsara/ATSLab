import plotly.express as px
import numpy as np
import pandas as pd
import polars as pl
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def facet_distribution_plot(
    df: pd.DataFrame,
    category: str, 
    value_vars: list[str], 
    dico_color: dict, 
    histnorm: str = 'probability', 
    title_x: str = "", 
    order: dict = None, 
    nbins: int = 50, 
    height: int = 400
):
    """
    Plot interactive histograms with faceted subplots and marginal boxplots.

    This function takes a Pandas dataframe and produces a Plotly figure consisting of:
    - Histograms faceted by multiple value variables (columns).
    - Colored by a categorical column.
    - Marginal boxplots (one per facet) for distribution overview.

    Parameters:
    -----------
        df (pd.DataFrame): Input DataFrame containing the data to plot.
        category (str): Column name in df to use for coloring (categorical grouping).
        value_vars (list[str]): List of numeric columns to facet into separate histograms.
        dico_color (dict): Mapping from category values to specific colors.
        histnorm (str): Normalization of histogram bars. Default to "probability".
                        Options: {"probability", "percent", "density", None}.
        title_x (str): X-axis title for all histograms. Default to "".
        order (dict): Dictionary specifying category order for consistent coloring/faceting. Default to None.
                      Example: {category: ["A", "B", "C"]}. 
        nbins (int): Number of bins for histograms. Default to 50.
        height (int): Height of the figure. Default to 400.

    Returns:
    --------
        plotly.graph_objects.Figure: Interactive Plotly figure with histograms and boxplots.

    Example:
    --------
        >>> fig = facet_distribution_plot(
        ...     df=my_dataframe,
        ...     category="group",
        ...     value_vars=["feature1", "feature2"],
        ...     dico_color={"A": "blue", "B": "red"}
        ... )
        >>> fig.show()
    """

    ## melt dataframe to long format: each value_var becomes "variable", values go in "value"
    df_melted = df.melt(id_vars=category, value_vars=value_vars)

    ## build interactive histogram with facets (per variable) and marginal boxplots
    fig = (
        px.histogram(
            df_melted,
            x="value",
            color=category,
            facet_col="variable",
            barmode="overlay",
            histnorm=histnorm, ## normalization type
            opacity=0.5,
            marginal="box", ## adds a boxplot above each histogram
            nbins=nbins, ## number of bin           
            height=height, ## maybe should include width too
            color_discrete_map=dico_color, ## the personnalized color
            category_orders=order, ## the personnalized order
            facet_col_spacing=0.03
        )
        .update_xaxes(
            matches=None, 
            autotickangles=45*np.ones(len(value_vars)),   ## rotate tick labels
            title=title_x
        )
        .update_yaxes(showticklabels=True)
    )

    ## remove redundant axis labels from marginal boxplots
    for trace in fig.data:
        if trace.type == "box":
            ## handle y-axis cleanup
            yaxis_attr = trace.yaxis
            if yaxis_attr == "y":
                yaxis_obj = fig.layout.yaxis
            else:
                num = yaxis_attr[1:]
                yaxis_obj = getattr(fig.layout, f"yaxis{num}")
            yaxis_obj.showticklabels = False
            yaxis_obj.showgrid = False
            yaxis_obj.title = ""

            ## handle x-axis cleanup
            xaxis_attr = trace.xaxis
            if xaxis_attr == "x":
                xaxis_obj = fig.layout.xaxis
            else:
                num = xaxis_attr[1:]
                xaxis_obj = getattr(fig.layout, f"xaxis{num}")
            xaxis_obj.showticklabels = False
            xaxis_obj.showgrid = False
            xaxis_obj.title = ""

    return fig



########
# TO DO: do something with the color axis but idk if it's better to have multiple coloraxis or a unique one.
########

########
# TO DO: parameters to choose the display order of groups
########

def plot_heatmap_by_group(
    df: pl.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    normalize: bool = False,
    colorscale: str = "Cividis"
):
    """
    Plots a series of heatmaps for each unique value in a specified grouping column.
    
    Each heatmap shows the count (or percentage if normalized) of occurrences 
    for combinations of `x_col` and `y_col` within the group.

    Warning: for floating data, the range need to be binned.
    
    Parameters:
    -----------
    df (pl.DataFrame): The input dataframe (Polars) containing the data to visualize.
    group_col (str): The name of the column to group by. A separate heatmap will be created for each unique value.
    x_col (str): The column to use for the x-axis of the heatmap.
    y_col (str): The column to use for the y-axis of the heatmap.
    normalize (bool): If True, counts are normalized to percentages within each group. Default to False.
    colorscale (str): The colorscale to use for the heatmaps. Default to "Cividis".
    
    Returns:
    --------
    plotly.graph_objects.Figure: A figure containing subplots for each group with heatmaps.

    Warning:
    --------
    For floating/continuous data in `x_col` or `y_col`, you must bin the values first before using this function.
    Otherwise, the heatmap axes may become too large or unreadable.
    
    Example:
    --------
        >>> fig = plot_heatmap_by_group(
        ...     df=df,
        ...     group_col="group",
        ...     x_col="x",
        ...     y_col="y",
        ...     normalize=True,
        ... )
        >>> fig.show()
    """
    
    ## compute counts of each combination of group, x, and y
    df_counts = (
        df.group_by([group_col, x_col, y_col])
          .len()
          .rename({"len": "COUNT"})
    )

    ## get unique values for groups, x-axis, and y-axis
    group_vals = df_counts[group_col].unique().to_list()
    all_x = df_counts[x_col].unique().to_list()
    all_y = df_counts[y_col].unique().to_list()

    ## initialize subplots: one column per group
    fig = make_subplots(
        rows=1, cols=len(group_vals),
        subplot_titles=[f"{group_col}={val}" for val in group_vals]
    )

    for i, val in enumerate(group_vals, start=1):
        ## filter data for the current group and pivot to create a 2D table
        df_filtered = (
            df_counts
            .filter(pl.col(group_col) == val)
            .to_pandas()
            .pivot(index=y_col, columns=x_col, values="COUNT")
            .reindex(index=all_y, columns=all_x)   ## ensure consistent axis order accross group
            .astype(float)
        )

        ## total count for normalization
        total = np.nansum(df_filtered.values)

        ## normalize to percentages if requested
        if normalize and total > 0:
            df_norm = df_filtered / total * 100.0
            is_normalized = True
        else:
            df_norm = df_filtered.copy()
            is_normalized = False

        z = df_norm.values

        ## set hover info and text formatting
        if is_normalized:
            hover = f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>PERCENT=%{{z:.2f}}%<extra></extra>"
            texttemplate_str = "%{z:.2f}"
        else:
            hover = f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>COUNT=%{{z:.0f}}<extra></extra>"
            texttemplate_str = "%{z:.0f}" 

        ## create the heatmap trace
        heatmap = go.Heatmap(
            z=z,
            x=df_norm.columns,
            y=df_norm.index,
            text=z,
            texttemplate=texttemplate_str,
            hovertemplate=hover,
            colorscale=colorscale,
            zmin=np.nanmin(z),
            zmax=np.nanmax(z),
            hoverongaps=False
        )

        ## add heatmap to the subplot
        fig.add_trace(heatmap, row=1, col=i)

        ## update axis titles and types
        fig.update_xaxes(title_text=x_col, row=1, col=i)
        fig.update_yaxes(title_text=y_col, row=1, col=i)
        fig.update_xaxes(type="category", row=1, col=i)
        fig.update_yaxes(type="category", row=1, col=i)

    return fig
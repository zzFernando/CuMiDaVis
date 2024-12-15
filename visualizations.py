import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

def plot_violin(data, selected_gene):
    """
    Plots a Violin Plot for the selected gene.
    
    Args:
        data (pd.DataFrame): Dataset with genomic data.
        selected_gene (str): Name of the selected gene for the Violin Plot.
    """
    selected_gene = str(selected_gene)
    
    fig_violin = px.violin(
        data, y=selected_gene, x='type', color='type', box=True, points="all",
        title=f"Violin Plot for {selected_gene}",
        labels={"type": "Type", selected_gene: "Gene Expression"}
    )
    st.plotly_chart(fig_violin, use_container_width=True)

def plot_2d_pca(data, normalized_data):
    pca_2d = PCA(n_components=2).fit_transform(normalized_data)
    data['PCA1'], data['PCA2'] = pca_2d.T
    fig = px.scatter(data, x='PCA1', y='PCA2', color='type', hover_data=['samples'],
                     title="PCA - 2 Components", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

def plot_tsne(data, normalized_data, perplexity):
    tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(normalized_data)
    data['tSNE1'], data['tSNE2'] = tsne_result.T
    fig = px.scatter(data, x='tSNE1', y='tSNE2', color='type', hover_data=['samples'],
                     title="t-SNE - 2 Components", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

def plot_umap(data, normalized_data, n_neighbors, min_dist):
    umap_result = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(normalized_data)
    data['UMAP1'], data['UMAP2'] = umap_result.T
    fig = px.scatter(data, x='UMAP1', y='UMAP2', color='type', hover_data=['samples'],
                     title="UMAP - 2 Components", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(normalized_data, selected_columns, heatmap_limit):
    corr_matrix = pd.DataFrame(normalized_data[:, :heatmap_limit], columns=selected_columns[:heatmap_limit]).corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", labels=dict(color="Correlation"))
    st.plotly_chart(fig, use_container_width=True)

def plot_boxplot(data, selected_gene):
    """
    Plots a boxplot for the expression of a selected gene.
    
    Args:
        data (pd.DataFrame): Dataset with genomic data.
        selected_gene (str): Name of the selected gene for the boxplot.
    """
    selected_gene = str(selected_gene)
    fig_box = px.box(
        data, x='type', y=selected_gene, color='type',
        title=f"Boxplot for {selected_gene}",
        labels={"type": "Type", selected_gene: "Gene Expression"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

def plot_histogram(data, selected_gene):
    """
    Plots a histogram for the distribution of a selected gene.
    
    Args:
        data (pd.DataFrame): Dataset with genomic data.
        selected_gene (str): Name of the selected gene for the histogram.
    """
    selected_gene = str(selected_gene)
    fig_hist = px.histogram(
        data, x=selected_gene, nbins=30, color='type',
        title=f"Histogram for {selected_gene}",
        labels={"type": "Type", selected_gene: "Gene Expression"}, marginal="box"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

def plot_scatterplot_matrix(data, normalized_data, selected_columns, scatterplot_limit):
    """
    Plots the Scatterplot Matrix with a configurable limit of genes.

    Args:
        data (pd.DataFrame): Genomic dataset.
        normalized_data (ndarray): Normalized data.
        selected_columns (list): List of selected columns.
        scatterplot_limit (int): Limit of genes to be included in the plot.
    """
    scatter_data = pd.DataFrame(normalized_data[:, :scatterplot_limit], columns=selected_columns[:scatterplot_limit])
    scatter_data['type'] = data['type']
    fig_matrix = px.scatter_matrix(
        scatter_data,
        dimensions=selected_columns[:scatterplot_limit],
        color="type",
        title="Scatterplot Matrix",
        labels={col: col for col in selected_columns[:scatterplot_limit]}
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

def plot_parallel_coordinates(data, normalized_data, selected_columns):
    parallel_data = pd.DataFrame(normalized_data[:, :10], columns=selected_columns[:10])
    parallel_data['type'] = data['type']
    parallel_data['type_numeric'] = pd.factorize(parallel_data['type'])[0]
    fig_parallel = px.parallel_coordinates(
        parallel_data, color='type_numeric',
        dimensions=selected_columns[:10],
        labels={col: col for col in selected_columns[:10]},
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel Coordinates Plot"
    )
    st.plotly_chart(fig_parallel, use_container_width=True)

def plot_bar_chart(data, selected_columns):
    """
    Plots a bar chart showing the average gene expression by type.
    """
    selected_gene = st.sidebar.selectbox("Select a Gene for the Bar Chart:", selected_columns)
    avg_values = data.groupby('type')[selected_gene].mean().reset_index()
    fig_bar = px.bar(
        avg_values, x='type', y=selected_gene,
        title=f"Average Expression of Gene '{selected_gene}' by Type",
        labels={"type": "Type", selected_gene: "Average Expression"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

def plot_line_chart(data, selected_columns):
    selected_gene = st.sidebar.selectbox("Select a Gene for the Line Chart:", selected_columns)
    fig_line = px.line(
        data, x='samples', y=selected_gene, color='type',
        title=f"Line Chart for {selected_gene}",
        labels={"samples": "Samples", selected_gene: "Gene Expression"}
    )
    st.plotly_chart(fig_line, use_container_width=True)

def plot_regression(data, selected_columns):
    reg_x = st.sidebar.selectbox("Select X-Axis (Regression):", selected_columns)
    reg_y = st.sidebar.selectbox("Select Y-Axis (Regression):", selected_columns)
    fig_reg = px.scatter(
        data, x=reg_x, y=reg_y, color='type', trendline="ols",
        title=f"Regression Scatter: {reg_x} vs {reg_y}",
        labels={"type": "Type", reg_x: "X-Axis", reg_y: "Y-Axis"}
    )
    st.plotly_chart(fig_reg, use_container_width=True)

def plot_correlation_scatter(data, selected_columns):
    """
    Plots a scatter plot for the correlation between two selected genes.

    Args:
        data (pd.DataFrame): Genomic dataset.
        selected_columns (list): List of selected columns.
    """
    gene_x = st.sidebar.selectbox("Select Gene for X-Axis:", selected_columns)
    gene_y = st.sidebar.selectbox("Select Gene for Y-Axis:", selected_columns)
    fig_corr_scatter = px.scatter(
        data, x=gene_x, y=gene_y, color='type',
        title=f"Correlation between {gene_x} and {gene_y}",
        labels={"type": "Type", gene_x: gene_x, gene_y: gene_y}
    )
    st.plotly_chart(fig_corr_scatter, use_container_width=True)

def plot_class_distribution(data):
    """
    Plots the distribution of classes (type).
    """
    fig_class_dist = px.histogram(
        data, x='type', color='type', 
        title="Class Distribution",
        labels={"type": "Type", "count": "Frequency"}
    )
    st.plotly_chart(fig_class_dist, use_container_width=True)

def plot_radar_chart(data, selected_columns):
    """
    Plots a radar chart comparing the average gene expression by type.

    Args:
        data (pd.DataFrame): Genomic dataset.
        selected_columns (list): List of selected columns.
    """
    radar_data = data.groupby('type')[selected_columns].mean().reset_index()
    fig_radar = px.line_polar(
        radar_data.melt(id_vars='type', var_name='gene', value_name='expression'),
        r='expression', theta='gene', color='type',
        line_close=True,
        title="Radar Chart - Gene Expression Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

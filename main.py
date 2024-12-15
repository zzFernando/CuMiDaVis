import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import load_data, setup_sidebar, get_top_genes
from visualizations import (
    plot_2d_pca,
    plot_tsne,
    plot_umap,
    plot_heatmap,
    plot_violin,
    plot_scatterplot_matrix,
    plot_parallel_coordinates,
    plot_bar_chart,
    plot_line_chart,
    plot_regression,
    plot_radar_chart,
    plot_class_distribution,
    plot_correlation_scatter
)

# Dashboard configuration and dataset loading
try:
    data, selected_columns, params, dataset_choice = setup_sidebar(get_top_genes)
    if not selected_columns:
        st.error("No columns selected for visualization.")
        st.stop()
except Exception as e:
    st.error(f"Error configuring the sidebar: {e}")
    st.stop()

# Data normalization
try:
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[selected_columns])
    if normalized_data is None or normalized_data.shape[1] == 0:
        st.error("Normalized data is empty or invalid.")
        st.stop()
except Exception as e:
    st.error(f"Error normalizing the data: {e}")
    st.stop()

# Main title
st.title(f"Visualization Dashboard - {dataset_choice}")
st.header("Visualizations")

# Displaying all visualizations
try:
    st.subheader("PCA - Principal Component Analysis")
    plot_2d_pca(data, normalized_data)

    st.subheader("t-SNE - 2D Visualization")
    plot_tsne(data, normalized_data, params['perplexity'])

    st.subheader("UMAP - 2D Visualization")
    plot_umap(data, normalized_data, params['n_neighbors'], params['min_dist'])

    st.subheader("Correlation Heatmap")
    plot_heatmap(normalized_data, selected_columns, params['heatmap_limit'])

    st.subheader("Violin Plot")
    plot_violin(data, params['selected_gene_for_violin'])

    st.subheader("Scatterplot Matrix")
    plot_scatterplot_matrix(data, normalized_data, selected_columns, params['scatterplot_limit'])

    st.subheader("Parallel Coordinates Plot")
    plot_parallel_coordinates(data, normalized_data, selected_columns)

    st.subheader("Bar Chart - Average Gene Expression")
    plot_bar_chart(data, selected_columns)

    st.subheader("Line Chart")
    plot_line_chart(data, selected_columns)

    st.subheader("Linear Regression")
    plot_regression(data, selected_columns)

    st.subheader("Radar Chart")
    plot_radar_chart(data, selected_columns[:10])  # Shows only the first 10 genes

    st.subheader("Class Distribution")
    plot_class_distribution(data)

    st.subheader("Cross-Correlation")
    plot_correlation_scatter(data, selected_columns)
except KeyError as ke:
    st.error(f"Key error generating visualizations: {ke}")
except Exception as e:
    st.error(f"Error rendering visualizations: {e}")

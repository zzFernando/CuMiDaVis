import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import load_data, setup_sidebar, get_top_genes
from visualizations import (
    plot_3d_pca,
    plot_2d_pca,
    plot_tsne,
    plot_umap,
    plot_heatmap,
    plot_violin,
    plot_scatterplot_matrix,
    plot_parallel_coordinates,
    plot_bar_chart,
    plot_line_chart,
    plot_regression
)

# Configurações do Dashboard e carregamento do dataset
try:
    data, selected_columns, params, dataset_choice = setup_sidebar(get_top_genes)
except Exception as e:
    st.error(f"Erro ao configurar o sidebar: {e}")
    st.stop()

# Normalização dos dados
try:
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[selected_columns])
except Exception as e:
    st.error(f"Erro ao normalizar os dados: {e}")
    st.stop()

# Título principal
st.title(f"Dashboard de Visualização - {dataset_choice}")
st.header("Visualizações")

# Chamando as visualizações
try:
    plot_3d_pca(data, normalized_data)
    plot_2d_pca(data, normalized_data)
    plot_tsne(data, normalized_data, params['perplexity'])
    plot_umap(data, normalized_data, params['n_neighbors'], params['min_dist'])
    plot_heatmap(normalized_data, selected_columns, params['heatmap_limit'])
    plot_violin(data, params['selected_gene_for_violin'])
    plot_scatterplot_matrix(data, normalized_data, selected_columns, params['scatterplot_limit'])
    plot_parallel_coordinates(data, normalized_data, selected_columns)
    plot_bar_chart(data, selected_columns)
    plot_line_chart(data, selected_columns)
    plot_regression(data, selected_columns)
except KeyError as ke:
    st.error(f"Erro de chave ao gerar visualizações: {ke}")
except Exception as e:
    st.error(f"Erro ao renderizar as visualizações: {e}")

import os
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scrapping import initialize_cache
from visualizations import (
    plot_2d_pca, plot_tsne, plot_umap, plot_heatmap, plot_violin,
    plot_scatterplot_matrix, plot_parallel_coordinates, plot_bar_chart,
    plot_line_chart, plot_regression
)
from config import setup_sidebar

CACHE_DIR = "normalized_cache"
initialize_cache()

def save_normalized_data(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_normalized_data(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def main():
    base_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/"
    data, selected_columns, params, selected_dataset = setup_sidebar(base_url)

    # Normalizar dados e usar cache
    normalized_cache_file = os.path.join(CACHE_DIR, f"{selected_dataset}_normalized.pkl")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if os.path.exists(normalized_cache_file):
        st.sidebar.markdown("**Carregando dados normalizados do cache...**")
        normalized_data = load_normalized_data(normalized_cache_file)
    else:
        st.sidebar.markdown("**Normalizando dados...**")
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data[selected_columns])
        save_normalized_data(normalized_cache_file, normalized_data)
        st.sidebar.markdown("**Dados normalizados salvos no cache.**")

    # Renderizar visualizações
    st.header("Visualizações")
    try:
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
    except Exception as e:
        st.error(f"Erro ao renderizar as visualizações: {e}")

if __name__ == "__main__":
    main()

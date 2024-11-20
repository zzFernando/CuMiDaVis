import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# URLs dos datasets
DATASETS = {
    "Liver": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv",
    "Breast": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE70947/Breast_GSE70947.csv",
    "Prostate": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Prostate/GSE6919_U95Av2/Prostate_GSE6919_U95Av2.csv",
}

@st.cache_data
def load_data(url):
    """Carrega os dados do URL fornecido."""
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def get_top_genes(data, target_column='type', n_genes=10):
    """Calcula os genes mais impactantes com base na importância do modelo."""
    # Prepara os dados para o modelo
    X = data.iloc[:, 2:]  # Assume que as colunas de genes começam no índice 2
    y = data[target_column]

    # Divide os dados para treinamento
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treina um RandomForestClassifier para calcular importâncias
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Calcula a importância dos genes
    importances = model.feature_importances_
    important_genes = X.columns[np.argsort(importances)[-n_genes:]].tolist()
    return important_genes

def setup_sidebar(get_top_genes):
    """Configurações do sidebar e carregamento do dataset."""
    st.sidebar.title("Configurações do Dashboard")
    
    # 1. Escolha do Dataset
    st.sidebar.subheader("1. Escolha do Dataset")
    dataset_choice = st.sidebar.selectbox(
        "Dataset:", options=list(DATASETS.keys()), index=0, key="dataset_choice"
    )
    data_url = DATASETS[dataset_choice]
    data = load_data(data_url)
    
    # 2. Configuração de Genes
    st.sidebar.subheader("2. Configuração de Genes")
    use_top_genes = st.sidebar.checkbox("Usar os 10 genes mais impactantes?", value=False, key="use_top_genes")
    
    if use_top_genes:
        st.sidebar.markdown("**Calculando os genes mais impactantes...**")
        top_genes = get_top_genes(data)
        selected_columns = top_genes
        st.sidebar.markdown(f"**Genes Selecionados:** {', '.join(top_genes)}")
    else:
        use_all_genes = st.sidebar.checkbox("Usar todos os genes?", value=True, key="use_all_genes")
        selected_columns = (
            data.columns[2:] if use_all_genes else st.sidebar.multiselect(
                "Selecione genes específicos:", data.columns[2:], default=data.columns[2:7], key="select_genes"
            )
        )
    
    # 3. Configurações de Gráficos
    st.sidebar.subheader("3. Configurações de Gráficos")
    pca_components = st.sidebar.slider("Número de Componentes para PCA (2D ou 3D):", 2, 3, 3, key="pca_components")
    perplexity = st.sidebar.slider("Perplexidade (t-SNE):", 5, 50, 30, key="tsne_perplexity")
    n_neighbors = st.sidebar.slider("Número de Vizinhos (UMAP):", 5, 50, 15, key="umap_neighbors")
    min_dist = st.sidebar.slider("Distância Mínima (UMAP):", 0.0, 1.0, 0.1, key="umap_min_dist")
    heatmap_limit = st.sidebar.slider(
        "Número Máximo de Genes no Heatmap:", 10, len(data.columns[2:]), 50, key="heatmap_limit"
    )
    selected_gene_for_violin = st.sidebar.selectbox(
        "Gene para Violin Plot:", options=selected_columns, key="violin_gene"
    )
    scatterplot_limit = st.sidebar.slider("Número de Genes no Scatterplot Matrix:", 2, 5, 3, key="scatterplot_limit")

    # 4. Resumo do Dataset
    st.sidebar.subheader("4. Resumo do Dataset")
    st.sidebar.markdown(f"""
    - **Dataset Selecionado:** {dataset_choice}
    - **Número de Genes Disponíveis:** {len(data.columns[2:])}
    - **Número de Amostras:** {len(data)}
    """)

    # Retorna os dados e parâmetros
    params = {
        'pca_components': pca_components,
        'perplexity': perplexity,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'heatmap_limit': heatmap_limit,
        'selected_gene_for_violin': selected_gene_for_violin,
        'scatterplot_limit': scatterplot_limit,
    }
    return data, selected_columns, params, dataset_choice

# Chamada Principal
data, selected_columns, params, dataset_choice = setup_sidebar(get_top_genes)

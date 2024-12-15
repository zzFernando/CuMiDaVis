import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

BASE_URL = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/"

@st.cache_data
def fetch_gene_types(base_url=BASE_URL):
    """Obtém os tipos de câncer disponíveis no CuMiDa."""
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [link["href"].strip("/") for link in soup.find_all("a", href=True) if link["href"].endswith("/")]

@st.cache_data
def fetch_datasets_for_gene_type(gene_type, base_url=BASE_URL):
    """Obtém os datasets disponíveis para um tipo de câncer, incluindo subpastas."""
    datasets = {}
    url = f"{base_url}{gene_type}/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".csv"):
            # Dataset direto no diretório
            datasets[href.replace(".csv", "")] = url + href
        elif href.endswith("/"):
            # Subdiretório, explorar recursivamente
            sub_url = url + href
            sub_response = requests.get(sub_url)
            sub_response.raise_for_status()
            sub_soup = BeautifulSoup(sub_response.text, "html.parser")
            for sub_link in sub_soup.find_all("a", href=True):
                if sub_link["href"].endswith(".csv"):
                    datasets[sub_link["href"].replace(".csv", "")] = sub_url + sub_link["href"]
    return datasets


@st.cache_data
def load_data(url):
    """Carrega os dados do URL fornecido."""
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def get_top_genes(data, target_column='type', n_genes=10):
    """Calcula os genes mais impactantes com base na importância do modelo."""
    X = data.iloc[:, 2:]  # Assume que as colunas de genes começam no índice 2
    y = data[target_column]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    important_genes = X.columns[np.argsort(importances)[-n_genes:]].tolist()
    return important_genes
def setup_sidebar(get_top_genes):
    """Configurações do sidebar e carregamento do dataset."""
    st.sidebar.title("Configurações do Dashboard")
    
    # 1. Escolha do Tipo de Câncer
    st.sidebar.subheader("1. Escolha o Tipo de Câncer")
    gene_types = fetch_gene_types()
    selected_gene_type = st.sidebar.selectbox(
        "Tipo de Câncer:", options=gene_types, index=0, key="gene_type"
    )
    
    # 2. Escolha do Dataset
    st.sidebar.subheader("2. Escolha o Dataset")
    datasets = fetch_datasets_for_gene_type(selected_gene_type)
    
    if datasets:  # Verifica se há datasets disponíveis
        dataset_choice = st.sidebar.selectbox(
            "Dataset:", options=list(datasets.keys()), key="dataset_choice"
        )
        data_url = datasets[dataset_choice]
        data = load_data(data_url)
    else:
        st.sidebar.warning(f"Nenhum dataset encontrado para o tipo de câncer: {selected_gene_type}")
        st.stop()  # Interrompe a execução caso não haja datasets
    
    # 3. Configuração de Genes
    st.sidebar.subheader("3. Configuração de Genes")
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
    
    # 4. Configurações de Gráficos
    st.sidebar.subheader("4. Configurações de Gráficos")
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

    # 5. Resumo do Dataset
    st.sidebar.subheader("5. Resumo do Dataset")
    st.sidebar.markdown(f"""
    - **Tipo de Câncer Selecionado:** {selected_gene_type}
    - **Dataset Selecionado:** {dataset_choice if datasets else "Nenhum"}
    - **Número de Genes Disponíveis:** {len(data.columns[2:]) if datasets else "0"}
    - **Número de Amostras:** {len(data) if datasets else "0"}
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
    return data, selected_columns, params, dataset_choice if datasets else None

# Chamada Principal
data, selected_columns, params, dataset_choice = setup_sidebar(get_top_genes)

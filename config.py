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
def fetch_cancer_types(base_url=BASE_URL):
    """Fetches the available cancer types in the CuMiDa dataset."""
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [link["href"].strip("/") for link in soup.find_all("a", href=True) if link["href"].endswith("/")]

@st.cache_data
def fetch_datasets_for_cancer_type(cancer_type, base_url=BASE_URL):
    """Fetches the available datasets for a specific cancer type, including subdirectories."""
    datasets = {}
    url = f"{base_url}{cancer_type}/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".csv"):
            # Dataset directly in the directory
            datasets[href.replace(".csv", "")] = url + href
        elif href.endswith("/"):
            # Subdirectory, explore recursively
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
    """Loads the dataset from the provided URL."""
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def get_top_genes(data, target_column='type', n_genes=10):
    """Calculates the most impactful genes based on feature importance."""
    X = data.iloc[:, 2:]  # Assumes gene columns start at index 2
    y = data[target_column]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    important_genes = X.columns[np.argsort(importances)[-n_genes:]].tolist()
    return important_genes

def setup_sidebar(get_top_genes):
    """Configures the sidebar and loads the dataset."""
    st.sidebar.title("Dashboard Configuration")

    # 1. Select Cancer Type
    st.sidebar.markdown("### 1. Select Cancer Type")
    cancer_types = fetch_cancer_types()
    selected_cancer_type = st.sidebar.selectbox(
        "Cancer Type:", options=cancer_types, index=0, key="cancer_type"
    )

    # 2. Select Dataset
    st.sidebar.markdown("### 2. Select Dataset")
    datasets = fetch_datasets_for_cancer_type(selected_cancer_type)

    if datasets:  # Check if datasets are available
        dataset_choice = st.sidebar.selectbox(
            "Dataset:", options=list(datasets.keys()), key="dataset_choice"
        )
        data_url = datasets[dataset_choice]
        data = load_data(data_url)
    else:
        st.sidebar.warning(f"No dataset found for cancer type: {selected_cancer_type}")
        st.stop()  # Stops execution if no datasets are found

    # Dataset Summary (Now displayed first in the sidebar)
    st.sidebar.markdown("## Dataset Summary")
    st.sidebar.markdown(f"""
    - **Selected Cancer Type:** {selected_cancer_type}
    - **Selected Dataset:** {dataset_choice if datasets else "None"}
    - **Number of Available Genes:** {len(data.columns[2:]) if datasets else "0"}
    - **Number of Samples:** {len(data) if datasets else "0"}
    """)

    # 3. Gene Configuration
    st.sidebar.markdown("### 3. Gene Configuration")
    use_top_genes = st.sidebar.checkbox("Use Top 10 Most Impactful Genes?", value=False, key="use_top_genes")

    if use_top_genes:
        st.sidebar.markdown("**Calculating the most impactful genes...**")
        top_genes = get_top_genes(data)
        selected_columns = list(top_genes)  # Convert to list to avoid ambiguity
        st.sidebar.markdown(f"**Selected Genes:** {', '.join(top_genes)}")
    else:
        use_all_genes = st.sidebar.checkbox("Use All Genes?", value=True, key="use_all_genes")
        if use_all_genes:
            selected_columns = list(data.columns[2:])  # Convert to list
        else:
            selected_columns = st.sidebar.multiselect(
                "Select Specific Genes:", list(data.columns[2:]), default=list(data.columns[2:7]), key="select_genes"
            )

    # Validate selected_columns
    if selected_columns is None or len(selected_columns) == 0:
        st.error("No genes selected for visualization.")
        st.stop()

    # Visualization Configurations
    st.sidebar.markdown("## Visualization Settings")

    # 4. Dimensionality Reduction Settings
    st.sidebar.markdown("### 4. PCA and Dimensionality Reduction")
    pca_components = st.sidebar.slider("Number of PCA Components:", 2, 3, 3, key="pca_components")
    perplexity = st.sidebar.slider("t-SNE Perplexity:", 5, 50, 30, key="tsne_perplexity")
    n_neighbors = st.sidebar.slider("UMAP Neighbors:", 5, 50, 15, key="umap_neighbors")
    min_dist = st.sidebar.slider("UMAP Minimum Distance:", 0.0, 1.0, 0.1, key="umap_min_dist")

    st.sidebar.markdown("### 5. Advanced Graph Settings")
    heatmap_limit = st.sidebar.slider(
        "Maximum Number of Genes in Heatmap:", 10, len(data.columns[2:]), 50, key="heatmap_limit"
    )
    selected_gene_for_violin = st.sidebar.selectbox(
        "Gene for Violin Plot:", options=selected_columns, key="violin_gene"
    )
    scatterplot_limit = st.sidebar.slider("Number of Genes in Scatterplot Matrix:", 2, 5, 3, key="scatterplot_limit")

    # Return the data and parameters
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

import streamlit as st
from scrapping import fetch_gene_types, fetch_datasets_for_gene_type, load_data
from utils import get_top_genes

def setup_sidebar(base_url: str):
    """
    Configura o sidebar e retorna os dados selecionados e parâmetros de visualização.

    Args:
        base_url (str): URL base para buscar os tipos de genes e datasets.

    Returns:
        tuple: Dados carregados, colunas selecionadas, parâmetros de visualização e dataset selecionado.
    """
    st.sidebar.title("Configurações do Dashboard")

    try:
        # Obter tipos de genes disponíveis
        gene_types = fetch_gene_types(base_url)
        if not gene_types:
            st.sidebar.error("Não há tipos de genes disponíveis.")
            st.stop()

        # Seleção do tipo de gene com valor padrão
        selected_gene_type = st.sidebar.selectbox(
            "Selecione o tipo de gene:",
            gene_types,
            index=gene_types.index("Brain") if "Brain" in gene_types else 0
        )

        # Obter datasets do tipo selecionado
        datasets = fetch_datasets_for_gene_type(selected_gene_type, base_url)
        if not datasets:
            st.sidebar.error(f"Não há datasets disponíveis para o tipo de gene: {selected_gene_type}.")
            st.stop()

        # Seleção do dataset com valor padrão
        dataset_keys = list(datasets.keys())
        selected_dataset = st.sidebar.selectbox(
            "Selecione o dataset:",
            dataset_keys,
            index=dataset_keys.index("Brain_GSE50161") if "Brain_GSE50161" in dataset_keys else 0
        )

        # Carregar dados do dataset selecionado
        data_url = datasets[selected_dataset]
        data = load_data(data_url)

        # Opção para selecionar os genes mais importantes
        use_top_genes = st.sidebar.checkbox("Usar os 10 genes mais impactantes?", value=False)
        selected_columns = data.columns[2:]
        if use_top_genes:
            try:
                top_genes = get_top_genes(data)
                selected_columns = top_genes
                st.sidebar.markdown(f"**Genes Selecionados:** {', '.join(top_genes)}")
            except Exception as e:
                st.sidebar.error(f"Erro ao calcular os genes mais impactantes: {e}")
                st.stop()

        # Configurações adicionais de visualização
        params = {
            'perplexity': st.sidebar.slider("Perplexidade (t-SNE):", 5, 50, 30),
            'n_neighbors': st.sidebar.slider("Número de Vizinhos (UMAP):", 5, 50, 15),
            'min_dist': st.sidebar.slider("Distância Mínima (UMAP):", 0.0, 1.0, 0.1),
            'heatmap_limit': st.sidebar.slider(
                "Número Máximo de Genes no Heatmap:", 
                10, 
                min(len(selected_columns), 50),  # Garantir que o limite seja o menor valor entre 50 e o número de colunas
                10
            ),
            'selected_gene_for_violin': st.sidebar.selectbox("Gene para Violin Plot:", options=selected_columns),
            'scatterplot_limit': st.sidebar.slider("Número de Genes no Scatterplot Matrix:", 2, 5, 3)
        }

        return data, selected_columns, params, selected_dataset

    except Exception as e:
        st.sidebar.error(f"Erro ao configurar o sidebar: {e}")
        st.stop()

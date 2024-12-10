import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt



def plot_3d_pca(data, normalized_data):
    pca_3d = PCA(n_components=3).fit_transform(normalized_data)
    data['PCA1'], data['PCA2'], data['PCA3'] = pca_3d.T
    fig = px.scatter_3d(data, x='PCA1', y='PCA2', z='PCA3', color='type', hover_data=['samples'],
                        title="Projeção PCA em 3D", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)



def plot_2d_pca(data, normalized_data):
    pca_2d = PCA(n_components=2).fit_transform(normalized_data)
    data['PCA1'], data['PCA2'] = pca_2d.T
    fig = px.scatter(data, x='PCA1', y='PCA2', color='type', hover_data=['samples'],
                     title="PCA - 2 Componentes", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)



def compute_tsne(normalized_data, perplexity):
    tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(normalized_data)
    return tsne_result



def plot_tsne(data, normalized_data, perplexity):
    tsne_result = compute_tsne(normalized_data, perplexity)
    data['tSNE1'], data['tSNE2'] = tsne_result.T
    fig = px.scatter(data, x='tSNE1', y='tSNE2', color='type', title="t-SNE Visualization")
    st.plotly_chart(fig, use_container_width=True)



def compute_umap(normalized_data, n_neighbors, min_dist):
    umap_result = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(normalized_data)
    return umap_result



def plot_umap(data, normalized_data, n_neighbors, min_dist):
    umap_result = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(normalized_data)
    data['UMAP1'], data['UMAP2'] = umap_result.T
    fig = px.scatter(data, x='UMAP1', y='UMAP2', color='type', hover_data=['samples'],
                     title="UMAP - 2 Componentes", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)



@st.cache_data
def plot_heatmap(normalized_data, _selected_columns, heatmap_limit):
    """
    Plota um heatmap de correlação baseado no número máximo de genes definido.

    Args:
        normalized_data (ndarray): Dados normalizados.
        _selected_columns (list): Colunas selecionadas para visualização.
        heatmap_limit (int): Número máximo de genes a serem exibidos no heatmap.
    """
    limited_columns = _selected_columns[:heatmap_limit]
    limited_data = normalized_data[:, :heatmap_limit]
    
    corr_matrix = pd.DataFrame(limited_data, columns=limited_columns).corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Heatmap de Correlação",
                    labels=dict(color="Correlação"))
    st.plotly_chart(fig, use_container_width=True)


def plot_violin(data, selected_gene):
    """
    Plota um Violin Plot para o gene selecionado.
    
    Args:
        data (pd.DataFrame): Dataset com os dados genômicos.
        selected_gene (str): Nome do gene selecionado para o Violin Plot.
    """
    # Garantir que `selected_gene` seja uma string
    selected_gene = str(selected_gene)
    
    fig_violin = px.violin(
        data, y=selected_gene, x='type', color='type', box=True, points="all",
        title=f"Violin Plot para {selected_gene}",
        labels={"type": "Tipo", selected_gene: "Expressão Gênica"}
    )
    st.plotly_chart(fig_violin, use_container_width=True)



def plot_scatterplot_matrix(data, normalized_data, selected_columns, scatterplot_limit):
    """
    Plota o Scatterplot Matrix com limite configurável de genes.

    Args:
        data (pd.DataFrame): Dataset genômico.
        normalized_data (ndarray): Dados normalizados.
        selected_columns (list): Lista de colunas selecionadas.
        scatterplot_limit (int): Limite de genes a serem incluídos no gráfico.
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
    Plota um gráfico de barras mostrando a média de expressão gênica por tipo.
    """
    selected_gene = st.sidebar.selectbox("Selecione um Gene para o Gráfico de Barras:", selected_columns)
    avg_values = data.groupby('type')[selected_gene].mean().reset_index()
    fig_bar = px.bar(
        avg_values, x='type', y=selected_gene,
        title=f"Média do Gene '{selected_gene}' por Tipo",
        labels={"type": "Tipo", selected_gene: "Expressão Média"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)



def plot_line_chart(data, selected_columns):
    selected_gene = st.sidebar.selectbox("Selecione um Gene para o Gráfico de Linhas:", selected_columns)
    fig_line = px.line(
        data, x='samples', y=selected_gene, color='type',
        title=f"Gráfico de Linhas para {selected_gene}",
        labels={"samples": "Amostras", selected_gene: "Expressão Gênica"}
    )
    st.plotly_chart(fig_line, use_container_width=True)



def plot_regression(data, selected_columns):
    reg_x = st.sidebar.selectbox("Selecione o Eixo X (Regressão):", selected_columns)
    reg_y = st.sidebar.selectbox("Selecione o Eixo Y (Regressão):", selected_columns)
    fig_reg = px.scatter(
        data, x=reg_x, y=reg_y, color='type', trendline="ols",
        title=f"Dispersão com Regressão: {reg_x} vs {reg_y}",
        labels={"type": "Tipo", reg_x: "Eixo X", reg_y: "Eixo Y"}
    )
    st.plotly_chart(fig_reg, use_container_width=True)

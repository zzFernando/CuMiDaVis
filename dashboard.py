import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
from io import StringIO

os.environ["NUMBA_THREADING_LAYER"] = "tbb"

# Função para baixar e carregar os dados
@st.cache_data
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Verifica se o download foi bem-sucedido
    csv_data = StringIO(response.text)
    data = pd.read_csv(csv_data)
    return data

# URLs dos datasets
datasets = {
    "Liver": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv",
    "Breast": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE70947/Breast_GSE70947.csv",
    "Prostate": "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Prostate/GSE6919_U95Av2/Prostate_GSE6919_U95Av2.csv",
}

# Seleção do dataset pelo usuário
st.sidebar.title("Configurações")
dataset_choice = st.sidebar.selectbox("Selecione o Dataset", options=list(datasets.keys()), index=0)
data_url = datasets[dataset_choice]

# Carregar os dados selecionados
st.title(f"Dashboard Interativo - {dataset_choice} Dataset")
st.sidebar.write(f"Carregando dataset: {dataset_choice}")
data = load_data_from_url(data_url)

# Configuração para seleção de genes
use_all_genes = st.sidebar.checkbox("Usar todos os genes?", value=True)

if use_all_genes:
    selected_columns = data.columns[2:]  # Todas as colunas de genes
else:
    selected_columns = st.sidebar.multiselect(
        "Selecione características (genes):",
        data.columns[2:],  # Todas as colunas de genes
        default=data.columns[2:7],  # Seleção inicial
    )

# Normalizar os dados
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[selected_columns])

# Gráfico 3D Inicial Giratório
st.header("Gráfico 3D Interativo")
st.subheader("Projeção em 3D com PCA")
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(normalized_data)
data['PCA1'] = pca_result_3d[:, 0]
data['PCA2'] = pca_result_3d[:, 1]
data['PCA3'] = pca_result_3d[:, 2]

fig_3d = px.scatter_3d(
    data, x='PCA1', y='PCA2', z='PCA3', color='type',
    title="Projeção PCA em 3D", hover_data=['samples'],
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
fig_3d.update_layout(scene=dict(
    xaxis_title="PCA1",
    yaxis_title="PCA2",
    zaxis_title="PCA3",
))
st.plotly_chart(fig_3d, use_container_width=True)

# PCA
st.subheader("PCA - 2 Componentes")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]
fig_pca = px.scatter(
    data, x='PCA1', y='PCA2', color='type', hover_data=['samples'],
    title="PCA - 2 Componentes", color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_pca, use_container_width=True)

# t-SNE
st.subheader("t-SNE - 2 Componentes")
perplexity = st.sidebar.slider("Perplexidade (t-SNE):", min_value=5, max_value=50, value=30)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(normalized_data)
data['tSNE1'] = tsne_result[:, 0]
data['tSNE2'] = tsne_result[:, 1]
fig_tsne = px.scatter(
    data, x='tSNE1', y='tSNE2', color='type', hover_data=['samples'],
    title="t-SNE - 2 Componentes", color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_tsne, use_container_width=True)

# UMAP
st.subheader("UMAP - 2 Componentes")
n_neighbors = st.sidebar.slider("Número de Vizinhos (UMAP):", min_value=5, max_value=50, value=15)
min_dist = st.sidebar.slider("Distância Mínima (UMAP):", min_value=0.0, max_value=1.0, value=0.1)
umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, verbose=False)
umap_result = umap_reducer.fit_transform(normalized_data)
data['UMAP1'] = umap_result[:, 0]
data['UMAP2'] = umap_result[:, 1]
fig_umap = px.scatter(
    data, x='UMAP1', y='UMAP2', color='type', hover_data=['samples'],
    title="UMAP - 2 Componentes", color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_umap, use_container_width=True)

# Heatmap de Correlação
st.subheader("Heatmap de Correlação")
heatmap_limit = st.sidebar.slider("Limite de Genes para Heatmap:", min_value=10, max_value=len(selected_columns), value=50)
with st.spinner("Gerando Heatmap..."):
    corr_data = pd.DataFrame(normalized_data[:, :heatmap_limit], columns=selected_columns[:heatmap_limit]).corr()
    fig_heatmap = px.imshow(
        corr_data, text_auto=True, title="Heatmap de Correlação",
        labels=dict(color="Correlação"), aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Gráfico de Barras (Média por Tipo)
st.subheader("Gráfico de Barras - Média por Tipo")
avg_values = data.groupby('type')[selected_columns].mean().reset_index()
fig_bar = px.bar(
    avg_values, x='type', y=selected_columns[0], title=f"Média do Gene '{selected_columns[0]}' por Tipo",
    labels={"value": "Valor Médio", "type": "Tipo"}
)
st.plotly_chart(fig_bar, use_container_width=True)

# Pairplot
st.subheader("Pairplot - Relações Entre Genes")
if len(selected_columns) > 1:
    pairplot_limit = st.sidebar.slider("Número de Genes no Pairplot:", min_value=2, max_value=min(len(selected_columns), 5), value=3)
    pairplot_data = pd.DataFrame(normalized_data[:, :pairplot_limit], columns=selected_columns[:pairplot_limit])
    pairplot_data['type'] = data['type']
    pairplot_fig = sns.pairplot(pairplot_data, hue="type")
    st.pyplot(pairplot_fig)

# Gráfico de Linhas por Amostra
st.subheader("Gráfico de Linhas por Amostra")
fig_line = px.line(
    data, x='samples', y=selected_columns[0], color='type',
    title=f"Gráfico de Linhas para {selected_columns[0]}"
)
st.plotly_chart(fig_line, use_container_width=True)

# Gráfico de Dispersão com Regressão
st.subheader("Dispersão com Regressão")
if len(selected_columns) > 1:
    reg_x = st.sidebar.selectbox("Selecione o Eixo X:", selected_columns, index=0)
    reg_y = st.sidebar.selectbox("Selecione o Eixo Y:", selected_columns, index=1)
    fig_reg = px.scatter(
        data, x=reg_x, y=reg_y, color='type', trendline="ols",
        title=f"Dispersão com Regressão: {reg_x} vs {reg_y}"
    )
    st.plotly_chart(fig_reg, use_container_width=True)

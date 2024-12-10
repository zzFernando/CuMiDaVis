import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/"
CACHE_DIR = "datasets_cache"

def initialize_cache(base_url=BASE_URL, cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for gene_type in fetch_gene_types(base_url):
        gene_cache_dir = os.path.join(cache_dir, gene_type)
        os.makedirs(gene_cache_dir, exist_ok=True)
        for dataset_name, dataset_url in fetch_datasets_for_gene_type(gene_type, base_url).items():
            local_path = os.path.join(gene_cache_dir, f"{dataset_name}.csv")
            if not os.path.exists(local_path):
                download_dataset(dataset_url, local_path)

def download_dataset(url, local_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)

def fetch_gene_types(base_url=BASE_URL):
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [link["href"].strip("/") for link in soup.find_all("a", href=True) if link["href"].endswith("/")]

def fetch_datasets_for_gene_type(gene_type, base_url=BASE_URL):
    datasets = {}
    url = f"{base_url}{gene_type}/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".csv"):
            datasets[href.replace(".csv", "")] = url + href
        elif href.endswith("/"):
            sub_url = url + href
            sub_response = requests.get(sub_url)
            sub_response.raise_for_status()
            sub_soup = BeautifulSoup(sub_response.text, "html.parser")
            for sub_link in sub_soup.find_all("a", href=True):
                if sub_link["href"].endswith(".csv"):
                    datasets[sub_link["href"].replace(".csv", "")] = sub_url + sub_link["href"]
    return datasets

def load_data(file_path):
    return pd.read_csv(file_path)

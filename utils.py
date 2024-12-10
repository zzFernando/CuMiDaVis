import numpy as np  # Certifique-se de importar numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_top_genes(data, target_column='type', n_genes=10):
    try:
        X = data.iloc[:, 2:]  # Assume que os dados de genes começam na 3ª coluna
        y = data[target_column]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        return X.columns[np.argsort(importances)[-n_genes:]].tolist()
    except Exception as e:
        raise RuntimeError(f"Erro ao calcular os genes mais importantes: {e}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    silhouette_score
)
import pickle


def classification_pipeline(
    data: pd.DataFrame,
    features: list,
    test_size: float = 0.5,
    scale: bool = True,
    algorithm: str = "knn",
    params: dict = None
) -> dict:
    """
    Executa treino e avaliação de modelo de classificação para DDoS (binário).
    Retorna dicionário:
      - model, X_test, y_test, y_pred
      - y_proba (ou None se multiclass)
      - metrics: accuracy, confusion_matrix, classification_report
        e opcionalmente roc_curve, auc
    """
    # Preparar dados
    X = data[features].copy()
    y = data['target'].copy()

    # One-hot encode de categoricais
    X = pd.get_dummies(X, drop_first=True)

    # Padronização
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Modelo
    params = params or {}
    if algorithm == "knn":
        model = KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
    elif algorithm == "dt":
        model = DecisionTreeClassifier(max_depth=params.get('max_depth', 5), random_state=42)
    elif algorithm == "rf":
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), random_state=42)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Probabilidades se binário
    y_proba = None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=[str(c) for c in model.classes_])
    }
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['roc_curve'] = (fpr, tpr)
        metrics['auc'] = auc(fpr, tpr)

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics
    }


def clustering_pipeline(
    data: pd.DataFrame,
    features: list,
    n_clusters: int = 3,
    scale: bool = True
) -> dict:
    """
    Executa K-Means e retorna labels, silhouette score e centros de cluster.
    """
    X = data[features].copy()
    X = pd.get_dummies(X, drop_first=True)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels)

    return {
        'labels': labels,
        'silhouette': sil_score,
        'cluster_centers': km.cluster_centers_
    }


def save_model(model, filepath: str):
    """Salva um objeto modelo em arquivo via pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filepath: str):
    """Carrega um modelo serializado via pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Exemplo de uso: binary Iris
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    # Filtra apenas 2 classes para binário
    df = df[df['target'] != 2]
    res = classification_pipeline(
        data=df,
        features=iris.feature_names,
        algorithm='dt',
        params={'max_depth':3},
        scale=False,
        test_size=0.3
    )
    print("Accuracy (Iris 2 classes):", res['metrics']['accuracy'])
    save_model(res['model'], 'iris_dt.pkl')
    print("Modelo Iris salvo com sucesso.")
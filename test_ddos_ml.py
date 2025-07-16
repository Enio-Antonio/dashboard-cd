# test_ddos_ml.py
import pandas as pd
from ml_module import classification_pipeline, clustering_pipeline, save_model, load_model

# 1) carregar dados
data = pd.read_csv("DDoS_dataset.csv")

# 2) teste de classificação
features = ["Packets/Time", "Transport Layer"]  # adapte pra colunas numéricas
# Se Transport Layer for categórica, você pode converter antes, e.g. pd.get_dummies()
data_num = pd.get_dummies(data[features])
result = classification_pipeline(
    data=pd.concat([data_num, data["target"]], axis=1),
    features=list(data_num.columns),
    test_size=0.3,
    scale=True,
    algorithm="dt",
    params={"max_depth": 4}
)

print(f"Acurácia: {result['metrics']['accuracy']:.2f}")
print("Confusion Matrix:\n", result['metrics']['confusion_matrix'])
print(result['metrics']['classification_report'])

# 3) salvar e recarregar
save_model(result["model"], "ddos_model.pkl")
loaded = load_model("ddos_model.pkl")
# conferir que a predição bate
assert (loaded.predict(result["y_test"].values.reshape(-1, len(data_num.columns))) 
        == result["y_pred"]).all()
print("Serialização OK ✅")

# 4) teste de clustering
cluster_features = ["Packets/Time", "Src Port"]  # substitua por 2 colunas numéricas
cluster_res = clustering_pipeline(
    data=data,
    features=cluster_features,
    n_clusters=4,
    scale=True
)
print(f"Silhouette Score: {cluster_res['silhouette']:.2f}")
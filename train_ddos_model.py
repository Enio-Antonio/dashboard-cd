# train_ddos_model.py

import os
import pandas as pd
import joblib
from ml_module import classification_pipeline, save_model

# Garante que models/ exista
os.makedirs('models', exist_ok=True)

# 1) Carrega os dados
csv_path = 'DDoS_dataset.csv'
print(f"Carregando dados de {csv_path}...")
data = pd.read_csv(csv_path)

# 2) Define as features, excluindo target e colunas de IP
cols_to_exclude = ['target', 'Source IP', 'Dest IP']
features = [c for c in data.columns if c not in cols_to_exclude]
print("Usando features:", features)

# 3) Executa o pipeline de classificação
res = classification_pipeline(
    data=data,
    features=features,
    test_size=0.3,
    scale=True,
    algorithm='rf',
    params={'n_estimators': 100}
)

# 4) Imprime métrica
acc = res['metrics']['accuracy']
print(f"Acurácia no conjunto de teste: {acc:.4f}")

# 5) Salva o modelo e os dados de teste
model_path = 'models/ddos_model.pkl'
test_path  = 'models/ddos_test.pkl'
save_model(res['model'], model_path)
joblib.dump({
    'X_test':  res['X_test'],
    'y_test':  res['y_test'],
    'metrics': res['metrics']
}, test_path)

print(f"✅ Modelo salvo em {model_path}")
print(f"✅ Dados de teste salvos em {test_path}")
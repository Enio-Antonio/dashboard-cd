## Uma análise sobre ataques DDoS.
* Link do dataset: https://www.kaggle.com/datasets/oktayrdeki/ddos-traffic-dataset
* Link da versão online: https://dashboard-cd.streamlit.app/

## Como rodar os códigos (no Windows) sem erro de versão:
* Criando o ambiente virtual:
```bash
python -m venv .venv
```
* Ativando o ambiente virtual:
```bash
./.venv/Scripts/Activate.ps1
```
* Instalação das dependências somente no ambiente virtual:
```bash
pip install -r requirements.txt
```
* Dependências adicionais (se necessário):
```bash
pip install scikit-learn joblib poltly shap
```
* Rodando a dashboard:
```bash
streamlit run dashboard.py
```

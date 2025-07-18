import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle 
from ml_module import classification_pipeline, clustering_pipeline, save_model, load_model
st.set_page_config(layout="centered", page_title="DDOS")

st.title('Análise de ataques DDOS')

@st.cache_resource
def load_ddos_model(path='models/ddos_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Carregar dados
data = pd.read_csv('DDoS_dataset.csv')

# Sidebar
st.sidebar.title("Navegação")

secao = st.sidebar.selectbox(
    "Selecione a seção:",
    ["Visualização", "Machine Learning"]
)

opcao = None

if secao == "Visualização":
    opcao = st.sidebar.selectbox(
        "Selecione o gráfico:",
        ["Pizza", "Barras", "Histograma", "Protocolo", "Top 10 IPs"]
    )

elif secao == "Machine Learning":
    import os
    import pickle
    import joblib

    st.header("▶ Resultados do Modelo DDoS (pré-treinado)")

    MODEL_PATH = "models/ddos_model.pkl"
    TEST_PATH  = "models/ddos_test.pkl"

    # 1) Verifica se os artefatos existem
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_PATH):
        st.error(
            "🚫 Modelo não encontrado.\n"
            "Execute localmente: python train_ddos_model.py\n"
            "para gerar em models/: ddos_model.pkl e ddos_test.pkl"
        )
    else:
        # 2) Carrega modelo e resultados de teste
        model     = load_model(MODEL_PATH)
        test_data = joblib.load(TEST_PATH)
        m         = test_data["metrics"]
        X_test    = test_data["X_test"]
        y_test    = test_data["y_test"]

        # random forest
        from sklearn.tree import plot_tree

        # Exibir a primeira árvore da floresta (índice 0)
        tree_index = 0
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model.estimators_[tree_index], 
                feature_names=feature_names, 
                class_names=["Normal", "DDoS"], 
                filled=False, ax=ax)
        st.pyplot(fig)

        # 3) Matriz de Confusão Normalizada (Plotly)
        import plotly.figure_factory as ff
        import numpy as np

        cm = m["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
        fig_norm = ff.create_annotated_heatmap(
            z=cm_norm,
            x=["Pred Normal","Pred DDoS"],
            y=["True Normal","True DDoS"],
            annotation_text=cm.astype(int),
            colorscale="Blues",
            showscale=True
        )
        fig_norm.update_layout(title="Matriz de Confusão (normalizada)")
        st.plotly_chart(fig_norm, use_container_width=True)

        # 5) Importância das Features (Random Forest)
        orig_feats = [c for c in data.columns if c not in ['target', 'Source IP', 'Dest IP']]
        dummy = pd.get_dummies(data[orig_feats], drop_first=True)
        colnames = dummy.columns.tolist()

        feat_imp = model.feature_importances_

        # ser = pd.Series(feat_imp, index=colnames).sort_values

        ser = pd.Series(feat_imp, index=colnames).sort_values(ascending=True)

        import plotly.express as px
        fig_fi = px.bar(
            ser,
            orientation="h",
            title="Importância das Features",
            labels={'index': 'Feature','value':'Importancia'}
        )
        st.plotly_chart(fig_fi, use_container_width=True)
            # 6) Exemplo de predição on-the-fly
        if st.sidebar.checkbox("Fazer predição de exemplo"):
            st.sidebar.write("📝 Insira valores na mesma ordem do teste:")
            # Determina quantas features existem
            if hasattr(X_test, "columns"):
                cols = list(X_test.columns)
                st.sidebar.write("Ordem de features:", cols)
                n_feats = len(cols)
            else:
                n_feats = X_test.shape[1]

            sample = []
            for i in range(n_feats):
                val = st.sidebar.text_input(f"Feature {i}", "0")
                try:
                    sample.append(float(val))
                except ValueError:
                    sample.append(0.0)

            # Gera predição
            pred = model.predict([sample])[0]
            st.markdown(f"**Predição:** {'🚨 DDoS' if pred == 1 else '✅ Normal'}")

if opcao == "Pizza":
    # Contagem de valores na coluna Target
    target_counts = data['target'].value_counts()

    # Nomes das categorias e proporções
    labels = ['Tráfego Normal', 'Ataques DDoS']
    sizes = [target_counts.get(0, 0), target_counts.get(1, 0)]

    # Criar o gráfico de pizza usando Plotly
    fig = px.pie(
        names=labels,
        values=sizes,
        color=labels,
        color_discrete_map={'Tráfego Normal': '#66b3ff', 'Ataques DDoS': '#ff9999'},
        title="Proporção de Tráfego Normal vs DDoS"
    )

    # Ajustar o layout do gráfico
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.1, 0]  # Destacar a fatia do tráfego normal
    )
    fig.update_layout(
        title_x=0.5,  # Centralizar o título
        showlegend=False  # Ocultar a legenda, já que as labels estão no gráfico
    )

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif opcao == "Barras":
    st.header("Média de Packets/Time: Tráfego Normal VS DDOS")

    # Calcular a média de Packets/Time por Target
    mean_packets_time = data.groupby('target')['Packets/Time'].mean().reset_index()
    mean_packets_time.columns = ['Target', 'Mean Packets/Time']

    # Mapear os nomes das categorias
    mean_packets_time['Category'] = mean_packets_time['Target'].map({0: 'Tráfego Normal', 1: 'Ataques DDoS'})

    # Criar o gráfico de barras
    fig = px.bar(
        mean_packets_time,
        x='Category',
        y='Mean Packets/Time',
        text='Mean Packets/Time',
        color='Category',
        color_discrete_map={'Tráfego Normal': '#66b3ff', 'Ataques DDoS': '#ff9999'},
        title='Intensidade do Tráfego por Categoria',
        height=500
    )

    # Ajustar layout
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Categoria de Tráfego",
        yaxis_title="Média de Packets/Time",
        title_x=0.5,  # Centralizar o título
        showlegend=False  # Ocultar legenda, pois as categorias já estão no eixo X
    )

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif opcao == "Histograma":
    st.header("Distribuição de Packets/Time por Tipo de Tráfego")
    # Criar histograma interativo
    fig = px.histogram(
        data, 
        x='Packets/Time', 
        color='target', 
        nbins=50, 
        labels={'color': 'Tráfego'},
        title="Distribuição de Packets/Time por Tipo de Tráfego",
        color_discrete_map={0: "blue", 1: "red"}
    )

    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

elif opcao == "Protocol":
    # Filtro para o usuário selecionar o protocolo
    protocols = data['Transport Layer'].unique()
    selected_protocols = st.multiselect(
        "Selecione os protocolos:",
        options=protocols,
        default=protocols  # Seleciona todos por padrão
    )

    # Filtrar os dados com base nos protocolos selecionados
    filtered_data = data[data['Transport Layer'].isin(selected_protocols)]

    # Agregar os dados para reduzir o volume
    aggregated_data = filtered_data.groupby(['Transport Layer', 'target']).size().reset_index(name='Count')

    # Criar gráfico de barras empilhado
    fig = px.bar(
        aggregated_data,
        x='Transport Layer',
        y='Count',
        color='target',
        barmode='stack',
        labels={'target': 'Tráfego', 'Transport Layer': 'Protocolo', 'Count': 'Contagem'},
        title="Contagem de Tráfego por Protocolo e Categoria",
        color_discrete_map={0: "blue", 1: "red"}
    )

    # Ajustar layout
    fig.update_layout(
        xaxis_title="Protocolo",
        yaxis_title="Contagem",
        legend_title="Categoria de Tráfego",
        bargap=0.2
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif opcao == "Top 10 IP's":
    # Filtrar apenas os tráfegos de ataque (Target = 1)
    attack_data = data[data['target'] == 1]

    # Contar a frequência de cada IP de destino
    dest_ip_counts = attack_data['Dest IP'].value_counts().head(10).reset_index()
    dest_ip_counts.columns = ['Dest IP', 'Count']

    # Criar o gráfico de barras
    fig = px.bar(
        dest_ip_counts,
        x='Dest IP',
        y='Count',
        text='Count',
        labels={'Dest IP': 'IP de Destino', 'Count': 'Número de Ataques'},
        title="Top 10 IPs de Destino Mais Atacados",
        color='Count',
        color_continuous_scale='Reds',
        height=500
    )

    # Ajustar layout
    fig.update_layout(
        xaxis_title="IP de Destino",
        yaxis_title="Número de Ataques",
        title_x=0.5,  # Centralizar o título
        margin=dict(t=100)
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)
elif opcao == "Machine Learning":
    st.header("▶ Resultados do Modelo DDoS (pré-treinado)")

    MODEL_PATH = "models/ddos_model.pkl"
    TEST_PATH  = "models/ddos_test.pkl"

    # 1) Verifica se o modelo e os dados de teste existem
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_PATH):
        st.error("Modelo ainda não treinado. Rode train_ddos_model.py para gerar os arquivos em models/.")
    else:
        # 2) Carrega modelo e resultados de teste
        model     = load_model(MODEL_PATH)
        test_data = joblib.load(TEST_PATH)
        m         = test_data['metrics']
        X_test    = test_data['X_test']
        y_test    = test_data['y_test']

        # 3) Exibe métricas
        st.subheader(f"Acurácia no teste: {m['accuracy']:.2f}")
        st.text(m['classification_report'])

        # 4) Matriz de Confusão
        cm = m['confusion_matrix']
        fig_cm, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i,j], ha='center', va='center')
        ax.set_xticks([0,1]); ax.set_xticklabels(['Normal','DDoS'])
        ax.set_yticks([0,1]); ax.set_yticklabels(['Normal','DDoS'])
        ax.set_title("Matriz de Confusão")
        st.pyplot(fig_cm)

        # 5) Curva ROC (se existir)
        if 'roc_curve' in m:
            fpr, tpr = m['roc_curve']
            fig_roc, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {m['auc']:.2f}")
            ax2.plot([0,1], [0,1], '--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig_roc)

        # 6) Exemplo de predição de um novo registro
        if st.sidebar.checkbox("Fazer predição de exemplo"):
            # cria um dicionário de exemplo com as mesmas features usadas no treino
            exemplo = {}
            for feat in test_data['X_test'].columns:
                exemplo[feat] = st.sidebar.text_input(feat, "0")
            df_ex = pd.DataFrame([exemplo]).astype(float)
            pred = model.predict(df_ex)[0]
            st.write("Predição:", "DDoS" if pred==1 else "Normal")

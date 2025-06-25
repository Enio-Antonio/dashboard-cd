import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered", page_title="DDOS")

st.title('Análise de ataques DDOS')

# Carregar dados
data = pd.read_csv('DDoS_dataset.csv')

# Sidebar
st.sidebar.title("Navegação")
opcao = st.sidebar.selectbox("Selecione o gráfico:", ["Pizza", "Barras", "Sunburst Chart"])

if opcao == "Pizza":
    st.header("Tráfego orgânico VS DDOS")
    
    # Contagem de valores na coluna Target
    target_counts = data['target'].value_counts()
    
    # Nomes das categorias
    labels = ['Tráfego Normal', 'Ataques DDoS']
    sizes = [target_counts.get(0, 0), target_counts.get(1, 0)]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # Destacar a fatia do tráfego normal
    
    # Criar o gráfico de pizza
    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')  # Garantir que o gráfico seja um círculo
    
    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)

elif opcao == "Barras":
    st.header("Média de Packets/Time: Tráfego Normal VS DDOS")
    
    # Calcular a média de Packets/Time por Target
    mean_packets_time = data.groupby('target')['Packets/Time'].mean()
    
    # Nomes das categorias
    labels = ['Tráfego Normal', 'Ataques DDoS']
    means = [mean_packets_time.get(0, 0), mean_packets_time.get(1, 0)]
    colors = ['#66b3ff', '#ff9999']
    
    # Criar o gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, means, color=colors, alpha=0.7)
    ax.set_ylabel('Média de Packets/Time')
    ax.set_title('Intensidade do Tráfego por Categoria')
    
    # Adicionar valores nas barras
    for i, v in enumerate(means):
        ax.text(i, v + max(means) * 0.02, f"{v:.2f}", ha='center', fontsize=10)
    
    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)
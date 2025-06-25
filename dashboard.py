import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="centered", page_title="DDOS")

st.title('Análise de ataques DDOS')

# Carregar dados
data = pd.read_csv('DDoS_dataset.csv')

# Sidebar
st.sidebar.title("Navegação")
opcao = st.sidebar.selectbox("Selecione o gráfico:", ["Pizza", "Barras", "Histograma", "Protocol"])

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
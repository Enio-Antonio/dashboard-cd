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
opcao = st.sidebar.selectbox("Selecione o gráfico:", ["Pizza", "Barras", "Histograma", "Protocol", "Top 10 IP's"])

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
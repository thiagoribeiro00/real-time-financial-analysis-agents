from crewai import Agent, Task, Crew
from agents import (
    data_historian_agent,
    market_sentiment_monitor_agent,
    macroeconomic_risk_assesor_agent,
)

historical_data_search_task = Task(
    description="""
    - Coleta dados históricos de preços de {empresa} de fontes confiáveis.
    - Utiliza APIs de exchanges e outras fontes para obter dados precisos e atualizados.
    - Armazena os dados em um formato estruturado para análise posterior.
    - É reconhecido por sua habilidade em coletar dados de alta qualidade e sua capacidade de lidar com grandes volumes de dados.""",
    expected_output="""
    Explique detalhadamente a {empresa} e seja conciso em sua explicação .
    """,
    agent=data_historian_agent,
)

market_sentiment_task = Task(
    description="""
    - Recebe os dados coletados pelo Agente de Coleta de Dados.
    - Limpa e trata os dados, lidando com valores ausentes, outliers e inconsistências.
    - Cria recursos (features) relevantes com base nos dados de preços.
    - Divide os dados em conjuntos de treinamento, validação e teste.
    - Normaliza ou padroniza os dados, se necessário.
    - Armazena os dados pré-processados em formato adequado para a modelagem.
    - É reconhecido por sua habilidade em lidar com dados complexos e sua capacidade de criar recursos úteis para a modelagem.""",
    expected_output="""
    Based on news and market information, analyze the general feeling about the {empresa}
    """,
    agent=market_sentiment_monitor_agent,
)

risk_assesor_task = Task(
    description="""
    - Recebe os dados pré-processados do Agente de Pré-processamento de Dados.
    - Carrega os dados no Autogluon e define a tarefa de regressão para previsão de preços.
    - Configura os hiperparâmetros do Autogluon e treina os modelos.
    - Avalia o desempenho dos modelos usando o conjunto de validação.
    - Ajusta os hiperparâmetros e retreina os modelos, se necessário.
    - Faz previsões de preços futuros usando o conjunto de teste.
    - Armazena os resultados das previsões para análise posterior.
    - É reconhecido por sua habilidade em treinar modelos complexos e sua capacidade de prever com precisão os preços futuros.""",
    expected_output="""
    A risk assessment report for {empresa} with:
        - Explanation of 1-year graph analysis
        - Risks of investing and why invest
        - Final decision (Buy or Not)
    """,
    agent=macroeconomic_risk_assesor_agent,
    output_file="stock_risk.md",
)
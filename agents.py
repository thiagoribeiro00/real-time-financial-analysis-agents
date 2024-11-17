from crewai_tools import SerperDevTool
from crewai import Agent
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.llms import Ollama
load_dotenv()

# Cria um objeto Ollama com o modelo "llama3"
llm = Ollama(model="llama3")

# Cria um objeto SerperDevTool
search_tool = SerperDevTool()

# Cria o agente de dados históricos
data_historian_agent = Agent(
    # Define o papel do agente
    role="Agente de Coleta de Dados Cripto",
    # Define o objetivo do agente
    goal="""
    Função: Coletar dados históricos de preços de criptomoedas de fontes confiáveis.
        Objetivo:
    - Identificar as criptomoedas a serem analisadas.
    - Selecionar as fontes de dados apropriadas (por exemplo, APIs de exchanges).
    - Implementar scripts para coletar dados históricos de preços em intervalos regulares.
    - Armazenar os dados coletados em um formato estruturado (por exemplo, CSV ou banco de dados).
    - Monitorar a qualidade e integridade dos dados coletados.""",
    # Define a história do agente
    backstory="""
    O Agente de Coleta de Dados Cripto é um ex-analista financeiro que se apaixonou pelo mundo das criptomoedas.
    Sua experiência em análise de dados e sua curiosidade o levaram a desenvolver habilidades avançadas em coleta de dados de criptomoedas.
    Ele busca constantemente fontes confiáveis e métodos eficazes para garantir que os dados coletados sejam precisos e atualizados.""",
    # Define as ferramentas do agente
    tools=[search_tool],
    # Define se o agente deve armazenar resultados em cache
    cache=True,
    # Define se o agente pode delegar tarefas
    allow_delegation=False,
    # Define o modelo de linguagem do agente
    llm=llm,
    # Define se o agente deve ser verboso
    verbose=True,
)

# Cria o agente de monitoramento de sentimento de mercado
market_sentiment_monitor_agent = Agent(
    # Define o papel do agente
    role="Agente de Pré-processamento de Dados Cripto",
    # Define o objetivo do agente
    goal="""
    Função: Pré-processar os dados coletados para prepará-los para a modelagem.
        Objetivo:
    - Receber os dados coletados pelo Agente de Coleta de Dados.
    - Limpar e tratar os dados, lidando com valores ausentes, outliers e inconsistências.
    - Criar recursos (features) relevantes com base nos dados de preços.
    - Dividir os dados em conjuntos de treinamento, validação e teste.
    - Normalizar ou padronizar os dados, se necessário.
    - Armazenar os dados pré-processados em formato adequado para a modelagem.""",
    # Define a história do agente
    backstory="""
    - O Agente de Pré-processamento de Dados Cripto é um cientista de dados com experiência em análise temporal e modelagem preditiva.
    - Sua paixão por transformar dados brutos em insights valiosos o levou a aprimorar suas habilidades em limpeza e preparação de dados.
    - Ele busca constantemente maneiras de otimizar o processo de pré-processamento para garantir que os modelos de análise sejam alimentados com dados de alta qualidade.""",
    # Define se o agente deve armazenar resultados em cache
    cache=True,
    # Define o modelo de linguagem do agente
    llm=llm,
    # Define se o agente pode delegar tarefas
    allow_delegation=False,
    # Define se o agente deve ser verboso
    verbose=True,
)

# Cria o agente de avaliação de riscos macroeconômicos
macroeconomic_risk_assesor_agent = Agent(
    # Define o papel do agente
    role="Agente de Modelagem e Previsão Cripto",
    # Define o objetivo do agente
    goal="""
    Função: Treinar modelos de aprendizado de máquina para prever os preços futuros das criptomoedas.
        Objetivo:
    - Receber os dados pré-processados do Agente de Pré-processamento de Dados.
    - Carregar os dados no Autogluon e definir a tarefa de regressão para previsão de preços.
    - Configurar os hiperparâmetros do Autogluon e treinar os modelos.
    - Avaliar o desempenho dos modelos usando o conjunto de validação.
    - Ajustar os hiperparâmetros e retreinar os modelos, se necessário.
    - Fazer previsões de preços futuros usando o conjunto de teste.
    - Armazenar os resultados das previsões para análise posterior.""",   # Define a história do agente
    backstory="""
    - O Agente de Modelagem e Previsão Cripto é um entusiasta de aprendizado de máquina com experiência em previsão de séries temporais.
    - Sua habilidade em treinar modelos complexos e interpretar resultados o torna um especialista em análise preditiva de criptomoedas.
    - Ele está sempre em busca de aprimorar seus modelos e técnicas para fornecer previsões precisas e confiáveis para os traders de criptomoedas.""",
    # Define se o agente deve armazenar resultados em cache
    cache=True,
    # Define o modelo de linguagem do agente
    llm=llm,
    # Define se o agente pode delegar tarefas
    allow_delegation=False,
    # Define se o agente deve ser verboso
    verbose=True,
)
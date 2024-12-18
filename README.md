# Real-Time Financial Analysis Agents

Este projeto é uma solução inovadora que utiliza agentes baseados em inteligência artificial para coletar, processar e analisar dados financeiros em tempo real. O objetivo principal é fornecer informações detalhadas e insights úteis para investidores, analistas financeiros e empresas tomarem decisões estratégicas mais informadas.

Por meio de uma arquitetura modular, os agentes realizam tarefas específicas, como análise de tendências históricas, monitoramento de sentimentos de mercado e avaliação de riscos macroeconômicos. Esta ferramenta combina aprendizado de máquina, processamento de linguagem natural (NLP) e técnicas avançadas de análise de dados para oferecer resultados confiáveis e acionáveis.

---

## Por que este projeto é importante? 

No cenário financeiro atual, onde decisões rápidas e bem-informadas são essenciais, ter acesso a análises precisas e em tempo real pode ser a diferença entre aproveitar uma oportunidade ou sofrer perdas. Este projeto automatiza processos que tradicionalmente exigiriam equipes inteiras de analistas, aumentando a eficiência e reduzindo os riscos de erro humano.

## Como funciona?

O sistema utiliza CrewAI, uma estrutura para criar e gerenciar agentes de inteligência artificial que realizam tarefas de forma sequencial ou paralela.

1. Entrada do Usuário: O nome da empresa é fornecido como entrada.
2- Execução das Tarefas:
  - **Coleta de dados históricos.** 
  - **Monitoramento de sentimentos de mercado.**
  - **Avaliação de riscos macroeconômicos.**
3. Geração de Resultados:
  - **Relatórios detalhados contendo previsões, tendências e recomendações.**
  - **Tradução automática dos resultados para português.**
Os resultados são salvos em arquivos Markdown (.md) para fácil leitura e compartilhamento.

---

## **Funcionalidades**

- **Coleta de Dados Históricos:**Busca informações sobre preços de ações, volumes de negociação e outros indicadores financeiros relevantes em fontes confiáveis, como Yahoo Finance e Google Finance. Identifica tendências com base nos últimos cinco anos de dados.
  
- **Monitoramento de Sentimento do Mercado:** Analisa dados em tempo real de notícias financeiras e redes sociais para avaliar o sentimento predominante (positivo, negativo ou neutro) sobre uma empresa. Alerta sobre mudanças significativas no sentimento que possam indicar oportunidades ou riscos.

- **Avaliação de Riscos Macroeconômicos:** Examina dados econômicos globais, como PIB, inflação e eventos políticos, para prever como fatores externos podem impactar o desempenho financeiro de uma empresa. Gera relatórios com recomendações de investimento, avaliando riscos e benefícios.

---

## **Arquitetura do Projeto**

O projeto é baseado em três principais agentes:

1. **Data Historian Agent (Analista de Mercado Financeiro):**
   - Coleta dados históricos.
   - Identifica padrões e tendências para decisões de investimento.
   - Fornece previsões baseadas em aprendizado de máquina.

2. **Market Sentiment Monitor Agent (Monitor de Sentimento de Mercado):**
   - Analisa dados de notícias e redes sociais.
   - Avalia o sentimento geral do mercado.
   - Gera alertas sobre mudanças abruptas no sentimento.

3. **Macroeconomic Risk Assessor Agent (Avaliador de Riscos Macroeconômicos):**
   - Avalia riscos econômicos e políticos que podem afetar os mercados financeiros.
   - Gera relatórios com recomendações de investimento.

---

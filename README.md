# Dashboard de Monitoramento Climático e Gestão de Abrigos
### Projeto para a Global Solution 2025 - FIAP

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

Um dashboard multifuncional desenvolvido em Python com a biblioteca Tkinter. Esta aplicação foi criada para oferecer uma solução integrada de monitoramento ambiental, previsão de temperatura com Machine Learning e gestão de recursos de emergência, como abrigos.

## 🌟 Principais Funcionalidades

O projeto é dividido em quatro abas principais, cada uma com uma finalidade específica:

#### 📊 Dashboard de Sensores em Tempo Real
- **Visualização Intuitiva:** Mostra dados simulados de sensores (temperatura, umidade e ocupação) através de medidores (gauges) dinâmicos.
- **Alertas Visuais:** Os medidores mudam de cor (de verde para vermelho) quando os valores ultrapassam um limite de alerta pré-definido, facilitando a rápida identificação de anomalias.
- **Atualização Automática:** Os dados são atualizados em tempo real a cada 2 segundos.

#### 🧠 Previsão de Temperatura com Machine Learning
- **Modelo Preditivo:** Utiliza um modelo de `RandomForestRegressor` (Scikit-learn) treinado com dados climáticos históricos de Delhi (do Kaggle) para prever a temperatura média.
- **Interface Interativa:** Permite que o usuário insira manualmente os dados (umidade, pressão, dia do ano, etc.) para gerar uma previsão.
- **Visualização de Dados:** Plota a temperatura prevista em um gráfico Matplotlib, comparando-a com os dados históricos para dar contexto à previsão.

#### 🏠 Simulador de Gestão de Abrigos
- **Busca Inteligente:** Ajuda a encontrar o melhor abrigo com base em critérios definidos pelo usuário, como número de pessoas necessitadas, recursos essenciais (água, comida, etc.) e nível de segurança desejado.
- **Recomendação Otimizada:** Utiliza um algoritmo de pontuação para recomendar o abrigo mais adequado, priorizando a segurança e a capacidade disponível.
- **Simulação Dinâmica:** Permite simular mudanças na ocupação dos abrigos para testar a resiliência do sistema de gestão.

#### 🔗 Links Úteis
- **Acesso Rápido:** Uma seção centralizada com botões que direcionam para recursos externos importantes, como o repositório do projeto, modelos de machine learning e outros links relevantes.

## 🛠️ Tecnologias e Bibliotecas

Este projeto foi construído utilizando as seguintes tecnologias e bibliotecas Python:

- **Python 3.x**
- **Tkinter (ttk.Notebook):** Para a construção de toda a interface gráfica do usuário (GUI), incluindo a navegação por abas.
- **Pandas:** Para o carregamento, manipulação e pré-processamento do dataset climático.
- **NumPy:** Para operações numéricas eficientes, especialmente durante o treinamento do modelo.
- **Scikit-learn:** Para a criação e treinamento do modelo de Machine Learning (`RandomForestRegressor`).
- **Matplotlib:** Para a incorporação de gráficos e visualizações de dados diretamente na interface Tkinter.
- **Webbrowser:** Para abrir links externos no navegador padrão do usuário.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/BrunoAraujo84/ERC20-Token-Newcoin/blob/main/LICENSE) | ![GitHub top language](https://img.shields.io/github/languages/top/BrunoAraujo84/ERC20-Token-Newcoin) | ![GitHub last commit](https://img.shields.io/github/last-commit/BrunoAraujo84/ERC20-Token-Newcoin) | ![Contribuições bem-vindas](https://img.shields.io/badge/contribuições-bem_vindas-brightgreen.svg?style=flat)

# Modelo de Previsão de Preço de Token

## Descrição
Este repositório contém um script Python desenvolvido por Bruno Araujo para prever os preços do token usando várias técnicas de aprendizado de máquina e séries temporais. Foi utilizado como exemplo o token MATIC.

## Bibliotecas e Ferramentas Utilizadas
- **numpy**: Manipulação numérica e álgebra linear.
- **pandas**: Manipulação e análise de dados.
- **matplotlib**: Criação de gráficos e visualizações.
- **pycoingecko**: API para obter dados de criptomoedas.
- **keras_turner**: Técnicas de otimização para Hiperparâmetros
- **tensorflow.keras**: Construção de modelos de redes neurais.
- **sklearn.preprocessing**: Pré-processamento e escalonamento de dados.
- **fredapi**: API para obter dados econômicos do FRED

## Modelos Utilizados

### Redes Neurais
- Usamos o framework TensorFlow para implementar um modelo de rede neural com camadas LSTM (Long Short-Term Memory) para capturar a temporalidade nos dados.
- Otimizador: Adam

### Gradient Boosting
- Modelo de boosting para criar um modelo forte a partir de vários modelos de árvores de decisão.
- Utilizado para capturar relações não-lineares nos dados.

### ARIMA (AutoRegressive Integrated Moving Average)
- Modelo de série temporal usado para prever valores futuros com base em valores passados.
- Utilizado para capturar sazonalidade e tendências nos dados.

### Otimização de Hiperparâmetros
- Usamos técnicas de otimização para encontrar os melhores hiperparâmetros para os modelos acima.

## Técnicas Estatísticas

As seguintes técnicas estatísticas e indicadores são usados para enriquecer o conjunto de dados e melhorar a eficácia dos modelos:

### RSI (Relative Strength Index)
- Índice de Força Relativa, utilizado para identificar condições de sobrecompra ou sobrevenda no preço de um ativo.

### Bollinger Bands
- **Bollinger_Mean**: Média móvel.
- **Bollinger_Upper**: Banda superior de Bollinger, usada para identificar sobrecompra.
- **Bollinger_Lower**: Banda inferior de Bollinger, usada para identificar sobrevenda.

### VWAP (Volume Weighted Average Price)
- Média Ponderada pelo Volume do Preço, usada para identificar a tendência de preço com base no volume negociado.

### MACD (Moving Average Convergence Divergence)
- Utilizado para identificar mudanças na força, direção, momentum e duração de uma tendência em um preço de ativo.

### Moving Average
- Média móvel dos preços, usada para identificar tendências.

### Weighted Moving Average
- Média móvel ponderada, dá mais peso aos preços mais recentes.

### Volatility
- Medida da variação do preço de um ativo ao longo do tempo.

### Garch Fitted
- Modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity) ajustado, usado para modelar a volatilidade do ativo.

## Dados e Fonte de Origem

Os dados utilizados para treinar os modelos e fazer as previsões são retirados da API CoinGecko. A API fornece dados históricos dos preços do token, que são então utilizados para treinar os modelos. Alguns dados utilizados são:
- **Marcketcap**: Volume de mercado de criptmoedas.
- **Preço Histórico**: Preço histórico como referencia: BTC, ETH e BNB além do token escolhido pelo usuário para relizar a projeção do valor.
- **Taxa de Inflação**: É resgatado o histórico de inflação da economia americana. A taxa de inflação é usada para ajustar o preço do ativo ao longo do tempo.
- **Taxa de Juros**: Taxa de juro americano é usada para descontar os fluxos de caixa futuros para o valor presente.

Este modelo tem como carater de estudo apenas.
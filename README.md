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

## Dados e Fonte de Origem

Os dados utilizados para treinar os modelos e fazer as previsões são retirados da API CoinGecko. A API fornece dados históricos dos preços do token, que são então utilizados para treinar os modelos. Alguns dados utilizados são:
- **Marcketcap**: Volume de mercado de criptmoedas.
- **Preço Histórico**: Preço histórico como referencia: BTC, ETH e BNB além do token escolhido pelo usuário para relizar a projeção do valor.
- **Inflação**: É resgatado o histórico de inflação da economia americana.

Este modelo tem como carater de estudo apenas.
# Bibliotecas para serem importadas no Google Colab ou Jupyter
# !pip install pycoingecko fredapi ta keras-tuner arch pmdarima numba
import numpy as np  # Biblioteca para manipulação numérica e álgebra linear
import pandas as pd  # Biblioteca para manipulação e análise de dados
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos e visualizações
import matplotlib.dates as mdates  # Ferramentas para trabalhar com datas em gráficos
from pycoingecko import CoinGeckoAPI  # API para obter dados de criptomoedas
import datetime as dt  # Biblioteca para trabalhar com datas e horas
from datetime import datetime
import time  # Funções para trabalhar com tempo
from tensorflow.keras.models import Sequential  # Modelo sequencial para construção de redes neurais
from tensorflow.keras.layers import Dense, LSTM, Dropout  # Camadas para redes neurais
from sklearn.preprocessing import MinMaxScaler  # Pré-processamento e escalonamento de dados
from tensorflow.keras import regularizers  # Regularizadores para redes neurais
from tensorflow.keras.optimizers.legacy import Adam  # Otimizador Adam para treinamento de redes neurais
from keras_tuner import RandomSearch  # Otimização de hiperparâmetros com pesquisa aleatória
from keras_tuner.engine.hyperparameters import HyperParameters  # Manipulação de hiperparâmetros
from tqdm import tqdm  # Barra de progresso para visualização do progresso de loops
import tensorflow as tf  # TensorFlow para construção e treinamento de modelos de aprendizado profundo
import os  # Manipulação de arquivos e diretórios
import warnings  # Controle de avisos e mensagens de erro
import shutil  # Operações de alto nível em arquivos e coleções de arquivos
import requests  # Fazer solicitações HTTP
from matplotlib import ticker  # Formatação e localização de ticks em gráficos
from tensorflow.keras.callbacks import \
    EarlyStopping  # Parada antecipada para interromper o treinamento quando o modelo não melhora
from sklearn.model_selection import TimeSeriesSplit  # Divisão de séries temporais para validação cruzada
from arch import arch_model  # Modelos ARCH e GARCH para modelagem de volatilidade
from sklearn.ensemble import RandomForestRegressor  # Modelo de regressão Random Forest
from sklearn.feature_selection import SelectFromModel  # Seleção de recursos com base em importâncias de recursos
from arch.__future__ import reindexing  # Ferramentas para reindexação de séries temporais
from ta.momentum import RSIIndicator  # Indicador de Força Relativa (RSI) para análise técnica
from fredapi import Fred  # API para obter dados econômicos do FRED
from keras.layers import Conv1D  # Camada convolucional 1D para aprendizado profundo
from pmdarima.arima import auto_arima  # Modelos ARIMA e seleção automática de hiperparâmetros
from sklearn.model_selection import train_test_split  # Divisão de dados em conjuntos de treinamento e teste
from sklearn.impute import SimpleImputer  # Imputação de valores ausentes em dados
from sklearn.metrics import mean_absolute_error  # Métrica de erro absoluto médio para avaliação de modelos
from keras.layers import TimeDistributed, Conv1D, Flatten  # Camadas adicionais para redes neurais
import keras_tuner as kt  # Biblioteca para otimização de hiperparâmetros de modelos de aprendizado profundo
from tensorflow.keras.models import load_model  # Função para carregar modelos salvos
import random  # Funções para gerar números aleatórios
import shutil  # Operações de alto nível em arquivos e coleções de arquivos
from sklearn.preprocessing import \
    StandardScaler  # O StandardScaler é utilizado para padronizar os recursos removendo a média e escalonando-os para a variância da unidade
from sklearn.linear_model import \
    Ridge  # Ridge é um modelo de regressão linear com regularização L2 que minimiza a soma dos resíduos ao quadrado com penalidade no tamanho dos coeficientes
from sklearn.ensemble import \
    GradientBoostingRegressor  # GradientBoostingRegressor é um modelo de aprendizado de máquina baseado em árvores de decisão que utiliza o algoritmo de boosting para minimizar o erro
from numba import jit, njit # Biblioteca que auxilia na aceleração da execução do código Python


# Configuração de variáveis de ambiente para modificar o comportamento de bibliotecas e frameworks
os.environ[
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Definir a implementação do Protocol Buffers para Python
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # Definir o nível mínimo de log do TensorFlow para exibir apenas erros e mensagens fatais

# Ignorar avisos
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Removendo o diretório 'random_search_lstm' do projeto para garantir que não haja conflitos ao executar a pesquisa
# de hiperparâmetros
shutil.rmtree('random_search_lstm', ignore_errors=True)

# Este trecho irá configurar o log do TensorFlow para mostrar apenas mensagens de erro, suprimindo avisos e outras
# mensagens menos importantes.
tf.get_logger().setLevel('ERROR')

# VARIÁVEIS GLOBAL
# MATIC - Rede BNB Chain
contratoToken = '0xcc42724c6683b7e57334c4e856f4c9965ed682bd'
combinacaoMaxima = 400
combinacaoTestada = 75
diasProjetar = 30
nomeToken = ''

# Classe para apresentar o progresso em barras do treinamento do modelo LSTM de acordo com o ajuste de
# hiperparâmetros usando o Keras Tuner
# Classe TqdmCallback herda de tf.keras.callbacks.Callback para criar uma barra de progresso personalizada
class TqdmCallback(tf.keras.callbacks.Callback):
    # Método chamado no início do treinamento
    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]  # Atribui o número de épocas
        # Inicializa a barra de progresso tqdm
        self.progress_bar = tqdm(total=self.epochs, desc="Tuner Progress", mininterval=1)

    # Método chamado no final de cada época
    def on_epoch_end(self, epoch, logs=None):
        # Atualiza a barra de progresso
        self.progress_bar.update(1)

    # Método chamado no final do treinamento
    def on_train_end(self, logs=None):
        self.progress_bar.close()  # Fecha a barra de progresso


# Função para limpar o diretório do Keras Tuner
def clear_tuner_directory(directory):
    try:
        shutil.rmtree(directory)  # Remove o diretório e todo o seu conteúdo
    except FileNotFoundError:
        # Caso o diretório não exista, imprime a mensagem de aviso
        print(f"O diretório {directory} não existe.")
    else:
        # Se o diretório for removido com sucesso, imprime a mensagem de confirmação
        print(f"Diretório {directory} limpo com sucesso.")


# As funções abaixo calculam diversos indicadores técnicos para análise financeira a partir de dados de preços e volume.
# As funções recebem os dados e uma janela de tempo como argumentos e retornam os valores calculados dos indicadores
# para cada ponto no tempo.
# Função para calcular a média móvel simples
def calculate_moving_average(data, window):
    return data['Price'].rolling(window=window).mean()


# Função para calcular o Índice de Força Relativa (RSI)
def calculate_rsi(data, window):
    rsi_indicator = RSIIndicator(data['Price'], window)
    return rsi_indicator.rsi()


# Função para calcular a Convergência/Divergência de Médias Móveis (MACD) e o sinal do MACD
def calculate_macd(data, short_window, long_window, signal_window):
    ema_short = data['Price'].ewm(span=short_window).mean()
    ema_long = data['Price'].ewm(span=long_window).mean()

    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window).mean()

    return macd, macd_signal


# Função para calcular a média móvel ponderada
def weighted_moving_average(data, window):
    weights = np.arange(1, window + 1)
    return data['Price'].rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# Função para calcular as Bandas de Bollinger
def calculate_bollinger_bands(data, window):
    rolling_mean = data['Price'].rolling(window=window).mean()
    rolling_std = data['Price'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band


# Função para calcular o Preço Médio Ponderado pelo Volume (VWAP)
def calculate_volume_weighted_average_price(data, window):
    price_times_volume = data['Price'] * data['Volume']
    sum_price_times_volume = price_times_volume.rolling(window=window).sum()
    sum_volume = data['Volume'].rolling(window=window).sum()
    vwap = sum_price_times_volume / sum_volume
    return vwap


# Função para obter os dados históricos de preços, market cap e volume de um token utilizando a CoinGecko API
def fetch_data(contract_address, start_date, end_date, sleep_time=10, max_retries=5):
    # Conectando com a API da CoinGecko e resgatando os dados
    cg = CoinGeckoAPI()
    token_info = cg.get_coin_info_from_contract_address_by_id(id='binance-smart-chain',
                                                              contract_address=contract_address)
    coin_id = token_info['id']

    if contract_address == contratoToken:
      nomeToken = token_info['symbol'].upper()


    for i in range(max_retries):
        try:
            data = cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency='usd', from_timestamp=start_date,
                                                        to_timestamp=end_date)
            break
        except requests.exceptions.RetryError as e:
            if i < max_retries - 1:
                wait_time = sleep_time * (i + 1)
                print(
                    f"Erro ao tentar obter dados da API, tentativa {i + 1}. Esperando {wait_time} segundos antes da próxima tentativa.")
                time.sleep(wait_time)
            else:
                raise e

    # Obter dados do preço do token que será analisado e projetado seus valores
    prices = data['prices']
    price_data = pd.DataFrame(prices, columns=['Date', 'Price'])
    price_data['Date'] = pd.to_datetime(price_data['Date'], unit='ms')
    price_data = price_data.set_index('Date')

    # Obter dados do MarketCap do token e adicionar ao DataFrame
    market_caps = data['market_caps']
    market_cap_data = pd.DataFrame(market_caps, columns=['Date', 'Market_Cap'])
    market_cap_data['Date'] = pd.to_datetime(market_cap_data['Date'], unit='ms')
    market_cap_data = market_cap_data.set_index('Date')
    price_data = price_data.join(market_cap_data)

    # Obter dados do Volume do token e adicionar ao DataFrame
    volumes = data['total_volumes']
    volume_data = pd.DataFrame(volumes, columns=['Date', 'Volume'])
    volume_data['Date'] = pd.to_datetime(volume_data['Date'], unit='ms')
    volume_data = volume_data.set_index('Date')
    price_data = price_data.join(volume_data)

    # Calcular os indicadores técnicos e adicionar ao DataFrame
    price_data['Moving_Average'] = calculate_moving_average(price_data, window=120)
    price_data['RSI'] = calculate_rsi(price_data, window=120)
    price_data['MACD'], price_data['MACD_Signal'] = calculate_macd(price_data, short_window=100, long_window=120,
                                                                   signal_window=80)
    price_data['Bollinger_Mean'], price_data['Bollinger_Upper'], price_data[
        'Bollinger_Lower'] = calculate_bollinger_bands(price_data, window=120)
    price_data['VWAP'] = calculate_volume_weighted_average_price(price_data, window=120)
    price_data['Weighted_Moving_Average'] = weighted_moving_average(price_data, window=120)

    # Tempo de espera para não impactar a sincronização com a API
    time.sleep(sleep_time)
    # Retorna um DataFrame contendo dados históricos e indicadores técnicos para o token selecionado
    return price_data


# Função para obter os dados históricos de preços do Bitcoin utilizando a CoinGecko API
def fetch_bitcoin_data(start_date, end_date, sleep_time=10):
    # Conectando com a API da CoinGecko e resgatando os dados
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_range_by_id(id='bitcoin', vs_currency='usd', from_timestamp=start_date,
                                                to_timestamp=end_date)

    # Obter os dados do preço do Bitcoin
    prices = data['prices']
    price_data = pd.DataFrame(prices, columns=['Date', 'Price'])
    price_data['Date'] = pd.to_datetime(price_data['Date'], unit='ms')
    price_data = price_data.set_index('Date')

    # Tempo de espera para não impactar a sincronização com a API
    time.sleep(sleep_time)
    # Retorna um DataFrame contendo os dados históricos de preços do Bitcoin
    return price_data


# Verificar a importancia dos dados e remover aqueles que não faz sentido
# Função para calcular a importância das características usando um modelo de Random Forest
def feature_selection(X, y):
    # Treinar um modelo Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=60)
    rf.fit(X, y)

    # Obter a importância dos recursos
    feature_importances = rf.feature_importances_

    # Selecionar recursos importantes usando um limiar
    selection = SelectFromModel(rf, threshold=0.01, prefit=True)

    # Retorna a importância das características calculadas pelo modelo Random Forest
    return feature_importances


# Função para selecionar as características mais importantes com base na importância calculada
def select_important_features(X, feature_importances, X_columns, threshold=0.02):
    column_importances = {col: imp for col, imp in zip(X_columns, feature_importances)}
    # Adicionar uma condição para verificar se a coluna está no dicionário column_importances
    important_columns = [col for col in X_columns if col in column_importances and column_importances[col] > threshold]
    important_indices = [i for i, col in enumerate(X_columns) if
                         col in column_importances and column_importances[col] > threshold]
    X_important = X[:, important_indices]
    # Retorna um subconjunto de X contendo apenas as características importantes e a lista das colunas importantes
    return X_important, important_columns


# A função prepare_data tem como objetivo tratar o conjunto de dados, lidar com valores NaN, selecionar
# características importantes e criar sequências temporais para alimentar o modelo. Após o processamento,
# a função retorna X e y ajustados e prontos para serem utilizados no treinamento do modelo.
def prepare_data(data, garch_fitted, time_steps):
    # Reiniciar o índice e calcular o número de dias desde 1970-01-01 para cada data
    data = data.reset_index()
    data['Date'] = data['Date'].map(lambda x: (x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D'))
    # Extrair valores ajustados do modelo GARCH e adicionar ao DataFrame
    fitted_values = garch_fitted.conditional_volatility  # Extrai os valores ajustados do resultado do modelo GARCH
    data['Garch_Fitted'] = fitted_values
    # Extrair recursos e rótulos do conjunto de dados
    X = data[['Date', 'Market_Cap', 'Volume', 'BTC_Price', 'ETH_Price', 'BNB_Price', 'RSI', 'Bollinger_Mean',
              'Bollinger_Upper', 'Bollinger_Lower', 'VWAP', 'MACD', 'Moving_Average', 'Weighted_Moving_Average',
              'Volatility', 'Inflation_rate', 'Garch_Fitted', 'Interest_rate']].values
    y = data['Price'].values

    X_columns = ['Date', 'Market_Cap', 'Volume', 'BTC_Price', 'ETH_Price', 'BNB_Price', 'RSI', 'Bollinger_Mean',
                 'Bollinger_Upper', 'Bollinger_Lower', 'VWAP', 'MACD', 'Moving_Average', 'Weighted_Moving_Average',
                 'Volatility', 'Inflation_rate', 'Garch_Fitted', 'Interest_rate']

    # Tratar valores NaN
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Analisar os dados e selecionar apenas aqueles que são importantes
    feature_importances = feature_selection(X, y)

    # Selecione apenas os recursos mais importantes
    X_important, important_columns = select_important_features(X, feature_importances, X_columns)

    # Crie sequências temporais
    n_vars = X_important.shape[1]
    X_temporal = np.zeros((X_important.shape[0] - time_steps + 1, time_steps, n_vars))
    y_temporal = np.zeros(X_important.shape[0] - time_steps + 1)
    for i in range(X_important.shape[0] - time_steps + 1):
        X_temporal[i] = X_important[i:i + time_steps]
        y_temporal[i] = y[i + time_steps - 1]

    # Imprimir a forma de X e y
    print(f"Na função prepare_data, X shape: {X_temporal.shape}, y shape: {y_temporal.shape}")
    # Retornar X e y após o processamento
    return X_temporal, y_temporal


# A função create_sequences cria sequências temporais de entrada (X) e saída (y) com base no número
# de time steps desejado.
def create_sequences(X, y, time_steps):
    # Inicializar listas para armazenar as sequências de entrada e saída
    Xs, ys = [], []
    # Loop para criar sequências de entrada e saída
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :, 0]  # Ajuste esta linha para selecionar a dimensão correta.
        Xs.append(v)
        ys.append(y[i + time_steps - 1])

    # Converter listas para arrays numpy e retornar
    return np.array(Xs), np.array(ys)


# A função normalize_3d_data normaliza os dados em 3 dimensões usando o escalonador MinMax. Retorna a matriz de dados
# normalizada e a lista de escalonadores para cada recurso.
def normalize_3d_data(X):
    # Inicializar matriz de dados normalizada e lista de escalonadores
    X_norm = np.zeros_like(X)
    scalers = []

    # Loop para normalizar cada recurso
    for i in range(X.shape[2]):
        # Criar e ajustar o escalonador MinMax para o recurso atual
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_2d = X[:, :, i]
        X_reshaped = X_2d.reshape(-1, 1)
        scaler.fit(X_reshaped)
        # Transformar os dados do recurso atual e atualizar a matriz normalizada
        X_norm[:, :, i] = scaler.transform(X_reshaped).reshape(X_2d.shape)
        # Adicionar o escalonador à lista de escalonadores
        scalers.append(scaler)

    # # Retornar a matriz de dados normalizada e a lista de escalonadores
    return X_norm, scalers


# A função scale_3d_data dimensiona os dados em 3 dimensões usando o escalonador MinMax para cada recurso e cada time
# step. Retorna a matriz de dados escalonados e a lista de escalonadores para cada recurso.
@njit
def scale_3d_data(X):
    # Inicializar a matriz de dados escalonados e a lista de escalonadores
    num_samples = X.shape[0]
    num_time_steps = X.shape[1]
    num_features = X.shape[2]

    X_scaled = np.zeros((num_samples, num_time_steps, num_features), dtype=np.float64)
    scalers_x = []

    # Loop para escalar cada recurso em cada time step
    for i in range(num_features):
        # Criar e adicionar o escalonador MinMax à lista de escalonadores
        scaler_i = MinMaxScaler(feature_range=(0, 1))
        scalers_x.append(scaler_i)

        # Loop para escalar cada time step para o recurso atual
        for j in range(num_time_steps):
            # Extrair, remodelar, dimensionar e atualizar a matriz X_scaled
            # Extrai a coluna do recurso i e time step j e a remodela para uma matriz 2D
            X_temp = X[:, j, i].reshape(-1, 1)

            # Dimensiona a coluna do recurso i e time step j e a remodela de volta para uma matriz 1D
            X_scaled_temp = scaler_i.fit_transform(X_temp).ravel()

            # Atualiza a matriz X_scaled com os valores escalonados
            X_scaled[:, j, i] = X_scaled_temp[:num_samples]

    # # Retornar a matriz de dados escalonados e a lista de escalonadores
    return X_scaled, scalers_x


# A função scale_1d_data dimensiona a saída (y) usando o escalonador MinMax. Retorna a saída escalonada e o escalonador.
def scale_1d_data(y):
    # Criar e ajustar o escalonador MinMax para a saída
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    # Transformar a saída e retornar a saída escalonada e o escalonador
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    return y_scaled, scaler_y


# A função cross_validate_lstm realiza a validação cruzada do modelo LSTM usando a divisão de séries temporais. Ela
# normaliza os dados, divide-os em subconjuntos de treinamento e validação, cria sequências temporais,
# treina o modelo LSTM e calcula a pontuação média de validação.
def cross_validate_lstm(X, y, hp, time_steps, lstm_units, dropout_rate):
    # Normalize os dados antes de dividir
    X_norm, _ = normalize_3d_data(X)

    # Defina o número de divisões e crie o objeto TimeSeriesSplit
    n_splits = hp.get('n_splits')
    kfold = TimeSeriesSplit(n_splits)
    # Inicializar lista para armazenar pontuações de validação
    validation_scores = []

    # Loop para cada divisão do TimeSeriesSplit
    for train_index, validation_index in kfold.split(X):
        # Divida os dados em treinamento e validação
        X_train, X_validation = X_norm[train_index], X_norm[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        # Crie sequências temporais para o conjunto de treinamento e validação
        X_train_lstm, y_train_lstm = create_sequences(X_train, y_train, time_steps)
        X_validation_lstm, y_validation_lstm = create_sequences(X_validation, y_validation, time_steps)

        # Treine o modelo LSTM e obtenha o objeto tuner
        model, tuner = train_model_lstm(X_train_lstm, y_train_lstm, hp, time_steps, lstm_units, dropout_rate)

        # Calcule a pontuação de validação para a divisão atual
        validation_score = model.evaluate(X_validation_lstm, y_validation_lstm, verbose=0)
        # Adicione a pontuação de validação à lista de pontuações de validação
        validation_scores.append(validation_score)

    # Calcule a pontuação média de validação e retorne
    mean_validation_score = np.mean(validation_scores)

    return mean_validation_score


# A função build_lstm_model constrói e compila um modelo LSTM de várias camadas com camadas de dropout para evitar
# overfitting. A função recebe os parâmetros para o modelo, como o número de time steps, o número de features,
# o número de unidades LSTM e a taxa de dropout. Retorna o modelo construído.
def build_lstm_model(hp, time_steps, num_features, lstm_units, dropout_rate):
    # Inicializar um modelo Sequential
    model = Sequential()

    # Adicionar camadas LSTM e Dropout
    # Primeira camada LSTM e Dropout
    # Camada LSTM com a opção de retorno de sequências para empilhar camadas LSTM
    model.add(LSTM(units=lstm_units, input_shape=(time_steps, num_features), return_sequences=True))
    # Camada Dropout para evitar overfitting
    model.add(Dropout(dropout_rate))

    # Segunda camada LSTM e Dropout
    # Camada LSTM com a opção de retorno de sequências para empilhar camadas LSTM
    model.add(LSTM(units=lstm_units, return_sequences=True))
    # Camada Dropout para evitar overfitting
    model.add(Dropout(dropout_rate))

    # Terceira camada LSTM e Dropout
    # Camada LSTM com a opção de retorno de sequências para empilhar camadas LSTM
    model.add(LSTM(units=lstm_units, return_sequences=True))
    # Camada Dropout para evitar overfitting
    model.add(Dropout(dropout_rate))

    # Quarta camada LSTM e Dropout
    # Camada LSTM com a opção de retorno de sequências para empilhar camadas LSTM
    model.add(LSTM(units=lstm_units, return_sequences=True))
    # Camada Dropout para evitar overfitting
    model.add(Dropout(dropout_rate))

    # Quinta camada LSTM e Dropout
    # Camada LSTM com a opção de retorno de sequências para empilhar camadas LSTM
    model.add(LSTM(units=lstm_units))
    # Camada Dropout para evitar overfitting
    model.add(Dropout(dropout_rate))

    # Camada Dense de saída
    model.add(Dense(1))

    # Compilar o modelo com a função de perda e o otimizador
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Retornar o modelo construído
    return model


# A função build_and_train_lstm constrói e treina um modelo LSTM, normaliza os dados, cria sequências temporais,
# treina o modelo LSTM, e calcula a pontuação média de validação usando validação cruzada. Retorna o modelo treinado,
# scaler_y, X_norm, scalers_x, time_steps e a pontuação média de validação.
def build_and_train_lstm(X, y, hp):
    # Obter hiperparâmetros a partir do objeto hp
    time_steps = hp.get('time_steps')
    lstm_units = hp.get('lstm_units')
    dropout_rate = hp.get('dropout_rate')

    # Normalizar os dados
    X_norm, scalers_x = normalize_3d_data(X)
    y_norm, scaler_y = scale_1d_data(y)

    # Criar sequências temporais para o conjunto de treinamento
    X_train, y_train = create_sequences(X_norm, y_norm, time_steps)
    # Treinar o modelo LSTM e obter o objeto tuner
    model, tuner = train_model_lstm(X_train, y_train, hp, time_steps, lstm_units, dropout_rate)
    # Calcular a pontuação média de validação usando validação cruzada
    mean_validation_score = cross_validate_lstm(X, y, hp, time_steps, lstm_units, dropout_rate)
    # Retornar o modelo, scaler_y, X_norm, scalers_x, time_steps e a pontuação média de validação
    return model, scaler_y, X_norm, scalers_x, time_steps, mean_validation_score


# A função get_tuner instancia um objeto RandomSearch do Keras Tuner, que é usado para otimizar os hiperparâmetros do
# modelo. Ele recebe os hiperparâmetros, input_shape, max_trials e executions_per_trial como argumentos e retorna o
# objeto tuner.
def get_tuner(hp, input_shape, lstm_units, dropout_rate, max_trials, executions_per_trial):
    # Definir o construtor do modelo para o tuner
    time_steps = input_shape[0]
    num_features = input_shape[1]
    model_builder = lambda hp: build_lstm_model(hp, time_steps, num_features, lstm_units, dropout_rate)
    # Instanciar um objeto RandomSearch com o construtor do modelo, o objetivo e outros parâmetros
    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        seed=42,
        directory="tuner_results",
        project_name="lstm_matic",
    )
    # Retornar o objeto tuner
    return tuner


# A função train_model_lstm treina o modelo LSTM usando o Keras Tuner para otimizar os hiperparâmetros. Ela define o
# input_shape, obtém os hiperparâmetros, instancia um objeto tuner, define um callback para exibir o progresso do
# treinamento e inicia a busca pelo melhor modelo.
def train_model_lstm(X_train, y_train, hp, time_steps, lstm_units, dropout_rate):
    # Obter o valor de num_features do shape de X_train
    num_features = X_train.shape[2]
    # Definir o input_shape para o modelo
    input_shape = (time_steps, num_features)
    # Obter os hiperparâmetros a partir do objeto hp
    max_trials = hp.Fixed("max_trials", combinacaoMaxima)
    executions_per_trial = combinacaoTestada
    epochs = hp.get("epochs")
    batch_size = hp.get("batch_size")
    # Instanciar um objeto tuner
    tuner = get_tuner(hp, input_shape, lstm_units, dropout_rate, max_trials, executions_per_trial)
    # Definir um callback para exibir o progresso do treinamento
    tqdm_callback = TqdmCallback()
    # Iniciar a busca pelo melhor modelo
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10), tqdm_callback],
    )

    # Obter o melhor modelo e carregar os pesos do melhor checkpoint
    best_trials = tuner.oracle.get_best_trials(num_trials=combinacaoMaxima)
    best_trial = best_trials[0]
    best_model = build_lstm_model(hp, time_steps, num_features, lstm_units, dropout_rate)
    best_model.load_weights(tuner._get_checkpoint_fname(best_trial.trial_id))

    # Retornar o melhor modelo e o objeto tuner
    return best_model, tuner


# A função train_arima treina um modelo ARIMA usando a função auto_arima, que escolhe automaticamente os melhores
# parâmetros do modelo. Ela retorna o modelo ajustado e as previsões no conjunto de treinamento.
def train_arima(y):
    # Construir e ajustar o modelo ARIMA automaticamente usando auto_arima
    arima_model = auto_arima(y, seasonal=False, suppress_warnings=True, stepwise=True)
    arima_model.fit(y)
    # Prever os valores da série temporal no conjunto de treinamento
    arima_preds = arima_model.predict_in_sample()
    # Retornar o modelo ajustado e as previsões
    return arima_model, arima_preds


# A função fit_garch_model treina um modelo GARCH nos dados fornecidos. Ele ajusta o modelo e retorna o modelo ajustado.
def fit_garch_model(data):
    # Construir o modelo GARCH(1,1) e ajustar aos dados
    garch_model = arch_model(data['Price'].pct_change().dropna() * 100, vol='Garch', p=1, q=1)
    garch_fitted = garch_model.fit(disp='off')
    # Retornar o modelo ajustado
    return garch_fitted


# A função add_volatility_to_data adiciona a volatilidade calculada a partir do modelo GARCH ajustado aos dados
# fornecidos e remove quaisquer linhas com valores NaN. Retorna os dados com a volatilidade adicionada.
def add_volatility_to_data(data, garch_fitted):
    # Calcular a volatilidade a partir do modelo GARCH ajustado e adicioná-la aos dados
    volatility = np.sqrt(garch_fitted.conditional_volatility)
    data['Volatility'] = volatility
    # Remover quaisquer linhas com valores NaN
    data.dropna(inplace=True)
    # Retornar os dados com a volatilidade adicionada
    return data


# A função prepare_input_data prepara os dados de entrada para o modelo LSTM, criando um array temporário e
# preenchendo-o com os valores de X. Retorna o array temporário.
def prepare_input_data(X, time_steps, num_features):
    # Criar um array temporário para armazenar os dados de entrada
    X_temp = np.zeros((1, time_steps, num_features))
    # Preencher o array temporário com os valores de X
    for i in range(time_steps):
        for j in range(num_features):
            try:
                X_temp[0, i, j] = X[
                    i, j, 0]  # Adicione o índice [0] para selecionar o primeiro elemento da dimensão extra
            except Exception as e:
                print(f"Erro ao atribuir o valor para i={i}, j={j}: {e}")
                print("Array X:", X)
                raise
    # Retornar o array temporário
    return X_temp


# Suavização Exponencial Simples (SES) às previsões do LSTM: A função exponential_smoothing aplica a suavização
# exponencial simples aos dados fornecidos usando o coeficiente alpha. Retorna os dados suavizados.
def exponential_smoothing(data, alpha):
    # Inicializar o array de dados suavizados com o primeiro valor de data
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    # Aplicar a suavização exponencial simples aos dados usando o coeficiente alpha
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]

    # Retornar os dados suavizados
    return smoothed_data


# Esta função treina um modelo de Regressão Ridge
def train_ridge(X_train, y_train, alpha=0.7):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# Esta função treina um modelo Gradient Boosting
def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model


# Esta função usa um modelo de Regressão Ridge
def ridge_future_predictions(ridge_model, X_input, n_days):
    future_preds = []
    for _ in range(n_days):
        pred = ridge_model.predict(X_input)
        future_preds.append(pred[0])
        X_input = np.roll(X_input, -1)
        X_input[-1] = pred
    return future_preds


# Esta função é semelhante à função ridge_future_predictions, mas usa um modelo Gradient Boosting treinado para fazer as previsões.
def gb_future_predictions(gb_model, X_input, n_days):
    future_preds = []
    for _ in range(n_days):
        pred = gb_model.predict(X_input)
        future_preds.append(pred[0])
        X_input = np.roll(X_input, -1)
        X_input[-1] = pred
    return future_preds


# Função principal do modelo
def main():
    # limpar os arquivos de checkpoint gerados pelo Keras Tuner antes de executar o código novamente
    clear_tuner_directory("tuner_results")

    # Obter os preços históricos do token usando a API CoinGecko (substitua os valores conforme necessário)
    contract_address = contratoToken  # Adicionando o endereço do Token que você quer projetar o preço
    end_date = int((dt.datetime.now() - dt.datetime(1970, 1, 1)).total_seconds())
    start_date = end_date - 86400 * 2190  # Seis anos atrás
    data = fetch_data(contract_address, start_date, end_date)

    # Obter os dados do Bitcoin e combinar com os dados do token
    btc_start_date = data.index.min().timestamp()  # Altere para a data do primeiro preço registrado do token
    btc_end_date = end_date
    btc_data = fetch_bitcoin_data(btc_start_date, btc_end_date)
    btc_data = btc_data.rename(columns={'Price': 'BTC_Price'})
    data = data.join(btc_data, how='outer')

    # Obter informação do TOKEN ETH - NA REDE BNB SMART CHAIN
    eth_contract_address = '0x2170ed0880ac9a755fd29b2688956bd959f933f8'  # Contrato do Ethereum
    eth_data = fetch_data(eth_contract_address, start_date, end_date)
    eth_data = eth_data.rename(columns={'Price': 'ETH_Price'})
    data = data.join(eth_data['ETH_Price'], how='outer')

    # Obter informação do TOKEN BNB - NA REDE BNB SMART CHAIN
    bnb_contract_address = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c'  # Contrato do Binance Coin (BNB)
    bnb_data = fetch_data(bnb_contract_address, start_date, end_date)
    bnb_data = bnb_data.rename(columns={'Price': 'BNB_Price'})
    data = data.join(bnb_data['BNB_Price'], how='outer')

    # Adicionando informações da inflação dos EUA
    start_date_fred = datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')
    end_date_fred = datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')
    fred = Fred(api_key='7f4981e0eb5f31d25801ea188988ac5d')  # Insira sua chave de API

    # Obtendo os dados da inflação
    inflation_data = fred.get_series('T10YIE', start_date_fred, end_date_fred)
    inflation_data = pd.DataFrame(inflation_data, columns=['Inflation_rate'])
    inflation_data.index.name = 'Date'
    data = data.merge(inflation_data, how='outer', left_index=True, right_index=True)
    # https://fred.stlouisfed.org/series/T10YIE

    # Adicionando informações da taxa de juros dos EUA
    interest_rate_data = fred.get_series('GS10', start_date_fred, end_date_fred)
    interest_rate_data = pd.DataFrame(interest_rate_data, columns=['Interest_rate'])
    interest_rate_data.index.name = 'Date'
    data = data.merge(interest_rate_data, how='outer', left_index=True, right_index=True)

    # Chamando a função do modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity).
    garch_fitted = fit_garch_model(data)

    # Preencher os valores NaN com a média da coluna
    data['Price'].fillna(data['Price'].mean(), inplace=True)
    data['BTC_Price'].fillna(data['BTC_Price'].mean(), inplace=True)
    data['ETH_Price'].fillna(data['ETH_Price'].mean(), inplace=True)
    data['BNB_Price'].fillna(data['BNB_Price'].mean(), inplace=True)
    data['Inflation_rate'].fillna(data['Inflation_rate'].mean(), inplace=True)
    data['Market_Cap'].fillna(data['Market_Cap'].mean(), inplace=True)
    data['Volume'].fillna(data['Volume'].mean(), inplace=True)

    # Preencher os valores NaN com o método 'pad'
    data.fillna(method='pad', inplace=True)

    # Adicionar média móvel ponderada ao DataFrame
    data['Weighted_Moving_Average'] = weighted_moving_average(data, window=120)

    # Avaliando a volatilidade
    data = add_volatility_to_data(data, garch_fitted)

    # Otimizar hiperparâmetros do modelo LSTM
    hp = HyperParameters()
    hp.Int('time_steps', min_value=60, max_value=120, step=10)
    hp.Int('epochs', min_value=200, max_value=800, step=10)
    hp.Int('batch_size', min_value=32, max_value=64, step=2)
    hp.Int('lstm_units', min_value=32, max_value=512, step=20)
    hp.Float('dropout_rate', min_value=0.2, max_value=0.9, step=0.1)
    hp.Int('n_splits', min_value=5, max_value=10, step=1)

    # Preparar os dados
    X, y = prepare_data(data, garch_fitted, hp.get('time_steps'))
    model, scaler, X_scaled, scaler_x, best_time_steps, mean_validation_score = build_and_train_lstm(X, y, hp)
    X_t, y_t = create_sequences(X, y, best_time_steps)

    # Avaliar o resultado pelom modelo ARIMA
    # Divida os dados em conjuntos de treinamento e teste
    train_size = int(len(X) * 0.8)
    X_train, X_test = X_t[:train_size], X_t[train_size:]
    y_train, y_test = y_t[:train_size], y_t[train_size:]

    # Remodelar os dados para 2D
    num_features = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)

    # Usar a primeira coluna de cada sequência
    X_train_2d_r = X_train[:, 0, :]
    X_test_2d_r = X_test[:, 0, :]

    # Antes de chamar as funções de treinamento, remodelar y_train para 2D
    y_train_2d = y_train.reshape(-1, 1)

    # Padronize os dados de treinamento e teste
    standard = StandardScaler()
    X_train_scaled_2d = standard.fit_transform(X_train_2d)
    X_test_scaled_2d = standard.transform(X_test_2d)

    # Remodelar os dados padronizados de volta para 3D
    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    # Criar sequências padronizadas para treinamento e teste
    X_train, y_train = create_sequences(X_train_scaled, y_train, best_time_steps)
    X_test, y_test = create_sequences(X_test_scaled, y_test, best_time_steps)

    # Treinar o modelo Ridge e fazer previsões
    ridge_model = train_ridge(X_train_2d_r, y_train_2d)
    ridge_test_preds = ridge_model.predict(X_test_2d_r)

    # Treinar o modelo Gradient Boosting e fazer previsões
    gb_model = train_gradient_boosting(X_train_2d_r, y_train_2d.ravel())
    gb_test_preds = gb_model.predict(X_test_2d_r)

    # Treine o modelo ARIMA e faça previsões
    arima_model, arima_preds = train_arima(y)
    # Faça previsões com o modelo LSTM
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2])
    lstm_test_preds = model.predict(X_test_reshaped)
    # Combine as previsões do LSTM e do ARIMA usando média ponderada
    lstm_weight = 0.6
    arima_weight = 1 - lstm_weight
    # Resultado do ANSEMBLE
    arima_test_preds = arima_model.predict(n_periods=len(y_test))

    # Definir pesos para os modelos Ridge e Gradient Boosting
    ridge_weight = 0.2
    gb_weight = 0.2

    # Ajustar os pesos dos outros modelos para que a soma seja 1
    lstm_weight *= (1 - ridge_weight - gb_weight)
    arima_weight *= (1 - ridge_weight - gb_weight)

    # Tranformando as Matrizes em 1D
    min_length = min(lstm_test_preds.shape[0], arima_test_preds.shape[0], ridge_test_preds.shape[0],
                     gb_test_preds.shape[0])
    lstm_test_preds = lstm_test_preds[:min_length]
    arima_test_preds = arima_test_preds[:min_length]
    ridge_test_preds = ridge_test_preds[:min_length]
    gb_test_preds = gb_test_preds[:min_length]

    # Combinar previsões usando média ponderada
    ensemble_test_preds = (lstm_weight * lstm_test_preds) + (arima_weight * arima_test_preds) + (
            ridge_weight * ridge_test_preds) + (gb_weight * gb_test_preds)

    # Prever os preços de acordo com o período escolhido
    future_dates = pd.date_range(start=data.index[-1], periods=diasProjetar)[1:]  # Escolher os próximos dias para projetar

    X_input = np.array(X_scaled[-best_time_steps:])

    # Previsões futuras para LSTM
    lstm_future_preds = []

    num_features = X_train.shape[2]  # Obter o número de recursos do conjunto de treinamento

    for _ in range(len(future_dates)):
        X_input_reshaped = prepare_input_data(X_input, best_time_steps, num_features)
        y_pred = model.predict(X_input_reshaped)
        y_pred_temp = np.zeros((1, num_features))
        y_pred_temp[0, 0] = y_pred[0, 0]
        lstm_future_preds.append(scaler.inverse_transform(y_pred_temp)[0, 0])
        y_pred_array = np.array(y_pred).reshape(-1, 1)
        y_pred_scaled = scaler.inverse_transform(y_pred_array)
        y_pred_value = y_pred_scaled[0, 0]
        new_X_input = np.zeros_like(X_input)
        new_X_input[:, 1:, :] = X_input[:, :-1, :]
        new_X_input[0, -1, 0] = y_pred_value
        X_input = new_X_input

    # Previsões futuras para ARIMA
    arima_future_preds = []
    for _ in range(len(future_dates)):
        arima_future_preds.append(arima_model.predict(n_periods=1)[0])
        y_train = np.concatenate((y_train, np.array([arima_future_preds[-1]])))

    # Previsões futuras para Ridge e Gradient Boosting
    ridge_future_preds = ridge_future_predictions(ridge_model, X_test_2d_r[-1].reshape(1, -1), len(future_dates))
    gb_future_preds = gb_future_predictions(gb_model, X_test_2d_r[-1].reshape(1, -1), len(future_dates))

    # Calcular previsões futuras do ensemble (LSTM + ARIMA)
    ensemble_future_preds = (lstm_weight * np.array(lstm_future_preds)) + (
                arima_weight * np.array(arima_future_preds)) + (ridge_weight * np.array(ridge_future_preds)) + (
                                        gb_weight * np.array(gb_future_preds))
    ensemble_future_preds = np.mean(ensemble_future_preds, axis=0).reshape(-1, 1)

    # Criar um DataFrame com as previsões do ensemble e os índices das datas futuras
    lstm_predictions_df = pd.DataFrame(lstm_future_preds, index=future_dates, columns=['LSTM_Prediction'])
    arima_predictions_df = pd.DataFrame(arima_future_preds, index=future_dates, columns=['ARIMA_Prediction'])
    ensemble_predictions_df = pd.DataFrame(ensemble_future_preds, index=future_dates, columns=['Ensemble_Prediction'])

    # Juntar os DataFrames de previsões LSTM e Ensemble (LSTM + ARIMA)
    all_predictions_df = pd.concat([lstm_predictions_df, arima_predictions_df, ensemble_predictions_df], axis=1)

    # Tecnica de Suavização Exponencial Simples (SES)
    alpha = 0.3  # Escolha um valor alfa adequado (entre 0 e 1)
    lstm_predictions = all_predictions_df['LSTM_Prediction']
    smoothed_lstm_predictions = exponential_smoothing(lstm_predictions, alpha)

    # Adicionar os valores suavizados ao DataFrame de previsões
    all_predictions_df['Smoothed_LSTM_Prediction'] = pd.Series(smoothed_lstm_predictions,
                                                               index=all_predictions_df.index)

    # Analisar a tendência de preços
    analyze_trend(data, all_predictions_df, ensemble_test_preds)

    # Imprimindo o resultado da validação cruzada do modelo LSTM
    print(f"Resultado da validação cruzada do modelo LSTM: {mean_validation_score}")

    # Plotar os resultados
    plot_results(data, all_predictions_df,
                 title="Previsão de Preços (LSTM + ARIMA + Ridge + Gradient Boosting) para o Token: " + nomeToken )
    plt.show()


# Analisando e apresentando o resultado
def analyze_trend(data, all_predictions_df, ensemble_test_preds):
    last_known_price = data['Price'].iloc[-1]
    predicted_price_in_3_months = all_predictions_df['Ensemble_Prediction'].iloc[-1]
    # Utilizar as previsões combinadas para calcular a diferença de preço
    price_difference = ensemble_test_preds[-1] - last_known_price

    if (price_difference > 0).all():
        trend = "subir"
    else:
        trend = "cair"

    print(f"O preço deve {trend} para ${predicted_price_in_3_months:.2f} nos próximos 30 dias.")


# Função para criar o gráfico de projeção
def plot_results(data, all_predictions_df, title):
    plt.plot(data.index, data['Price'], label='Preços históricos')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    # plt.plot(all_predictions_df.index, all_predictions_df['Smoothed_LSTM_Prediction'], label="Previsão LSTM",
    # linewidth=2, linestyle='--') plt.plot(all_predictions_df.index, all_predictions_df['ARIMA_Prediction'],
    # label="Previsão ARIMA", linewidth=2, linestyle='-.')
    plt.plot(all_predictions_df.index, all_predictions_df['Ensemble_Prediction'], label="Previsão Ensemble",
             linewidth=2, linestyle=':')

    # IMPRESSÃO DOS RESULTADOS DO INDICADOR
    # print(f"Projeção Ensemble (LSTM + ARIMA + Ridge + Gradient Boosting) - {nomeToken}")
    # print(all_predictions_df['Ensemble_Prediction'])

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()


if __name__ == "__main__":
    main()

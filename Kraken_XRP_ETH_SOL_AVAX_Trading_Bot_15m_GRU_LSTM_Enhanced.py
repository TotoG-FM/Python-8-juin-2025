import os
import yaml
import logging  # 🔄 Doit être avant l'utilisation de `logger`

# Initialiser le logger dès le début
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Initialisation du logging.")

# Ensuite les autres imports
from dotenv import load_dotenv
import krakenex
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import csv
import tensorflow as tf
from numba import jit
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
import signal
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import traceback
import fcntl
from google.cloud import storage
import optuna
from newsapi.newsapi_client import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from arch import arch_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from hmmlearn.hmm import GaussianHMM
from flask import Flask
import threading
import importlib


# Charger les variables d’environnement et initialiser le client Kraken
load_dotenv()
client = krakenex.API()
client.load_key('.env')  # ⚠️ Assure-toi que le fichier `kraken.key` existe ou adapte selon ton usage

# Activer XLA pour accélérer les calculs
tf.config.optimizer.set_jit(True)

# Vérifier les dispositifs physiques disponibles (Metal GPU ou CPU)
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponibles :", physical_devices)
if physical_devices:
    logger.info(f"GPU Metal détecté : {physical_devices[0]}")
else:
    logger.warning("Aucun GPU Metal détecté, utilisation du CPU.")


# Configuration des avertissements
warnings.filterwarnings("ignore", category=UserWarning, module="arch.univariate.base")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# Gestion de l'interruption
def signal_handler(signum, frame):
    logger.info("Arrêt manuel détecté via Ctrl+C.")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Initialisation du logging.")


# Charger la configuration depuis config.yaml
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


config = load_config()
CONFIG = config.get('trading', {})
MODEL_CONFIG = config.get('model', {})

# Paramètres globaux (fusion avec config.yaml)
HISTORICAL_PERIODS = CONFIG.get('historical_periods', 20000)  # ≈208 jours
BACKTEST_PERIODS = CONFIG.get('backtest_periods', 30000)  # ≈1 an
WINDOW_SIZE = CONFIG.get('window_size', 500)  # ≈5 jours pour prédictions live
TRADING_FEE = CONFIG.get('trading_fee', 0.0016)
SLIPPAGE_RATE = CONFIG.get('slippage_rate', 0.001)  # 0.1% de slippage par défaut
T = 1 / 8760
N = 100
DT = T / N
M = 1000
INTERVAL = CONFIG.get('interval', 900)  # 15 minutes
DATA_SAVE_INTERVAL = CONFIG.get('data_save_interval', 900)
RSI_PERIOD = CONFIG.get('rsi_period', 14)
RSI_OVERBOUGHT = CONFIG.get('rsi_overbought', 85)
PRICE_THRESHOLD = CONFIG.get('price_threshold', 0.005)
CORRELATION_WINDOW = CONFIG.get('correlation_window', 48)
BTC_THRESHOLD = CONFIG.get('btc_threshold', -0.02)
MIN_NOTIONAL = CONFIG.get('min_notional', 10)
EWMA_LAMBDA = CONFIG.get('ewma_lambda', 0.94)
DRAWDOWN_LIMIT = CONFIG.get('drawdown_limit', -0.25)
DRAWDOWN_WARNING = CONFIG.get('drawdown_warning', -0.20)
NEGLIGIBLE_POSITION_THRESHOLD = CONFIG.get('negligible_position_threshold', 2)
RISK_AVERSION = CONFIG.get('risk_aversion', 0.3)
MIN_RETURNS_FOR_EWMA = CONFIG.get('min_returns_for_ewma', 50)
DEFAULT_VOLATILITY = CONFIG.get('default_volatility', 0.7)
MIN_WEIGHT = CONFIG.get('min_weight', 0.05)
PNL_SELL_THRESHOLD = CONFIG.get('pnl_sell_threshold', -0.10)
TREND_LOOKBACK = CONFIG.get('trend_lookback', 60)
API_CALL_DELAY = CONFIG.get('api_call_delay', 1.0)
MAX_RETRIES = CONFIG.get('max_retries', 10)
RETRY_DELAY = CONFIG.get('retry_delay', 2)
MIN_DATA_POINTS = TREND_LOOKBACK + 1
TRAILING_PERCENT = CONFIG.get('trailing_percent', 0.02)
SYNC_POSITION_INTERVAL = CONFIG.get('sync_position_interval', 600)
SENTIMENT_UPDATE_INTERVAL = CONFIG.get('sentiment_update_interval', 1800)
MONTHLY_CHECK_INTERVAL = CONFIG.get('monthly_check_interval', 6 * 60 * 60)  # 6 heures
MODEL_TRAINING_INTERVAL = CONFIG.get('model_training_interval', 6 * 60 * 60)  # 6 heures

# Paramètres des modèles
GRU_UNITS = MODEL_CONFIG.get('gru_units', 64)
LEARNING_RATE = MODEL_CONFIG.get('learning_rate', 0.001)
DROPOUT_RATE = MODEL_CONFIG.get('dropout_rate', 0.2)
MIN_TRADING_DAYS = MODEL_CONFIG.get('min_trading_days', 30)
LEARNING_CONFIDENCE_THRESHOLD = MODEL_CONFIG.get('learning_confidence_threshold', 0.75)

# Date de trading
TRADING_START_TIME = datetime(2025, 6, 7, 22, 0, 0)  # Début à maintenant
END_DATE = datetime(2025, 7, 7, 22, 0, 0)  # Fin après 1 mois

# Cache et stockage
price_cache = {}
balance_cache = {'USD': None}
balance_cache_expiry = {'USD': 0}
price_cache_expiry = {}
highest_prices = {pair['symbol']: 0 for pair in CONFIG['pairs']}
if not os.path.exists(CONFIG['cache_dir']):
    os.makedirs(CONFIG['cache_dir'])
    logger.info(f"Répertoire {CONFIG['cache_dir']} créé.")

if not os.path.exists(CONFIG['csv_file']):
    with open(CONFIG['csv_file'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'action', 'quantity', 'price', 'profit'])
    logger.info(f"Fichier {CONFIG['csv_file']} créé.")
if not os.path.exists(CONFIG['data_log_file']):
    with open(CONFIG['data_log_file'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'last_price', 'volume', 'volatility', 'success_rate', 'monthly_return',
                         'sharpe_ratio', 'max_drawdown', 'total_budget', 'losing_trades'])
    logger.info(f"Fichier {CONFIG['data_log_file']} créé.")


# Versions des packages
def log_package_versions():
    packages = ['python-dotenv', 'krakenex', 'requests', 'numpy', 'pandas', 'tensorflow-macos', 'optuna',
                'newsapi-python', 'nltk', 'google-cloud-storage', 'flask', 'arch', 'scikit-learn', 'pycoingecko',
                'numba', 'aiohttp', 'statsmodels', 'hmmlearn', 'pyyaml']
    logger.info("Versions des packages :")
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            logger.info(f"  {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            logger.warning(f"  {package}: Non installé")


# Récupérer les filtres des paires
async def get_symbol_filters_async(session, kraken_symbol):
    try:
        async with session.get(f"https://api.kraken.com/0/public/AssetPairs?pair={kraken_symbol}",
                               ssl=False) as response:
            data = await response.json()
            if 'error' in data and data['error']:
                raise Exception(f"Erreur Kraken : {data['error']}")
            pair_info = data['result'][kraken_symbol]
            return {'minQty': float(pair_info['ordermin']), 'maxQty': 1000000.0,
                    'stepSize': 10 ** -int(pair_info['lot_decimals'])}
    except Exception as e:
        logger.error(f"Erreur filtres {kraken_symbol}: {e}")
        return {'minQty': 0.1, 'maxQty': 1000000.0, 'stepSize': 0.1}


async def get_symbol_filters():
    kraken_symbols = ','.join([pair['kraken_symbol'] for pair in CONFIG['pairs']])
    logger.info(f"Regroupement paires : {kraken_symbols}")
    async with aiohttp.ClientSession() as session:
        tasks = [get_symbol_filters_async(session, pair['kraken_symbol']) for pair in CONFIG['pairs']]
        results = await asyncio.gather(*tasks)
        return {pair['symbol']: result for pair, result in zip(CONFIG['pairs'], results)}


# Charger l'historique des transactions
def load_transaction_history(positions, average_entry_prices):
    if not os.path.exists(CONFIG['csv_file']):
        logger.warning(f"Fichier {CONFIG['csv_file']} absent.")
        return positions, average_entry_prices
    try:
        with open(CONFIG['csv_file'], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    symbol = row['symbol']
                    action = row['action']
                    quantity = float(row['quantity'])
                    price = float(row['price'])
                    profit = float(row['profit']) if row['profit'] else 0.0
                    if action == 'buy':
                        current_qty = positions[symbol]
                        current_price = average_entry_prices[symbol]
                        new_qty = current_qty + quantity
                        if new_qty > 0:
                            average_entry_prices[symbol] = (current_price * current_qty + price * quantity) / new_qty
                        positions[symbol] = new_qty
                    elif action == 'sell':
                        positions[symbol] -= quantity
                        if positions[symbol] <= 0:
                            positions[symbol] = 0
                            average_entry_prices[symbol] = 0
                except (ValueError, KeyError) as e:
                    logger.warning(f"Erreur lecture {symbol}: {e}")
    except Exception as e:
        logger.error(f"Erreur ouverture {CONFIG['csv_file']}: {e}")
    logger.info(f"Positions : {positions}")
    return positions, average_entry_prices


# Gestion du cache
def load_cached_historical_data(symbol, required_periods):
    cache_file = os.path.join(CONFIG['cache_dir'], f"{symbol}_historical.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Data'])
            # Vérifier les colonnes minimales
            required_columns = ['Data', 'price', 'Volume', 'Returns', 'Volatility', 'Momentum', 'Volume_Anomaly']
            if all(col in df.columns for col in required_columns):
                df['Volatility'] = pd.to_numeric(df['Volatility'], errors='coerce')
                if len(df) >= required_periods:
                    logger.info(f"Cache chargé pour {symbol}: {len(df)} périodes")
                    return df.tail(required_periods)[required_columns]
            else:
                logger.warning(f"Cache pour {symbol} invalide ou incomplet, régénération forcée.")
                return None
        except Exception as e:
            logger.error(f"Erreur lecture cache pour {symbol}: {e}")
            return None
    return None


def save_historical_data_to_cache(symbol, df):
    cache_file = os.path.join(CONFIG['cache_dir'], f"{symbol}_historical.csv")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, index=False,
              columns=['Data', 'price', 'Volume', 'Returns', 'Volatility', 'Momentum', 'Volume_Anomaly'])
    logger.info(f"Données pour {symbol} sauvegardées dans {cache_file}")


# Récupérer les données historiques
async def fetch_historical_data_async(session, symbol, limit):
    coin_id = CONFIG['coin_mapping'].get(symbol)
    if not coin_id:
        logger.error(f"Symbole {symbol} non supporté par CoinGecko")
        return pd.DataFrame()
    days = max(1, int(limit * 15 / (60 * 24)))
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params, ssl=False) as response:
                data = await response.json()
                if 'error' in data or 'prices' not in data:
                    logger.error(f"Erreur CoinGecko {symbol}: {data.get('error', 'Invalide')}")
                    return pd.DataFrame()
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['Volume'] = [vol[1] for vol in data.get('total_volumes', [0] * len(data['prices']))]
                df['Data'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['Data', 'price', 'Volume']].tail(limit)
                df = df.resample('15min', on='Data').mean().interpolate().reset_index()
                df['Returns'] = np.log(df['price'] / df['price'].shift(1)).fillna(0)
                df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(8760)
                df['Momentum'] = df['price'].pct_change(periods=24).fillna(0)  # Momentum sur 6 heures
                df['Volume_Anomaly'] = (df['Volume'] - df['Volume'].rolling(48).mean()) / df['Volume'].rolling(
                    48).std()  # Anomalie de volume
                df.dropna(subset=['Returns', 'Volatility', 'Momentum', 'Volume_Anomaly'], inplace=True)
                df[['price', 'Volume', 'Volatility', 'Momentum', 'Volume_Anomaly']] = df[
                    ['price', 'Volume', 'Volatility', 'Momentum', 'Volume_Anomaly']].ffill()
                if (df['price'] <= 0).any():
                    logger.error(f"Données invalides {symbol}")
                    return pd.DataFrame()
                # Sauvegarder avec les nouvelles features
                save_historical_data_to_cache(symbol, df[
                    ['Data', 'price', 'Volume', 'Returns', 'Volatility', 'Momentum', 'Volume_Anomaly']])
                logger.info(f"Données {symbol}: {len(df)} périodes")
                return df
        except Exception as e:
            logger.warning(f"Tentative {attempt + 1}/{MAX_RETRIES} {symbol} échouée: {e}")
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
    return pd.DataFrame()


async def fetch_historical_data(symbol, interval='15min', limit=HISTORICAL_PERIODS):
    cached_df = load_cached_historical_data(symbol, limit)
    if cached_df is None:
        logger.info(f"Pas de cache valide pour {symbol}, téléchargement en cours...")
        async with aiohttp.ClientSession() as session:
            return await fetch_historical_data_async(session, symbol, limit)
    return cached_df


# Récupérer le prix actuel
def get_current_price(symbol):
    current_time = time.time()
    if symbol in price_cache and current_time < price_cache_expiry.get(symbol, 0):
        return price_cache[symbol]
    kraken_symbol = next(p['kraken_symbol'] for p in CONFIG['pairs'] if p['symbol'] == symbol)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.query_public('Ticker', {'pair': kraken_symbol})
            logger.debug(f"Réponse brute pour {symbol} : {response}")  # Ajout pour débogage
            if 'error' in response and response['error']:
                raise Exception(f"Erreur API Kraken: {response['error']}")
            if not response['result'] or kraken_symbol not in response['result']:
                raise Exception("Résultat vide ou paire non trouvée")
            price = float(response['result'][kraken_symbol]['c'][0])
            price_cache[symbol] = price
            price_cache_expiry[symbol] = current_time + 60
            time.sleep(API_CALL_DELAY)
            return price
        except Exception as e:
            logger.error(f"Erreur prix {symbol}: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    logger.error(f"Échec prix {symbol} après {MAX_RETRIES} tentatives")
    return None  # Retourner None si toutes les tentatives échouent


# Calculer mu et sigma
def calculate_mu_sigma(data, symbol=None):
    if len(data) < MIN_RETURNS_FOR_EWMA:
        return 0, DEFAULT_VOLATILITY
    if 'Returns' not in data.columns:
        logger.error(
            f"Colonne 'Returns' manquante dans les données pour {symbol}. Colonnes disponibles : {data.columns.tolist()}")
        return 0, DEFAULT_VOLATILITY
    returns = data['Returns'].values
    returns = np.clip(returns, -0.2, 0.2)
    if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
        logger.warning(f"Données invalides (NaN ou infini) dans 'Returns' pour {symbol}. Remplacement par 0.")
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten() * 1000
    weights = np.array([EWMA_LAMBDA ** i for i in range(len(returns) - 1, -1, -1)])
    weights /= weights.sum()
    mu = np.sum(returns * weights) * 8760 / 1000
    try:
        model = arch_model(scaled_returns, mean='Zero', vol='Garch', p=1, q=1, dist='normal', rescale=True)
        start_time = time.time()
        res = model.fit(disp='off', options={'maxiter': 1000, 'tol': 1e-6})
        if time.time() - start_time > 30:
            raise Exception("Timeout GARCH")
        sigma = res.conditional_volatility[-1] / 1000 * np.sqrt(8760)
    except Exception as e:
        logger.warning(f"Échec GARCH {symbol}: {e}. Utilisation d'EWMA.")
        sigma = np.sqrt(np.mean(returns ** 2) * 8760)
    return mu, sigma


# Calculer la corrélation
def calculate_correlation(data1, data2):
    if data1.empty or data2.empty:
        return 0.0
    returns1 = data1['Returns'].values
    returns2 = data2['Returns'].values
    n = min(len(returns1), len(returns2), CORRELATION_WINDOW)
    if n < 2:
        return 0.0
    mean1, mean2 = np.mean(returns1[:n]), np.mean(returns2[:n])
    cov = np.sum((returns1[:n] - mean1) * (returns2[:n] - mean2)) / n
    std1, std2 = np.std(returns1[:n]), np.std(returns2[:n])
    return cov / (std1 * std2) if std1 * std2 != 0 else 0.0


# Simulation Monte Carlo
@jit(nopython=True)
def monte_carlo_sim(S0, mu, sigma, T, N, DT, M):
    price_paths = np.zeros((N + 1, M))
    price_paths[0] = S0
    for t in range(1, N + 1):
        Z = np.random.standard_normal(M)
        price_paths[t] = price_paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * DT + sigma * np.sqrt(DT) * Z)
    return price_paths


# Calculer le RSI avec validation
def calculate_rsi(data, window=RSI_PERIOD):
    if 'price' not in data.columns:
        raise ValueError(f"Colonne 'price' absente dans le DataFrame pour RSI.")
    delta = data['price'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-8)  # Éviter division par zéro
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0


# Calculer les moyennes mobiles
@jit(nopython=True)
def calculate_moving_averages(data):
    short = np.mean(data['price'].values[-9:])
    long = np.mean(data['price'].values[-21:])
    return 1 if short > long and data['price'].values[-10] <= data['price'].values[-22] else -1 if short < long and \
                                                                                                   data['price'].values[
                                                                                                       -10] >= \
                                                                                                   data['price'].values[
                                                                                                       -22] else 0

# Calculer le MACD
@jit(nopython=True)
def calculate_macd(data):
    exp1 = pd.Series(data['price']).ewm(span=12, adjust=False).mean().values[-1]
    exp2 = pd.Series(data['price']).ewm(span=26, adjust=False).mean().values[-1]
    macd = exp1 - exp2
    signal = pd.Series([macd]).ewm(span=9, adjust=False).mean().values[-1]
    return 1 if macd > signal and pd.Series(data['price']).ewm(span=12, adjust=False).mean().values[-2] <= \
                pd.Series(data['price']).ewm(span=26, adjust=False).mean().values[-2] else -1 if macd < signal and \
                                                                                                 pd.Series(
                                                                                                     data['price']).ewm(
                                                                                                     span=12,
                                                                                                     adjust=False).mean().values[
                                                                                                     -2] >= pd.Series(
        data['price']).ewm(span=26, adjust=False).mean().values[-2] else 0


# Créer le modèle GRU
def create_gru_model(input_shape):
    # Détecter le device (Metal GPU ou CPU fallback)
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print("Device utilisé :", device)

    with tf.device(device):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, activation='tanh'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.GRU(GRU_UNITS, return_sequences=False, activation='tanh'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # Compilation avec XLA activé
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            jit_compile=True  # Active XLA ici, pas besoin du décorateur @tf.function
        )

    return model

# ----------------------
# Fonction globale : optimize_horizon
# ----------------------
def optimize_horizon(data, symbol, horizon):
    logger.info(f"Optimisation pour {symbol} avec horizon : {horizon}")

    def objective(trial):
        device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        gru_units = trial.suggest_int('gru_units', 32, 128)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 0.01, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        epochs = trial.suggest_int('epochs', 10, 20)

        with tf.device(device):
            model = create_gru_model((TREND_LOOKBACK, 5))
            returns = data['Returns'].values
            volume = data['Volume'].values
            volatility = data['Volatility'].values
            momentum = data['Momentum'].values
            volume_anomaly = data['Volume_Anomaly'].values

            X, y = prepare_sequences(returns, volume, volatility, momentum, volume_anomaly, horizon)
            if len(X) == 0 or len(y) == 0 or np.any(np.isnan(X)) or np.any(np.isnan(y)):
                return float('inf')

            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
            loss = model.evaluate(X, y, verbose=0)
            return loss if np.isfinite(loss) else float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    with tf.device("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"):
        best_model = create_gru_model((TREND_LOOKBACK, 5))
        returns = data['Returns'].values
        volume = data['Volume'].values
        volatility = data['Volatility'].values
        momentum = data['Momentum'].values
        volume_anomaly = data['Volume_Anomaly'].values
        X, y = prepare_sequences(returns, volume, volatility, momentum, volume_anomaly, horizon)
        best_model.fit(X, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)
    return best_model

# ----------------------
# Fonction asynchrone : optimize_models_async
# ----------------------
async def optimize_models_async(data, symbol):
    best_models = {}
    horizons = ['15min', '1h', '4h']
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, optimize_horizon, data, symbol, horizon) for horizon in horizons]
        for horizon, task in zip(horizons, tasks):
            try:
                model = await task
                best_models[f'{symbol}_{horizon}'] = model
            except Exception as e:
                logger.error(f"Erreur d'optimisation pour {symbol} {horizon}: {e}")

    return best_models



def prepare_sequences(returns, volume, volatility, momentum, volume_anomaly, horizon):
    logger.info(f"Préparation des séquences avec horizon : {horizon}")
    offset = {'15min': 1, '1h': 4, '4h': 16}[horizon]
    series_len = len(returns)
    max_start = min(series_len - TREND_LOOKBACK - offset + 1, WINDOW_SIZE - TREND_LOOKBACK - offset + 1)

    if max_start <= 0:
        logger.warning("Pas assez de données pour créer des séquences.")
        return np.array([]), np.array([])

    # Stack features
    features = np.stack([
        returns,
        volume,
        volatility,
        momentum,
        volume_anomaly
    ], axis=1)

    X = np.lib.stride_tricks.sliding_window_view(features, (TREND_LOOKBACK, 5))  # (N, 1, TREND_LOOKBACK, 5)
    X = X.squeeze(1)  # Remove singleton dimension

    # Limiter à max_start
    X = X[:max_start]

    # Y = valeur future de retour après offset
    y_indices = np.arange(TREND_LOOKBACK + offset - 1, TREND_LOOKBACK + offset - 1 + max_start)
    y = returns[y_indices]

    # Nettoyage
    mask = ~np.any(np.isnan(X), axis=(1, 2)) & ~np.isnan(y)
    X = X[mask].astype(np.float32)
    y = y[mask].astype(np.float32)

    return X, y



def prepare_last_sequence(returns, volume, volatility, momentum, volume_anomaly):
    return np.array([np.column_stack((returns[-TREND_LOOKBACK:], volume[-TREND_LOOKBACK:], volatility[-TREND_LOOKBACK:],
                                      momentum[-TREND_LOOKBACK:], volume_anomaly[-TREND_LOOKBACK:]))], dtype=np.float32)


@tf.function(jit_compile=True)  # Activer XLA pour l'entraînement
def train_model(model, X, y):
    # Détecter et choisir device (Metal GPU ou fallback CPU)
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    with tf.device(device):
        model.fit(X, y, epochs=15, batch_size=64, verbose=0, validation_split=0.2)


def save_models(models):
    for name, model in models.items():
        path = os.path.join(CONFIG['model_save_path'], f"{name}_weights.h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)  # Sauvegarde au format .h5
        logger.info(f"Modèle sauvegardé pour {name}")


async def retrain_models(historical_dfs):
    global gru_models
    new_gru_models = {}

    # Optimisation asynchrone des 3 horizons par paire
    tasks = [optimize_models_async(historical_dfs[pair['symbol']], pair['symbol']) for pair in CONFIG['pairs']]
    results = await asyncio.gather(*tasks)

    # Fusionner les modèles
    for pair, models in zip(CONFIG['pairs'], results):
        new_gru_models.update(models)

    gru_models = new_gru_models
    save_models(gru_models)

    # Fine-tuning uniquement sur 15min
    for symbol, df in historical_dfs.items():
        live_df = df.tail(1000)
        if len(live_df) < MIN_DATA_POINTS:
            continue

        X, y = prepare_sequences(
            live_df['Returns'].values,
            live_df['Volume'].values,
            live_df['Volatility'].values,
            live_df['Momentum'].values,
            live_df['Volume_Anomaly'].values,
            '15min'
        )
        if X.size and y.size and not (np.any(np.isnan(X)) or np.any(np.isnan(y))):
            model = gru_models.get(f'{symbol}_15min')
            if model:
                train_model(model, X, y)



# Prédire la tendance
def predict_price_trend(data, symbol):
    if len(data) < MIN_DATA_POINTS:
        return 0, 0.0

    live_df = data.tail(WINDOW_SIZE)
    X, y = prepare_sequences(
        live_df['Returns'].values,
        live_df['Volume'].values,
        live_df['Volatility'].values,
        live_df['Momentum'].values,
        live_df['Volume_Anomaly'].values,
        '15min'
    )

    if not X.size or not y.size or np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return 0, 0.0

    model_path = os.path.join(CONFIG['model_save_path'], f"{symbol}_15min_weights.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = gru_models.get(f'{symbol}_15min') or create_gru_model((TREND_LOOKBACK, 5))
        train_model(model, X, y)
        save_models({f'{symbol}_15min': model})

    last_sequence = prepare_last_sequence(
        live_df['Returns'].values,
        live_df['Volume'].values,
        live_df['Volatility'].values,
        live_df['Momentum'].values,
        live_df['Volume_Anomaly'].values
    )

    @tf.function(jit_compile=True)
    def predict_batch():
        return model.predict(last_sequence, verbose=0)

    preds = [predict_batch()[0][0] for _ in range(3)]
    pred = np.mean(preds)
    confidence = 1 / (1 + np.std(preds)) if np.std(preds) > 0 else 1.0

    return (1 if pred > 0.001 else -1 if pred < -0.001 else 0), confidence



# Prédire la sortie
def predict_exit(data, symbol):
    if f'{symbol}_15min' not in gru_models:
        return False
    live_df = data.tail(WINDOW_SIZE)
    last_sequence = prepare_last_sequence(live_df['Returns'].values, live_df['Volume'].values,
                                          live_df['Volatility'].values, live_df['Momentum'].values,
                                          live_df['Volume_Anomaly'].values)

    @tf.function(jit_compile=True)  # Activer XLA pour la prédiction
    def predict_batch():
        return gru_models[f'{symbol}_15min'].predict(last_sequence, verbose=0)

    preds = [predict_batch()[0][0] for _ in range(3)]
    return np.mean(preds) < -0.001


# Backtesting avec validation croisée
def backtest(data, symbol, budget, take_profit, stop_loss, rsi_oversold, price_threshold):
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    positions, cash = 0, budget
    equity, trades = [budget], []
    avg_entry_price = 0

    logger.info(f"Backtest pour {symbol} (train: {len(train_data)}, test: {len(test_data)})")

    for df in [train_data, test_data]:
        for i in range(max(MIN_RETURNS_FOR_EWMA, RSI_PERIOD, CORRELATION_WINDOW), len(df)):
            df_slice = df.iloc[:i + 1]
            current_price = df_slice['price'].iloc[-1]
            if current_price <= 0:
                logger.warning(f"Prix invalide pour {symbol} à l'index {i}")
                continue

            mu, sigma = calculate_mu_sigma(df_slice)
            rsi = calculate_rsi(df_slice)
            price_paths = monte_carlo_sim(current_price, mu, sigma, T, N, DT, M)
            median_price = np.median(price_paths[-1])

            buy_signal = (
                cash > 0 and
                rsi < rsi_oversold and
                current_price < median_price * (1 + price_threshold)
            )

            if buy_signal:
                effective_price = current_price * (1 + SLIPPAGE_RATE + TRADING_FEE)
                quantity = (cash / effective_price) * (1 - TRADING_FEE)
                if quantity * current_price >= MIN_NOTIONAL:
                    # Mise à jour positions
                    new_total = positions + quantity
                    avg_entry_price = ((avg_entry_price * positions) + (current_price * quantity)) / new_total
                    positions = new_total
                    cash -= quantity * effective_price
                    trades.append({
                        'entry_price': current_price,
                        'phase': 'train' if i < train_size else 'test'
                    })

            # Signal de sortie
            sell_signal = (
                positions > 0 and (
                    current_price >= median_price * (1 + take_profit) or
                    rsi > RSI_OVERBOUGHT or
                    current_price <= avg_entry_price * (1 + stop_loss)
                )
            )

            if sell_signal:
                effective_price = current_price * (1 - SLIPPAGE_RATE - TRADING_FEE)
                cash += positions * effective_price
                profit = (
                    (current_price - trades[-1]['entry_price']) / trades[-1]['entry_price']
                    - 2 * SLIPPAGE_RATE - 2 * TRADING_FEE
                )
                trades[-1]['profit'] = profit
                trades[-1]['exit_price'] = current_price
                positions = 0

            equity.append(cash + positions * current_price)

    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
    annualized_return = (equity[-1] / equity[0]) ** (8760 / len(equity)) - 1 if len(equity) > 1 else 0
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(8760) if np.std(returns) else 0
    drawdowns = (np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)
    max_drawdown = np.max(drawdowns) if drawdowns.size else 0
    success_rate = np.mean([t.get('profit', 0) > 0 for t in trades]) if trades else 0

    if max_drawdown > 0:
        logger.warning(f"Backtest {symbol}: Max Drawdown = {max_drawdown:.2%}")

    return annualized_return, sharpe_ratio, max_drawdown, success_rate



# Benchmark buy & hold
def benchmark_buy_and_hold(data, initial_budget):
    if len(data) < 2:
        logger.warning("Données insuffisantes pour Buy & Hold.")
        return 0.0

    start_price = data['price'].iloc[0]
    end_price = data['price'].iloc[-1]
    final_value = initial_budget * (end_price / start_price) * (1 - SLIPPAGE_RATE - TRADING_FEE)

    annualized_return = (final_value / initial_budget) ** (8760 / len(data)) - 1
    logger.info(f"Benchmark Buy & Hold: AR = {annualized_return:.2%}")
    return annualized_return



# Optimisation du portefeuille
def optimize_portfolio(mu_values, sigma_values, corr_matrix):
    mu_values = np.asarray(mu_values, dtype=np.float64)
    sigma_values = np.asarray(sigma_values, dtype=np.float64)
    n = len(mu_values)
    weights = np.full(n, 1 / n)
    max_weight = min(0.7, 1 / (1 + np.mean(sigma_values)))

    for _ in range(100):
        portfolio_var = np.sum(weights[:, None] * weights * corr_matrix * np.outer(sigma_values, sigma_values))
        if portfolio_var == 0 or np.isnan(portfolio_var):
            logger.warning("Variance de portefeuille nulle ou NaN.")
            break

        grad = -mu_values / np.sqrt(portfolio_var) + 2 * RISK_AVERSION * weights * portfolio_var
        grad_norm = np.linalg.norm(grad) or 1  # éviter division par zéro
        weights -= 0.01 * grad / grad_norm
        weights = np.clip(weights, MIN_WEIGHT, max_weight)
        weights /= weights.sum()

    return weights



# Calculer le stop-loss suiveur
@jit(nopython=True)
def calculate_trailing_stop(current_price, highest_price):
    return highest_price * (1 - TRAILING_PERCENT)


# Simuler un ordre d'achat
def place_buy_order(symbol, quantity, precision, price, positions):
    executed_qty = round(quantity, precision)
    executed_price = price * (1 + SLIPPAGE_RATE + TRADING_FEE)

    # Écriture du trade
    with open(CONFIG['csv_file'], 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        csv.writer(f).writerow([datetime.now(), symbol, 'buy', executed_qty, price, 0.0])
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    logger.info(f"Achat simulé : {symbol}, Qté = {executed_qty}, Prix exécuté = {executed_price:.4f}")
    positions[symbol] += executed_qty
    return {'executedQty': executed_qty}, positions[symbol]



# Simuler un ordre de vente
def place_sell_order(symbol, quantity, precision, price, positions, entry_price):
    executed_qty = round(quantity, precision)
    executed_price = price * (1 - SLIPPAGE_RATE - TRADING_FEE)

    if entry_price > 0:
        raw_profit_pct = (executed_price - entry_price) / entry_price
        net_profit_pct = raw_profit_pct - (2 * SLIPPAGE_RATE + 2 * TRADING_FEE)
    else:
        net_profit_pct = 0.0

    # Écriture du trade
    with open(CONFIG['csv_file'], 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        csv.writer(f).writerow([datetime.now(), symbol, 'sell', executed_qty, price, net_profit_pct])
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    logger.info(f"Vente simulée : {symbol}, Qté = {executed_qty}, Profit net = {net_profit_pct:.4%}")
    positions[symbol] -= executed_qty
    return {'executedQty': executed_qty}, positions[symbol]



# Ajuster les paramètres
def adjust_parameters(symbol, trading_history):
    trades = [t for t in trading_history if t['symbol'] == symbol and 'profit' in t]
    if len(trades) < MIN_TRADING_DAYS:
        logger.info(f"Pas assez de trades pour ajuster {symbol}")
        return next(p for p in CONFIG['pairs'] if p['symbol'] == symbol)

    win_rate = np.mean([t['profit'] > 0 for t in trades])
    pair = next(p for p in CONFIG['pairs'] if p['symbol'] == symbol).copy()

    if win_rate > LEARNING_CONFIDENCE_THRESHOLD:
        pair['take_profit'] = round(min(pair['take_profit'] * 1.05, 0.03), 4)
        logger.info(f"Ajustement take_profit {symbol} -> {pair['take_profit']}")
    elif win_rate < 0.5 and np.mean([t['profit'] for t in trades if t['profit'] < 0]) < -0.03:
        pair['stop_loss'] = round(max(pair['stop_loss'] * 0.95, -0.10), 4)
        logger.info(f"Ajustement stop_loss {symbol} -> {pair['stop_loss']}")

    return pair



# Modèles VAR
def train_var(data):
    model = VAR(data[['Returns', 'Volatility', 'Momentum', 'Volume_Anomaly']].dropna())
    return model.fit(maxlags=10)

def predict_var(model, steps=1):
    forecast = model.forecast(model.endog, steps)
    return forecast[-1][0]



# Modèle HMM
def train_hmm(data):
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    features = data[['Returns', 'Volatility', 'Momentum', 'Volume_Anomaly']].dropna().values
    model.fit(features)
    return model

def predict_hmm(model, data):
    features = data[['Returns', 'Volatility', 'Momentum', 'Volume_Anomaly']].dropna().values
    states = model.predict(features)
    return np.mean(features[states == np.bincount(states).argmax(), 0])



# Modèle ARIMA
def train_arima(data):
    model = ARIMA(data['Returns'].dropna(), order=(1, 0, 1))
    return model.fit()

def predict_arima(model, steps=1):
    forecast = model.forecast(steps=steps)
    return forecast.iloc[-1]



# Calculer VaR, Calmar, Sortino
def calculate_risk_metrics(returns):
    var_95 = np.percentile(returns, 5)
    calmar = np.mean(returns) / np.max(np.abs(returns)) if np.max(np.abs(returns)) else 0
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(8760) if len(downside_returns) > 0 else 0
    return var_95, calmar, sortino

#Visualisation du Trading Bot

async def trading_bot():
    global gru_models, last_model_save_time, last_training_time

    log_package_versions()
    logger.info("Démarrage du bot de trading avec budget initial de 200 USD")

    # Initialisation
    positions = {pair['symbol']: pair['initial_position'] for pair in CONFIG['pairs']}
    average_entry_prices = {pair['symbol']: 0 for pair in CONFIG['pairs']}
    positions, average_entry_prices = load_transaction_history(positions, average_entry_prices)
    historical_dfs = {pair['symbol']: await fetch_historical_data(pair['symbol']) for pair in CONFIG['pairs']}

    # Chargement ou initialisation des modèles GRU
    gru_models = {}
    for pair in CONFIG['pairs']:
        model_path = os.path.join(CONFIG['model_save_path'], f"{pair['symbol']}_15min_weights.h5")
        if os.path.exists(model_path):
            gru_models[f'{pair["symbol"]}_15min'] = load_model(model_path)
        else:
            models = await optimize_models_async(historical_dfs[pair['symbol']], pair['symbol'])
            gru_models.update(models)

    # Backtests
    for pair in CONFIG['pairs']:
        if not historical_dfs[pair['symbol']].empty:
            backtest(historical_dfs[pair['symbol']], pair['symbol'], pair['budget'], pair['take_profit'],
                     pair['stop_loss'], pair['rsi_oversold'], PRICE_THRESHOLD)
            benchmark_buy_and_hold(historical_dfs[pair['symbol']], pair['budget'])

    # Modèles traditionnels
    var_models = {p['symbol']: train_var(historical_dfs[p['symbol']]) for p in CONFIG['pairs']}
    hmm_models = {p['symbol']: train_hmm(historical_dfs[p['symbol']]) for p in CONFIG['pairs']}
    arima_models = {p['symbol']: train_arima(historical_dfs[p['symbol']]) for p in CONFIG['pairs']}

    last_sync = last_save = last_sentiment = last_budget = time.time()
    initial_budget = float(sum(p['budget'] for p in CONFIG['pairs']))
    trading_history = []
    portfolio = PortfolioManager(CONFIG['pairs'], initial_budget)

    while datetime.fromtimestamp(time.time()) < END_DATE:
        try:
            current_time = time.time()

            # --- Synchronisations périodiques ---
            if current_time - last_sync_time >= SYNC_POSITION_INTERVAL:
                logger.info("Synchronisation des positions.")
                last_sync_time = current_time

            if current_time - last_sentiment_update_time >= SENTIMENT_UPDATE_INTERVAL:
                sentiment = await get_market_sentiment()
                last_sentiment_update_time = current_time

            if current_time - last_training_time >= MODEL_TRAINING_INTERVAL:
                await retrain_models(historical_dfs)
                last_training_time = current_time

            # --- Mise à jour des données historiques ---
            tasks = [fetch_historical_data(pair['symbol']) for pair in CONFIG['pairs']]
            historical_dfs.update({
                pair['symbol']: df for pair, df in zip(CONFIG['pairs'], await asyncio.gather(*tasks)) if not df.empty
            })

            # --- Réallocation du portefeuille ---
            mu_values, sigma_values = zip(*[
                calculate_mu_sigma(historical_dfs[pair['symbol']])
                for pair in CONFIG['pairs'] if
                pair['symbol'] in historical_dfs and not historical_dfs[pair['symbol']].empty
            ])
            corr_matrix = np.array([[calculate_correlation(historical_dfs.get(p1['symbol'], pd.DataFrame()),
                                                           historical_dfs.get(p2['symbol'], pd.DataFrame()))
                                     for p2 in CONFIG['pairs']] for p1 in CONFIG['pairs']])
            weights = portfolio_manager.reallocate(np.array(mu_values), np.array(sigma_values), corr_matrix)

            total_budget = float(sum(float(p['budget']) for p in CONFIG['pairs']))
            adjusted_budgets = {
                p['symbol']: total_budget * w for p, w in zip(CONFIG['pairs'], weights)
            }

            for pair in CONFIG['pairs']:
                pair['budget'] = adjusted_budgets[pair['symbol']]
                pair.update(adjust_parameters(pair['symbol'], trading_history))

            # --- Logique de trading pour chaque paire ---
            for pair in CONFIG['pairs']:
                symbol = pair['symbol']
                df = historical_dfs.get(symbol, pd.DataFrame())
                if df.empty or len(df) < MIN_DATA_POINTS:
                    continue

                current_price = get_current_price(symbol)
                if not current_price:
                    continue

                trend, confidence = predict_price_trend(df, symbol)
                should_trade = should_enter_trade(symbol, df, trend, confidence, current_price)
                quantity = compute_position_size(pair, current_price, confidence)

                if quantity > 0 and should_trade:
                    order, pos = place_buy_order(symbol, quantity, pair['quantity_precision'], current_price, positions)
                    if order:
                        update_after_buy(order, symbol, pos, current_price, positions,
                                         average_entry_prices, trading_history, portfolio_manager)
                        highest_prices[symbol] = current_price

                # Gestion de sortie
                if positions[symbol] > 0:
                    should_exit = should_exit_trade(symbol, df, current_price, average_entry_prices[symbol])
                    if should_exit:
                        order, pos = place_sell_order(symbol, positions[symbol], pair['quantity_precision'],
                                                      current_price, positions)
                        if order:
                            update_after_sell(order, symbol, current_price, average_entry_prices,
                                              positions, trading_history, portfolio_manager)
                            highest_prices[symbol] = 0

            # --- Suivi budget et drawdown ---
            current_budget = portfolio_manager.capital + sum(
                positions[s] * get_current_price(s)
                for s in positions if get_current_price(s)
            )
            drawdown = (current_budget - initial_budget) / initial_budget
            check_drawdown(drawdown)

            if current_time - last_data_save_time >= DATA_SAVE_INTERVAL:
                generate_progress_report(trading_history, historical_dfs)
                last_data_save_time = current_time

            if current_time - last_budget_check_time >= MONTHLY_CHECK_INTERVAL:
                logger.info(
                    f"Budget total à {datetime.now()}: {current_budget:.2f} USD (progression: {(current_budget - initial_budget) / initial_budget:.2%})"
                )
                last_budget_check_time = current_time

            logger.info(f"Attente {INTERVAL}s avant prochain cycle...")
            await asyncio.sleep(INTERVAL)

        except Exception as e:
            logger.error(f"Erreur principale : {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(INTERVAL)


# PortfolioManager
class PortfolioManager:
    def __init__(self, pairs, initial_capital):
        self.pairs = pairs
        self.capital = float(initial_capital)
        self.weights = {pair['symbol']: 1.0 / len(pairs) for pair in pairs}
        self.positions = {pair['symbol']: 0.0 for pair in pairs}

    def reallocate(self, mu_values, sigma_values, corr_matrix):
        try:
            weights = optimize_portfolio(np.array(mu_values), np.array(sigma_values), np.array(corr_matrix))
            weights = [float(w) for w in weights]  # Conversion explicite
            for i, pair in enumerate(self.pairs):
                self.weights[pair['symbol']] = weights[i]
            return weights
        except Exception as e:
            logger.error(f"Erreur dans la réallocation du portefeuille : {e}")
            return [self.weights[pair['symbol']] for pair in self.pairs]

    def update_positions(self, symbol, quantity):
        try:
            quantity = float(quantity)
            price = get_current_price(symbol)
            if price is None or not isinstance(price, (int, float)):
                raise ValueError(f"Prix invalide pour {symbol} : {price}")
            price = float(price)
            cost = quantity * price * (1 + TRADING_FEE + SLIPPAGE_RATE)

            if cost > self.capital:
                logger.warning(f"Pas assez de capital pour acheter {symbol} : coût={cost:.2f} > capital={self.capital:.2f}")
                return False

            self.positions[symbol] += quantity
            self.capital -= cost
            return True
        except Exception as e:
            logger.error(f"Erreur dans update_positions pour {symbol} : {e}")
            return False



# Tests unitaires
def test_calculate_rsi():
    data = pd.DataFrame({'price': [100] * (RSI_PERIOD + 1)})
    data.iloc[RSI_PERIOD] = 101
    rsi = calculate_rsi(data)
    assert 0 <= rsi <= 100, f"RSI hors plage : {rsi}"
    logger.info("✅ Test RSI réussi")

def test_monte_carlo_sim():
    result = monte_carlo_sim(100.0, 0.01, 0.1, T, N, DT, M)
    assert result.shape == (101, M), f"Shape Monte Carlo incorrect : {result.shape}"
    logger.info("✅ Test Monte Carlo réussi")

def test_portfolio_manager():
    pairs = [{'symbol': 'ETH/USD'}, {'symbol': 'LTC/USD'}]
    manager = PortfolioManager(pairs, 1000.0)
    assert abs(sum(manager.weights.values()) - 1.0) < 1e-6
    success = manager.update_positions('ETH/USD', 0.01)
    assert isinstance(success, bool), "update_positions ne retourne pas un booléen"
    logger.info("✅ Test PortfolioManager réussi")



if __name__ == "__main__":
    import os

    # Écrire config.yaml uniquement s'il n'existe pas
    if not os.path.exists('config.yaml'):
        config_yaml = """
        trading:
          historical_periods: 20000
          backtest_periods: 30000
          window_size: 500
          trading_fee: 0.0016
          slippage_rate: 0.001
          interval: 900
          data_save_interval: 900
          rsi_period: 14
          rsi_overbought: 85
          price_threshold: 0.005
          correlation_window: 48
          btc_threshold: -0.02
          min_notional: 10
          ewma_lambda: 0.94
          drawdown_limit: -0.25
          drawdown_warning: -0.20
          negligible_position_threshold: 2
          risk_aversion: 0.3
          min_returns_for_ewma: 50
          default_volatility: 0.7
          min_weight: 0.05
          pnl_sell_threshold: -0.10
          trend_lookback: 60
          api_call_delay: 1.0
          max_retries: 10
          retry_delay: 2
          trailing_percent: 0.02
          sync_position_interval: 600
          sentiment_update_interval: 1800
          monthly_check_interval: 21600
          model_training_interval: 21600
          cache_dir: historical_data_cache
          model_save_path: gru_model_weights
          csv_file: transactions.csv
          data_log_file: historical_data_log.csv
          pairs:
            - symbol: ETH/USD
              kraken_symbol: XETHZUSD
              budget: 66.67
              take_profit: 0.01
              stop_loss: -0.03
              quantity_precision: 4
              rsi_oversold: 80
              initial_position: 0
            - symbol: DOGE/USD
              kraken_symbol: XDGUSD
              budget: 66.67
              take_profit: 0.015
              stop_loss: -0.05
              quantity_precision: 2
              rsi_oversold: 80
              initial_position: 0
            - symbol: LTC/USD
              kraken_symbol: XLTCZUSD
              budget: 66.66
              take_profit: 0.01
              stop_loss: -0.03
              quantity_precision: 4
              rsi_oversold: 80
              initial_position: 0
          coin_mapping:
            ETH/USD: ethereum
            DOGE/USD: dogecoin
            LTC/USD: litecoin
        model:
          gru_units: 64
          learning_rate: 0.001
          dropout_rate: 0.2
          min_trading_days: 30
          learning_confidence_threshold: 0.75
        """
        with open("config.yaml", "w") as f:
            f.write(config_yaml.strip())
        print("✅ config.yaml créé automatiquement.")

    # Lancer serveur Flask pour monitoring santé
    from flask import Flask
    import threading

    app = Flask(__name__)

    @app.route('/')
    def health_check():
        return "Bot is running", 200

    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8081, use_reloader=False), daemon=True).start()
    print("✅ Serveur Flask lancé sur http://localhost:8081")

    # Exécuter les tests unitaires
    try:
        test_calculate_rsi()
        test_monte_carlo_sim()
        test_portfolio_manager()
        print("✅ Tests unitaires passés avec succès.")
    except AssertionError as e:
        print(f"❌ Erreur de test : {e}")

    # Initialiser le bot
    try:
        portfolio_manager = PortfolioManager(CONFIG['pairs'], sum(float(p['budget']) for p in CONFIG['pairs']))
        print("✅ PortfolioManager initialisé.")
        asyncio.run(trading_bot())
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage du bot : {e}")

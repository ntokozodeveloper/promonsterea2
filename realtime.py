from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import yfinance as yf
from datetime import datetime
import pandas as pd
import json
#Keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
# import MetaTrader5 as mt5
import talib
import talib as ta
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import logging
import os
from datetime import datetime, timedelta
import asyncio
from deriv_api import DerivAPI


# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app, cors_allowed_origins="*")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Deriv Connection


async def initialize_deriv_api(app_id, api_token):
    try:
        api = DerivAPI(
            app_id=app_id, endpoint=f'wss://ws.derivws.com/websockets/v3?app_id={app_id}&auth_token={api_token}')
        await api.api_connect()
        logging.info("Successfully connected to DerivAPI")
        return api
    except Exception as e:
        logging.error(f"Failed to connect to DerivAPI: {e}")
        return None


async def fetch_deriv_data(api, symbol, interval, start_date, end_date):
    try:
        interval_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }

        # Get Deriv interval
        deriv_interval = interval_mapping.get(interval, '1d')

         # Ensure start_date and end_date are datetime objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        if deriv_interval == '1d':
            start_date = start_date - timedelta(days=1)
        elif deriv_interval == '1w':
            start_date = start_date - timedelta(days=7)
        elif deriv_interval == '1M':
            start_date = start_date - timedelta(days=30)

        # Convert start_date to UNIX timestamp
        start_timestamp = int(start_date.timestamp())
        
        # For end_date, use 'latest' as a string
        end_param = 'latest'  # API often uses 'latest' for the end date

        # Log the request details. Symbol, Interval, Start and End Date Logging.
        logging.debug(
            f"Fetching Deriv data for {symbol} with interval {interval} from {start_date} to {end_date}")

        ticks_history_request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 1000,
            "start": 1,  # Use start date directly as a string
            "end": "latest",  # Use "latest" directly
            "style": "candles"
        }

        response = await api.send(ticks_history_request)
        if 'error' in response:
            logging.error(
                f"Error fetching Deriv data: {response['error']['message']}")
            return pd.DataFrame()

        if 'candles' not in response:
            logging.warning("Response does not contain 'candles' data.")
            return pd.DataFrame()

        data = response['candles']
        df = pd.DataFrame(data)

        # Ensure 'volume' field is present in the DataFrame
        if 'volume' not in df.columns:
            logging.warning("'volume' column is missing in the response data.")
            df['volume'] = 0  # Add a default volume column if missing

        df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                  'close': 'Close', 'volume': 'Volume'}, inplace=True)
        logging.info(
            f"Successfully fetched Deriv data for {symbol} with interval {interval}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logging.error(f"Error fetching Deriv data: {e}")
        return pd.DataFrame()


def combine_data(yf_data, deriv_data):
    if yf_data is None or yf_data.empty:
        logging.warning(
            "Yahoo Finance data is empty or not available. Using only Deriv data.")
        return deriv_data
    if deriv_data is None or deriv_data.empty:
        logging.warning(
            "Deriv data is empty or not available. Using only Yahoo Finance data.")
        return yf_data

    combined_data = pd.concat([yf_data, deriv_data])
    combined_data = combined_data[~combined_data.index.duplicated(
        keep='first')]
    combined_data.sort_index(inplace=True)
    logging.info("Successfully combined data from Yahoo Finance and Deriv")
    return combined_data


def calculate_probability(atr, close_price):
    max_atr = close_price * 0.1
    probability = (atr / max_atr) * 100
    return round(probability, 2)


def recommend_lot_size(probability, account_balance):
    risk_per_trade = 0.01
    if probability >= 80:
        lot_size_factor = 0.02
    elif probability >= 60:
        lot_size_factor = 0.015
    elif probability >= 40:
        lot_size_factor = 0.01
    else:
        lot_size_factor = 0.005

    lot_size = account_balance * risk_per_trade * lot_size_factor
    return round(lot_size, 2)


def download_data(ticker_symbol, interval, start_date, end_date, retries=3):
    logging.info(
        f"Downloading Yahoo Finance data for {ticker_symbol} from {start_date} to {end_date} with interval {interval}")
    for attempt in range(retries):
        try:
            data = yf.download(ticker_symbol, start=start_date,
                               end=end_date, interval=interval)
            if data.empty:
                logging.warning(
                    f"No data available for {ticker_symbol} at interval {interval} on attempt {attempt + 1}")
                continue
            logging.info(
                f"Successfully downloaded data for {ticker_symbol} at interval {interval} on attempt {attempt + 1}")
            return data
        except Exception as e:
            logging.error(
                f"Error downloading data for {ticker_symbol} at interval {interval} on attempt {attempt + 1}: {e}")
    logging.error(
        f"Failed to download data for {ticker_symbol} after {retries} attempts")
    return None


def process_data(data_frames, data, ticker_symbol, interval, start_date, end_date):
    signals = {}

    for interval in intervals:
        # Assuming data is a dictionary with intervals as keys
        interval_data = data[interval]

        # Calculate candlestick patterns
        interval_data['Engulfing'] = ta.CDLENGULFING(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])
        interval_data['Doji'] = ta.CDLDOJI(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])
        interval_data['Hammer'] = ta.CDLHAMMER(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])
        interval_data['ShootingStar'] = ta.CDLSHOOTINGSTAR(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])
        interval_data['MorningStar'] = ta.CDLMORNINGSTAR(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])
        interval_data['EveningStar'] = ta.CDLEVENINGSTAR(
            interval_data['Open'], interval_data['High'], interval_data['Low'], interval_data['Close'])

        # Calculate MACD
        interval_data['MACD'], interval_data['MACD_Signal'], interval_data['MACD_Hist'] = ta.MACD(
            interval_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Calculate Stochastic Oscillator
        interval_data['%K'], interval_data['%D'] = ta.STOCH(
            interval_data['High'], interval_data['Low'], interval_data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)

        # Calculate Moving Averages
        interval_data['SMA_20'] = ta.SMA(interval_data['Close'], timeperiod=20)
        interval_data['SMA_50'] = ta.SMA(interval_data['Close'], timeperiod=50)
        interval_data['SMA_100'] = ta.SMA(
            interval_data['Close'], timeperiod=100)
        interval_data['EMA_20'] = ta.EMA(interval_data['Close'], timeperiod=20)
        interval_data['EMA_50'] = ta.EMA(interval_data['Close'], timeperiod=50)
        interval_data['EMA_100'] = ta.EMA(
            interval_data['Close'], timeperiod=100)
        interval_data['EMA_200'] = ta.EMA(
            interval_data['Close'], timeperiod=200)

        # Calculate RSI
        interval_data['RSI'] = ta.RSI(interval_data['Close'], timeperiod=14)

        # Calculate Bollinger Bands
        interval_data['Upper_BB'], interval_data['Middle_BB'], interval_data['Lower_BB'] = ta.BBANDS(
            interval_data['Close'], timeperiod=20)

        # Calculate ATR
        interval_data['ATR'] = ta.ATR(
            interval_data['High'], interval_data['Low'], interval_data['Close'], timeperiod=14)

        # Calculate OBV
        interval_data['OBV'] = ta.OBV(
            interval_data['Close'], interval_data['Volume'])
        
        #Other TAs for the AI
        data['EMA_50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
        data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
        data['Upper_BB'], data['Lower_BB'] = ta.volatility.BollingerBands(data['Close'], window=20).bollinger_hband(), ta.volatility.BollingerBands(data['Close'], window=20).bollinger_lband()
        data['%K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3).stoch()
        data['%D'] = data['%K'].rolling(window=3).mean()

        # Example signals generation
        if len(interval_data) >= 50:
            interval_data = add_all_ta_features(
                interval_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        if len(interval_data) >= 15:
            interval_data = add_all_ta_features(
                interval_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Define buy and sell conditions (Active: EMA50 & 100, Stochastic and BB)
        buy_conditions = [
            (interval_data['EMA_50'] < interval_data['Close']),
            #(interval_data['RSI'] < 30),
            #(interval_data['MACD_Hist'] > 0),
            (interval_data['Close'] < interval_data['Lower_BB']),
            (interval_data['%K'] < interval_data['%D']),
        ]

        sell_conditions = [
            (interval_data['EMA_50'] > interval_data['Close']),
            #(interval_data['RSI'] > 70),
            #(interval_data['MACD_Hist'] < 0),
            (interval_data['Close'] > interval_data['Upper_BB']),
            (interval_data['%K'] > interval_data['%D']),
        ]

        # Generate buy and sell signals
        interval_data['Buy_Signal'] = np.where(
            np.all(buy_conditions, axis=0), 1, 0)
        interval_data['Sell_Signal'] = np.where(
            np.all(sell_conditions, axis=0), -1, 0)

        #AI Compent
        interval_data['Buy_Signal'] = np.where(np.all(buy_conditions, axis=0), 1, 0)
        interval_data['Sell_Signal'] = np.where(np.all(sell_conditions, axis=0), -1, 0)
        interval_data['Signal'] = interval_data['Buy_Signal'] + interval_data['Sell_Signal']
        interval_data['Position'] = interval_data['Signal'].diff()

        # Combine buy and sell signals into a single signal
        interval_data['Signal'] = interval_data['Buy_Signal'] + \
            interval_data['Sell_Signal']
        interval_data['Position'] = interval_data['Signal'].diff()

        signals[interval] = interval_data.iloc[-1] if not interval_data.empty else None
        logging.info(
            f"Processed data for interval {interval}: {signals[interval]}")

    return signals


intervals = ['1d', '4h', '2h', '1h', '30m', '15m']


def generate_trade_signals(signals):
    logging.info("Generating trade signals")

    # Define default values
    trade_signal = 'hold'
    entry_price = None
    probability = 0
    interval = 'No signal'

    # Iterate over the specified intervals to find the most recent signal
    for i in ['1d', '1h', '2h', '4h', '30m', '15m']:
        if signals.get(i) is not None:
            recent_signal = signals[i]
            trade_signal = 'buy' if recent_signal['Signal'] == 1 else 'sell'
            entry_price = recent_signal['Close']
            probability = calculate_probability(
                recent_signal['ATR'], recent_signal['Close'])
            logging.info(f"Trade signal: {trade_signal} for interval {i}")
            interval = i
            break  # Exit loop once a signal is found

    if trade_signal == 'hold':
        logging.info("No trade signal generated")

    return trade_signal, entry_price, probability, interval

## Integrating EA with AI
# Training The AI Model (Intergrating AI with the EA)

# Function to create and train a RandomForestRegressor model
def create_and_train_model(data, target):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Function to download data from Yahoo Finance
def download_data(ticker, interval, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval)

# Function to combine data (placeholder)
def combine_data(yf_data, deriv_data):
    return yf_data

# Function to process data (placeholder)
def process_data(*args):
    return pd.DataFrame()

# Function to generate trade signals (placeholder)
def generate_trade_signals(*args):
    return 'BUY', 1.0, 95, '1d'

# Training the Model on Yahoo Finance Data 
forex_pairs = [
    'GBPJPY',
    'EURJPY',
    'AUDJPY',
    'EURUSD',
    'GBPUSD',
    'AUDUSD',
    'USDJPY',
    'USDZAR',
    'USDCAD',
    'USDCHF',
    'EURGBP',
    'EURCHF',
    'GBPAUD',
    'EURNZD',
    'NZDJPY',
    'CADCHF',
    'GBPNZD'
]

if __name__ == "__main__":
    interval = '1d'  # You can change this to your desired interval
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    for symbol in forex_pairs:
        ticker_symbol = f'{symbol}=X'

    yf_data = download_data(ticker_symbol, interval, start_date_str, end_date_str)
    deriv_data = None  # Assume we have another function to fetch this data
    combined_data = combine_data(yf_data, deriv_data)

    if combined_data is not None:
        signals = process_data(None, {"1d": combined_data}, ticker_symbol, interval, start_date_str, end_date_str)
        trade_signal, entry_price, probability, interval = generate_trade_signals(signals)
        logging.info(f"Trade Signal: {trade_signal}, Entry Price: {entry_price}, Probability: {probability}%, Interval: {interval}")

        # Training data using EMA_50, Stochastic, and Bollinger Bands (Simple, Forex Rule. Buy Low, Sell High. Sell High and Buy Low)
            # Ensure the indicators are in the data before using them
        if all(indicator in combined_data.columns for indicator in ['EMA_50', 'RSI', 'Lower_BB', 'Upper_BB', '%K', '%D']):
                features = combined_data[['Close', 'Volume', 'EMA_50', 'RSI', 'Lower_BB', 'Upper_BB', '%K', '%D']]
                target = combined_data['Close'].shift(-1)  # Predict next day's close price

                features.dropna(inplace=True)
                target = target[features.index]

                model, scaler = create_and_train_model(features, target)
                logging.info("Model training completed")
        else:
                logging.error("Indicators are missing from the data.")

# Posting the data to the Server (placeholder)
# The idea with Yahoo Finance is getting historical data from the past 60 Days
# Supported by YF by Default

@app.route('/api/trade', methods=['POST'])
async def trade():
    logging.info("Received request at /api/trade")
    data = request.json

    symbol = data.get('symbol')
    amount = data.get('amount')
    contract_type = data.get('contract_type')
    app_id = data.get('app_id')
    api_token = data.get('api_token')
    stop_loss_percent = data.get('stop_loss_percent', 1)
    take_profit_percent = data.get('take_profit_percent', 2)

    if not symbol or not amount or not contract_type or not app_id or not api_token:
        return jsonify({'errors': 'Invalid input'}), 400

    ticker_symbol = f'{symbol}=X'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']
    data_frames = {}  # Initialize as an empty dictionary

    api = await initialize_deriv_api(app_id, api_token)
    if api is None:
        return jsonify({'errors': 'Failed to connect to DerivAPI'}), 500

    for interval in intervals:
        yf_data = download_data(ticker_symbol, interval, start_date, end_date)
        deriv_data = await fetch_deriv_data(api, symbol, interval, start_date, 'latest') if yf_data is None or len(yf_data) < 50 else None
        combined_data = combine_data(yf_data, deriv_data)
        if combined_data is not None and not combined_data.empty:
            data_frames[interval] = combined_data

    signals = process_data(data, ticker_symbol, interval, start_date, end_date)
    trade_signal, entry_price, probability, timeframe_displayed = generate_trade_signals(
        signals)

    if trade_signal != 'hold':
        stop_loss = entry_price * \
            (1 - stop_loss_percent / 100) if trade_signal == 'buy' else entry_price * \
            (1 + stop_loss_percent / 100)
        take_profit = entry_price * \
            (1 + take_profit_percent / 100) if trade_signal == 'buy' else entry_price * \
            (1 - take_profit_percent / 100)
    else:
        stop_loss = None
        take_profit = None

    lot_sizes = {f'Account Balance {balance}': recommend_lot_size(
        probability, balance) for balance in [100, 200, 500, 1000, 2000, 5000, 10000]}

    result = {
        'status': 'success',
        'message': f'Trade signal: {trade_signal}',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'probability': f'{probability}%',
        'recommended_lot_sizes': lot_sizes,
        'timeframe_displayed': timeframe_displayed,
    }

    return jsonify(result)

def get_real_trade_data():
    # Placeholder for actual data retrieval logic / This Should not affect how Trading Signals are displayed.
    trade_signal = 'BUY'  # Retrieve actual trade signal
    entry_price = 1.2345  # Retrieve actual entry price
    stop_loss = 1.2300  # Retrieve actual stop loss
    take_profit = 1.2400  # Retrieve actual take profit
    probability = 85.0  # Calculate actual probability
    lot_sizes = [0.01, 0.1, 1.0]  # Calculate recommended lot sizes
    timeframe_displayed = '1h'  # Retrieve actual timeframe

    return {
        'trade_signal': trade_signal,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'probability': probability,
        'lot_sizes': lot_sizes,
        'timeframe_displayed': timeframe_displayed
    }

@app.route('/getTradeData', methods=['GET'])
def get_trade_data():
    trade_data = get_real_trade_data()

    result = {
        'status': 'success',
        'message': f'Trade signal: {trade_data["trade_signal"]}',
        'entry_price': trade_data['entry_price'],
        'stop_loss': trade_data['stop_loss'],
        'take_profit': trade_data['take_profit'],
        'probability': f'{trade_data["probability"]}%',
        'recommended_lot_sizes': trade_data['lot_sizes'],
        'timeframe_displayed': trade_data['timeframe_displayed'],
    }

    return jsonify(result)

# Sending Order Request to MT5

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
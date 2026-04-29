import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Standard implementation of common technical indicators."""
    
    @staticmethod
    def calculate_ema(prices, period=9):
        """Exponential Moving Average."""
        if len(prices) < period:
            return pd.Series(prices).expanding().mean()
        return pd.Series(prices).ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index."""
        if len(prices) < period:
            return pd.Series([50] * len(prices))
        
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        """Moving Average Convergence Divergence."""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_atr(df, period=14):
        """Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_adx(df, period=14):
        """Average Directional Index."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Bollinger Bands."""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_obv(df):
        """On-Balance Volume."""
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

class DataPreprocessor:
    """Pipeline for fetching, cleaning, and preparing trading data for ML."""
    
    def __init__(self, stock_map=None):
        self.stock_map = stock_map or {}
        self.scaler = StandardScaler()

    def get_mapped_symbol(self, symbol):
        """Map common names to yfinance symbols."""
        symbol = symbol.strip().upper()
        if symbol in self.stock_map:
            return self.stock_map[symbol]
        
        # Default logic if not in map
        if '.' not in symbol and '=' not in symbol and not symbol.startswith('^'):
            # Check if it's a known US stock, otherwise default to .NS
            us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'}
            if symbol not in us_stocks:
                return symbol + ".NS"
        return symbol

    def fetch_data(self, symbol, period='1y', interval='1d'):
        """Fetch historical data using yfinance."""
        mapped = self.get_mapped_symbol(symbol)
        print(f"Fetching data for {symbol} (Mapped: {mapped})...")
        df = yf.download(mapped, period=period, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for {mapped}")
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df

    def clean_data(self, df):
        """Basic data cleaning: handling NaNs and duplicates."""
        df = df.copy()
        # Drop rows where critical columns are NaN
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        # Forward fill any other gaps
        df = df.ffill()
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        return df

    def enrich_features(self, df):
        """Add technical indicators to the dataframe."""
        df = df.copy()
        ti = TechnicalIndicators()
        
        # Trend
        df['EMA_9'] = ti.calculate_ema(df['Close'], 9)
        df['EMA_21'] = ti.calculate_ema(df['Close'], 21)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Momentum
        df['RSI'] = ti.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ti.calculate_macd(df['Close'])
        
        # Volatility
        df['ATR'] = ti.calculate_atr(df)
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = ti.calculate_bollinger_bands(df['Close'])
        
        # Strength
        df['ADX'] = ti.calculate_adx(df)
        
        # Volume
        df['OBV'] = ti.calculate_obv(df)
        df['Vol_Avg_20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg_20']
        
        # Price Change
        df['Returns'] = df['Close'].pct_change()
        
        return df

    def create_labels(self, df, window=5, target_pct=0.02, stop_pct=0.01):
        """
        Create labels for ML training (Triple Barrier Method).
        1: Profit target hit
        -1: Stop loss hit
        0: Neither hit (neutral/time exit)
        """
        df = df.copy()
        labels = []
        
        prices = df['Close'].values
        for i in range(len(prices)):
            if i + window >= len(prices):
                labels.append(0)
                continue
                
            entry_p = prices[i]
            label = 0
            for j in range(1, window + 1):
                future_p = prices[i + j]
                ret = (future_p - entry_p) / entry_p
                
                if ret >= target_pct:
                    label = 1
                    break
                elif ret <= -stop_pct:
                    label = -1
                    break
            labels.append(label)
            
        df['Target'] = labels
        return df

    def prepare_ml_ready_data(self, symbol, period='2y', fit_scaler=True):
        """Full pipeline from raw symbol to scaled features and labels."""
        df = self.fetch_data(symbol, period=period)
        df = self.clean_data(df)
        df = self.enrich_features(df)
        df = self.create_labels(df)
        
        # Select features (similar to AIEngine logic)
        features = [
            'RSI', 'MACD', 'MACD_Hist', 'ADX', 'ATR', 
            'EMA_9', 'EMA_21', 'Vol_Ratio', 'Returns'
        ]
        
        # Drop first few rows with NaN from indicators
        df_ml = df.dropna().copy()
        
        X = df_ml[features].values
        y = df_ml['Target'].values
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y, df_ml

# ── Example Usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import map from existing app if available
    try:
        from ai_prediction_app import STOCK_MAP
    except ImportError:
        STOCK_MAP = {'RELIANCE': 'RELIANCE.NS', 'NIFTY': '^NSEI'}

    preprocessor = DataPreprocessor(stock_map=STOCK_MAP)
    
    symbol = "RELIANCE"
    try:
        X, y, df = preprocessor.prepare_ml_ready_data(symbol, period='1mo')
        print(f"Preprocessing complete for {symbol}")
        print(f"Feature Matrix Shape: {X.shape}")
        print(f"Label Distribution: {pd.Series(y).value_counts().to_dict()}")
        print("\nLast 5 rows of enriched data:")
        print(df[['Close', 'RSI', 'MACD', 'Target']].tail())
        
        # Save preprocessed data
        output_dir = "preprocessed_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        csv_path = f"{output_dir}/{symbol}_preprocessed.csv"
        df.to_csv(csv_path)
        print(f"\nSaved preprocessed data to {csv_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")


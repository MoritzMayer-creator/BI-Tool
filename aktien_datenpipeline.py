import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentiment_analyzer import SentimentAnalyzer
from reddit_sentiment import RedditSentimentAnalyzer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, Attention

class AktienDatenPipeline:
   def __init__(self):
       """Initialisierung der Aktien-Datenpipeline"""
       self.daten = None
       self.symbol = None
       self.sentiment_analyzer = SentimentAnalyzer()
       self.reddit_analyzer = RedditSentimentAnalyzer()
       self.news_sentiment_score = 0
       self.combined_sentiment_score = 0
       
   def hole_daten(self, symbol, zeitraum="2y"):
       """
       Lädt historische Aktienkursdaten sowie Sentiment-Analysen von Yahoo Finance und Reddit.
       
       Parameter:
           symbol (str): Das Aktiensymbol (z.B. 'AAPL' für Apple)
           zeitraum (str): Zeitraum für historische Daten (Standard: '2y' für 2 Jahre)
           
       Returns:
           pandas.DataFrame: DataFrame mit den geladenen Aktiendaten
       """
       self.symbol = symbol
       
       # Hole Aktiendaten für 2 Jahre statt 1 Jahr
       aktie = yf.Ticker(symbol)
       self.daten = aktie.history(period=zeitraum, interval='1d')
       
       # Falls nicht genug Daten, versuche 5 Jahre
       if len(self.daten) < 100:  # Mindestanforderung nicht erfüllt
           print(f"Zu wenig Daten für {zeitraum}, versuche 5 Jahre...")
           self.daten = aktie.history(period="5y", interval='1d')
       
       if not self.daten.empty:
           heute = pd.Timestamp.now().date()
           if self.daten.index[-1].date() < heute:
               neueste_daten = aktie.history(period='1d', interval='1m')
               if not neueste_daten.empty:
                   self.daten = pd.concat([self.daten, neueste_daten.resample('1d').last()])
       
       # Hole News Sentiment
       try:
           nachrichten_df = self.sentiment_analyzer.hole_nachrichten(symbol, max_nachrichten=5)
           if not nachrichten_df.empty:
               self.sentiment_score, self.sentiment_positiv = self.sentiment_analyzer.berechne_gesamt_sentiment(nachrichten_df)
               self.news_sentiment_score = self.sentiment_score
               self.sentiment_nachrichten = nachrichten_df
           else:
               self.sentiment_score = 0
               self.news_sentiment_score = 0
               self.sentiment_positiv = 0
               self.sentiment_kategorie = "Neutral"
               self.sentiment_nachrichten = pd.DataFrame()
       except Exception as e:
           print(f"Fehler beim Laden der Sentiment-Daten: {str(e)}")
           self.sentiment_score = 0
           self.news_sentiment_score = 0
           self.sentiment_positiv = 0
           self.sentiment_kategorie = "Neutral"
           self.sentiment_nachrichten = pd.DataFrame()

       # Hole Reddit Sentiment
       try:
           reddit_df = self.reddit_analyzer.hole_reddit_posts(symbol)
           if not reddit_df.empty:
               reddit_df = self.reddit_analyzer.analysiere_reddit_sentiment(reddit_df)
               self.reddit_sentiment_score, self.reddit_positiv, self.reddit_stats = (
                   self.reddit_analyzer.berechne_reddit_gesamt_sentiment(reddit_df)
               )
               self.reddit_posts = (
                   reddit_df.sort_values('score', ascending=False)
                   .head(5)
                   .to_dict('records')
               )
               
               # Calculate combined sentiment (60% news, 40% reddit)
               self.combined_sentiment_score = (self.news_sentiment_score * 0.6) + (self.reddit_sentiment_score * 0.4)
               self.sentiment_kategorie = self.sentiment_analyzer.klassifiziere_sentiment(self.combined_sentiment_score)
           else:
               self.reddit_sentiment_score = 0
               self.reddit_positiv = 0
               self.reddit_stats = {}
               self.reddit_posts = []
               self.combined_sentiment_score = self.news_sentiment_score
       except Exception as e:
           print(f"Fehler beim Laden der Reddit-Daten: {str(e)}")
           self.reddit_sentiment_score = 0
           self.reddit_positiv = 0
           self.reddit_stats = {}
           self.reddit_posts = []
           self.combined_sentiment_score = self.news_sentiment_score
           
       return self.daten
   
   def berechne_indikatoren(self):
       """
       Berechnet verschiedene technische Indikatoren für die Aktienanalyse.
       
       Enthält:
       - Tägliche Rendite und Volatilität
       - Gleitende Durchschnitte (20 und 50 Tage)
       - RSI (Relative Strength Index)
       - MACD (Moving Average Convergence Divergence)
       - Bollinger Bänder
       - Handelssignale basierend auf Durchschnitten
       
       Raises:
           ValueError: Wenn keine Daten geladen wurden
       """
       if self.daten is None:
           raise ValueError("Keine Daten verfügbar. Bitte zuerst Daten laden.")
           
       # Grundlegende Kennzahlen
       self.daten['Tägliche_Rendite'] = self.daten['Close'].pct_change()
       self.daten['GD20'] = self.daten['Close'].rolling(window=20).mean()
       self.daten['GD50'] = self.daten['Close'].rolling(window=50).mean()
       self.daten['Volatilität'] = self.daten['Tägliche_Rendite'].rolling(window=20).std()
       self.daten['Momentum'] = self.daten['Close'].pct_change(periods=20)
       
       # RSI(14) mit korrekter EMA-Berechnung
       delta = self.daten['Close'].diff()
       gain = delta.where(delta > 0, 0)
       loss = -delta.where(delta < 0, 0)
       
       # Erste 14 Perioden für Initialisierung
       avg_gain = gain.rolling(window=14).mean()
       avg_loss = loss.rolling(window=14).mean()
       
       # Weitere Perioden mit korrekter EMA-Berechnung
       for i in range(14, len(gain)):
           avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
           avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
       
       rs = avg_gain / avg_loss
       self.daten['RSI(14)'] = 100 - (100 / (1 + rs))
       
       # MACD
       exp1 = self.daten['Close'].ewm(span=12, adjust=False).mean()
       exp2 = self.daten['Close'].ewm(span=26, adjust=False).mean()
       self.daten['MACD'] = exp1 - exp2
       self.daten['Signal'] = self.daten['MACD'].ewm(span=9, adjust=False).mean()
       self.daten['MACD_Hist'] = self.daten['MACD'] - self.daten['Signal']
       
       # Bollinger Bänder
       self.daten['BB_Mitte'] = self.daten['Close'].rolling(window=20).mean()
       std20 = self.daten['Close'].rolling(window=20).std()
       self.daten['BB_Oben'] = self.daten['BB_Mitte'] + (std20 * 2)
       self.daten['BB_Unten'] = self.daten['BB_Mitte'] - (std20 * 2)
       
       # Handelssignale
       self.daten['Signal_GD'] = 0
       self.daten.loc[self.daten['GD20'] > self.daten['GD50'], 'Signal_GD'] = 1
       self.daten.loc[self.daten['GD20'] < self.daten['GD50'], 'Signal_GD'] = -1
       
       return self.daten
   
   def hole_zusammenfassung(self):
       """
       Erstellt eine kompakte Übersicht der wichtigsten Kennzahlen und Analysen.
       
       Berechnet und gibt zurück:
       - Aktuelle Kursdaten
       - Performance-Metriken
       - Technische Indikatoren
       - Sentiment-Analyse
       - Handelssignale
       
       Raises:
           ValueError: Wenn keine Daten geladen wurden
       """
       if self.daten is None:
           raise ValueError("Keine Daten verfügbar. Bitte zuerst Daten laden.")
           
       statistiken = {
           'symbol': self.symbol,
           'aktueller_preis': round(self.daten['Close'].iloc[-1], 2),
           'tages_rendite': round(self.daten['Tägliche_Rendite'].iloc[-1] * 100, 2),
           'volatilität': round(self.daten['Tägliche_Rendite'].std() * np.sqrt(252) * 100, 2),
           'jahres_hoch': round(self.daten['High'].max(), 2),
           'jahres_tief': round(self.daten['Low'].min(), 2),
           'rsi': round(self.daten['RSI(14)'].iloc[-1], 2),
           'macd': round(self.daten['MACD'].iloc[-1], 2),
           'sentiment_score': self.combined_sentiment_score,
           'sentiment_positiv': self.sentiment_positiv,
           'sentiment_kategorie': self.sentiment_kategorie,
           'trend_signal': 'Kaufen' if self.daten['Signal_GD'].iloc[-1] == 1 else 
                         'Verkaufen' if self.daten['Signal_GD'].iloc[-1] == -1 else 'Halten'
       }
       
       return statistiken

   def erweiterte_feature_engineering(self):
       """
       Führt erweitertes Feature-Engineering für präzisere LSTM-Prognosen durch.
       
       Berechnet zusätzliche Indikatoren:
       - Momentum-Indikatoren (ROC, MFI)
       - Trend-Indikatoren (ADX, Supertrend)
       - Volatilitäts-Indikatoren (ATR, Bollinger-Band-Breite)
       - Volumen-Indikatoren (OBV, VWAP)
       - Saisonale Merkmale
       
       Raises:
           ValueError: Wenn keine Basisdaten verfügbar sind
       """
       if self.daten is None:
           raise ValueError("Keine Daten verfügbar")

       # Technische Indikatoren
       # Momentum Indikatoren
       self.daten['ROC'] = self.daten['Close'].pct_change(periods=12)  # Rate of Change
       self.daten['MFI'] = self._calculate_mfi(14)  # Money Flow Index
       
       # Trend Indikatoren
       self.daten['ADX'] = self._calculate_adx(14)  # Average Directional Index
       self.daten['Supertrend'] = self._calculate_supertrend(10, 3)
       
       # Volatilitäts Indikatoren
       self.daten['ATR'] = self._calculate_atr(14)  # Average True Range
       self.daten['BB_Width'] = (self.daten['BB_Oben'] - self.daten['BB_Unten']) / self.daten['BB_Mitte']
       
       # Volumen Indikatoren
       self.daten['OBV'] = self._calculate_obv()  # On Balance Volume
       self.daten['VWAP'] = self._calculate_vwap()  # Volume Weighted Average Price
       
       # Saisonale Features
       self.daten['DayOfWeek'] = self.daten.index.dayofweek
       self.daten['MonthOfYear'] = self.daten.index.month
       self.daten['QuarterOfYear'] = self.daten.index.quarter

       return self.daten

   def _calculate_mfi(self, period):
       """
       Berechnet den Money Flow Index (MFI).
       
       Der MFI ist ein technischer Oszillator, der Preis und Volumen kombiniert,
       um überkaufte oder überverkaufte Bedingungen zu identifizieren.
       """
       typical_price = (self.daten['High'] + self.daten['Low'] + self.daten['Close']) / 3
       money_flow = typical_price * self.daten['Volume']
       
       positive_flow = pd.Series(0.0, index=self.daten.index)
       negative_flow = pd.Series(0.0, index=self.daten.index)
       
       # Berechne positive und negative Flows
       price_diff = typical_price.diff()
       positive_flow[price_diff > 0] = money_flow[price_diff > 0]
       negative_flow[price_diff < 0] = money_flow[price_diff < 0]
       
       positive_mf = positive_flow.rolling(window=period).sum()
       negative_mf = negative_flow.rolling(window=period).sum()
       
       mfi = 100 - (100 / (1 + positive_mf / negative_mf))
       return mfi

   def _calculate_adx(self, period):
       """
       Berechnet den Average Directional Index (ADX).
       
       Der ADX misst die Stärke eines Trends, unabhängig von seiner Richtung.
       Werte über 25 deuten auf einen starken Trend hin.
       """
       plus_dm = self.daten['High'].diff()
       minus_dm = self.daten['Low'].diff()
       
       plus_dm[plus_dm < 0] = 0
       minus_dm[minus_dm > 0] = 0
       
       tr = pd.DataFrame(index=self.daten.index)
       tr['h-l'] = self.daten['High'] - self.daten['Low']
       tr['h-c'] = abs(self.daten['High'] - self.daten['Close'].shift())
       tr['l-c'] = abs(self.daten['Low'] - self.daten['Close'].shift())
       tr['tr'] = tr[['h-l', 'h-c', 'l-c']].max(axis=1)
       
       plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr['tr'].ewm(alpha=1/period).mean())
       minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr['tr'].ewm(alpha=1/period).mean())
       
       dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
       adx = dx.ewm(alpha=1/period).mean()
       
       return adx

   def _calculate_supertrend(self, period, multiplier):
       """
       Berechnet den Supertrend-Indikator.
       
       Ein Trendfolge-Indikator, der Unterstützungs- und Widerstandslinien
       basierend auf ATR und Preisbewegungen generiert.
       """
       hl2 = (self.daten['High'] + self.daten['Low']) / 2
       atr = self._calculate_atr(period)
       
       upperband = hl2 + (multiplier * atr)
       lowerband = hl2 - (multiplier * atr)
       
       supertrend = pd.Series(index=self.daten.index)
       supertrend.iloc[0] = upperband.iloc[0]
       
       for i in range(1, len(self.daten.index)):
           curr = self.daten['Close'].iloc[i]
           prev = supertrend.iloc[i-1]
           
           if curr > upperband.iloc[i]:
               supertrend.iloc[i] = lowerband.iloc[i]
           elif curr < lowerband.iloc[i]:
               supertrend.iloc[i] = upperband.iloc[i]
           else:
               supertrend.iloc[i] = prev
                
       return supertrend

   def _calculate_atr(self, period):
       """
       Berechnet den Average True Range (ATR).
       
       Ein Volatilitätsindikator, der die durchschnittliche Preisspanne
       über einen bestimmten Zeitraum misst.
       """
       tr = pd.DataFrame(index=self.daten.index)
       tr['h-l'] = self.daten['High'] - self.daten['Low']
       tr['h-c'] = abs(self.daten['High'] - self.daten['Close'].shift())
       tr['l-c'] = abs(self.daten['Low'] - self.daten['Close'].shift())
       tr['tr'] = tr[['h-l', 'h-c', 'l-c']].max(axis=1)
       
       return tr['tr'].rolling(window=period).mean()

   def _calculate_obv(self):
       """
       Berechnet das On Balance Volume (OBV).
       
       Ein kumulativer Indikator, der Volumenänderungen in Relation zu
       Preisbewegungen analysiert und Trendsignale generiert.
       """
       obv = pd.Series(0.0, index=self.daten.index)
       obv.iloc[0] = self.daten['Volume'].iloc[0]
       
       for i in range(1, len(self.daten.index)):
           if self.daten['Close'].iloc[i] > self.daten['Close'].iloc[i-1]:
               obv.iloc[i] = obv.iloc[i-1] + self.daten['Volume'].iloc[i]
           elif self.daten['Close'].iloc[i] < self.daten['Close'].iloc[i-1]:
               obv.iloc[i] = obv.iloc[i-1] - self.daten['Volume'].iloc[i]
           else:
               obv.iloc[i] = obv.iloc[i-1]
                
       return obv

   def _calculate_vwap(self):
       """
       Berechnet den Volume Weighted Average Price (VWAP).
       
       Ein gewichteter Durchschnittspreis, der das Handelsvolumen
       berücksichtigt und wichtige Preisniveaus identifiziert.
       """
       typical_price = (self.daten['High'] + self.daten['Low'] + self.daten['Close']) / 3
       return (typical_price * self.daten['Volume']).cumsum() / self.daten['Volume'].cumsum()

if __name__ == "__main__":
    """
    Beispielhafte Ausführung der Datenpipeline mit Apple-Aktien (AAPL).
    Demonstriert die grundlegende Verwendung der Klasse.
    """
    # Test der Pipeline
    pipeline = AktienDatenPipeline()
    df = pipeline.hole_daten('AAPL')
    df = pipeline.berechne_indikatoren()
    stats = pipeline.hole_zusammenfassung()
    print("\nZusammenfassung:")
    for key, value in stats.items():
        print(f"{key}: {value}")
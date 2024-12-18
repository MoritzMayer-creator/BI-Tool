import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import tensorflow as tf

class AktienPredictor:
    def __init__(self):
        # Setze festen Seed für reproduzierbare Ergebnisse
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.sequence_length = 100

    def prepare_data(self, df):
        """Bereitet die Daten für das Training vor, indem die Schlusskurse normalisiert 
        und in Sequenzen umgewandelt werden"""
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Erstellt das LSTM-Modell mit der angegebenen Architektur"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(input_shape, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """Trainiert das Modell mit den gegebenen Parametern und Early Stopping"""
        X, y = self.prepare_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model(self.sequence_length)
        
        # Early Stopping Konfiguration
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, df, days_ahead=2):
        """Erstellt Vorhersagen für die angegebene Anzahl von Tagen
        
        Args:
            df: DataFrame mit historischen Daten
            days_ahead: Anzahl der Tage für die Vorhersage
            
        Returns:
            Dictionary mit Vorhersagen und prozentualer Preisänderung
        """
        if self.model is None:
            raise ValueError("Modell muss erst trainiert werden!")
        
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Verwende die letzten sequence_length Tage für die erste Vorhersage
        last_sequence = scaled_data[-self.sequence_length:]
        current_sequence = last_sequence.copy()
        predictions = []
        
        # Erstelle Vorhersagen für die gewünschte Anzahl von Tagen
        for _ in range(days_ahead):
            current_batch = current_sequence.reshape((1, self.sequence_length, 1))
            next_pred = self.model.predict(current_batch)[0]
            predictions.append(next_pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # Transformiere die skalierten Werte zurück
        predictions = np.array(predictions).reshape(-1, 1)
        original_predictions = self.scaler.inverse_transform(predictions)[:, 0]
        
        return {
            'predictions': original_predictions,
            'price_change': ((original_predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        }
    
    def save_model(self, file_path):
        """Speichert das trainierte Modell in einer Datei"""
        if self.model is None:
            raise ValueError("Kein trainiertes Modell vorhanden")
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Lädt ein gespeichertes Modell aus einer Datei"""
        self.model = load_model(file_path)

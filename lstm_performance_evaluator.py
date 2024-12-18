import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from lstm_predictor import AktienPredictor
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

class LstmPerformanceEvaluator:
    def __init__(self):
        """Initialisierung des Performance Evaluators"""
        # Setze festen Seed für reproduzierbare Ergebnisse
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.performance_metrics = {}
        
    def evaluate_predictions(self, df, evaluation_start_date=None):
        """
        Evaluiert die LSTM-Prognosen (10-Tage-Vorhersage)
        """
        try:
            # Kopiere und bereinige DataFrame
            df_clean = df.copy()
            df_clean.index = df_clean.index.tz_localize(None)
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            
            # Nur Close-Preis verwenden
            data = df_clean['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            sequence_length = 100
            test_size = 252  # Ein Handelsjahr
            
            # Beschränke die Testdaten auf das letzte Jahr
            scaled_data = scaled_data[-test_size-sequence_length:]
            
            # Testdaten erstellen
            x_test = []
            y_test = []
            
            # Vergleiche die Vorhersage mit dem tatsächlichen Wert 2 Tage später
            for i in range(sequence_length, len(scaled_data) - 2):
                x_test.append(scaled_data[i-sequence_length:i])
                y_test.append(scaled_data[i+2, 0])  # +2 für 2-Tage-Vorhersage
            
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            
            # Modell trainieren
            print("\nTrainiere LSTM-Modell...")
            predictor = AktienPredictor()
            predictor.train(df_clean, epochs=20, batch_size=32, validation_split=0.1)
            
            # Für jeden Tag eine 2-Tage-Vorhersage machen
            y_pred = []
            for x in x_test:
                x_reshaped = x.reshape(1, sequence_length, 1)
                pred = predictor.predict(pd.DataFrame({'Close': scaler.inverse_transform(x.reshape(-1, 1)).flatten()}, 
                                                    index=range(len(x))))
                y_pred.append(scaler.transform([[pred['predictions'][1]]])[0, 0])  # Den 2. Tag der Vorhersage nehmen
            
            y_pred = np.array(y_pred)
            
            # Umformen für inverse_transform
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            
            # Rücktransformation der skalierten Werte
            y_pred = scaler.inverse_transform(y_pred_reshaped).flatten()
            y_test = scaler.inverse_transform(y_test_reshaped).flatten()
            
            # Zukunftsprognose erstellen
            future_forecast = predictor.predict(df_clean)
            future_predictions = future_forecast['predictions']
            
            # Zukünftige Handelstage erstellen
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=4,  # Reduziert von 15 auf 4
                freq='D'
            )
            future_dates = [date for date in future_dates if date.weekday() < 5][:2]  # Nur 2 Handelstage
            
            # Terminal-Ausgabe der Prognose
            print("\nZukunftsprognose für die nächsten 2 Handelstage:")
            print("Datum         | Prognose")
            print("-" * 25)
            for date, pred in zip(future_dates, future_predictions):
                print(f"{date.strftime('%Y-%m-%d')} | ${pred:8.2f}")

            # Länge der Vorhersagen definieren
            predictions_length = len(y_pred)
            
            # Originale Datumsindizes vom DataFrame verwenden
            historical_index = df_clean.index[-predictions_length-2:-2]  # -2 wegen 2-Tage-Vorhersage
            
            # Performance-Metriken speichern
            self.performance_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                'mae': mean_absolute_error(y_test, y_pred),
                'correlation': np.corrcoef(y_test, y_pred)[0, 1],
                'lstm_forecasts': pd.Series(y_pred, index=historical_index),
                'actual_values': pd.Series(y_test, index=historical_index),
                'future_forecast': pd.Series(future_predictions, index=future_dates),
                'mean_predicted_price': y_pred.mean(),
                'mean_actual_price': y_test.mean(),
                'price_prediction_std': y_pred.std(),
                'price_actual_std': y_test.std()
            }

            return self.performance_metrics
            
        except Exception as e:
            print(f"Fehler in der Performance-Evaluation: {str(e)}")
            return None

    def plot_performance(self, save_path=None):
        """
        Erstellt Visualisierungen der LSTM Performance mit täglichen Werten
        """
        if not self.performance_metrics:
            raise ValueError("Führen Sie zuerst evaluate_predictions aus!")

        # Erstelle Figure mit 2 Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Historische Daten und Vorhersagen
        historical = self.performance_metrics['actual_values']
        predictions = self.performance_metrics['lstm_forecasts']
        
        # Historische tatsächliche Werte - jetzt ohne Marker
        ax1.plot(
            historical.index,
            historical,
            label='Tatsächliche Werte',
            alpha=0.7,
            color='blue',
            linewidth=1.5
        )
        
        # Historische Vorhersagen - jetzt ohne Marker
        ax1.plot(
            predictions.index,
            predictions,
            label='LSTM Vorhersagen (2 Tage)',
            alpha=0.7,
            color='red',
            linewidth=1.5
        )
        
        # Legende hinzufügen
        ax1.legend(loc='upper left', fontsize=10)
        
        ax1.set_title('LSTM Preisprognose', fontsize=16, pad=20)
        
        # Formatierung verbessern
        ax1.set_ylabel('Preis ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Datum-Formatierung verbessern
        ax1.xaxis.set_major_locator(plt.MaxNLocator(15))  # Mehr Datumsbeschriftungen
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Streudiagramm der 2-Tage-Vorhersagen
        min_len = min(len(historical), len(predictions))
        ax2.scatter(historical[:min_len], predictions[:min_len], alpha=0.5)
        max_val = max(historical.max(), predictions.max())
        min_val = min(historical.min(), predictions.min())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax2.set_title('2-Tage-Vorhersage vs. Tatsächlicher Preis', fontsize=16)
        ax2.set_xlabel('Tatsächlicher Preis ($)', fontsize=12)
        ax2.set_ylabel('Vorhergesagter Preis (2 Tage)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Metriken hinzufügen
        metrics_text = (
            f"Performance Metriken:\n"
            f"RMSE: ${self.performance_metrics['rmse']:.2f}\n"
            f"MAPE: {self.performance_metrics['mape']:.2f}%\n"
            f"MAE: ${self.performance_metrics['mae']:.2f}\n"
            f"Korrelation: {self.performance_metrics['correlation']:.2f}"
        )
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def generate_performance_report(self):
        """
        Generiert einen detaillierten Performance-Bericht
        """
        if not self.performance_metrics:
            raise ValueError("Führen Sie zuerst evaluate_predictions aus!")
            
        report = {
            'Grundlegende Metriken': {
                'RMSE': f"${self.performance_metrics['rmse']:.2f}",
                'MAPE': f"{self.performance_metrics['mape']:.2f}%",
                'Korrelation': f"{self.performance_metrics['correlation']:.2f}"
            },
            'Zusätzliche Statistiken': {
                'Mittlerer Prognosewert': f"${self.performance_metrics['mean_predicted_price']:.2f}",
                'Mittlerer tatsächlicher Wert': f"${self.performance_metrics['mean_actual_price']:.2f}",
                'Prognose Standardabweichung': f"${self.performance_metrics['price_prediction_std']:.2f}",
                'Tatsächliche Standardabweichung': f"${self.performance_metrics['price_actual_std']:.2f}"
            }
        }
        
        return pd.DataFrame.from_dict({(i,j): report[i][j] 
                                     for i in report.keys() 
                                     for j in report[i].keys()}, 
                                    orient='index', columns=['Wert'])

def main():
    """
    Hauptprogramm zur Ausführung der LSTM Performance-Evaluation
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("LSTM Preis-Prognose Performance Evaluator")
    print("-" * 50)
    
    # Benutzereingaben
    while True:
        symbol = input("\nBitte geben Sie das Aktiensymbol ein (z.B. AAPL, MSFT): ").upper()
        try:
            # Daten abrufen - jetzt nur 2 Jahre
            print(f"\nLade Daten für {symbol}...")
            aktie = yf.Ticker(symbol)
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')  # 2 Jahre
            df = aktie.history(start=start_date)
            
            # Grundlegende Datenprüfung
            if df.empty:
                print("Keine Daten gefunden. Bitte versuchen Sie es mit einem anderen Symbol.")
                continue
            
            if len(df) < 252:  # Mindestens ein Handelsjahr
                print("Zu wenig Daten für eine aussagekräftige Analyse. Bitte wählen Sie ein anderes Symbol.")
                continue
            
            # Zeitzone entfernen
            df.index = df.index.tz_localize(None)
            
            # NaN-Werte behandeln
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if df['Close'].isnull().any():
                print("Fehlerhafte Daten gefunden. Bitte wählen Sie ein anderes Symbol.")
                continue
                
            break
        except Exception as e:
            print(f"Fehler beim Laden der Daten: {e}")
            print("Bitte versuchen Sie es erneut.")
    
    # Evaluation durchführen
    evaluator = LstmPerformanceEvaluator()
    
    print("\nFühre Performance-Evaluation durch...")
    print(f"Evaluationszeitraum: {start_date} bis heute")
    print("Prognosehorizont: 10 Tage")
    
    try:
        # Performance evaluieren
        metrics = evaluator.evaluate_predictions(
            df,
            evaluation_start_date=None,
        )
        
        if metrics is None:
            print("Keine ausreichenden Daten für die Evaluation.")
            return
        
        # Bericht generieren
        print("\nPerformance-Bericht:")
        print("-" * 50)
        report = evaluator.generate_performance_report()
        print(report)
        
        # Plots erstellen
        print("\nErstelle Visualisierungen...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'lstm_performance_{symbol}_10d_{timestamp}.png'
        evaluator.plot_performance(save_path=save_path)
        print(f"Visualisierungen wurden gespeichert unter: {save_path}")
        
        # Detaillierte Metriken ausgeben
        print("\nDetaillierte Performance-Metriken:")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Korrelation: {metrics['correlation']:.2f}")
        
    except Exception as e:
        print(f"\nFehler bei der Evaluation: {e}")
        print("Details:", str(e))
        return
    
    print("\nEvaluation abgeschlossen!")

if __name__ == "__main__":
    main() 
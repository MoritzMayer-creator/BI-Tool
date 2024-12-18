"""
Evaluiert die Performance des ARIMA-Modells und erstellt Visualisierungen
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from arima_predictor import ArimaPredictor
import pytz
from datetime import datetime

class ArimaPerformanceEvaluator:
    def __init__(self):
        self.arima_predictor = ArimaPredictor()
        self.performance_metrics = {}
        
    def evaluate_predictions(self, df, evaluation_start_date, forecast_days=1, test_size=100):
        """
        Evaluiert die ARIMA-Vorhersagen mit Rolling-Window-Backtesting
        """
        try:
            # Kopiere und bereinige DataFrame
            df = df.copy()
            df.index = df.index.tz_localize(None)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Konvertiere Close-Preise zu NumPy Array
            close_prices = np.array(df['Close'].values, dtype=float)
            
            if np.isnan(close_prices).any():
                raise ValueError("Dataset still contains NaN values after cleaning")
            
            # Rolling Forecast für test_size Handelstage
            test_size = min(test_size, len(df) - 252)
            step_size = 1  # Jeden Tag eine Vorhersage machen
            rolling_predictions = []
            realized_prices = []
            forecast_dates = []
            
            print("\nCalculating Rolling Forecast...")
            for i in range(0, test_size, step_size):
                if i % 25 == 0:  # Häufigere Updates
                    print(f"Fortschritt: {i/test_size*100:.1f}%")
                    
                train_idx = len(df) - test_size + i
                if train_idx < 252:  # Mindestens 1 Jahr Trainingsdaten
                    continue
                    
                current_date = df.index[train_idx]
                
                # Überspringe doppelte Daten
                if forecast_dates and current_date == forecast_dates[-1]:
                    continue
                    
                train_df = df.iloc[:train_idx].copy()
                
                try:
                    result = self.arima_predictor.erstelle_prognose(train_df, prognose_tage=1)
                    if result is not None and 'endpreis' in result:  # Vereinfachte Validierung
                        # Finde den realisierten Preis 1 Tag später
                        future_idx = min(train_idx + 1, len(close_prices) - 1)
                        realized_future_price = float(close_prices[future_idx])
                        
                        rolling_predictions.append(float(result['endpreis']))
                        realized_prices.append(realized_future_price)
                        forecast_dates.append(current_date)
                except Exception as e:
                    continue  # Stille Fehlerbehandlung
            
            if len(rolling_predictions) < 2:
                raise ValueError("Nicht genügend gültige Vorhersagen")
            
            # Konvertiere Listen zu Arrays/Series mit korrekten Datumsindizes
            predictions = pd.Series(rolling_predictions, index=forecast_dates)
            realized = pd.Series(realized_prices, index=forecast_dates)
            
            # Berechne Änderungen
            pred_changes = predictions.pct_change().dropna()
            real_changes = realized.pct_change().dropna()
            
            # Berechne Metriken
            rmse = float(np.sqrt(np.mean((predictions - realized) ** 2)))
            mape = float(np.mean(np.abs((realized - predictions) / realized)) * 100)
            mae = float(np.mean(np.abs(realized - predictions)))
            correlation = float(np.corrcoef(predictions, realized)[0, 1])
            direction_accuracy = float(np.mean(np.sign(pred_changes) == np.sign(real_changes)) * 100)
            
            self.performance_metrics = {
                'rmse': rmse,
                'mape': mape,
                'mae': mae,
                'correlation': correlation,
                'directional_accuracy': direction_accuracy,
                'arima_forecasts': predictions,
                'current_prices': realized,
                'mean_pred_change': float(pred_changes.mean() * 100),
                'mean_real_change': float(real_changes.mean() * 100)
            }
            
            if len(rolling_predictions) >= 5:
                print("\nLetzte 5 ARIMA Vorhersagen und realisierte Preise:")
                for i in range(-5, 0):
                    print(f"Datum: {forecast_dates[i]}, "
                          f"ARIMA-Vorhersage: ${rolling_predictions[i]:.2f}, "
                          f"Realisierter Preis: ${realized_prices[i]:.2f}, "
                          f"Differenz: ${(rolling_predictions[i] - realized_prices[i]):.2f}")
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"Kritischer Fehler in evaluate_predictions: {str(e)}")
            raise

    def plot_performance(self, save_path=None):
        """
        Erstellt einen detaillierten Performance-Plot mit täglichen Datenpunkten
        """
        if not self.performance_metrics:
            raise ValueError("Führen Sie zuerst evaluate_predictions aus!")
            
        # Erstelle Figure mit 2 Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Zeitreihe mit Vorhersagen
        realized = self.performance_metrics['arima_forecasts']
        predictions = self.performance_metrics['current_prices']
        
        # Verschiebe die Vorhersagen um 1 Tag nach vorne
        predictions.index = predictions.index + pd.Timedelta(days=1)
        
        # Berechne Zeitfenster für 100 Tage
        last_date = predictions.index[-1]
        first_date = last_date - pd.DateOffset(days=100)
        
        # Filtere die Daten für die letzten 100 Tage
        mask = (realized.index >= first_date) & (realized.index <= last_date)
        realized_year = realized[mask]
        predictions_year = predictions[mask]
        
        # Plotte die gefilterten Daten ohne Marker, nur mit Linien
        ax1.plot(realized_year.index, realized_year.values, 'b-', label='Realisierte Preise', 
                linewidth=1.5, alpha=0.8)
        ax1.plot(predictions_year.index, predictions_year.values, 'r-', label='ARIMA Prognosen (1 Tag)', 
                linewidth=1.5, alpha=0.8)
        
        ax1.set_title('ARIMA Rolling Forecast Performance', fontsize=16, pad=20)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Datum', fontsize=12)
        ax1.set_ylabel('Preis ($)', fontsize=12)
        ax1.set_xlim(first_date, last_date)
        
        # Formatiere x-Achse für bessere Lesbarkeit
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Streudiagramm mit allen Punkten
        ax2.scatter(realized, predictions, alpha=0.5, color='blue', s=30)
        max_val = max(realized.max(), predictions.max())
        min_val = min(realized.min(), predictions.min())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax2.set_title('Prognose vs. Realisierte Preise', fontsize=16)
        ax2.set_xlabel('Realisierte Preise ($)', fontsize=12)
        ax2.set_ylabel('Prognostizierte Preise ($)', fontsize=12)
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
                 verticalalignment='top', bbox=dict(boxstyle='round', 
                 facecolor='white', alpha=0.8), fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    import yfinance as yf
    from datetime import datetime, timedelta
    import pytz
    
    while True:
        symbol = input("\nBitte Aktiensymbol eingeben (oder 'exit' zum Beenden): ").upper()
        if symbol == 'EXIT':
            break
            
        try:
            # Lade 500 Tage (ca. 2 Jahre) für ausreichend Trainingsdaten
            end_date = datetime.now(pytz.UTC).date()
            while end_date.weekday() > 4:
                end_date -= timedelta(days=1)
            start_date = end_date - timedelta(days=500)  # 500 Tage für Training + 250 Tage Test
            
            print(f"\nLade Daten für {symbol}...")
            print(f"Zeitraum: {start_date} bis {end_date}")
            
            df = yf.download(symbol, 
                           start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'),
                           progress=False)
            
            if df.empty:
                print(f"Keine Daten für {symbol} gefunden.")
                continue
            
            # Stelle sicher, dass wir nur Handelstage haben
            df = df.asfreq('B')  # Business days
            df = df.fillna(method='ffill')
            
            print(f"\nErster verfügbarer Tag: {df.index[0].strftime('%Y-%m-%d')}")
            print(f"Letzter verfügbarer Tag: {df.index[-1].strftime('%Y-%m-%d')}")
            
            print(f"\nAnalysiere Performance für {symbol}...")
            
            evaluator = ArimaPerformanceEvaluator()
            metrics = evaluator.evaluate_predictions(
                df=df,
                evaluation_start_date=start_date,
                forecast_days=5
            )
            
            # Speichere Plot
            save_path = f'arima_performance_{symbol}.png'
            evaluator.plot_performance(save_path=save_path)
            print(f"\nPlot gespeichert als: {save_path}")
            
            # Zeige detaillierte Metriken
            print("\nDetaillierte Performance Metriken:")
            for key, value in metrics.items():
                if not isinstance(value, (pd.Series, np.ndarray)):
                    print(f"{key}: {value:.2f}")
                    
            # Zeige die letzten beiden Prognosewerte
            historical_predictions = metrics['current_prices']
            last_two_predictions = historical_predictions.tail(2)
            print("\nLetzte zwei Prognosewerte:")
            for date, value in last_two_predictions.items():
                print(f"Datum: {date.strftime('%Y-%m-%d')}, Prognose: ${value:.2f}")
            
        except Exception as e:
            print(f"Fehler bei der Analyse von {symbol}: {str(e)}")
            continue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from volatility_predictor import VolatilityPredictor
import seaborn as sns
from datetime import datetime, timedelta
from arch import arch_model

class GarchPerformanceEvaluator:
    def __init__(self):
        """Initialisierung des Performance Evaluators"""
        self.volatility_predictor = VolatilityPredictor()
        self.performance_metrics = {}
        
    def calculate_realized_volatility(self, df, window=21):
        """
        Berechnung der realisierten Volatilität - einfache Returns-basierte Methode
        """
        try:
            # Berechne prozentuale Returns
            returns = 100 * df['Close'].pct_change().dropna()
            
            # Berechne rollierende Standardabweichung und annualisiere
            realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            return realized_vol
            
        except Exception as e:
            print(f"Fehler bei der Volatilitätsberechnung: {str(e)}")
            return None

    def evaluate_predictions(self, df, evaluation_start_date=None, forecast_horizon=2, silent=False):
        """
        Evaluiert die GARCH Prognosen
        
        Args:
            df: DataFrame mit den Aktiendaten
            evaluation_start_date: Startdatum für die Evaluation
            forecast_horizon: Anzahl der Tage für die Prognose
            silent: Wenn True, werden keine Ausgaben im Terminal angezeigt
        """
        try:
            # Kopiere und bereinige DataFrame
            df_clean = df.copy()
            df_clean.index = df_clean.index.tz_localize(None)
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            
            if df_clean['Close'].isnull().any():
                raise ValueError("Datensatz enthält immer noch NaN-Werte nach der Bereinigung")
            
            # Änderung der Testgröße auf 1 Jahr (ca. 252 Handelstage)
            test_size = 252
            rolling_predictions = []
            rolling_dates = []
            
            print("\nBerechne Rolling Forecast...")
            
            # Bestimme Start- und Enddatum für die Evaluation
            end_date = df_clean.index[-1]
            start_date = end_date - pd.DateOffset(years=1)
            
            # Erstelle eine Liste aller Handelstage im Evaluationszeitraum
            evaluation_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Für jeden Handelstag eine Vorhersage machen
            for i, current_date in enumerate(evaluation_dates):
                if i % 5 == 0:  # Fortschrittsanzeige alle 5 Tage
                    print(f"Fortschritt: {i/len(evaluation_dates)*100:.1f}%")
                
                # Finde den Index des aktuellen Datums in df_clean
                current_data = df_clean[df_clean.index <= current_date]
                
                if len(current_data) < 252:  # Mindestens 1 Jahr Daten für Training
                    continue
                    
                forecast = self.volatility_predictor.garch_forecast(current_data, forecast_days=2)
                
                if forecast is not None:
                    rolling_predictions.append(forecast['forecast_values'][0])  # Nur den ersten Vorhersagewert nehmen
                    rolling_dates.append(current_date)
            
            if not rolling_predictions:
                raise ValueError("Keine Prognosen generiert")
            
            # Erstelle Series mit den täglichen Vorhersagen
            rolling_predictions = pd.Series(rolling_predictions, index=rolling_dates)
            
            # Berechne realisierte Volatilität
            realized_vol = self.calculate_realized_volatility(df_clean)
            
            # Stelle sicher, dass beide Serien den gleichen Index haben
            common_dates = rolling_predictions.index.intersection(realized_vol.index)
            rolling_predictions = rolling_predictions[common_dates]
            realized_vol = realized_vol[common_dates]
            
            # Entferne NaN-Werte aus beiden Serien
            valid_data = pd.concat([rolling_predictions, realized_vol], axis=1).dropna()
            if len(valid_data) == 0:
                raise ValueError("Keine überlappenden, gültigen Daten gefunden")
            
            predictions = valid_data.iloc[:, 0]  # Rolling Predictions
            realized = valid_data.iloc[:, 1]     # Realized Volatility
            
            # Berechne Metriken
            rmse = np.sqrt(mean_squared_error(realized, predictions))
            mape = mean_absolute_percentage_error(realized, predictions) * 100
            mae = mean_absolute_error(realized, predictions)  # Neue Metrik
            correlation = np.corrcoef(realized, predictions)[0, 1]
            
            self.performance_metrics = {
                'rmse': rmse,
                'mape': mape,
                'mae': mae,  # Neue Metrik hinzugefügt
                'correlation': correlation,
                'rolling_predictions': rolling_predictions,
                'rolling_realized': realized_vol,
                'rolling_dates': rolling_dates,
                'mean_predicted_vol': predictions.mean(),
                'mean_realized_vol': realized.mean(),
                'vol_prediction_std': predictions.std(),
                'vol_realized_std': realized.std()
            }
            
            # Ausgabe nur wenn nicht silent
            if not silent:
                print("\nLetzte 5 GARCH Prognosen und realisierte Werte:")
                print("-" * 60)
                print("Datum        |   Prognose | Realisiert |  Differenz")
                print("-" * 60)
                
                for i in range(-5, 0):
                    pred_date = predictions.index[i]
                    pred_val = predictions.iloc[i]  # Bereits in Prozent
                    real_val = realized.iloc[i]     # Bereits in Prozent
                    diff = real_val - pred_val      # Differenz in Prozentpunkten
                    
                    print(f"{pred_date.strftime('%Y-%m-%d')} | {pred_val:10.2f}% | {real_val:10.2f}% | {diff:+10.2f}%")
            
            return self.performance_metrics
            
        except Exception as e:
            if not silent:
                print(f"Fehler in GARCH Performance Evaluation: {str(e)}")
            return None

    def plot_performance(self, save_path=None):
        """
        Erstellt Visualisierungen der Rolling Forecast Performance mit korrigierter Zeitachse
        """
        if not self.performance_metrics:
            raise ValueError("Führen Sie zuerst evaluate_predictions aus!")

        # Erstelle Figure mit 2 Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Rolling Forecast Zeitreihe mit korrigierter Zeitverschiebung
        realized_vol = self.performance_metrics['rolling_realized']
        predictions = self.performance_metrics['rolling_predictions']
        
        # Plotte die Linien für tatsächliche Werte
        ax1.plot(
            realized_vol.index,
            realized_vol,
            label='Realisierte Volatilität',
            color='blue',
            alpha=0.7,
            linewidth=1.5  # Dickere Linie für bessere Sichtbarkeit
        )
        
        # Plotte die Linien für Vorhersagen
        ax1.plot(
            predictions.index,
            predictions,
            label='GARCH Vorhersagen (2 Tage)',
            color='red',
            alpha=0.7,
            linewidth=1.5  # Dickere Linie für bessere Sichtbarkeit
        )
        
        ax1.set_title('GARCH Rolling Forecast Volatilitätsprognose', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.set_ylabel('Annualisierte Volatilität (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Zoom auf das letzte Jahr für bessere Sichtbarkeit
        last_date = realized_vol.index[-1]
        one_year_ago = last_date - pd.DateOffset(years=1)
        ax1.set_xlim(one_year_ago, last_date)
        
        # Plot 2: Streudiagramm
        valid_mask = ~(predictions.isna() | realized_vol.isna())
        realized = realized_vol[valid_mask]
        predicted = predictions[valid_mask]
        
        ax2.scatter(realized, predicted, alpha=0.5, color='blue', s=15)
        max_val = max(realized.max(), predicted.max())
        min_val = min(realized.min(), predicted.min())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax2.set_title('2-Tage-Vorhersage vs. Realisierte Volatilität', fontsize=16)
        ax2.set_xlabel('Realisierte Volatilität (%)', fontsize=12)
        ax2.set_ylabel('Vorhergesagte Volatilität (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Füge Metriken-Box hinzu
        metrics_text = (
            f"Performance Metriken:\n"
            f"RMSE: {self.performance_metrics['rmse']:.2f}%\n"
            f"MAPE: {self.performance_metrics['mape']:.2f}%\n"
            f"MAE: {self.performance_metrics['mae']:.2f}%\n"
            f"Korrelation: {self.performance_metrics['correlation']:.2f}"
        )
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_forecast(self, forecast_result):
        """
        Erstellt einen Plot der 7-Tage-Volatilitätsprognose
        """
        plt.figure(figsize=(10,4))
        plt.plot(forecast_result['forecast_volatility'])
        plt.title('Volatilitätsprognose - Nächste 7 Tage', fontsize=20)
        plt.ylabel('Prognostizierte Volatilität', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.show()

    def generate_performance_report(self):
        """
        Generiert einen detaillierten Performance-Bericht basierend auf dem Rolling Forecast
        """
        if not self.performance_metrics:
            raise ValueError("Führen Sie zuerst evaluate_predictions aus!")
            
        report = {
            'Grundlegende Metriken': {
                'RMSE (Prozentpunkte)': f"{self.performance_metrics['rmse']:.2f}",
                'MAPE (%)': f"{self.performance_metrics['mape']:.2f}",
                'Korrelation': f"{self.performance_metrics['correlation']:.2f}"
            },
            'Zusätzliche Statistiken': {
                'Mittlere Prognosevolatilität (%)': f"{self.performance_metrics['mean_predicted_vol']:.2f}",
                'Mittlere realisierte Volatilität (%)': f"{self.performance_metrics['mean_realized_vol']:.2f}",
                'Prognose Standardabweichung (%)': f"{self.performance_metrics['vol_prediction_std']:.2f}",
                'Realisierte Standardabweichung (%)': f"{self.performance_metrics['vol_realized_std']:.2f}"
            }
        }
        
        return pd.DataFrame.from_dict({(i,j): report[i][j] 
                                     for i in report.keys() 
                                     for j in report[i].keys()}, 
                                    orient='index', columns=['Wert']) 

def main():
    """
    Hauptprogramm zur Ausführung der GARCH Performance-Evaluation
    """
    import yfinance as yf
    from datetime import datetime
    
    print("GARCH Volatilitäts-Prognose Performance Evaluator")
    print("-" * 50)
    
    # Benutzereingaben
    while True:
        symbol = input("\nBitte geben Sie das Aktiensymbol ein (z.B. AAPL, MSFT): ").upper()
        try:
            # Daten abrufen
            print(f"\nLade Daten für {symbol}...")
            aktie = yf.Ticker(symbol)
            start_date = '2020-01-01'  # Längerer Zeitraum für bessere Modellierung
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
            
            # Behandlung fehlender Werte
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if df['Close'].isnull().any():
                print("Fehlerhafte Daten gefunden. Bitte wählen Sie ein anderes Symbol.")
                continue
                
            break
        except Exception as e:
            print(f"Fehler beim Laden der Daten: {e}")
            print("Bitte versuchen Sie es erneut.")
    
    # Evaluation durchführen
    evaluator = GarchPerformanceEvaluator()
    
    print("\nFühre Performance-Evaluation durch...")
    print(f"Evaluationszeitraum: {start_date} bis heute")
    print("Prognosehorizont: 2 Tage")  # Aktualisierte Information
    
    try:
        # Performance evaluieren
        evaluation_start = '2021-01-01'
        metrics = evaluator.evaluate_predictions(
            df,
            evaluation_start_date=evaluation_start,
            forecast_horizon=2  # Auf 2 Tage reduziert
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
        save_path = f'garch_performance_{symbol}_7d_{timestamp}.png'  # Angepasster Dateiname
        evaluator.plot_performance(save_path=save_path)
        print(f"Visualisierungen wurden gespeichert unter: {save_path}")
        
        # Detaillierte Metriken ausgeben
        print("\nDetaillierte Performance-Metriken:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Korrelation: {metrics['correlation']:.2f}")
        
    except Exception as e:
        print(f"\nFehler bei der Evaluation: {e}")
        print("Details:", str(e))
        return
    
    print("\nEvaluation abgeschlossen!")

if __name__ == "__main__":
    main()
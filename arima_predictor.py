import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import logging
from joblib import Parallel, delayed
from scipy import stats

# Logger für die Protokollierung einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ArimaPredictor:
    def __init__(self, seasonal_periods=[5, 20, 60], parallel=True):
        """
        Initialisiert den ARIMA-Predictor mit saisonalen Perioden und Parallelverarbeitung.
        
        Args:
            seasonal_periods: Liste der zu prüfenden saisonalen Perioden (Standard: [5, 20, 60])
            parallel: Aktiviert parallele Verarbeitung (Standard: True)
        """
        self.model = None
        self.model_fit = None
        self.differencing_order = 0
        self.best_params = None
        self.mape = None
        self.seasonal_periods = seasonal_periods
        self.parallel = parallel
        
    def _check_stationarity(self, data):
        """
        Überprüft die Stationarität der Zeitreihe und ermittelt die optimale Differenzierungsordnung.
        
        Args:
            data: Eingangszeitreihe
            
        Returns:
            Optimale Differenzierungsordnung (d)
        """
        max_d = 2
        d = 0
        data_diff = data.copy()
        
        while d <= max_d:
            adf_test = adfuller(data_diff, autolag='AIC')
            if adf_test[1] < 0.05:  # Stationarität erreicht wenn p-Wert < 0.05
                break
            data_diff = np.diff(data_diff)
            d += 1
            
        return d if d <= max_d else 1

    def _prepare_data(self, df):
        """
        Bereitet die Eingangsdaten für die ARIMA-Analyse vor.
        - Konvertiert Daten in das richtige Format
        - Behandelt fehlende Werte durch Interpolation
        - Entfernt extreme Ausreißer
        
        Args:
            df: DataFrame oder Array mit den Eingangsdaten
            
        Returns:
            Bereinigte Zeitreihe als NumPy-Array
        """
        try:
            # Konvertiere zu NumPy Array
            if isinstance(df, pd.DataFrame):
                prices = np.array(df['Close'].values, dtype=float)
            elif isinstance(df, np.ndarray):
                prices = df.astype(float)
            else:
                prices = np.array(df, dtype=float)
            
            # Behandle fehlende Werte
            if np.isnan(prices).any():
                logger.warning(f"Gefunden {np.isnan(prices).sum()} fehlende Werte")
                mask = np.isnan(prices)
                prices[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), prices[~mask])
            
            # Behandle extreme Ausreißer
            z_scores = stats.zscore(prices)
            mask = np.abs(z_scores) > 5
            if mask.any():
                prices[mask] = np.median(prices)
            
            return prices

        except Exception as e:
            logger.error(f"Fehler in _prepare_data: {str(e)}")
            raise

    def _find_best_parameters(self, data):
        """
        Ermittelt die optimalen ARIMA-Parameter mittels auto_arima.
        
        Args:
            data: Vorbereitete Zeitreihe
            
        Returns:
            Tuple mit (order, seasonal_order) der optimalen Parameter
        """
        try:
            # Verwende auto_arima mit erweiterten Optionen
            model = auto_arima(data,
                              start_p=0, max_p=5,
                              start_q=0, max_q=5,
                              start_P=0, max_P=2,
                              start_Q=0, max_Q=2,
                              m=5,  # Wöchentliche Saisonalität
                              seasonal=True,
                              d=None,  # Automatische Differenzierung
                              D=None,  # Automatische saisonale Differenzierung
                              trace=False,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True,
                              random_state=42)
            
            return model.order, model.seasonal_order
        except Exception as e:
            logger.error(f"Fehler bei Parameter-Suche: {str(e)}")
            return None, None

    def _fit_single_arima(self, data, seasonal_period):
        """
        Führt eine einzelne ARIMA-Modellierung mit gegebener saisonaler Periode durch.
        
        Args:
            data: Zeitreihendaten
            seasonal_period: Zu prüfende saisonale Periode
            
        Returns:
            Tuple mit (order, seasonal_order, AIC-Wert)
        """
        try:
            model = auto_arima(
                data,
                start_p=1, max_p=5,  # Reduzierte Komplexität
                start_q=1, max_q=5,
                m=seasonal_period,
                seasonal=seasonal_period is not None,
                d=None, max_d=2,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                random_state=42,
                n_fits=50,
                maxiter=50,          # Begrenzte Iterationen
                method='nm'          # Schnellere Optimierungsmethode
            )
            return (model.order, 
                    model.seasonal_order if seasonal_period else None,
                    model.aic())
        except Exception as e:
            logger.error(f"Fehler in ARIMA-Anpassung: {str(e)}")
            return (2,1,1), None, float('inf')

    def _validate_forecast(self, test_data, forecast):
        """
        Berechnet verschiedene Qualitätsmetriken für die Prognose.
        
        Args:
            test_data: Tatsächliche Werte
            forecast: Prognostizierte Werte
            
        Returns:
            Dictionary mit verschiedenen Fehlermetriken (MAPE, RMSE, MAE, Richtungsgenauigkeit)
        """
        metrics = {
            'mape': mean_absolute_percentage_error(test_data, forecast) * 100,
            'rmse': np.sqrt(mean_squared_error(test_data, forecast)),
            'mae': np.mean(np.abs(test_data - forecast)),
            'direction_accuracy': np.mean(np.sign(np.diff(test_data)) == 
                                       np.sign(np.diff(forecast[:-1]))) * 100
        }
        
        return metrics

    def erstelle_prognose(self, df, prognose_tage=1):
        """
        Erstellt eine ARIMA-basierte Prognose für die gegebene Zeitreihe.
        
        Args:
            df: DataFrame mit historischen Daten
            prognose_tage: Anzahl der zu prognostizierenden Tage (Standard: 1)
            
        Returns:
            Dictionary mit Prognosewerten, Endpreis und prozentuale Preisänderung
            None bei Fehler
        """
        try:
            logger.info("Starte ARIMA-Prognose")
            
            # Verwende ein festes Zeitfenster von 252 Tagen (1 Jahr)
            df_subset = df.tail(252).copy()
            data_values = self._prepare_data(df_subset)
            
            if len(data_values) < 30:
                logger.error("Zu wenig Daten für Prognose")
                return None
            
            # Vereinfachte ARIMA Modellierung mit festen Parametern
            try:
                model = auto_arima(data_values,
                                 start_p=1, max_p=2,  # Reduzierte Parameter
                                 start_q=1, max_q=2,
                                 m=5,
                                 seasonal=True,
                                 d=1,
                                 D=1,
                                 stepwise=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 maxiter=10)  # Begrenzte Iterationen
                
                # Erstelle Prognose
                forecast = model.predict(n_periods=prognose_tage)
                forecast_mean = np.array(forecast, dtype=float)
                
                if len(forecast_mean) == 0:
                    logger.error("Leere Prognose")
                    return None
                
                # Berechne Endpreis ohne zusätzliche Validierungen
                prognose_endpreis = float(forecast_mean[-1])
                
                return {
                    'forecast': forecast_mean,
                    'endpreis': prognose_endpreis,
                    'preisänderung': float(((prognose_endpreis - data_values[-1]) / data_values[-1]) * 100)
                }
                
            except Exception as e:
                logger.error(f"Fehler beim ARIMA Fit/Forecast: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Kritischer Fehler bei ARIMA-Prognose: {str(e)}")
            return None

    def get_model_summary(self):
        """
        Erstellt eine umfassende Zusammenfassung des trainierten ARIMA-Modells.
        
        Returns:
            Dictionary mit Modellzusammenfassung, Parametern und Qualitätsmetriken
            None wenn kein Modell trainiert wurde
        """
        if self.model_fit is not None:
            summary = {
                'model_summary': self.model_fit.summary(),
                'parameters': self.best_params,
                'differencing_order': self.differencing_order,
                'mape': self.mape,
                'aic': self.model_fit.aic,
                'bic': self.model_fit.bic
            }
            return summary
        return None
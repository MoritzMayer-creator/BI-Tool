import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictor:
    def __init__(self):
        """Initialisierung des Volatilitäts-Prognosemodells"""
        self.best_model = None
        self.model_metrics = {}
        
    def _select_best_garch_params(self, returns, max_p=3, max_q=3):
        """
        Wählt die besten GARCH-Parameter durch Grid Search aus
        
        Args:
            returns: Renditezeitreihe
            max_p: Maximale Ordnung für ARCH-Term
            max_q: Maximale Ordnung für GARCH-Term
            
        Returns:
            Tuple mit optimalen p,q Parametern
        """
        best_aic = np.inf
        best_params = (1, 1)
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = arch_model(returns, vol='Garch', p=p, q=q, dist='skewt')
                    results = model.fit(disp='off', show_warning=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, q)
                except:
                    continue
                    
        return best_params

    def _detect_outliers(self, returns, method='iqr'):
        """
        Erkennt und behandelt Ausreißer in den Renditen
        
        Args:
            returns: Renditezeitreihe
            method: Methode zur Ausreißererkennung ('iqr' oder 'zscore')
            
        Returns:
            Bereinigte Renditezeitreihe
        """
        if method == 'iqr':
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            returns_clean = returns.clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            returns_clean = returns.mask(z_scores > 3, returns.mean())
            
        return returns_clean

    def _calculate_regime_probabilities(self, volatility):
        """
        Berechnet die Wahrscheinlichkeiten für verschiedene Volatilitätsregime
        
        Args:
            volatility: Volatilitätszeitreihe
            
        Returns:
            Tuple mit aktuellem Regime und Regime-Wahrscheinlichkeiten
        """
        kmeans = KMeans(n_clusters=3, random_state=42)
        scaled_vol = StandardScaler().fit_transform(volatility.values.reshape(-1, 1))
        labels = kmeans.fit_predict(scaled_vol)
        
        current_regime = labels[-1]
        regime_counts = np.bincount(labels)
        probabilities = regime_counts / len(labels)
        
        return current_regime, probabilities

    def _detect_volatility_regime(self, returns, window=63):
        """
        Erkennt das aktuelle Volatilitätsregime basierend auf historischen Daten
        
        Args:
            returns: Renditezeitreihe
            window: Fenstergröße für die rollierende Standardabweichung
            
        Returns:
            String mit Regimebezeichnung ('low', 'medium', 'high')
        """
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        current_vol = vol.iloc[-1]
        vol_quantiles = vol.quantile([0.25, 0.75])
        
        if current_vol < vol_quantiles[0.25]:
            return 'low'
        elif current_vol > vol_quantiles[0.75]:
            return 'high'
        else:
            return 'medium'

    def garch_forecast(self, df, forecast_days=5):
        """
        Erstellt eine GARCH-Volatilitätsprognose für die nächsten n Tage
        
        Args:
            df: DataFrame mit Kursdaten
            forecast_days: Anzahl der Prognosetage
            
        Returns:
            Dictionary mit Prognosewerten und -daten oder None bei Fehler
        """
        try:
            # Datenvorverarbeitung
            df_clean = df.copy()
            df_clean.index = df_clean.index.tz_localize(None)
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            
            returns = 100 * df_clean['Close'].pct_change().dropna()
            
            if len(returns) < 252:  # Mindestens 1 Jahr Daten erforderlich
                return None
            
            # GARCH-Modell mit festen Parametern
            model = arch_model(returns, vol='Garch', p=2, q=2, dist='normal')
            results = model.fit(disp='off')
            
            forecast = results.forecast(horizon=forecast_days)
            # Umrechnung in annualisierte Volatilität in Prozent
            forecast_values = np.sqrt(forecast.variance.values[-1]) * np.sqrt(252)
            
            # Erstellung der Prognosedaten
            forecast_dates = pd.date_range(
                start=df_clean.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='B'  # Nur Geschäftstage
            )
            
            return {
                'forecast_dates': forecast_dates,
                'forecast_values': forecast_values,
                'current_volatility': returns.std() * np.sqrt(252),
                'model_results': results
            }
            
        except Exception as e:
            print(f"Fehler bei der GARCH-Prognose: {str(e)}")
            return None

    def calculate_rolling_forecast(self, df, test_size=500):
        """
        Erstellt rollierende Prognosen für Backtesting
        
        Args:
            df: DataFrame mit Kursdaten
            test_size: Größe des Testzeitraums in Handelstagen
            
        Returns:
            Zeitreihe mit rollierenden Prognosen
        """
        returns = 100 * df['Close'].pct_change().dropna()
        returns = self._detect_outliers(returns)
        rolling_predictions = []
        
        for i in range(test_size):
            train = returns[:-(test_size-i)]
            if len(train) < 252:  # Mindestanforderung für Training bleibt ein Jahr
                rolling_predictions.append(np.nan)
                continue
            
            try:
                model = arch_model(
                    train,
                    vol='GARCH',
                    p=2,
                    q=2,
                    dist='normal',
                    rescale=True
                )
                model_fit = model.fit(disp='off')
                # 5-Tage-Prognose
                pred = model_fit.forecast(horizon=5)
                # Nehme den 5-Tage-Wert für die Evaluation
                vol_pred = np.sqrt(pred.variance.values[-1,-1]) * np.sqrt(252)
                rolling_predictions.append(vol_pred)
            except:
                rolling_predictions.append(np.nan)
        
        return pd.Series(rolling_predictions, index=returns.index[-test_size:])

    def analyze_volatility_patterns(self, df):
        """
        Analysiert Volatilitätsmuster für verbesserte GARCH-Prognosen
        
        Args:
            df: DataFrame mit Kursdaten
            
        Returns:
            Dictionary mit Analyseergebnissen (Cluster, Saisonalität, Regime)
        """
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=21).std() * np.sqrt(252) * 100
        
        # Volatilitäts-Cluster identifizieren
        kmeans = KMeans(n_clusters=3, random_state=42)
        vol_clusters = kmeans.fit_predict(volatility.values.reshape(-1, 1))
        
        # Saisonalität in Volatilität
        volatility_by_month = volatility.groupby(volatility.index.month).mean()
        volatility_by_weekday = volatility.groupby(volatility.index.dayofweek).mean()
        
        # Volatilitäts-Regime-Switching
        high_vol = volatility > volatility.quantile(0.75)
        low_vol = volatility < volatility.quantile(0.25)
        
        regime_persistence = {
            'high_vol_persistence': (high_vol.shift() & high_vol).mean(),
            'low_vol_persistence': (low_vol.shift() & low_vol).mean()
        }
        
        return {
            'clusters': {
                'centers': kmeans.cluster_centers_.flatten(),
                'distribution': np.bincount(vol_clusters) / len(vol_clusters)
            },
            'seasonality': {
                'monthly': volatility_by_month.to_dict(),
                'weekly': volatility_by_weekday.to_dict()
            },
            'regime_persistence': regime_persistence,
            'current_regime': 'high' if volatility.iloc[-1] > volatility.quantile(0.75) else 'low' if volatility.iloc[-1] < volatility.quantile(0.25) else 'medium'
        } 
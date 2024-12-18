import openai
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from dotenv import load_dotenv
import json
import yfinance as yf

class DashboardAI:
    def __init__(self):
        """Initialisierung der KI mit xAI Grok Integration"""
        self.xai_client = openai.OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1"
        )

    def _beschreibe_zeitraum(self, tage: int) -> str:
        """Erstellt eine beschreibende Zeitraum-Bezeichnung"""
        return f"{tage} Tage"

    def _analyze_fundamentals(self, kennzahlen: Dict[str, float], horizont: str) -> Dict[str, Any]:
        """Analysiert Fundamentaldaten mit Grok"""
        prompt = f"""Analysiere folgende Fundamentalkennzahlen für einen {horizont}en Anlagehorizont:

KGV: {kennzahlen.get('kgv', 'N/A')}
ROE: {kennzahlen.get('roe', 'N/A')}%
Verschuldungsgrad: {kennzahlen.get('verschuldungsgrad', 'N/A')}%
Dividendenrendite: {kennzahlen.get('dividendenrendite', 'N/A')}%
Free Cashflow: {kennzahlen.get('free_cashflow', 'N/A')}M
Umsatzwachstum: {kennzahlen.get('umsatzwachstum', 'N/A')}%
EBIT-Marge: {kennzahlen.get('ebit_marge', 'N/A')}%
Eigenkapitalquote: {kennzahlen.get('eigenkapitalquote', 'N/A')}%

Antworte nur mit einem JSON-Objekt in diesem Format:
{{
    "bewertungen": {{
        "kgv": {{
            "score": X,
            "begründung": "..."
        }},
        "roe": {{
            "score": X,
            "begründung": "..."
        }},
        "verschuldungsgrad": {{
            "score": X,
            "begründung": "..."
        }},
        "dividendenrendite": {{
            "score": X,
            "begründung": "..."
        }},
        "free_cashflow": {{
            "score": X,
            "begründung": "..."
        }},
        "umsatzwachstum": {{
            "score": X,
            "begründung": "..."
        }},
        "ebit_marge": {{
            "score": X,
            "begründung": "..."
        }},
        "eigenkapitalquote": {{
            "score": X,
            "begründung": "..."
        }}
    }},
    "gesamtempfehlung": "..."
}}"""
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            print(f"Fehler bei der Fundamental-Analyse: {str(e)}")
            return {}

    def _analyze_technical_indicators(self, technical_data: Dict[str, Any], horizont: str) -> Dict[str, Any]:
        """Analysiert technische Indikatoren mit Grok"""
        rsi = technical_data.get('rsi', 50)
        macd = technical_data.get('macd', 0)
        volatility = technical_data.get('volatilität', 0)
        
        prompt = f"""Analysiere folgende technische Indikatoren für einen {horizont}en Anlagehorizont:

RSI: {rsi}
MACD: {macd}
Volatilität: {volatility}%

Antworte nur mit einem JSON-Objekt in diesem Format:
{{
    "indikatoren": {{
        "rsi": {{
            "signal_stärke": X,
            "interpretation": "..."
        }},
        "macd": {{
            "signal_stärke": X,
            "interpretation": "..."
        }},
        "volatilität": {{
            "signal_stärke": X,
            "interpretation": "..."
        }}
    }},
    "gesamtbewertung": "..."
}}"""
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            print(f"Fehler bei der technischen Analyse: {str(e)}")
            return {}

    def _analyze_sentiment(self, sentiment_data: Dict[str, Any], horizont: str) -> Dict[str, Any]:
        """Analysiert Sentiment-Daten mit Grok"""
        prompt = f"""Analysiere folgende Sentiment-Daten für einen {horizont}en Anlagehorizont:

Gesamt-Sentiment: {sentiment_data.get('score', 'N/A')}
News-Sentiment: {sentiment_data.get('news_sentiment', 'N/A')}
Social Media-Sentiment: {sentiment_data.get('social_sentiment', 'N/A')}

Antworte nur mit einem JSON-Objekt in diesem Format:
{{
    "quellen": {{
        "news": {{
            "bewertung": X,
            "interpretation": "..."
        }},
        "social": {{
            "bewertung": X,
            "interpretation": "..."
        }}
    }},
    "gesamteinschätzung": "..."
}}"""
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            print(f"Fehler bei der Sentiment-Analyse: {str(e)}")
            return {}

    def _analyze_forecasts(self, forecasts: Dict[str, Any], horizont: str) -> Dict[str, Any]:
        """Analysiert die verschiedenen Prognosen mit Grok"""
        
        # Extrahiere GARCH Start- und Endwerte für Trendanalyse
        garch_data = forecasts.get('garch_volatility_change', {})
        if isinstance(garch_data, dict):
            garch_start = garch_data.get('start_volatility', 'N/A')
            garch_end = garch_data.get('end_volatility', 'N/A')
            garch_change = garch_data.get('volatility_change', 'N/A')
        else:
            garch_start = 'N/A'
            garch_end = 'N/A'
            garch_change = garch_data if garch_data is not None else 'N/A'

        prompt = f"""Analysiere folgende Prognosen für einen {horizont}en Anlagehorizont:

LSTM Prognose: {forecasts.get('lstm_change', 'N/A')}% Änderung
ARIMA Prognose: {forecasts.get('arima_change', 'N/A')}% Änderung

GARCH Volatilitätsprognose (annualisiert):
- Start Volatilität: {garch_start}% p.a.
- End Volatilität: {garch_end}% p.a.
- Volatilitätsänderung: {garch_change}%

Beachte bei der GARCH-Analyse:
- Die Werte sind annualisierte Volatilitäten (% pro Jahr)
- Ein Anstieg der Volatilität deutet auf zunehmende Marktunsicherheit hin
- Ein Rückgang der Volatilität deutet auf Marktberuhigung hin
- Nutze den Trend der Volatilität (steigend/fallend) als zusätzlichen Indikator für die Marktrichtung

Antworte nur mit einem JSON-Objekt in diesem Format:
{{
    "prognosen": {{
        "lstm": {{
            "signal_stärke": X,
            "interpretation": "..."
        }},
        "arima": {{
            "signal_stärke": X,
            "interpretation": "..."
        }},
        "garch": {{
            "signal_stärke": X,
            "interpretation": "...",
            "volatilitätstrend": "steigend/fallend/stabil",
            "marktimplikation": "..."
        }}
    }},
    "gesamteinschätzung": "..."
}}"""
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            print(f"Fehler bei der Prognose-Analyse: {str(e)}")
            return {}

    def predict(self, kpis: Dict[str, Any], df: pd.DataFrame, symbol: str, tage: int = None, 
               lstm_change: float = None, arima_change: float = None, 
               garch_volatility_change: float = None) -> Dict[str, Any]:
        """Generiert eine Handlungsempfehlung basierend auf allen verfügbaren Daten"""
        try:
            zeitraum = self._beschreibe_zeitraum(tage) if tage else "unbestimmter Zeitraum"
            
            # Hole alle Analysen
            fundamental_analysis = self._analyze_fundamentals(kpis.get('fundamental', {}), zeitraum)
            technical_analysis = self._analyze_technical_indicators(kpis.get('technical', {}), zeitraum)
            sentiment_analysis = self._analyze_sentiment(kpis.get('sentiment', {}), zeitraum)
            
            # Vorbereitung der Prognosedaten
            forecast_data = {
                'lstm_change': lstm_change,
                'arima_change': arima_change if arima_change is not None else 0.0,
                'garch_volatility_change': (
                    garch_volatility_change.get('volatility_change', 0.0) 
                    if isinstance(garch_volatility_change, dict) 
                    else garch_volatility_change if garch_volatility_change is not None 
                    else 0.0
                )
            }
            
            forecast_analysis = self._analyze_forecasts(forecast_data, zeitraum)
            
            # Bereite die Analysedaten für das Prompt vor
            analysis_data = {
                'fundamental': fundamental_analysis,
                'technical': technical_analysis,
                'sentiment': sentiment_analysis,
                'forecasts': forecast_analysis
            }
            
            # Definiere das erwartete JSON-Format
            json_format = '''{
    "decision": "Kaufen/Halten/Verkaufen",
    "confidence": 0.XX,
    "reasoning": "Deine ausführliche Begründung",
    "unternehmen": "Kurze Beschreibung was das Unternehmen macht"
}'''
            
            # Finale Empfehlung mit allen Daten
            prompt = f"""Du bist ein erfahrener Finanzanalyst. Analysiere {symbol} für einen Anlagehorizont von {zeitraum}.
Beschreibe kurz was das Unternehmen macht und gib eine klare Handelsempfehlung.

Technische Daten:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Gib eine klare Handelsempfehlung ab. Sei dir deiner Empfehlung auf einer Skala von 0.0 bis 1.0 sicher.
Je mehr Faktoren in die gleiche Richtung zeigen, desto höher sollte deine Konfidenz sein.

Antworte nur mit einem JSON-Objekt im folgenden Format:
{json_format}"""

            try:
                response = self.xai_client.chat.completions.create(
                    model="grok-beta",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                
                # Bereinige das JSON
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                try:
                    recommendation = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON Parsing Fehler: {str(e)}")
                    print(f"Erhaltener Inhalt: {content}")
                    return {
                        'decision': 'Halten',
                        'confidence': 0.5,
                        'reasoning': 'Fehler beim Parsen der KI-Antwort',
                        'forecast_details': {}
                    }
                
                return {
                    'decision': recommendation.get('decision', 'Halten'),
                    'confidence': float(recommendation.get('confidence', 0.5)),
                    'reasoning': recommendation.get('reasoning', 'Keine Begründung verfügbar'),
                    'unternehmen': recommendation.get('unternehmen', ''),
                    'forecast_details': {
                        'fundamental': fundamental_analysis,
                        'technical': technical_analysis,
                        'sentiment': sentiment_analysis,
                        'forecasts': forecast_analysis,
                        'investment_period': zeitraum
                    }
                }
                
            except Exception as e:
                print(f"API-Fehler: {str(e)}")
                return {
                    'decision': 'Halten',
                    'confidence': 0.5,
                    'reasoning': f'Fehler bei der API-Anfrage: {str(e)}',
                    'forecast_details': {}
                }
                
        except Exception as e:
            print(f"Allgemeiner Fehler: {str(e)}")
            return {
                'decision': 'Halten',
                'confidence': 0.33,
                'reasoning': f'Technischer Fehler: {str(e)}',
                'forecast_details': {}
            }

if __name__ == "__main__":
    # Test der KI
    ai = DashboardAI()
    test_kpis = {
        'fundamental': {
            'kgv': 15.5,
            'roe': 12.3,
            'verschuldungsgrad': 45.6,
            'dividendenrendite': 2.8,
            'free_cashflow': 500,
            'umsatzwachstum': 8.5,
            'ebit_marge': 15.2,
            'eigenkapitalquote': 40.5
        },
        'technical': {
            'rsi': 55,
            'macd': 0.5,
            'volatilität': 15
        },
        'sentiment': {
            'score': 0.6,
            'news_sentiment': 0.7,
            'social_sentiment': 0.5
        }
    }
    
    # Test mit 30 Tagen Anlagehorizont
    result = ai.predict(test_kpis, pd.DataFrame(), 'AAPL', tage=30)
    print("\nTest-Ergebnis:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
import yfinance as yf
import pandas as pd
import datetime
from transformers import pipeline
import torch
from reddit_sentiment import RedditSentimentAnalyzer

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialisierung des FinBERT Sentiment Analyzers"""
        print("Initialisiere FinBERT Sentiment Analyzer...")
        # Prüfe ob GPU verfügbar ist
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
        print("FinBERT Modell geladen!")

    def analysiere_text(self, text):
        """
        Analysiert den Sentiment eines einzelnen Textes
        Rückgabe: Score zwischen -1 und 1
        """
        try:
            ergebnis = self.sentiment_pipeline(text)[0]
            # Konvertiere FinBERT Labels in numerische Scores
            if ergebnis['label'] == 'positive':
                return ergebnis['score']
            elif ergebnis['label'] == 'negative':
                return -ergebnis['score']
            else:
                return 0
        except Exception as e:
            print(f"Fehler bei Textanalyse: {str(e)}")
            return 0

    def hole_nachrichten(self, symbol, max_nachrichten=5):
        """Holt und analysiert die neuesten Nachrichten für ein bestimmtes Symbol"""
        try:
            aktie = yf.Ticker(symbol)
            nachrichten = aktie.news
            
            if not nachrichten:
                print(f"Keine Nachrichten gefunden für {symbol}")
                return pd.DataFrame()
            
            nachrichten = nachrichten[:max_nachrichten]
            ergebnisse = []
            
            for nachricht in nachrichten:
                try:
                    # Kombiniere Titel und Zusammenfassung für die Analyse
                    text = f"{nachricht['title']} {nachricht.get('summary', '')}"
                    sentiment_score = self.analysiere_text(text)
                    
                    ergebnisse.append({
                        'datum': datetime.datetime.fromtimestamp(nachricht['providerPublishTime']),
                        'titel': nachricht['title'],
                        'sentiment_score': sentiment_score,
                        'quelle': nachricht.get('publisher', 'Unbekannt'),
                        'url': nachricht['link']
                    })
                except Exception as e:
                    print(f"Fehler bei Nachrichtenverarbeitung: {str(e)}")
                    continue
            
            return pd.DataFrame(ergebnisse)
        
        except Exception as e:
            print(f"Fehler beim Abrufen der Nachrichten: {str(e)}")
            return pd.DataFrame()

    def berechne_gesamt_sentiment(self, df):
        """
        Berechnet den gewichteten Sentiment-Score aus allen Nachrichten
        Neuere Nachrichten werden stärker gewichtet
        """
        if df.empty:
            return 0, 0
        
        # Gewichte neuere Nachrichten stärker
        max_datum = df['datum'].max()
        df['gewicht'] = df['datum'].apply(lambda x: 1 / (1 + (max_datum - x).days))
        
        # Berechne gewichteten Durchschnitt
        gewichteter_score = (df['sentiment_score'] * df['gewicht']).sum() / df['gewicht'].sum()
        
        # Berechne die prozentuale Verteilung der positiven Sentiments
        positiv = len(df[df['sentiment_score'] > 0]) / len(df) * 100
        
        return round(gewichteter_score, 3), round(positiv, 1)

    def klassifiziere_sentiment(self, score):
        """
        Klassifiziert einen Sentiment-Score in verschiedene Kategorien
        Rückgabe: Textuelle Beschreibung des Sentiments
        """
        if score >= 0.5:
            return "Sehr Positiv"
        elif score >= 0.2:
            return "Positiv"
        elif score <= -0.5:
            return "Sehr Negativ"
        elif score <= -0.2:
            return "Negativ"
        else:
            return "Neutral"

class CombinedSentimentAnalyzer:
    def __init__(self):
        """Initialisierung des kombinierten Sentiment-Analyzers für News und Social Media"""
        self.news_analyzer = SentimentAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer()
        
    def analysiere_gesamt_sentiment(self, symbol):
        """
        Analysiert das Gesamtsentiment aus allen verfügbaren Quellen
        Kombiniert News- und Social-Media-Sentiment mit Gewichtung
        """
        # News Sentiment analysieren
        news_df = self.news_analyzer.hole_nachrichten(symbol)
        if not news_df.empty:
            news_score, news_positiv = self.news_analyzer.berechne_gesamt_sentiment(news_df)
        else:
            news_score, news_positiv = 0, 0
            
        # Reddit Sentiment analysieren
        reddit_df = self.reddit_analyzer.hole_reddit_posts(symbol)
        if not reddit_df.empty:
            reddit_df = self.reddit_analyzer.analysiere_reddit_sentiment(reddit_df)
            reddit_score, reddit_positiv, subreddit_stats = self.reddit_analyzer.berechne_reddit_gesamt_sentiment(reddit_df)
        else:
            reddit_score, reddit_positiv, subreddit_stats = 0, 0, {}
            
        # Kombiniere die Sentiments (Gewichtung: 60% News, 40% Social Media)
        gesamt_score = (news_score * 0.6) + (reddit_score * 0.4)
        gesamt_positiv = (news_positiv * 0.6) + (reddit_positiv * 0.4)
        
        return {
            'gesamt': {
                'score': round(gesamt_score, 3),
                'positiv_prozent': round(gesamt_positiv, 1),
                'kategorie': self.news_analyzer.klassifiziere_sentiment(gesamt_score)
            },
            'news': {
                'score': news_score,
                'positiv_prozent': news_positiv,
                'nachrichten': news_df.to_dict('records') if not news_df.empty else []
            },
            'social': {
                'score': reddit_score,
                'positiv_prozent': reddit_positiv,
                'subreddit_stats': subreddit_stats,
                'posts': reddit_df.to_dict('records') if not reddit_df.empty else []
            }
        }

if __name__ == "__main__":
    # Teste den kombinierten Analyzer
    analyzer = CombinedSentimentAnalyzer()
    symbol = 'AAPL'
    
    print(f"\nAnalysiere Gesamtsentiment für {symbol}...")
    ergebnis = analyzer.analysiere_gesamt_sentiment(symbol)
    
    print("\nGesamtergebnis:")
    print(f"Score: {ergebnis['gesamt']['score']}")
    print(f"Positiv: {ergebnis['gesamt']['positiv_prozent']}%")
    print(f"Kategorie: {ergebnis['gesamt']['kategorie']}")
    
    print("\nNews Sentiment:")
    print(f"Score: {ergebnis['news']['score']}")
    print(f"Positiv: {ergebnis['news']['positiv_prozent']}%")
    
    print("\nSocial Media Sentiment:")
    print(f"Score: {ergebnis['social']['score']}")
    print(f"Positiv: {ergebnis['social']['positiv_prozent']}%")
    
    print("\nSentiment nach Subreddit:")
    for subreddit, stats in ergebnis['social']['subreddit_stats'].items():
        print(f"\n{subreddit}:")
        print(f"Sentiment: {stats['sentiment_score']:.3f}")
        print(f"Aktivität: {stats['score']} Upvotes, {stats['kommentare']} Kommentare")
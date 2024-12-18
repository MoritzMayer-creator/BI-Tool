from praw import Reddit
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

class RedditSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialisiert den Reddit Sentiment Analyzer mit dem angegebenen Modell"""
        self.reddit = Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='FinanceAnalyzer/1.0'
        )
        
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            max_length=512,
            truncation=True
        )
        
        self.finance_subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'stockmarket',
            'options'
        ]

    def _clean_text(self, text):
        """Bereinigt den Text und kürzt ihn auf eine geeignete Länge"""
        # Begrenze die Textlänge auf 450 Zeichen (sicher unter dem 512 Token Limit)
        text = text[:450] if text else ""
        # Entferne überflüssige Leerzeichen und Zeilenumbrüche
        text = ' '.join(text.split())
        return text

    def hole_reddit_posts(self, symbol, tage=7, max_posts=100):
        """
        Holt Reddit-Beiträge für ein bestimmtes Aktiensymbol
        
        Args:
            symbol: Das zu suchende Aktiensymbol
            tage: Zeitraum in Tagen für die Suche
            max_posts: Maximale Anzahl der Posts pro Subreddit
        """
        ende_datum = datetime.utcnow()
        start_datum = ende_datum - timedelta(days=tage)
        
        alle_posts = []
        
        for subreddit_name in self.finance_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Suche nach Posts mit dem Symbol
                suche = subreddit.search(
                    f"{symbol}",
                    syntax='lucene',
                    time_filter='week',
                    sort='relevance',
                    limit=max_posts
                )
                
                for post in suche:
                    erstelldatum = datetime.fromtimestamp(post.created_utc)
                    if start_datum <= erstelldatum <= ende_datum:
                        # Nehme nur den Titel für die Sentiment-Analyse
                        text = self._clean_text(post.title)
                        
                        alle_posts.append({
                            'datum': erstelldatum,
                            'titel': post.title,
                            'text': text,
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'kommentare': post.num_comments,
                            'url': f"https://reddit.com{post.permalink}"
                        })
            except Exception as e:
                print(f"Fehler beim Abrufen von r/{subreddit_name}: {str(e)}")
                continue
        
        return pd.DataFrame(alle_posts)

    def analysiere_reddit_sentiment(self, posts_df):
        """
        Führt die Sentiment-Analyse für Reddit-Beiträge durch
        
        Berechnet einen normalisierten Sentiment-Score für jeden Beitrag,
        der durch die Popularität des Beitrags gewichtet wird
        """
        if posts_df.empty:
            return pd.DataFrame()
            
        for index, row in posts_df.iterrows():
            try:
                if row['text']:
                    sentiment = self.sentiment_pipeline(row['text'])[0]
                    
                    # Basis-Score von -1 bis 1
                    if sentiment['label'] == 'positive':
                        base_score = sentiment['score']
                    elif sentiment['label'] == 'negative':
                        base_score = -sentiment['score']
                    else:
                        base_score = 0
                    
                    # Gewichtung basierend auf Popularität (0.5 bis 1.5)
                    gewichtung = 0.5 + min((row['score'] + row['kommentare']) / 2000, 1.0)
                    
                    # Gewichteter Score, garantiert im Bereich -1 bis 1
                    posts_df.at[index, 'sentiment_score'] = base_score * gewichtung
                    
                else:
                    posts_df.at[index, 'sentiment_score'] = 0
                    
            except Exception as e:
                print(f"Fehler bei der Sentiment-Analyse: {str(e)}")
                posts_df.at[index, 'sentiment_score'] = 0
                
        return posts_df

    def berechne_reddit_gesamt_sentiment(self, posts_df):
        """
        Berechnet das Gesamtsentiment aller Reddit-Beiträge
        
        Returns:
            - Gewichteter Gesamtscore (-1 bis 1)
            - Prozentualer Anteil positiver Sentiments
            - Sentiment-Statistiken pro Subreddit
        """
        if posts_df.empty:
            return 0, 0, {}
            
        # Gewichte neuere Posts stärker
        max_datum = posts_df['datum'].max()
        posts_df['zeitgewicht'] = posts_df['datum'].apply(
            lambda x: 1 / (1 + (max_datum - x).days)
        )
        
        # Berechne gewichtetes Durchschnittssentiment
        posts_mit_sentiment = posts_df[posts_df['sentiment_score'] != 0]
        if not posts_mit_sentiment.empty:
            gewichteter_score = (
                posts_mit_sentiment['sentiment_score'] * posts_mit_sentiment['zeitgewicht']
            ).sum() / posts_mit_sentiment['zeitgewicht'].sum()
            
            # Berechne Sentiment-Verteilung
            positiv = len(posts_mit_sentiment[posts_mit_sentiment['sentiment_score'] > 0]) / len(posts_mit_sentiment) * 100
        else:
            gewichteter_score = 0
            positiv = 0
        
        # Sentiment pro Subreddit
        subreddit_sentiment = posts_df.groupby('subreddit').agg({
            'sentiment_score': 'mean',
            'score': 'sum',
            'kommentare': 'sum'
        }).to_dict('index')
        
        return round(gewichteter_score, 3), round(positiv, 1), subreddit_sentiment

if __name__ == "__main__":
    # Beispielausführung des Reddit Sentiment Analyzers
    analyzer = RedditSentimentAnalyzer()
    symbol = 'AAPL'
    
    print(f"\nAnalysiere Reddit-Beiträge für {symbol}...")
    posts_df = analyzer.hole_reddit_posts(symbol)
    
    if not posts_df.empty:
        posts_df = analyzer.analysiere_reddit_sentiment(posts_df)
        score, positiv, subreddit_stats = analyzer.berechne_reddit_gesamt_sentiment(posts_df)
        
        print(f"\nReddit Sentiment-Score: {score}")
        print(f"Prozent positiv: {positiv}%")
        
        print("\nSentiment nach Subreddit:")
        for subreddit, stats in subreddit_stats.items():
            print(f"\n{subreddit}:")
            print(f"Durchschnittliches Sentiment: {stats['sentiment_score']:.3f}")
            print(f"Gesamte Upvotes: {stats['score']}")
            print(f"Gesamte Kommentare: {stats['kommentare']}")
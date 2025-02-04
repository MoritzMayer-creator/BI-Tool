from dash import Dash, html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import callback_context
from aktien_datenpipeline import AktienDatenPipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from volatility_predictor import VolatilityPredictor
from fundamental_analyzer import FundamentalAnalyzer
from dashboard_fundamental import erstelle_fundamental_bereich, register_fundamental_callbacks
from trading_ai import DashboardAI
from lstm_predictor import AktienPredictor
import sys
from arch import arch_model
from arima_predictor import ArimaPredictor
from arima_performance_evaluator import ArimaPerformanceEvaluator
from garch_performance_evaluator import GarchPerformanceEvaluator
from lstm_performance_evaluator import LstmPerformanceEvaluator

# Globale Variablen für die Anwendung
global_lstm_change = None
current_df = None
current_stats = None
df = None

# Initialisiere die Dash-App mit Bootstrap und FontAwesome Styling
external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

def erstelle_sentiment_gauge(score):
    """Erstellt einen Gauge-Chart mit Emojis zur Visualisierung des Sentiments"""
    gauge_wert = ((score + 1) / 2) * 4 + 1

    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = gauge_wert,
        domain = {'x': [0.1, 0.9], 'y': [0, 0.85]},
        title = {'text': "Sentiment Score", 'font': {'size': 24}},
        gauge = {
            'axis': {
                'range': [1, 5],
                'tickwidth': 2,
                'tickcolor': "black",
                'tickmode': 'array',
                'ticktext': ['😡', '😕', '😐', '🙂', '😊'],
                'tickvals': [1, 2, 3, 4, 5],
                'tickfont': {'size': 16},
                'tickangle': 0
            },
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [1, 2], 'color': "rgb(255, 0, 0)"},
                {'range': [2, 3], 'color': "rgb(255, 165, 0)"},
                {'range': [3, 4], 'color': "rgb(255, 255, 0)"},
                {'range': [4, 5], 'color': "rgb(0, 255, 0)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_wert
            }
        }
    ))

    fig.update_layout(
        height=338,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.add_annotation(
        x=0.5,
        y=0.3,
        text=f"{score:.2f}",
        showarrow=False,
        font=dict(size=24, color="black")
    )

    return fig

def erstelle_hauptchart(df, symbol):
    """Erstellt den Hauptchart mit Kerzendiagramm und MACD-Indikator"""
    df = df.sort_index()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Kurs'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange', width=1.5)
    ), row=2, col=1)

    colors = ['red' if val < 0 else 'green' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Hist'],
        name='MACD Hist',
        marker_color=colors
    ), row=2, col=1)

    fig.update_layout(
        title=f'{symbol} Aktienkurs',
        yaxis_title='Preis ($)',
        yaxis2_title='MACD',
        height=600,
        hovermode='x unified',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    last_date = df.index[-1]
    six_months_ago = last_date - pd.DateOffset(months=6)
    fig.update_xaxes(range=[six_months_ago, last_date])

    fig.update_yaxes(title_text="Preis ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)

    return fig

def erstelle_prognose_chart(df, predictions, title="Preisprognose"):
    """Erstellt einen Chart mit historischen Daten und zukünftiger Prognose"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Historische Daten',
        line=dict(color='blue', width=2),
        hovertemplate='%{x}<br>Preis: $%{y:.2f}<extra></extra>'
    ))

    future_dates = pd.date_range(start=df.index[-1], periods=len(predictions)+1, freq='B')[1:]

    fig.add_trace(go.Scatter(
        x=[df.index[-1], future_dates[0]],
        y=[df['Close'].iloc[-1], predictions[0]],
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Prognose',
        line=dict(color='red', width=2),
        hovertemplate='%{x}<br>Prognose: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Preis ($)',
        xaxis_title='Datum',
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    return fig

def erstelle_volatilitäts_chart(df, forecast_data):
    """Erstellt den Volatilitätschart basierend auf GARCH-Prognosen"""
    try:
        # Initialisiere den GARCH Performance Evaluator für 2-Tage-Prognosen
        evaluator = GarchPerformanceEvaluator()
        
        # Berechne die Metriken ohne Konsolenausgabe
        metrics = evaluator.evaluate_predictions(
            df=df,
            evaluation_start_date=df.index[0],
            forecast_horizon=2,
            silent=True
        )
        
        if metrics is None:
            raise ValueError("Keine GARCH-Metriken verfügbar")
            
        # Hole die letzten 3 Prognosen (einen Tag davor + 2 Tage Prognose)
        rolling_predictions = metrics['rolling_predictions']
        realized_vol = metrics['rolling_realized']
        
        # Hole die letzten 3 Werte (einer davor + 2 Prognosen)
        last_predictions = rolling_predictions.iloc[-3:]
        
        # Erstelle Zukunftsdaten für die Visualisierung (3 Tage: 1 davor + 2 Prognose)
        future_dates = pd.date_range(
            start=df.index[-1],  # Start beim letzten historischen Datum
            periods=3,  # 3 Tage total
            freq='B'  # Geschäftstage
        )
        
        # Figure erstellen
        fig = go.Figure()
        
        # Plot die GARCH Prognose (alle 3 Werte)
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=last_predictions.values,
            name='GARCH Prognose',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title={
                'text': 'GARCH 2-Tage Volatilitätsprognose',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title='Volatilität (%)',
            xaxis_title='Datum',
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

        # Berechne Änderung zwischen erstem und letztem Prognosetag
        vol_change = last_predictions.iloc[-1] - last_predictions.iloc[0]

        # Info-Karten
        info_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Start Volatilität", className="card-title"),
                    html.H3(f"{last_predictions.iloc[0]:.2f}%", 
                           className="text-primary")
                ])
            ])),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("End Volatilität 2d", className="card-title"),
                    html.H3(f"{last_predictions.iloc[-1]:.2f}%", 
                           className="text-primary")
                ])
            ])),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("2-Tage Änderung", className="card-title"),
                    html.H3(
                        f"{vol_change:+.2f}%",
                        className=f"text-{'success' if vol_change < 0 else 'danger'}"
                    )
                ])
            ]))
        ])

        return fig, info_cards
        
    except Exception as e:
        print(f"Fehler beim Erstellen des GARCH-Charts: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Keine GARCH-Prognose verfügbar",
            annotations=[{
                'text': str(e),
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        error_cards = html.Div("Fehler bei der Volatilitätsprognose")
        return empty_fig, error_cards

def erstelle_sentiment_details(stats, pipeline):
    """Erstellt eine detaillierte Ansicht der Sentiment-Analyse mit News und Social Media Daten"""
    sentiment_info = html.Div([
        html.H5("News & Social Media Sentiment", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Sentiment Kategorie", className="card-subtitle mb-2 text-muted"),
                        html.H4(
                            pipeline.sentiment_kategorie if hasattr(pipeline, 'sentiment_kategorie') else "N/A",
                            className="card-title"
                        )
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("News Sentiment", className="card-subtitle mb-2 text-muted"),
                        html.H4(
                            f"{pipeline.news_sentiment_score:.2f}" if hasattr(pipeline, 'news_sentiment_score') else "N/A",
                            className="card-title"
                        )
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Reddit Sentiment", className="card-subtitle mb-2 text-muted"),
                        html.H4(
                            f"{pipeline.reddit_sentiment_score:.2f}" if hasattr(pipeline, 'reddit_sentiment_score') else "N/A",
                            className="card-title"
                        )
                    ])
                ])
            ], width=4)
        ], className="mb-4"),

        html.H5("Aktuelle Nachrichten", className="mb-3"),
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H6(
                        row['datum'].strftime('%Y-%m-%d'),
                        className="card-subtitle mb-2 text-muted"
                    ),
                    html.A(
                        row['titel'],
                        href=row.get('url', '#'),
                        target="_blank",
                        className="text-primary"
                    ),
                    html.P([
                        "Sentiment: ",
                        html.Span(
                            f"{row['sentiment_score']:.2f}",
                            className=f"text-{'success' if row['sentiment_score'] > 0 else 'danger'}"
                        )
                    ], className="card-text mt-2")
                ])
            ], className="mb-3") for _, row in pipeline.sentiment_nachrichten.iterrows()
        ] if hasattr(pipeline, 'sentiment_nachrichten') and not pipeline.sentiment_nachrichten.empty else
          [html.P("Keine aktuellen Nachrichten verfügbar")]),

        html.H5("Reddit Diskussionen", className="mb-3 mt-4"),
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6(
                                f"r/{post['subreddit']} • {post['datum'].strftime('%Y-%m-%d')}",
                                className="card-subtitle mb-2 text-muted"
                            ),
                            html.A(
                                post['titel'],
                                href=post['url'],
                                target="_blank",
                                className="text-primary"
                            ),
                        ], width=9),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-arrow-up", style={'fontSize': '0.8rem'}),
                                html.Span(f"{post['score']}", style={'fontSize': '0.8rem', 'marginRight': '8px'}),
                                html.I(className="fas fa-comment", style={'fontSize': '0.8rem'}),
                                html.Span(f"{post['kommentare']}", style={'fontSize': '0.8rem'})
                            ], className="text-muted text-right", style={'whiteSpace': 'nowrap'})
                        ], width=3)
                    ]),
                    html.P([
                        "Sentiment: ",
                        html.Span(
                            f"{post['sentiment_score']:.2f}",
                            className=f"text-{'success' if post['sentiment_score'] > 0 else 'danger'}"
                        )
                    ], className="card-text mt-2")
                ])
            ], className="mb-3") for post in pipeline.reddit_posts[:5]
        ] if hasattr(pipeline, 'reddit_posts') and pipeline.reddit_posts else
          [html.P("Keine Reddit Diskussionen verfügbar")])
    ])

    return sentiment_info

def erstelle_indikator_karte(titel, wert, zusatz="", farbe="primary"):
    """Erstellt eine Anzeigekomponente für einen einzelnen Indikator"""
    return dbc.Card([
        dbc.CardBody([
            html.H4(titel, className="card-title"),
            html.H2(f"{wert}{zusatz}")
        ])
    ], color=farbe, inverse=True)

def erstelle_lstm_prognose(df, prognose_tage=10, epochs=50):
    """Erstellt eine LSTM-basierte Preisprognose für die angegebene Anzahl von Tagen"""
    try:
        # Verwende die gleiche Evaluierung wie im Performance Evaluator
        evaluator = LstmPerformanceEvaluator()
        metrics = evaluator.evaluate_predictions(df)
        
        if metrics is None:
            raise ValueError("Keine gültige Prognose generiert")
        
        # Erstelle Plot
        prognose_fig = go.Figure()
        
        # Plot historische Daten (nur die letzten 30 Tage)
        last_30_days = df.last('30D')
        prognose_fig.add_trace(go.Scatter(
            x=last_30_days.index,
            y=last_30_days['Close'],
            name='Historische Daten',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Preis: $%{y:.2f}<extra></extra>'
        ))
        
        # Hole die Zukunftsprognose
        future_predictions = metrics['future_forecast']
        
        # Füge Verbindungslinie hinzu
        prognose_fig.add_trace(go.Scatter(
            x=[df.index[-1], future_predictions.index[0]],
            y=[df['Close'].iloc[-1], future_predictions.iloc[0]],
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Plot die Zukunftsprognose
        prognose_fig.add_trace(go.Scatter(
            x=future_predictions.index,
            y=future_predictions,
            name='LSTM Prognose',
            line=dict(color='red', width=2),
            hovertemplate='%{x}<br>Prognose: $%{y:.2f}<extra></extra>'
        ))
        
        # Layout anpassen
        prognose_fig.update_layout(
            title={
                'text': "LSTM 2-Tages Preisprognose",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title='Preis ($)',
            xaxis_title='Datum',
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        prognose_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        prognose_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # Berechne die Preisänderung
        last_prediction = future_predictions.iloc[-1]
        price_change = ((last_prediction - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        
        # Info-Karten erstellen
        info_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Prognostizierter Endpreis", className="card-title"),
                    html.H3(f"${last_prediction:.2f}", className="text-primary")
                ])
            ])),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Erwartete Änderung", className="card-title"),
                    html.H3(
                        f"{price_change:+.1f}%",
                        className=f"text-{'success' if price_change > 0 else 'danger'}"
                    )
                ])
            ]))
        ])
        
        return prognose_fig, info_cards
        
    except Exception as e:
        print(f"Fehler bei LSTM-Prognose: {e}")
        return go.Figure(), html.Div("Fehler bei der LSTM-Prognose")

def update_lstm_forecast():
    """Aktualisiert die LSTM-Prognose mit den aktuellen Daten"""
    global df
    if df is None:
        return None
        
    try:
        # LSTM Vorhersage
        predictor = AktienPredictor()
        predictor.train(df, epochs=20, batch_size=32, validation_split=0.1)  # Gleiche Parameter wie im Evaluator
        
        # 2-Tage-Vorhersage statt 10 Tage
        forecast = predictor.predict(df, days_ahead=2)
        predictions = forecast['predictions']
        
        # Zukunftsdaten für 2 Handelstage
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=4,  # Reduziert von 15 auf 4
            freq='D'
        )
        future_dates = [date for date in future_dates if date.weekday() < 5][:2]  # Nur 2 Handelstage
        
        # Performance Evaluation
        evaluator = LstmPerformanceEvaluator()
        metrics = evaluator.evaluate_predictions(df)
        
        if metrics:
            rmse = metrics['rmse']
            mape = metrics['mape']
            correlation = metrics['correlation']
            
            # Update der Metriken im Dashboard
            metrics_text = (
                f"LSTM Performance Metriken (2-Tage-Vorhersage):\n"  # Geändert von 10 auf 2
                f"RMSE: ${rmse:.2f}\n"
                f"MAPE: {mape:.2f}%\n"
                f"Korrelation: {correlation:.2f}"
            )
            
            # Update der Vorhersage-Tabelle
            forecast_table = pd.DataFrame({
                'Datum': future_dates,
                'Prognose': predictions,
                'Änderung (%)': [(pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100 
                                for pred in predictions]
            })
            
            return {
                'metrics': metrics_text,
                'forecast_table': forecast_table.to_dict('records'),
                'lstm_plot': evaluator.plot_performance()  # Dies nutzt nun auch die 2-Tage-Parameter
            }
            
    except Exception as e:
        print(f"Fehler in LSTM-Vorhersage: {str(e)}")
        return None

# Definition des Dashboard-Layouts mit verschiedenen Bereichen
app.layout = dbc.Container([
    # Kopfbereich
    dbc.Row([
        dbc.Col(html.H1("Aktienanalyse Dashboard", className="text-center mb-4"), width=12)
    ]),

    # Eingabebereich für Aktiensymbol
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id="aktien-input", 
                        placeholder="Aktiensymbol eingeben (z.B. AAPL)", 
                        type="text"
                    )
                ], width=9),
                dbc.Col([
                    dbc.Button(
                        "Analysieren", 
                        id="analyse-button", 
                        color="primary", 
                        className="w-100"
                    )
                ], width=3)
            ], className="g-2")
        ], width={"size": 6, "offset": 3})
    ]),

    # Kennzahlen
    dbc.Row([
        dbc.Col(html.Div(id="zusammenfassung-stats"), className="mt-4")
    ]),

    # Hauptbereich mit Sentiment und Charts
    dbc.Row([
        # Linke Spalte: Sentiment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id="sentiment-gauge", style={'height': '338px'}),
                    html.Hr(),
                    html.Div(id="sentiment-details")
                ])
            ], className="mt-4 me-3", style={'height': '2200px'})
        ], width=4),

        # Rechte Spalte: Charts und Prognosen
        dbc.Col([
            # Aktienchart oben
            dcc.Graph(id="aktien-chart", style={'height': '600px'}, className="mt-4"),

            # Prognosebereich
            dbc.Card([
                dbc.CardHeader(html.H4("Prognosen", className="mb-0")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-forecasts",
                        type="circle",
                        children=[
                            dcc.Graph(id="prognose-chart", style={'height': '400px'}),
                            html.Div(id="prognose-info"),
                            dcc.Graph(id="arima-prognose-chart", style={'height': '400px'}),
                            html.Div(id="arima-prognose-info"),
                            dcc.Graph(id="garch-prognose-chart", style={'height': '400px'}),
                            html.Div(id="garch-prognose-info")
                        ]
                    )
                ])
            ], className="mt-4")
        ], width=8)
    ], className="g-0"),

    # Fundamentalanalyse Bereich
    erstelle_fundamental_bereich(),

    # KI-Handelsempfehlung (am Ende)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("KI-Handelsempfehlung", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="horizont-input",
                                placeholder="Anlagehorizont in Tagen",
                                type="number",
                                min=1,
                                value="",  # Kein Standardwert
                                className="mb-2"
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                "KI-Analyse starten", 
                                id="ki-analyse-button", 
                                color="primary",
                                className="w-100"
                            )
                        ], width=4)
                    ]),
                    html.Div(id="ki-empfehlung", className="mt-4")
                ])
            ], className="mt-4 mb-4")
        ], width=12)
    ])
], fluid=True)

# Callback-Funktionen für die Interaktivität
@app.callback(
    [Output("zusammenfassung-stats", "children"),
     Output("aktien-chart", "figure"),
     Output("sentiment-gauge", "figure"),
     Output("sentiment-details", "children"),
     Output("prognose-chart", "figure"),
     Output("prognose-info", "children"),
     Output("arima-prognose-chart", "figure"),
     Output("arima-prognose-info", "children"),
     Output("garch-prognose-chart", "figure"),
     Output("garch-prognose-info", "children")],
    [Input("analyse-button", "n_clicks")],
    [State("aktien-input", "value")]
)
def aktualisiere_dashboard(n_clicks, symbol):
    """Hauptfunktion zur Aktualisierung aller Dashboard-Komponenten"""
    if not symbol or not n_clicks:
        raise PreventUpdate

    try:
        pipeline = AktienDatenPipeline()
        df = pipeline.hole_daten(symbol, zeitraum="2y")
        df = pipeline.berechne_indikatoren()
        stats = pipeline.hole_zusammenfassung()

        # Speichere Daten für KI-Empfehlung
        global current_df
        global current_stats
        current_df = df
        current_stats = stats

        # Basisdaten
        übersicht_karten = dbc.Row([
            dbc.Col(erstelle_indikator_karte(
                "Aktueller Preis",
                f"${stats['aktueller_preis']}"
            )),
            dbc.Col(erstelle_indikator_karte(
                "Tagesrendite",
                stats['tages_rendite'],
                "%",
                "success" if stats['tages_rendite'] > 0 else "danger"
            )),
            dbc.Col(erstelle_indikator_karte(
                "RSI",
                stats['rsi'],
                "",
                "danger" if stats['rsi'] > 70 else "success" if stats['rsi'] < 30 else "primary"
            )),
            dbc.Col(erstelle_indikator_karte(
                "MACD Trend",
                "Bullish" if stats['macd'] > 0 else "Bearish",
                "",
                "success" if stats['macd'] > 0 else "danger"
            ))
        ])

        # Charts und Prognosen
        fig_aktie = erstelle_hauptchart(df, symbol)
        gauge_fig = erstelle_sentiment_gauge(stats['sentiment_score'])
        sentiment_details = erstelle_sentiment_details(stats, pipeline)
        
        # LSTM Prognose
        lstm_fig, lstm_info = erstelle_lstm_prognose(df, 10, 50)
        
        # Speichere den LSTM Change als globale Variable
        global global_lstm_change
        if isinstance(lstm_info, dbc.Row):
            # Extrahiere den Wert aus der Karte
            for card in lstm_info.children:
                if "Erwartete Änderung" in str(card):
                    text = str(card)
                    # Extrahiere die Zahl aus dem Text
                    import re
                    match = re.search(r'([+-]?\d+\.?\d*)%', text)
                    if match:
                        global_lstm_change = float(match.group(1))
                        break
        
        # ARIMA Prognose
        try:
            evaluator = ArimaPerformanceEvaluator()
            metrics = evaluator.evaluate_predictions(
                df=df,
                evaluation_start_date=None,
                forecast_days=1,
                test_size=60
            )
            
            # Hole die letzte Vorhersage aus den Metriken
            predictions = metrics['arima_forecasts']
            historical_predictions = predictions[-1:]
            
            # Berechne den nächsten Handelstag
            last_date = df.index[-1]
            next_trading_day = last_date + pd.Timedelta(days=1)
            while next_trading_day.weekday() >= 5:  # Überspringe Wochenenden
                next_trading_day += pd.Timedelta(days=1)
            
            # Setze den Index der Vorhersage auf den nächsten Handelstag
            historical_predictions.index = [next_trading_day]
            
            # Plot erstellen...
            arima_fig = go.Figure()
            
            # Plot nur die letzten 30 Tage der historischen Daten
            last_30_days = df.last('30D')
            arima_fig.add_trace(go.Scatter(
                x=last_30_days.index,
                y=last_30_days['Close'],
                name='Historische Daten',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Preis: $%{y:.2f}<extra></extra>'
            ))
            
            # Füge Verbindungslinie hinzu
            connection_x = [df.index[-1], historical_predictions.index[0]]
            connection_y = [df['Close'].iloc[-1], historical_predictions.iloc[0]]
            arima_fig.add_trace(go.Scatter(
                x=connection_x,
                y=connection_y,
                name='Verbindung',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Plot die ARIMA Prognosen
            arima_fig.add_trace(go.Scatter(
                x=historical_predictions.index,
                y=historical_predictions,
                name='ARIMA Prognosen',
                line=dict(color='red', width=2),
                hovertemplate='%{x}<br>Prognose: $%{y:.2f}<extra></extra>'
            ))

            last_prediction = historical_predictions.iloc[-1]
            arima_change = ((last_prediction - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100

            # Layout anpassen wie bei den anderen Charts
            arima_fig.update_layout(
                title={
                    'text': "ARIMA 1-Tages Preisprognose",  # Titel angepasst
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                yaxis_title='Preis ($)',
                xaxis_title='Datum',
                height=400,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                )
            )

            arima_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            arima_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

            arima_info = dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Prognostizierter Endpreis (ARIMA)", className="card-title"),
                        html.H3(f"${last_prediction:.2f}", className="text-primary")
                    ])
                ])),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Erwartete Änderung (ARIMA)", className="card-title"),
                        html.H3(
                            f"{arima_change:+.1f}%",
                            className=f"text-{'success' if arima_change > 0 else 'danger'}"
                        )
                    ])
                ]))
            ])

        except Exception as e:
            print(f"Fehler in ARIMA-Prognose: {str(e)}")
            arima_fig = go.Figure()
            arima_info = html.Div("Fehler bei der ARIMA-Prognose")

        # GARCH Prognose
        try:
            volatility_predictor = VolatilityPredictor()
            garch_forecast = volatility_predictor.garch_forecast(df)
            
            if garch_forecast is not None:
                garch_fig, garch_info = erstelle_volatilitäts_chart(df, garch_forecast)
            else:
                raise ValueError("Keine GARCH-Prognose verfügbar")
            
        except Exception as e:
            print(f"Fehler in GARCH-Prognose: {str(e)}")
            garch_fig = go.Figure()
            garch_info = html.Div("Fehler bei der Volatilitätsprognose")

        # Debug-Ausgabe hinzufügen
        print("\nDashboard ARIMA Vorhersagen:")
        print(historical_predictions)

        return (
            übersicht_karten, 
            fig_aktie, 
            gauge_fig, 
            sentiment_details,
            lstm_fig,
            lstm_info,
            arima_fig,
            arima_info,
            garch_fig,
            garch_info
        )

    except Exception as e:
        print(f"Fehler im Dashboard: {str(e)}")
        # Erstelle leere Figuren für die Graphen
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Keine Daten verfügbar",
            annotations=[{
                'text': "Fehler beim Laden der Daten",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        error_msg = html.Div("Fehler beim Laden der Daten")
        
        return [
            error_msg,  # zusammenfassung-stats
            empty_fig,  # aktien-chart
            empty_fig,  # sentiment-gauge
            error_msg,  # sentiment-details
            empty_fig,  # prognose-chart
            error_msg,  # prognose-info
            empty_fig,  # arima-prognose-chart
            error_msg,  # arima-prognose-info
            empty_fig,  # garch-prognose-chart
            error_msg   # garch-prognose-info
        ]

@app.callback(
    Output("ki-empfehlung", "children"),
    [Input("ki-analyse-button", "n_clicks")],
    [State("horizont-input", "value"),
     State("aktien-input", "value")]
)
def update_ki_empfehlung(n_clicks, anlagehorizont, symbol):
    """Generiert eine KI-basierte Handelsempfehlung basierend auf allen verfügbaren Daten"""
    if not n_clicks or not symbol:
        return None
        
    if not hasattr(sys.modules[__name__], 'current_df') or current_df is None:
        return html.Div("Bitte führen Sie zuerst eine Aktienanalyse durch.")
            
    tage = int(anlagehorizont) if anlagehorizont else None  # Kein Standardwert
    
    try:
        # Get fundamental data from warehouse through FundamentalAnalyzer
        fundamental_analyzer = FundamentalAnalyzer()
        fundamental_data = fundamental_analyzer.hole_fundamentaldaten(symbol)
        
        # Prepare KPIs dictionary
        current_kpis = {
            'fundamental': fundamental_data,  # Hier kommen die Kennzahlen aus dem Warehouse
            'technical': {
                'rsi': current_stats['rsi'],
                'macd': current_stats['macd'],
                'volatilität': current_stats.get('volatilität', 0)
            },
            'sentiment': {
                'score': current_stats['sentiment_score'],
                'news_sentiment': current_stats.get('sentiment_score', 0),
                'social_sentiment': current_stats.get('sentiment_score', 0)
            }
        }
        
        # Get global forecast changes
        lstm_change = global_lstm_change if 'global_lstm_change' in globals() else None
        
        # Angepasste ARIMA-Prognose
        arima_predictor = ArimaPredictor()
        arima_ergebnis = arima_predictor.erstelle_prognose(current_df)
        arima_change = arima_ergebnis.get('preisänderung', 0.0) if arima_ergebnis else 0.0
        
        # Angepasste GARCH-Prognose
        volatility_predictor = VolatilityPredictor()
        garch_ergebnis = volatility_predictor.garch_forecast(current_df)
        
        # GARCH-Daten für die KI aufbereiten
        if isinstance(garch_ergebnis, dict):
            garch_data = {
                'start_volatility': garch_ergebnis.get('start_volatility', 0.0),
                'end_volatility': garch_ergebnis.get('end_volatility', 0.0),
                'volatility_change': garch_ergebnis.get('volatility_change', 0.0)
            }
        else:
            garch_data = garch_ergebnis
        
        # Get AI prediction with all forecasts and symbol
        ai = DashboardAI()
        signal = ai.predict(
            current_kpis, 
            current_df, 
            symbol=symbol,
            tage=tage,
            lstm_change=lstm_change,
            arima_change=arima_change,
            garch_volatility_change=garch_data  # Übergebe strukturierte GARCH-Daten
        )
        
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Analyseergebnis", className="mb-0")
            ], className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Empfehlung", className="text-muted mb-2"),
                                html.H2(
                                    signal['decision'], 
                                    className=f"text-{'success' if signal['decision'] == 'Kaufen' else 'danger' if signal['decision'] == 'Verkaufen' else 'warning'}"
                                )
                            ], className="text-center")
                        ], className="h-100")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Konfidenz", className="text-muted mb-2"),
                                html.H2(
                                    f"{signal['confidence']:.0%}",
                                    className="display-4"
                                )
                            ], className="text-center")
                        ], className="h-100")
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Unternehmensprofil", className="mb-3"),
                        html.P(signal['unternehmen'], className="lead")
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Analysebegründung", className="mb-3"),
                        html.P(signal['reasoning'], style={
                            'line-height': '1.6',
                            'font-size': '1.1rem'
                        })
                    ])
                ])
            ], className="px-4 py-4")
        ], className="shadow-sm")

    except Exception as e:
        print(f"Fehler in KI-Analyse: {str(e)}")
        return dbc.Card([
            dbc.CardBody([
                html.H4("Fehler bei der KI-Analyse", className="text-danger mb-3"),
                html.P("Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.", 
                      className="lead")
            ])
        ], className="shadow-sm border-danger")

# Registriere die Callbacks für den Fundamentalanalyse-Bereich
register_fundamental_callbacks(app)

# Startet den Server im Debug-Modus
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    

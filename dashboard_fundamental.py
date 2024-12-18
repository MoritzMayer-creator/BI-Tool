from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from fundamental_analyzer import FundamentalAnalyzer

def erstelle_fundamental_karte(titel, wert, zusatz="", farbe="primary", beschreibung=None):
    """Erstellt eine Karte für fundamentale Kennzahlen mit optionaler Beschreibung"""
    return dbc.Card([
        dbc.CardBody([
            html.H5(titel, className="card-title"),
            html.H3(f"{wert:,.2f}{zusatz}", className="mb-2"),
            html.P(beschreibung, className="text-muted small mb-0") if beschreibung else None
        ])
    ], color=farbe, inverse=True, className="mb-3")

def erstelle_fundamental_bereich():
    """Erstellt den Bereich für die Fundamentalanalyse"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(
                    html.H4("Fundamentalkennzahlen", className="mb-0"), 
                    width=8,
                    className="d-flex align-items-center"
                ),
                dbc.Col(
                    html.H6(
                        id="geschaeftsjahr-info", 
                        className="text-muted text-right mb-0"
                    ), 
                    width=4,
                    className="d-flex align-items-center justify-content-end"
                )
            ], className="align-items-center")
        ]),
        dbc.CardBody([
            # Erste Reihe
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("KGV", className="card-title"),
                            html.H2(id="kgv-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("ROE", className="card-title"),
                            html.H2(id="roe-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Verschuldungsgrad", className="card-title"),
                            html.H2(id="verschuldungsgrad-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Dividendenrendite", className="card-title"),
                            html.H2(id="dividende-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
            ], className="mb-3"),
            # Zweite Reihe
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Free Cashflow", className="card-title"),
                            html.H2(id="cashflow-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Umsatzwachstum", className="card-title"),
                            html.H2(id="umsatzwachstum-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("EBIT-Marge", className="card-title"),
                            html.H2(id="ebit-marge-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Eigenkapitalquote", className="card-title"),
                            html.H2(id="eigenkapitalquote-wert")
                        ])
                    ], color="primary", inverse=True)
                , width=3)
            ])
        ])
    ], className="mt-4")

def register_fundamental_callbacks(app):
    @app.callback(
        [Output("kgv-wert", "children"),
         Output("roe-wert", "children"),
         Output("verschuldungsgrad-wert", "children"),
         Output("dividende-wert", "children"),
         Output("cashflow-wert", "children"),
         Output("umsatzwachstum-wert", "children"),
         Output("ebit-marge-wert", "children"),
         Output("eigenkapitalquote-wert", "children"),
         Output("geschaeftsjahr-info", "children")],
        [Input("analyse-button", "n_clicks")],
        [State("aktien-input", "value")]
    )
    def update_fundamental_analyse(n_clicks, symbol):
        if not n_clicks or not symbol:
            raise PreventUpdate

        try:
            analyzer = FundamentalAnalyzer()
            kennzahlen = analyzer.hole_fundamentaldaten(symbol)
            
            # Hole das Geschäftsjahr aus den Kennzahlen
            geschaeftsjahr = kennzahlen.get('geschaeftsjahr', 'N/A')
            geschaeftsjahr_text = f"Geschäftsjahr: {geschaeftsjahr}"

            def format_value(key, with_percent=False, with_m=False):
                try:
                    value = kennzahlen.get(key)
                    if value is None:
                        return "N/A"
                    if value == 0:
                        return "0.00" + ("%" if with_percent else "M" if with_m else "")
                    if abs(value) < 0.01 and value != 0:
                        # Für sehr kleine Werte wissenschaftliche Notation verwenden
                        if with_percent:
                            return f"{value:.4f}%"
                        elif with_m:
                            return f"{value:.4f}M"
                        else:
                            return f"{value:.4f}"
                    # Normale Formatierung für größere Werte
                    suffix = '%' if with_percent else 'M' if with_m else ''
                    return f"{value:.2f}{suffix}"
                except:
                    return "N/A"

            return (
                format_value('kgv'),
                format_value('roe', True),
                format_value('verschuldungsgrad', True),
                format_value('dividendenrendite', True),
                format_value('free_cashflow', with_m=True),
                format_value('umsatzwachstum', True),
                format_value('ebit_marge', True),
                format_value('eigenkapitalquote', True),
                geschaeftsjahr_text
            )

        except Exception as e:
            print(f"Fehler in der Fundamentalanalyse: {str(e)}")
            return ["N/A"] * 8 + ["Geschäftsjahr: N/A"]
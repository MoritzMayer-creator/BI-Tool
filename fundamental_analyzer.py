import yfinance as yf
from typing import Dict, Any
import pandas as pd
from data_warehouse import DataWarehouse

class FundamentalAnalyzer:
    def __init__(self):
        self.kennzahlen = None
        self.symbol = None
        
    def hole_fundamentaldaten(self, symbol: str) -> dict:
        """Holt die Fundamentaldaten für ein Symbol vom letzten Geschäftsjahr"""
        try:
            warehouse = DataWarehouse()
            aktie = yf.Ticker(symbol)
            financials = aktie.financials
            
            # Geschäftsjahr bestimmen
            if not financials.empty:
                latest_fiscal_year = financials.columns[0]
                fiscal_year_start = latest_fiscal_year.year - 1
                fiscal_year_end = latest_fiscal_year.year
                geschaeftsjahr = f"{fiscal_year_start}/{fiscal_year_end}"
                print(f"Aktuelles Geschäftsjahr: {geschaeftsjahr}")
            else:
                geschaeftsjahr = 'N/A'
                print("Keine Finanzdaten verfügbar für Geschäftsjahr")
                return {}  # Wenn keine Finanzdaten verfügbar sind, brechen wir ab
            
            heute = pd.Timestamp.now()
            cached_data = warehouse.hole_fundamental_daten(symbol)
            
            if cached_data is not None and not cached_data.empty:
                letzte_aktualisierung = pd.Timestamp(cached_data.iloc[0]['aktualisiert_am'])
                
                # Prüfe, ob eine Aktualisierung nötig ist
                aktualisierung_nötig = False
                
                if latest_fiscal_year > letzte_aktualisierung:
                    aktualisierung_nötig = True
                    print(f"Neues Geschäftsjahr verfügbar: {latest_fiscal_year}")
                
                if not aktualisierung_nötig:
                    row = cached_data.iloc[0]
                    return {
                        'kgv': row['kgv'],
                        'roe': row['roe'],
                        'verschuldungsgrad': row['verschuldungsgrad'],
                        'dividendenrendite': row['dividendenrendite'],
                        'free_cashflow': row['free_cashflow'],
                        'umsatzwachstum': row['umsatzwachstum'],
                        'ebit_marge': row['ebit_marge'],
                        'eigenkapitalquote': row['eigenkapitalquote'],
                        'geschaeftsjahr': geschaeftsjahr  # Hier verwenden wir das aktuelle Geschäftsjahr
                    }
            
            print(f"Hole neue Fundamentaldaten für {symbol}")
            # Wenn keine aktuellen Daten vorhanden, hole neue von Yahoo Finance
            info = aktie.info
            financials = aktie.financials
            balance_sheet = aktie.balance_sheet
            cashflow = aktie.cashflow

            if not financials.empty and not balance_sheet.empty:
                latest_fiscal_year = financials.columns[0]  # Letztes Geschäftsjahr
                previous_fiscal_year = financials.columns[1]  # Vorjahr
                latest_bs_year = balance_sheet.columns[0]  # Letztes Bilanzjahr
                
                try:
                    # 1. KGV (Kurs-Gewinn-Verhältnis) berechnen
                    aktueller_kurs = info.get('regularMarketPrice', info.get('currentPrice'))
                    if 'Net Income' in financials.index and aktueller_kurs:
                        net_income = financials.loc['Net Income', latest_fiscal_year]
                        shares = info.get('sharesOutstanding')
                        if shares and shares > 0:
                            eps = net_income / shares
                            kgv = aktueller_kurs / eps if eps > 0 else None
                            print(f"Berechnetes KGV: {kgv:.2f} (Kurs: {aktueller_kurs:.2f}, EPS: {eps:.2f})")
                        else:
                            kgv = info.get('trailingPE')
                    else:
                        kgv = info.get('trailingPE')

                    # 2. ROE (Eigenkapitalrendite) berechnen
                    if 'Net Income' in financials.index:
                        net_income = financials.loc['Net Income', latest_fiscal_year]
                        # Versuche verschiedene mögliche Bezeichnungen für das Eigenkapital
                        if 'Total Stockholder Equity' in balance_sheet.index:
                            equity = balance_sheet.loc['Total Stockholder Equity', latest_bs_year]
                        elif 'Stockholders Equity' in balance_sheet.index:
                            equity = balance_sheet.loc['Stockholders Equity', latest_bs_year]
                        elif 'Total Equity' in balance_sheet.index:
                            equity = balance_sheet.loc['Total Equity', latest_bs_year]
                        else:
                            equity = None

                        if equity and equity != 0:
                            roe = (net_income / equity) * 100
                            print(f"Berechnetes ROE: {roe:.2f}% (Nettogewinn: {net_income:.2f}, Eigenkapital: {equity:.2f})")
                        else:
                            roe = None
                    else:
                        roe = None

                    # 3. Verschuldungsgrad berechnen
                    # Versuche verschiedene mögliche Bezeichnungen für die Verbindlichkeiten
                    if 'Total Liabilities' in balance_sheet.index:
                        liabilities = balance_sheet.loc['Total Liabilities', latest_bs_year]
                    elif 'Total Liabilities Net Minority Interest' in balance_sheet.index:
                        liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_bs_year]
                    else:
                        liabilities = None

                    # Verwende das bereits gefundene Eigenkapital von oben
                    if liabilities is not None and equity is not None and equity != 0:
                        verschuldungsgrad = (liabilities / equity) * 100
                        print(f"Berechneter Verschuldungsgrad: {verschuldungsgrad:.2f}% (Verbindlichkeiten: {liabilities:.2f}, Eigenkapital: {equity:.2f})")
                    else:
                        verschuldungsgrad = None

                    # 4. Dividendenrendite berechnen
                    if aktueller_kurs:
                        # Versuche zuerst die jährliche Dividende aus den Info-Daten zu holen
                        if 'trailingAnnualDividendRate' in info and info['trailingAnnualDividendRate']:
                            dividendenrendite = (info['trailingAnnualDividendRate'] / aktueller_kurs) * 100
                            print(f"Berechnete Dividendenrendite (aus Info): {dividendenrendite:.2f}%")
                        # Alternative: Berechne aus den Cashflow-Daten
                        elif 'Total Dividends Paid' in cashflow.index:
                            jahres_dividenden = abs(cashflow.loc['Total Dividends Paid', cashflow.columns[0]])
                            shares = info.get('sharesOutstanding')
                            if shares:
                                dividende_pro_aktie = jahres_dividenden / shares
                                dividendenrendite = (dividende_pro_aktie / aktueller_kurs) * 100
                                print(f"Berechnete Dividendenrendite (aus Cashflow): {dividendenrendite:.2f}%")
                            else:
                                dividendenrendite = None
                        else:
                            dividendenrendite = None
                    else:
                        dividendenrendite = None

                    # 5. Free Cashflow (Freier Cashflow) berechnen
                    if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                        operating_cf = cashflow.loc['Operating Cash Flow', cashflow.columns[0]]
                        capex = cashflow.loc['Capital Expenditure', cashflow.columns[0]]
                        free_cashflow = (operating_cf + capex) / 1_000_000  # In Millionen
                        print(f"Berechneter Free Cashflow: {free_cashflow:.2f}M")
                    else:
                        free_cashflow = None

                    # 6. Umsatzwachstum berechnen
                    if 'Total Revenue' in financials.index:
                        umsatz_aktuell = financials.loc['Total Revenue', latest_fiscal_year]
                        umsatz_vorjahr = financials.loc['Total Revenue', previous_fiscal_year]
                        umsatzwachstum = ((umsatz_aktuell - umsatz_vorjahr) / umsatz_vorjahr) * 100
                        print(f"Berechnetes Umsatzwachstum: {umsatzwachstum:.2f}%")
                    else:
                        umsatzwachstum = None

                    # 7. EBIT-Marge berechnen
                    if 'EBIT' in financials.index and 'Total Revenue' in financials.index:
                        ebit = financials.loc['EBIT', latest_fiscal_year]
                        umsatz = financials.loc['Total Revenue', latest_fiscal_year]
                        ebit_marge = (ebit / umsatz) * 100 if umsatz != 0 else None
                        print(f"Berechnete EBIT-Marge: {ebit_marge:.2f}% (EBIT: {ebit:.2f}, Umsatz: {umsatz:.2f})")
                    else:
                        ebit_marge = None

                    # 8. Eigenkapitalquote berechnen
                    try:
                        # Versuche verschiedene mögliche Bezeichnungen für das Eigenkapital
                        if 'Total Stockholder Equity' in balance_sheet.index:
                            eigenkapital = balance_sheet.loc['Total Stockholder Equity', latest_bs_year]
                        elif 'Stockholders Equity' in balance_sheet.index:
                            eigenkapital = balance_sheet.loc['Stockholders Equity', latest_bs_year]
                        elif 'Total Equity' in balance_sheet.index:
                            eigenkapital = balance_sheet.loc['Total Equity', latest_bs_year]
                        else:
                            eigenkapital = None

                        # Versuche verschiedene mögliche Bezeichnungen für die Bilanzsumme
                        if 'Total Assets' in balance_sheet.index:
                            bilanzsumme = balance_sheet.loc['Total Assets', latest_bs_year]
                        else:
                            bilanzsumme = None

                        if eigenkapital is not None and bilanzsumme is not None and bilanzsumme != 0:
                            eigenkapitalquote = (eigenkapital / bilanzsumme) * 100
                            print(f"Berechnete Eigenkapitalquote: {eigenkapitalquote:.2f}% (Eigenkapital: {eigenkapital:.2f}, Bilanzsumme: {bilanzsumme:.2f})")
                        else:
                            eigenkapitalquote = None
                    except Exception as e:
                        print(f"Fehler bei der Berechnung der Eigenkapitalquote: {str(e)}")
                        eigenkapitalquote = None

                    kennzahlen = {
                        'kgv': kgv,
                        'roe': roe,
                        'verschuldungsgrad': verschuldungsgrad,
                        'dividendenrendite': dividendenrendite,
                        'free_cashflow': free_cashflow,
                        'umsatzwachstum': umsatzwachstum,
                        'ebit_marge': ebit_marge,
                        'eigenkapitalquote': eigenkapitalquote,
                        'geschaeftsjahr': geschaeftsjahr
                    }

                    # Speichere die neuen Daten im Warehouse
                    warehouse.speichere_fundamental_daten(symbol, kennzahlen)
                    
                    # Hole und speichere auch die Metadaten
                    meta_daten = {
                        'name': aktie.info.get('longName', ''),
                        'sektor': aktie.info.get('sector', ''),
                        'branche': aktie.info.get('industry', ''),
                        'land': aktie.info.get('country', ''),
                        'waehrung': aktie.info.get('currency', 'USD'),
                        'boerse': aktie.info.get('exchange', 'NYSE')
                    }
                    warehouse.speichere_meta_daten(symbol, meta_daten)
                    
                    return kennzahlen

                except Exception as e:
                    print(f"Fehler bei der Berechnung der Kennzahlen: {str(e)}")
                    return {}
            else:
                print("Keine Finanzdaten verfügbar")
                return {}
        
        except Exception as e:
            print(f"Fehler beim Holen der Fundamentaldaten: {str(e)}")
            return {}

    def _hole_neue_fundamentaldaten(self, symbol: str) -> Dict[str, Any]:
        """Die ursprüngliche Implementierung der Fundamentaldaten-Abfrage"""
        self.symbol = symbol
        aktie = yf.Ticker(symbol)
        
        try:
            info = aktie.info
            balance_sheet = aktie.balance_sheet
            income_stmt = aktie.financials
            cashflow = aktie.cashflow
            
            latest_fiscal_year = balance_sheet.columns[0]
            print(f"\nAnalyse für {symbol} - Geschäftsjahr: {latest_fiscal_year.strftime('%Y-%m-%d')}")
            
            # 1. Berechne den Verschuldungsgrad mit Gesamtverbindlichkeiten
            if not balance_sheet.empty:
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_fiscal_year]
                total_equity = balance_sheet.loc['Stockholders Equity', latest_fiscal_year]
                verschuldungsgrad = (total_liabilities / total_equity) * 100 if total_equity else 0
                
                print("\nVerschuldungsanalyse:")
                print(f"Gesamtverbindlichkeiten: ${total_liabilities:,.2f}")
                print(f"Eigenkapital: ${total_equity:,.2f}")
                print(f"Verschuldungsgrad: {verschuldungsgrad:.2f}%")
            else:
                verschuldungsgrad = 0
                
            # 2. Berechne das KGV (Kurs-Gewinn-Verhältnis)
            kgv = info.get('trailingPE', 0)
            
            # 3. Berechne die Eigenkapitalrendite (ROE)
            if not income_stmt.empty and total_equity:
                net_income = income_stmt.loc['Net Income', latest_fiscal_year]
                roe = (net_income / total_equity) * 100
                print(f"\nROE-Analyse:")
                print(f"Nettogewinn: ${net_income:,.2f}")
                print(f"ROE: {roe:.2f}%")
            else:
                roe = 0
            
            # 4. Berechne die Dividendenrendite
            div_yield = info.get('dividendYield', 0)
            if div_yield is not None:
                div_yield = div_yield * 100
            
            # 5. Berechne den Free Cashflow
            if not cashflow.empty:
                operating_cf = cashflow.loc['Operating Cash Flow', latest_fiscal_year]
                capex = cashflow.loc['Capital Expenditure', latest_fiscal_year]
                free_cashflow = (operating_cf + capex) / 1_000_000  # In Millionen
                
                print(f"\nCashflow-Analyse:")
                print(f"Operativer Cashflow: ${operating_cf:,.2f}")
                print(f"Investitionsausgaben: ${capex:,.2f}")
                print(f"Free Cashflow: ${free_cashflow:,.2f}M")
            else:
                free_cashflow = 0
            
            self.kennzahlen = {
                'kgv': kgv,
                'roe': roe,
                'verschuldungsgrad': verschuldungsgrad,
                'dividendenrendite': div_yield,
                'free_cashflow': free_cashflow
            }
            
            print("\nZusammenfassung der Kennzahlen:")
            print(f"KGV: {kgv:.2f}")
            print(f"ROE: {roe:.2f}%")
            print(f"Verschuldungsgrad: {verschuldungsgrad:.2f}%")
            print(f"Dividendenrendite: {div_yield:.2f}%")
            print(f"Free Cashflow: ${free_cashflow:.2f}M")
            
            return self.kennzahlen
            
        except Exception as e:
            print(f"Fehler beim Laden der Fundamentaldaten für {symbol}: {str(e)}")
            return {
                'kgv': 0,
                'roe': 0,
                'verschuldungsgrad': 0,
                'dividendenrendite': 0,
                'free_cashflow': 0
            }

if __name__ == "__main__":
    # Testausführung
    analyzer = FundamentalAnalyzer()
    data = analyzer.hole_fundamentaldaten('AAPL')
import sqlite3
import pandas as pd
from datetime import datetime
import threading
from functools import lru_cache
from typing import Dict, Any, Optional
import logging

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataWarehouse:
    def __init__(self, db_path="aktien_warehouse.db"):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()
        self.setup_database()
        
    def __enter__(self):
        """Context Manager für sicheres Verbindungshandling"""
        self._get_connection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatisches Schließen der Verbindung"""
        self._close_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Thread-sichere Verbindungsverwaltung"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            # Aktiviere Write-Ahead Logging für bessere Konkurrenz
            self._connection.execute('PRAGMA journal_mode=WAL')
            # Aktiviere Foreign Keys
            self._connection.execute('PRAGMA foreign_keys=ON')
        return self._connection

    def _close_connection(self):
        """Schließt die Datenbankverbindung"""
        if self._connection:
            self._connection.close()
            self._connection = None
        
    def setup_database(self):
        """Erweiterte Datenbankinitialisierung mit Indizes"""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fundamental Daten mit korrigiertem Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    symbol TEXT,
                    datum DATE,
                    kgv REAL,
                    roe REAL,
                    verschuldungsgrad REAL,
                    dividendenrendite REAL,
                    free_cashflow REAL,
                    umsatzwachstum REAL,
                    ebit_marge REAL,
                    eigenkapitalquote REAL,
                    geschaeftsjahr TEXT,
                    aktualisiert_am TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, datum)
                )
            """)
            
            # Prüfen ob die Spalte geschaeftsjahr existiert, wenn nicht hinzufügen
            cursor.execute("PRAGMA table_info(fundamental_data)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'geschaeftsjahr' not in columns:
                cursor.execute("""
                    ALTER TABLE fundamental_data 
                    ADD COLUMN geschaeftsjahr TEXT
                """)
            
            # Aktien Metadaten mit erweitertem Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aktien_meta (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sektor TEXT,
                    branche TEXT,
                    land TEXT,
                    waehrung TEXT,
                    boerse TEXT,
                    letzte_aktualisierung DATE,
                    aktualisiert_am TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Technische Indikatoren Tabelle
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technische_indikatoren (
                    symbol TEXT,
                    datum DATE,
                    rsi REAL,
                    macd REAL,
                    signal REAL,
                    volumen REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    aktualisiert_am TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, datum),
                    FOREIGN KEY (symbol) REFERENCES aktien_meta(symbol)
                )
            """)
            
            # Erstelle Indizes für häufig genutzte Abfragen
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_datum 
                ON fundamental_data(symbol, datum)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_technische_symbol_datum 
                ON technische_indikatoren(symbol, datum)
            """)
            
            conn.commit()
            
    @lru_cache(maxsize=100)
    def hole_fundamental_daten(self, symbol: str, tage: int = 365) -> Optional[pd.DataFrame]:
        """Zwischengespeicherter Abruf von Fundamentaldaten"""
        try:
            with self._lock, self._get_connection() as conn:
                query = f"""
                    SELECT * FROM fundamental_data 
                    WHERE symbol = ? 
                    AND datum >= date('now', '-{tage} days')
                    ORDER BY datum DESC
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                return df if not df.empty else None
                
        except Exception as e:
            logger.error(f"Fehler beim Abruf der Fundamentaldaten für {symbol}: {e}")
            return None

    def speichere_fundamental_daten(self, symbol: str, kennzahlen: Dict[str, Any]) -> bool:
        """Thread-sichere Speicherung von Fundamentaldaten"""
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                datum = datetime.now().date()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO fundamental_data 
                    (symbol, datum, kgv, roe, verschuldungsgrad, dividendenrendite, free_cashflow, 
                     umsatzwachstum, ebit_marge, eigenkapitalquote, geschaeftsjahr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, datum,
                    kennzahlen.get('kgv', 0),
                    kennzahlen.get('roe', 0),
                    kennzahlen.get('verschuldungsgrad', 0),
                    kennzahlen.get('dividendenrendite', 0),
                    kennzahlen.get('free_cashflow', 0),
                    kennzahlen.get('umsatzwachstum', 0),
                    kennzahlen.get('ebit_marge', 0),
                    kennzahlen.get('eigenkapitalquote', 0),
                    kennzahlen.get('geschaeftsjahr', 'N/A')
                ))
                
                conn.commit()
                # Lösche den Cache für dieses Symbol
                self.hole_fundamental_daten.cache_clear()
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Fundamentaldaten für {symbol}: {e}")
            return False

    def speichere_meta_daten(self, symbol: str, meta_daten: Dict[str, Any]) -> bool:
        """Erweiterte Speicherung von Metadaten"""
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO aktien_meta 
                    (symbol, name, sektor, branche, land, waehrung, boerse, letzte_aktualisierung)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    meta_daten.get('name', ''),
                    meta_daten.get('sektor', ''),
                    meta_daten.get('branche', ''),
                    meta_daten.get('land', ''),
                    meta_daten.get('waehrung', 'USD'),
                    meta_daten.get('boerse', 'NYSE'),
                    datetime.now().date()
                ))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Metadaten für {symbol}: {e}")
            return False

    def speichere_technische_daten(self, symbol: str, technische_daten: Dict[str, Any]) -> bool:
        """Speichert technische Indikatoren"""
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                datum = datetime.now().date()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO technische_indikatoren
                    (symbol, datum, rsi, macd, signal, volumen, sma_20, sma_50)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, datum,
                    technische_daten.get('rsi', 0),
                    technische_daten.get('macd', 0),
                    technische_daten.get('signal', 0),
                    technische_daten.get('volumen', 0),
                    technische_daten.get('sma_20', 0),
                    technische_daten.get('sma_50', 0)
                ))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der technischen Daten für {symbol}: {e}")
            return False 
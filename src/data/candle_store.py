"""SQLite-backed persistent store for normalized candle data.

Schema note: the unique constraint on (symbol, timeframe, timestamp, provider)
is the deduplication key. INSERT OR IGNORE silently skips duplicates.

Migration path: swap this class for a PostgresCandleStore that honours the
same public interface — no callers need to change.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.data.models import Candle

logger = logging.getLogger(__name__)

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    resolved_symbol TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    open            REAL    NOT NULL,
    high            REAL    NOT NULL,
    low             REAL    NOT NULL,
    close           REAL    NOT NULL,
    volume          REAL    NOT NULL,
    provider        TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
    UNIQUE (symbol, timeframe, timestamp, provider)
);
"""

_DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (symbol, timeframe, timestamp, provider);
"""


class CandleStore:
    """Persists and queries candles in a local SQLite database.

    Designed for single-writer use. For concurrent or distributed use,
    replace with PostgresCandleStore behind the same interface.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(_DDL_TABLE)
                conn.execute(_DDL_INDEX)
            logger.info("database_init: db=%s", self._db_path)
        except sqlite3.Error as exc:
            logger.error(
                "database_error: init failed | %s "
                "| Hint: Check file permissions for the data/ directory.",
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_candles(self, candles: List[Candle]) -> Tuple[int, int]:
        """Insert candles, ignoring duplicates.

        Returns (saved_count, duplicate_count).
        """
        if not candles:
            return 0, 0

        saved = 0
        dupes = 0

        try:
            with self._connect() as conn:
                for c in candles:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO candles
                            (symbol, resolved_symbol, timeframe, timestamp,
                             open, high, low, close, volume, provider)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            c.symbol,
                            c.resolved_symbol or c.symbol,
                            c.timeframe,
                            c.timestamp.isoformat(),
                            c.open,
                            c.high,
                            c.low,
                            c.close,
                            c.volume,
                            c.provider,
                        ),
                    )
                    if cur.rowcount == 1:
                        saved += 1
                    else:
                        dupes += 1

            logger.info(
                "candles_saved: saved=%d duplicates_skipped=%d "
                "symbol=%s timeframe=%s provider=%s",
                saved,
                dupes,
                candles[0].symbol,
                candles[0].timeframe,
                candles[0].provider,
            )
        except sqlite3.Error as exc:
            logger.error(
                "database_error: save_candles failed | %s "
                "| Hint: Check disk space and database file permissions.",
                exc,
            )

        return saved, dupes

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Candle]:
        """Load candles ordered by timestamp, optionally bounded by date range."""
        query = "SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp ASC"

        try:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("database_error: load_candles | %s", exc)
            return []

        return [self._row_to_candle(r) for r in rows]

    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
        provider: str,
    ) -> Optional[datetime]:
        """Return the most recent stored candle timestamp for the given key."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(timestamp) AS ts FROM candles
                    WHERE symbol = ? AND timeframe = ? AND provider = ?
                    """,
                    (symbol, timeframe, provider),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.error("database_error: get_latest_timestamp | %s", exc)
            return None

        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"])
        return None

    def count_candles(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """Return stored candle count, optionally filtered by symbol/timeframe."""
        conditions: List[str] = []
        params: list = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) FROM candles{where}"

        try:
            with self._connect() as conn:
                return conn.execute(query, params).fetchone()[0]
        except sqlite3.Error as exc:
            logger.error("database_error: count_candles | %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_candle(row: sqlite3.Row) -> Candle:
        return Candle(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            provider=row["provider"],
            resolved_symbol=row["resolved_symbol"],
        )

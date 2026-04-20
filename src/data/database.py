from typing import Optional
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parents[2] / "data" / "prospects.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            player_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            dob         TEXT,
            nationality TEXT,
            position    TEXT,
            height_cm   REAL,
            weight_kg   REAL,
            shoots      TEXT,
            source      TEXT,
            draft_year  INTEGER,
            draft_round INTEGER,
            draft_pick  INTEGER,
            draft_team  TEXT
        );

        CREATE TABLE IF NOT EXISTS seasons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id   TEXT NOT NULL,
            season      TEXT NOT NULL,
            league      TEXT NOT NULL,
            team        TEXT,
            gp          INTEGER,
            goals       INTEGER,
            assists     INTEGER,
            points      INTEGER,
            pim         INTEGER,
            plus_minus  INTEGER,
            pp_goals    INTEGER,
            pp_assists  INTEGER,
            gp_rate     REAL,
            ppg         REAL,
            age         INTEGER,
            nhle_ppg    REAL,
            nhle_gpg    REAL,
            draft_year  INTEGER,
            draft_label TEXT,
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            UNIQUE(player_id, season, league)
        );

        CREATE TABLE IF NOT EXISTS nhl_outcomes (
            player_id       TEXT PRIMARY KEY,
            draft_year      INTEGER,
            draft_round     INTEGER,
            draft_pick      INTEGER,
            draft_team      TEXT,
            nhl_gp          INTEGER DEFAULT 0,
            nhl_goals       INTEGER DEFAULT 0,
            nhl_assists     INTEGER DEFAULT 0,
            nhl_points      INTEGER DEFAULT 0,
            is_nhler        INTEGER DEFAULT 0,
            is_star         INTEGER DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            player_id           TEXT PRIMARY KEY,
            nhle_ppg            REAL,
            nhler_probability   REAL,
            star_probability    REAL,
            projected_career_pts REAL,
            rank_score          REAL,
            rank_position       INTEGER,
            top_comparable      TEXT,
            updated_at          TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );
        """)

        # Migration: add missing columns to players / seasons for pre-existing
        # DBs. SQLite has no "ADD COLUMN IF NOT EXISTS" — PRAGMA the schema.
        def _ensure_cols(table: str, wanted: list[tuple[str, str]]):
            have = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col, col_type in wanted:
                if col not in have:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

        _ensure_cols("players", [
            ("draft_year",  "INTEGER"),
            ("draft_round", "INTEGER"),
            ("draft_pick",  "INTEGER"),
            ("draft_team",  "TEXT"),
        ])
        _ensure_cols("seasons", [
            ("age",         "INTEGER"),
            ("nhle_ppg",    "REAL"),
            ("nhle_gpg",    "REAL"),
            ("draft_year",  "INTEGER"),
            ("draft_label", "TEXT"),
        ])


def upsert_players(df: pd.DataFrame):
    with get_conn() as conn:
        df.to_sql("players", conn, if_exists="append", index=False,
                  method=_upsert_method("player_id"))


def upsert_seasons(df: pd.DataFrame):
    with get_conn() as conn:
        df.to_sql("seasons", conn, if_exists="append", index=False,
                  method=_upsert_method(None))


def upsert_outcomes(df: pd.DataFrame):
    with get_conn() as conn:
        df.to_sql("nhl_outcomes", conn, if_exists="append", index=False,
                  method=_upsert_method("player_id"))


def upsert_predictions(df: pd.DataFrame):
    with get_conn() as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False,
                  method=_upsert_method("player_id"))


def load_players() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM players", conn)


def load_seasons() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM seasons", conn)


def load_outcomes() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM nhl_outcomes", conn)


def load_predictions() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("""
            SELECT p.*, pl.name, pl.position, pl.height_cm, pl.weight_kg
            FROM predictions p
            JOIN players pl ON p.player_id = pl.player_id
            ORDER BY rank_position
        """, conn)


def _upsert_method(pk: Optional[str]):
    def method(table, conn, keys, data_iter):
        # pandas 2.x passes a sqlite3.Cursor directly; older versions passed a
        # sqlite3.Connection. Support both.
        executor = conn.cursor() if hasattr(conn, "cursor") else conn

        # Filter to columns that actually exist on the target table — lets
        # upstream DataFrames carry extra columns without crashing the insert.
        executor.execute(f"PRAGMA table_info({table.name})")
        table_cols = {row[1] for row in executor.fetchall()}
        valid_indices = [i for i, k in enumerate(keys) if k in table_cols]
        valid_keys = [keys[i] for i in valid_indices]

        if not valid_keys:
            return  # nothing in common with the table — nothing to insert

        placeholders = ", ".join(["?"] * len(valid_keys))
        cols = ", ".join(valid_keys)
        sql = f"INSERT OR REPLACE INTO {table.name} ({cols}) VALUES ({placeholders})"
        filtered = ([row[i] for i in valid_indices] for row in data_iter)
        executor.executemany(sql, filtered)
    return method

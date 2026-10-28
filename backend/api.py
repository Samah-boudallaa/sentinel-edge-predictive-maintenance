import sqlite3
from contextlib import contextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

DB = "sentinel.db"
app = FastAPI(title="SentinelEdge API")


class Measure(BaseModel):
    ts_ms: int
    mode: str
    std_g: float
    p2p_g: float
    ratio: float
    freq_hz: float
    level: str


@contextmanager
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS measures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        received_at TEXT DEFAULT CURRENT_TIMESTAMP,
        ts_ms INTEGER, mode TEXT, std_g REAL, p2p_g REAL,
        ratio REAL, freq_hz REAL, level TEXT)""")

INSERT = ("INSERT INTO measures (ts_ms, mode, std_g, p2p_g, ratio, freq_hz, level) "
          "VALUES (?,?,?,?,?,?,?)")


def row(m: Measure):
    return (m.ts_ms, m.mode, m.std_g, m.p2p_g, m.ratio, m.freq_hz, m.level)


@app.post("/measure")
def add_measure(m: Measure):
    with db() as c:
        c.execute(INSERT, row(m))
    return {"status": "ok"}


@app.post("/measures/batch")
def add_batch(ms: List[Measure]):
    with db() as c:
        c.executemany(INSERT, [row(m) for m in ms])
    return {"status": "ok", "count": len(ms)}


@app.get("/measures")
def get_measures(limit: int = 2000):
    with db() as c:
        rows = c.execute("SELECT * FROM measures ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]


@app.get("/summary")
def summary():
    with db() as c:
        rows = c.execute("""SELECT mode, COUNT(*) AS n, ROUND(AVG(std_g),4) AS avg_std,
                            ROUND(AVG(freq_hz),1) AS avg_freq, ROUND(MAX(ratio),2) AS max_ratio
                            FROM measures GROUP BY mode""").fetchall()
    return [dict(r) for r in rows]


@app.delete("/measures")
def clear():
    with db() as c:
        c.execute("DELETE FROM measures")
    return {"status": "cleared"}

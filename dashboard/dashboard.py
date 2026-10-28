import html
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="SentinelEdge", layout="wide", initial_sidebar_state="expanded")

COLS = ["ms", "mode", "std_g", "p2p_g", "ratio", "freq_hz", "niveau"]
RATIO_ZERO = 8.0
RANK = {"NORMAL": 0, "ATTENTION": 1, "CRITIQUE": 2}
API_DEFAULT = os.environ.get("SENTINEL_API_URL", "http://localhost:8000")
STALE_S = 10.0
PAGES = ["Overview", "Live Telemetry", "AI Diagnostics", "Event History", "API & Database", "Settings"]
SRC_DEMO, SRC_CSV, SRC_API = "Démonstration", "Fichier CSV", "API (SQLite)"
SIM_MODES = ["Normal", "Instable", "Panne", "Dérive progressive"]
RANGES = {"Tout": None, "1 h": 3600, "6 h": 21600, "24 h": 86400, "7 j": 604800}
LEVEL_LABEL = {"NORMAL": "NORMAL", "ATTENTION": "ATTENTION", "CRITIQUE": "CRITIQUE"}


# Thèmes

THEMES = {
    "Midnight": dict(bg="#050816", glow1="#1a1148", glow2="#0a2a44", side="#080D1B", panel="rgba(11,18,36,.82)",
                     panel2="#0B1224", line="#192747", txt="#E8EEFF", muted="#8998B8", brand="#8B5CF6",
                     tele="#22D3EE", good="#34D399", warn="#FBBF24", bad="#FF4567", grid="rgba(137,152,184,.09)",
                     tpl="plotly_dark", glowpct="55%", shadow="0 14px 40px -20px rgba(0,0,0,.9)"),
    "Obsidian Gold": dict(bg="#060508", glow1="#2b220d", glow2="#1b150a", side="#0a0806", panel="rgba(27,22,13,.78)",
                          panel2="#17120a", line="#3a2f14", txt="#F4EAD3", muted="#B3A27A", brand="#D4AF37",
                          tele="#F1D27A", good="#3DDC97", warn="#F5A524", bad="#FF5D6C",
                          grid="rgba(212,175,55,.08)", tpl="plotly_dark", glowpct="50%",
                          shadow="0 14px 40px -20px rgba(0,0,0,.95)"),
    "Light": dict(bg="#EEF2F7", glow1="#DCE6F7", glow2="#EFE6FA", side="#E3E9F2", panel="rgba(255,255,255,.9)",
                  panel2="#FFFFFF", line="#C9D4E5", txt="#0B1F3A", muted="#566B87", brand="#6D4FD1",
                  tele="#0369A1", good="#059669", warn="#B45309", bad="#E11D48", grid="rgba(15,40,80,.08)",
                  tpl="plotly_white", glowpct="0%", shadow="0 12px 30px -18px rgba(15,40,80,.35)"),
}

ss = st.session_state
ss.setdefault("pos", 12)
ss.setdefault("play", True)
ss.setdefault("last_level", None)
ss.setdefault("sim", "Dérive progressive")
ss.setdefault("acked", set())
ss.setdefault("api_ok", None)
ss.setdefault("api_last_ms", None)
ss.setdefault("api_last_change", time.time())

# ----------------------------------------------------------------------------
# Barre latérale : marque, navigation, thème
# ----------------------------------------------------------------------------
st.sidebar.markdown("<div class='brand-s'>SENTINELEDGE</div><div class='brand-sub'>Predictive Maintenance</div>",
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

names = list(THEMES)
qp_theme = st.query_params.get("theme")
theme = st.sidebar.radio("Thème", names, index=names.index(qp_theme) if qp_theme in names else 0,
                         horizontal=True)
if qp_theme != theme:
    st.query_params["theme"] = theme
T = THEMES[theme]
LV = {"NORMAL": T["good"], "ATTENTION": T["warn"], "CRITIQUE": T["bad"]}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,.stApp{font-family:'Inter',sans-serif}
.stApp{background:radial-gradient(1100px 560px at 8% -10%,var(--glow1) 0%,transparent 60%),
 radial-gradient(900px 480px at 100% 0%,var(--glow2) 0%,transparent 55%),
 linear-gradient(var(--grid) 1px,transparent 1px) 0 0/48px 48px,
 linear-gradient(90deg,var(--grid) 1px,transparent 1px) 0 0/48px 48px,var(--bg)}
header[data-testid="stHeader"]{background:transparent}
.stApp header *{color:var(--muted)!important}
.stApp header svg{fill:var(--muted)!important}
.block-container{padding-top:3rem;max-width:1720px}
section[data-testid="stSidebar"]{background:var(--side)!important;border-right:1px solid var(--line)}
.stApp :is(p,label,span,li,div,small,summary,h1,h2,h3,h4){color:var(--txt)!important}
.stApp [data-testid="stCaptionContainer"] *{color:var(--muted)!important}
.stApp input{color:var(--txt)!important;background:transparent!important}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"]>div,.stApp [data-testid="stFileUploaderDropzone"]{
 background:var(--panel2)!important;border-color:var(--line)!important;border-radius:8px!important}
.stApp [data-testid="stExpander"]{background:var(--panel);border:1px solid var(--line);border-radius:12px}
.stApp [data-testid="stSlider"] [role="slider"]{background:var(--brand)!important;box-shadow:none}
.stApp button{border:1px solid var(--line)!important;background:var(--panel2)!important;border-radius:8px!important;
 font-weight:600;letter-spacing:.02em;transition:background .2s,border-color .2s}
.stApp button:hover{border-color:var(--brand)!important}
.stApp button[data-testid="stBaseButton-primary"]{background:var(--brand)!important;border-color:var(--brand)!important}
.stApp button[data-testid="stBaseButton-primary"] *{color:#fff!important}
.stApp iframe{border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.stApp [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}

.stApp .brand{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.6rem;letter-spacing:.3em;
 background:linear-gradient(90deg,var(--brand),var(--txt) 60%,var(--tele));-webkit-background-clip:text;
 background-clip:text;-webkit-text-fill-color:transparent;color:transparent!important}
.stApp .brand-s{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;letter-spacing:.26em;
 color:var(--brand)!important;margin-top:4px}
.stApp .brand-sub{color:var(--muted)!important;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px}
.stApp .tagline{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-top:22px;
 background:linear-gradient(135deg,var(--panel2),transparent);color:var(--muted)!important;font-size:.85rem;font-style:italic}

.stApp .topbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;
 padding:12px 18px;margin-bottom:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel);
 box-shadow:var(--shadow);backdrop-filter:blur(10px)}
.stApp .chips{display:flex;flex-wrap:wrap;gap:8px}
.stApp .chip{display:inline-flex;align-items:center;gap:8px;padding:4px 11px;border:1px solid var(--line);
 border-radius:999px;font-size:.8rem;color:var(--muted)!important;background:var(--panel2)}
.stApp .chip b{color:var(--txt)!important;font-weight:600}
.chip i{width:8px;height:8px;border-radius:50%;background:var(--c);box-shadow:0 0 8px var(--c)}
.stApp .clock{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.95rem;color:var(--muted)!important}
.stApp .pagetitle{display:flex;align-items:baseline;gap:14px;margin:4px 0 12px}
.stApp .pagetitle .brand{font-size:1.3rem}
.stApp .sub{color:var(--muted)!important;font-size:.9rem}
.stApp .live{color:var(--bad)!important;animation:blink 1.2s infinite;font-weight:700;font-size:.8rem;letter-spacing:.1em}
.stApp .paused{color:var(--warn)!important;font-weight:700;font-size:.8rem;letter-spacing:.1em}
.stApp .simtag{color:var(--brand)!important;font-weight:700;font-size:.8rem;letter-spacing:.1em}
@keyframes blink{50%{opacity:.3}}

.stApp .banner{position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:center;gap:20px;
 padding:16px 24px;margin-bottom:12px;border:1px solid var(--c);border-left:5px solid var(--c);border-radius:12px;
 background:var(--panel);box-shadow:0 0 28px -12px var(--c);backdrop-filter:blur(8px)}
.banner.crit{animation:alarm 1.4s ease-in-out infinite}
@keyframes alarm{50%{box-shadow:0 0 34px -4px var(--c)}}
.stApp .banner .b1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;letter-spacing:.1em;
 color:var(--c)!important;display:flex;align-items:center}
.stApp .banner .b2{color:var(--muted)!important;font-size:.95rem;margin-top:5px}
.stApp .banner .b3{color:var(--muted)!important;font-size:.8rem;margin-top:3px}
.dot{width:11px;height:11px;border-radius:50%;background:var(--c);margin-right:14px;
 box-shadow:0 0 10px color-mix(in srgb,var(--c) var(--glowpct),transparent);animation:blink 1.6s infinite}
.wave{width:320px;height:54px;flex:none}
.wave polyline{fill:none;stroke:var(--c);stroke-width:1.6;stroke-dasharray:520 80;animation:dash 6s linear infinite;
 filter:drop-shadow(0 0 3px var(--c))}
@keyframes dash{to{stroke-dashoffset:-600}}

.kpi{position:relative;padding:14px 18px 12px;min-height:150px;border:1px solid var(--line);border-radius:12px;
 background:var(--panel);box-shadow:var(--shadow);backdrop-filter:blur(8px);transition:border-color .25s,transform .25s}
.kpi:hover{border-color:var(--c);transform:translateY(-2px)}
.kpi::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:3px;border-radius:3px;background:var(--c)}
.stApp .kl{color:var(--muted)!important;font-size:.82rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.stApp .kv{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2rem;color:var(--c)!important;margin:4px 0 2px;
 white-space:nowrap;text-shadow:0 0 16px color-mix(in srgb,var(--c) var(--glowpct),transparent)}
.stApp .ks{color:var(--muted)!important;font-size:.85rem}
.stApp .tr{color:var(--c)!important;font-size:.82rem;font-weight:600}
.arc{width:100%;max-width:190px;display:block;margin:0 auto}
.arc-bg,.arc-fg{fill:none;stroke-width:8;stroke-linecap:round}
.arc-bg{stroke:var(--line)}
.arc-fg{transition:stroke-dasharray .6s}
.arc-t{font:700 26px 'Space Grotesk',sans-serif}

div[data-testid="stPlotlyChart"],.panel{position:relative;border:1px solid var(--line);border-radius:12px;
 background:var(--panel);box-shadow:var(--shadow);backdrop-filter:blur(8px);padding:4px}
.panel{padding:12px 16px}
.stApp .ptitle{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.95rem;letter-spacing:.06em;margin-bottom:6px}
.stApp .cap{color:var(--muted)!important;font-size:.85rem}
.stApp .ov{display:inline-block;margin:6px 8px 0 0;padding:6px 12px;border:1px solid var(--line);border-radius:8px;
 background:var(--panel2);font-size:.8rem;color:var(--muted)!important}
.stApp .ov b{color:var(--c)!important;font-size:1rem;margin-left:6px}

.stApp .sec{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.9rem;letter-spacing:.2em;color:var(--brand)!important;
 margin:22px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--line)}
.stApp .alert{display:flex;align-items:center;gap:12px;padding:9px 14px;margin-bottom:6px;border:1px solid var(--line);
 border-left:3px solid var(--c);border-radius:8px;background:var(--panel2);font-size:.88rem}
.stApp .alert .a-t{color:var(--muted)!important;min-width:70px;font-variant-numeric:tabular-nums}
.stApp .alert .a-d{flex:1}
.stApp .alert .a-m{color:var(--muted)!important;font-size:.8rem}
.stApp .badge{padding:2px 10px;border-radius:999px;border:1px solid var(--c);color:var(--c)!important;font-size:.72rem;font-weight:700;
 letter-spacing:.06em;white-space:nowrap}
.stApp .foot{color:var(--muted)!important;font-size:.82rem;text-align:center;margin-top:26px}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
"""
root = ":root{" + ";".join(f"--{k}:{T[k]}" for k in
                           ["bg", "side", "panel", "panel2", "line", "txt", "muted", "brand", "tele", "good", "warn",
                            "bad", "grid", "glow1", "glow2", "glowpct", "shadow"]) + "}"
st.markdown("<style>" + root + CSS + "</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Données : lecture, validation, démonstration, API
# ----------------------------------------------------------------------------
def clean(df):
    """Valide le schéma ; tolère l'ancien format sans freq_hz et ignore les lignes invalides."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLS)
    d = df.copy()
    for c in COLS:
        if c not in d:
            d[c] = 0.0 if c == "freq_hz" else np.nan
    d = d[COLS]
    for c in ["ms", "std_g", "p2p_g", "ratio", "freq_hz"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["mode"] = d["mode"].astype(str)
    d["niveau"] = d["niveau"].astype(str).str.upper().str.strip()
    d = d.dropna(subset=["ms", "std_g", "p2p_g", "ratio"])
    d["freq_hz"] = d["freq_hz"].fillna(0.0)
    d = d[d["niveau"].isin(RANK)]
    d["ms"] = d["ms"].astype("int64")
    return d.drop_duplicates("ms").reset_index(drop=True)


def parse_serial(text):
    rows = []
    for line in text.splitlines():
        p = line.strip().split(",")
        if not p or not p[0].isdigit():
            continue
        try:
            if len(p) == 7:
                rows.append([int(p[0]), p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5]), p[6]])
            elif len(p) == 6:  # ancien format sans fréquence
                rows.append([int(p[0]), p[1], float(p[2]), float(p[3]), float(p[4]), 0.0, p[5]])
        except ValueError:
            continue
    return clean(pd.DataFrame(rows, columns=COLS))


@st.cache_data
def make_demo(scenario):
    """Données SIMULÉES (jamais mélangées aux mesures réelles)."""
    rng = np.random.default_rng(7)
    rows, t = [], 4000
    for i in range(190):
        if scenario == "Normal":
            mode, amp, freq = "NORMALE", 0.02, 25 + rng.normal(0, 1.0)
        elif scenario == "Instable":
            if i < 40:
                mode, amp, freq = "NORMALE", 0.02, 25 + rng.normal(0, 1.0)
            else:
                mode, amp, freq = "INSTABLE", 0.02 + 0.08 * rng.uniform(), 32 + rng.normal(0, 6)
        elif scenario == "Panne":
            if i < 50:
                mode, amp, freq = "NORMALE", 0.02, 25 + rng.normal(0, 1.0)
            else:
                mode, amp, freq = "PANNE", 0.38 * rng.normal(1, 0.1), 78 + rng.normal(0, 2)
        else:  # Dérive progressive
            if i < 40:
                mode, p = "NORMALE", 0.0
            else:
                mode, p = "DERIVE", (i - 40) / 150
            amp, freq = 0.02 + 0.33 * p * p, 25 + 53 * p + rng.normal(0, 1.2)
        std = 0.645 * amp * rng.normal(1, 0.08)
        ratio = std / 0.0128
        lvl = "CRITIQUE" if ratio >= 6 else ("ATTENTION" if ratio >= 2.5 else "NORMAL")
        rows.append([t, mode, round(std, 4), round(std * 3.8, 4), round(ratio, 2), round(max(freq, 0), 1), lvl])
        t += 527
    return pd.DataFrame(rows, columns=COLS)


def fetch_api(url):
    try:
        resp = requests.get(f"{url}/measures", params={"limit": 5000}, timeout=3)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()).rename(columns={"ts_ms": "ms", "level": "niveau"})
        df = clean(df)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def analyse(raw, contam):
    d = raw.sort_values("ms").reset_index(drop=True).copy()
    d["t_s"] = (d["ms"] - d["ms"].min()) / 1000
    has_freq = bool(d["freq_hz"].abs().sum() > 0)
    feats = ["std_g", "p2p_g"] + (["freq_hz"] if has_freq else [])
    train = d[d["mode"] == "NORMALE"] if (d["mode"] == "NORMALE").sum() >= 10 else d
    model = IsolationForest(n_estimators=200, contamination=contam, random_state=42).fit(train[feats])
    d["score"] = model.decision_function(d[feats])
    d["ml_anomalie"] = model.predict(d[feats]) == -1
    d["seuil_anomalie"] = d["niveau"] != "NORMAL"
    h_ratio = (1 - (d["ratio"] - 1) / (RATIO_ZERO - 1)).clip(0, 1) * 100
    h_ml = ((d["score"] + 0.15) / 0.30).clip(0, 1) * 100
    d["health"] = (0.7 * h_ratio + 0.3 * h_ml).ewm(span=8).mean()
    base = float((d["std_g"] / d["ratio"].replace(0, np.nan)).median())
    return d, has_freq, base, feats, len(train)


def estimate_rul(d, thr, win):
    """Régression linéaire de la santé ; intervalle approximatif à 95 % sur la pente."""
    t_last = float(d["t_s"].max())
    r = d[d["t_s"] >= t_last - win]
    if len(r) < 8:
        return None
    try:
        (slope, icpt), cov = np.polyfit(r["t_s"], r["health"], 1, cov=True)
        se = float(np.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        slope, icpt = np.polyfit(r["t_s"], r["health"], 1)
        se = 0.0
    now = float(slope * t_last + icpt)

    def rul_for(s):
        if now <= thr:
            return 0.0
        return np.inf if s > -0.05 else (now - thr) / (-s)

    return {"rul": rul_for(slope), "lo": rul_for(slope - 1.96 * se), "hi": rul_for(slope + 1.96 * se),
            "slope": float(slope), "now": now, "t_last": t_last}


def first_sustained(mask, t, k=3):
    """Première détection valide : k mesures consécutives signalées (évite les fausses alertes isolées)."""
    arr = (mask.astype(int).rolling(k).sum() == k).to_numpy()
    if not arr.any():
        return np.nan
    return float(t.iloc[int(arr.argmax()) - (k - 1)])


def build_events(v, acked):
    """Alertes issues des vrais changements d'état : une seule entrée par transition, jamais par rafraîchissement."""
    ch = v[v["niveau"] != v["niveau"].shift()].reset_index(drop=True)
    rows, prev = [], "NORMAL"
    for i, r in ch.iterrows():
        lvl = r["niveau"]
        if lvl == prev:
            continue
        up = RANK[lvl] > RANK[prev]
        nxt = ch.iloc[i + 1:]["niveau"].tolist()
        if up:
            if "NORMAL" in nxt:
                status = "Résolue"
            elif nxt and RANK[nxt[0]] > RANK[lvl]:
                status = "Escaladée"
            else:
                status = "Acquittée" if int(r["ms"]) in acked else "Active"
            desc = f"Passage {prev} → {lvl} (mode {r['mode']})"
        else:
            status = "Info"
            desc = "Retour à la normale" if lvl == "NORMAL" else f"Amélioration {prev} → {lvl}"
        rows.append({"id": int(r["ms"]), "temps (s)": round(float(r["t_s"]), 1), "Sévérité": lvl,
                     "Événement": desc,
                     "Mesure": f"std {r['std_g']:.3f} g | ratio x{r['ratio']:.1f} | {r['freq_hz']:.0f} Hz",
                     "Statut": status})
        prev = lvl
    cols = ["id", "temps (s)", "Sévérité", "Événement", "Mesure", "Statut"]
    return pd.DataFrame(rows, columns=cols).iloc[::-1].reset_index(drop=True)


# Composants visuels

def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style(fig, h=320, title=None):
    fig.update_layout(
        template=T["tpl"], height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color=T["muted"]),
        margin=dict(l=48, r=24, t=52 if title else 20, b=38),
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=14, color=T["txt"]), x=0.02),
        hoverlabel=dict(bgcolor=T["panel2"], font=dict(color=T["txt"]), bordercolor=T["line"]),
        legend=dict(orientation="h", y=-0.24, bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"])))
    fig.update_xaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["line"])
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["line"])
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def section(text):
    st.markdown(f"<div class='sec'>{text}</div>", unsafe_allow_html=True)


def chip(label, value, color):
    return f"<span class='chip' style='--c:{color}'><i></i>{label} <b>{value}</b></span>"


def trend(series, n, good_when_up, unit="", fmt="{:+.1f}"):
    """Tendance uniquement s'il y a assez d'historique."""
    if len(series) < n + 1:
        return "<span class='ks'>tendance : historique insuffisant</span>"
    delta = float(series.iloc[-1] - series.iloc[-1 - n])
    if abs(delta) < 1e-9:
        return "<span class='tr' style='--c:var(--muted)'>■ stable</span>"
    up = delta > 0
    col = T["good"] if up == good_when_up else T["bad"]
    return f"<span class='tr' style='--c:{col}'>{'▲' if up else '▼'} {fmt.format(delta)}{unit} sur {n} mesures</span>"


def kpi(label, value, color, sub="", tip="", extra=""):
    return (f"<div class='kpi' style='--c:{color}' title=\"{html.escape(tip, quote=True)}\">"
            f"<div class='kl'>{label}</div><div class='kv'>{value}</div>{extra}<div class='ks'>{sub}</div></div>")


def kpi_gauge(label, value, color, sub, tip):
    p = f"M10 60 A50 50 0 0 1 110 60"
    svg = (f"<svg viewBox='0 0 120 66' class='arc'><path d='{p}' pathLength='100' class='arc-bg'/>"
           f"<path d='{p}' pathLength='100' class='arc-fg' style='stroke:{color};stroke-dasharray:{value:.1f} 100'/>"
           f"<text x='60' y='54' text-anchor='middle' class='arc-t' style='fill:{color}'>{value:.0f}</text></svg>")
    return (f"<div class='kpi' style='--c:{color}' title=\"{html.escape(tip, quote=True)}\">"
            f"<div class='kl'>{label}</div>{svg}<div class='ks' style='text-align:center'>{sub}</div></div>")


def wave_svg(ratio):
    amp = min(5 + ratio * 3, 24)
    x = np.linspace(0, 400, 140)
    y = 30 + amp * np.sin(x / 9) * np.sin(x / 61 + 1) + 0.15 * amp * np.sin(x / 3)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return f"<svg class='wave' viewBox='0 0 400 60' preserveAspectRatio='none'><polyline points='{pts}'/></svg>"


def gauge(value, rng, title, color, steps, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "font": {"family": "Space Grotesk", "size": 42, "color": color}},
        gauge={"axis": {"range": rng, "tickcolor": T["muted"], "tickfont": {"color": T["muted"]}},
               "bar": {"color": color, "thickness": 0.3}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": steps}))
    return style(fig, 260, title)


TWIN_HTML = """
<div id="w" style="position:relative;width:100%;height:__H__px;border-radius:12px;overflow:hidden;
background:radial-gradient(ellipse at 50% 45%,__GLOW__ 0%,transparent 70%)">
<div id="c" style="position:absolute;inset:0"></div>
<div id="msg" style="position:absolute;left:12px;bottom:8px;color:#8998B8;font:11px Inter,sans-serif">
Glisser pour orienter le moteur</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
if(typeof THREE==='undefined'){document.getElementById('msg').textContent='Three.js indisponible (hors ligne) : jumeau 3D non affiché';return;}
const reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const el=document.getElementById('c');const W=el.clientWidth||600,H=el.clientHeight||__H__;
const r=new THREE.WebGLRenderer({antialias:true,alpha:true});r.setSize(W,H);r.setPixelRatio(Math.min(window.devicePixelRatio||1,2));el.appendChild(r.domElement);
const s=new THREE.Scene();const cam=new THREE.PerspectiveCamera(38,W/H,.1,100);cam.position.set(0,2.4,8.6);cam.lookAt(0,0,0);
const col=new THREE.Color('__C__');
s.add(new THREE.AmbientLight(0x8899bb,.6));
const key=new THREE.DirectionalLight(0xffffff,1.2);key.position.set(4,6,5);s.add(key);
const rim=new THREE.PointLight(col,2.4,30);rim.position.set(-5,2,-3);s.add(rim);
const rim2=new THREE.PointLight(col,1.3,30);rim2.position.set(5,-1,4);s.add(rim2);
const g=new THREE.Group();s.add(g);
const steel=new THREE.MeshStandardMaterial({color:0x4a5878,metalness:.85,roughness:.32});
const dark=new THREE.MeshStandardMaterial({color:0x1b2338,metalness:.7,roughness:.45});
const chrome=new THREE.MeshStandardMaterial({color:0xcfd8ee,metalness:1,roughness:.18});
const glow=new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:.9,metalness:.3,roughness:.4});
function cyl(rt,rb,h,m){return new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h,48),m);}
function box(w,h,d,m,x,y,z){const b=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);b.position.set(x,y,z);g.add(b);return b;}
const body=cyl(1.2,1.2,3.4,steel);body.rotation.z=Math.PI/2;g.add(body);
for(let i=0;i<13;i++){const f=new THREE.Mesh(new THREE.TorusGeometry(1.22,.045,10,64),dark);f.rotation.y=Math.PI/2;f.position.x=-1.5+i*.25;g.add(f);}
const front=cyl(1.28,1.28,.3,dark);front.rotation.z=Math.PI/2;front.position.x=1.85;g.add(front);
const rear=cyl(1.15,1.15,.7,dark);rear.rotation.z=Math.PI/2;rear.position.x=-2.0;g.add(rear);
const ring=new THREE.Mesh(new THREE.TorusGeometry(1.0,.03,8,64),glow);ring.rotation.y=Math.PI/2;ring.position.x=-2.37;g.add(ring);
box(1.1,.5,.9,dark,-.2,1.45,0);box(.7,.08,.5,glow,-.2,1.72,0);
box(2.8,.25,.5,steel,0,-1.28,-.8);box(2.8,.25,.5,steel,0,-1.28,.8);box(4.8,.15,2.6,dark,0,-1.5,0);
const rotor=new THREE.Group();rotor.position.x=2.0;g.add(rotor);
const shaft=cyl(.2,.2,1.6,chrome);shaft.rotation.z=Math.PI/2;shaft.position.x=.8;rotor.add(shaft);
const disk=cyl(.55,.55,.12,chrome);disk.rotation.z=Math.PI/2;disk.position.x=1.1;rotor.add(disk);
for(let k=0;k<6;k++){const a=k*Math.PI/3;const m=new THREE.Mesh(new THREE.BoxGeometry(.14,.14,.14),glow);m.position.set(1.1,Math.cos(a)*.45,Math.sin(a)*.45);rotor.add(m);}
let drag=false,px=0,py=0,ry=-.55,rx=.15;
r.domElement.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY;});
window.addEventListener('pointerup',()=>{drag=false;});
window.addEventListener('pointermove',e=>{if(!drag)return;ry+=(e.clientX-px)*.008;rx+=(e.clientY-py)*.005;rx=Math.max(-.6,Math.min(.8,rx));px=e.clientX;py=e.clientY;});
const speed=__SPEED__*(reduce?.2:1),amp=reduce?0:__AMP__;let t=0;
(function loop(){t+=1;rotor.rotation.x+=speed;g.rotation.y=ry+(drag?0:Math.sin(t*.004)*.12);g.rotation.x=rx;
g.position.y=Math.sin(t*1.9)*amp;g.position.z=Math.cos(t*2.3)*amp*.6;r.render(s,cam);requestAnimationFrame(loop);})();
})();
</script>
"""


def twin(freq, ratio, color, height=360):
    """Moteur 3D procédural. La vitesse suit la fréquence dominante (proxy), la secousse suit le ratio de vibration.
    Les valeurs sont arrondies pour ne pas recharger l'iframe à chaque rafraîchissement."""
    f = round(min(max(freq, 10), 100) / 5) * 5
    rr = round(min(ratio, 10) * 2) / 2
    speed = 0.04 + f / 100 * 0.35
    amp = 0.004 + rr * 0.006
    glow = rgba(color, 0.16)
    code = (TWIN_HTML.replace("__H__", str(height)).replace("__C__", color).replace("__GLOW__", glow)
            .replace("__SPEED__", f"{speed:.3f}").replace("__AMP__", f"{amp:.3f}"))
    if hasattr(st, "iframe"):  # Streamlit récent : components.html est déprécié
        st.iframe(code, height=height + 4)
    else:
        components.html(code, height=height + 4)


def badge(text, color):
    return f"<span class='badge' style='--c:{color}'>{text}</span>"


STATUS_COL = {"Active": "bad", "Acquittée": "tele", "Résolue": "good", "Escaladée": "warn", "Info": "muted"}


def alerts_html(ev, n=5):
    if ev.empty:
        return "<div class='cap'>Aucun événement : la machine n'a pas quitté l'état normal.</div>"
    out = []
    for _, r in ev.head(n).iterrows():
        out.append(f"<div class='alert' style='--c:{LV[r['Sévérité']]}'>"
                   f"<span class='a-t'>T+{r['temps (s)']:.0f} s</span>{badge(r['Sévérité'], LV[r['Sévérité']])}"
                   f"<span class='a-d'>{r['Événement']}<div class='a-m'>{r['Mesure']}</div></span>"
                   f"{badge(r['Statut'], T[STATUS_COL[r['Statut']]])}</div>")
    return "".join(out)


def style_levels(df, level_col, status_col=None):
    def lv(x):
        c = LV.get(x)
        return f"color:{c};font-weight:700" if c else ""

    def stt(x):
        c = STATUS_COL.get(x)
        return f"color:{T[c]};font-weight:700" if c else ""

    sty = df.style.map(lv, subset=[level_col])
    if status_col:
        sty = sty.map(stt, subset=[status_col])
    return sty


# Barre latérale : données, lecture, analyse

st.sidebar.markdown("**Source des données**")
src = st.sidebar.radio("Source", [SRC_DEMO, SRC_CSV, SRC_API], label_visibility="collapsed")
is_api = src == SRC_API
raw, url = pd.DataFrame(columns=COLS), API_DEFAULT
if src == SRC_DEMO:
    raw = make_demo(ss.sim)
elif src == SRC_CSV:
    up = st.sidebar.file_uploader("Sortie du moniteur série (Wokwi)", type=["csv", "txt"])
    if up:
        raw = parse_serial(up.read().decode("utf-8", errors="ignore"))
        if raw.empty:
            st.sidebar.error("Aucune ligne valide : colonnes attendues ms,mode,std_g,p2p_g,ratio,freq_hz,niveau.")
else:
    url = st.sidebar.text_input("URL de l'API", API_DEFAULT).rstrip("/")
    api = fetch_api(url)
    if api is None:
        st.sidebar.error("API injoignable ou base vide.")
    else:
        raw = api

st.sidebar.markdown("**Lecture et rafraîchissement**")
st.sidebar.toggle("Lecture automatique", key="play", disabled=is_api,
                  help="En mode API, la page interroge l'API en continu.")
interval = st.sidebar.slider("Intervalle de rafraîchissement (s)", 0.2, 5.0, 0.8, 0.1)
step = st.sidebar.slider("Mesures ajoutées par rafraîchissement", 1, 10, 2, disabled=is_api)
playing = bool(ss.play) and not is_api
if not raw.empty and not is_api and len(raw) > 12:
    ss.scrub = int(min(max(ss.pos, 1), len(raw)))
    st.sidebar.slider("Position de lecture", 1, len(raw), key="scrub", disabled=playing,
                      on_change=lambda: ss.__setitem__("pos", ss.scrub))


def replay():
    ss.pos, ss.last_level, ss.play, ss.acked = 12, None, True, set()


st.sidebar.button("Rejouer depuis le début", on_click=replay, width="stretch", disabled=is_api)

st.sidebar.markdown("**Paramètres d'analyse**")
contam = st.sidebar.slider("Contamination (Isolation Forest)", 0.01, 0.30, 0.10, 0.01)
threshold = st.sidebar.slider("Seuil de panne (santé)", 5, 50, 20)
window_s = st.sidebar.slider("Fenêtre de tendance (s)", 10, 90, 30)

if raw.empty:
    st.markdown("<div class='topbar'><div><div class='brand'>SENTINELEDGE</div>"
                "<div class='sub'>Aucune donnée à afficher</div></div></div>", unsafe_allow_html=True)
    st.info("Charge un fichier CSV issu de Wokwi, démarre l'API, ou choisis la source Démonstration.")
    st.markdown("<div class='tagline'>Anticipate failure. Preserve performance.</div>", unsafe_allow_html=True)
    st.stop()

if not is_api:
    sig = (src, len(raw), int(raw["ms"].min()), ss.sim)
    if ss.get("sig") != sig:
        ss.sig, ss.pos, ss.last_level, ss.acked = sig, min(12, len(raw)), None, set()
    d0 = analyse(raw, contam)[0]
    st.sidebar.download_button("Télécharger l'analyse (CSV)", d0.to_csv(index=False).encode("utf-8"),
                               "sentinel_analyse.csv", "text/csv", width="stretch")

st.sidebar.markdown("<div class='tagline'>Anticipate failure.<br>Preserve performance.</div>", unsafe_allow_html=True)


# Pages

def header(ctx):
    src_col = T["brand"] if src == SRC_DEMO else T["tele"]
    chips = [chip("Source", f"{src}{' (simulée)' if src == SRC_DEMO else ''}", src_col)]
    if is_api:
        if not ctx["api_ok"]:
            chips.append(chip("API", "injoignable", T["bad"]))
            chips.append(chip("Réseau", "déconnecté", T["bad"]))
        elif ctx["stale"]:
            chips.append(chip("API", "joignable", T["good"]))
            chips.append(chip("Données", f"obsolètes (> {STALE_S:.0f} s)", T["warn"]))
        else:
            chips.append(chip("API", "joignable", T["good"]))
            chips.append(chip("Données", "à jour", T["good"]))
    else:
        chips.append(chip("API", "non utilisée", T["muted"]))
        chips.append(chip("Réseau", "hors ligne (rejeu local)", T["muted"]))
    chips.append(chip("Thème", theme, T["brand"]))
    if ctx["finished"]:
        tag = "<span class='paused'>TERMINÉ</span>"
    elif is_api and ctx["api_ok"] and not ctx["stale"]:
        tag = "<span class='live'>EN DIRECT</span>"
    elif is_api:
        tag = "<span class='paused'>HORS LIGNE</span>"
    elif playing:
        tag = "<span class='simtag'>SIMULATION</span>" if src == SRC_DEMO else "<span class='simtag'>REJEU</span>"
    else:
        tag = "<span class='paused'>PAUSE</span>"
    st.markdown(
        f"<div class='topbar'><div><div class='brand'>SENTINELEDGE</div>"
        f"<div class='sub'>{page} | T+ {ctx['t_now']:,.1f} s | {ctx['pos']} / {ctx['n']} mesures</div></div>"
        f"<div class='chips'>{''.join(chips)}</div>"
        f"<div style='text-align:right'>{tag}<div class='clock'>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div></div></div>",
        unsafe_allow_html=True)


def page_overview(c):
    v, d, last = c["v"], c["d"], c["last"]
    health, res, lvl = c["health"], c["res"], c["lvl"]
    t_now, t_end = c["t_now"], c["t_end"]
    ev = c["ev"]
    n_active = int((ev["Statut"] == "Active").sum())

    rank = max(RANK[lvl], 0 if health >= 70 else 1 if health >= 40 else 2)
    state, col, cls, msg = [
        ("FONCTIONNEMENT NORMAL", T["good"], "", "Vibrations dans la plage de référence."),
        ("ATTENTION : COMPORTEMENT ANORMAL", T["warn"], "",
         "Vibration au-dessus de la référence ; surveiller l'évolution."),
        ("CRITIQUE : INTERVENTION REQUISE", T["bad"], "crit",
         "Anomalie sévère détectée ; inspection recommandée.")][rank]
    if res is None:
        rul_txt, rul_sub = "n/d", "pas assez de mesures"
    elif res["rul"] == 0:
        rul_txt, rul_sub = "0 s", "seuil de panne atteint"
    elif np.isinf(res["rul"]):
        rul_txt, rul_sub = "Stable", "aucune dégradation estimée"
    else:
        hi = f"{res['hi']:.0f}" if np.isfinite(res["hi"]) else "plus"
        rul_txt, rul_sub = f"~{res['rul']:.0f} s", f"estimation ; intervalle 95 % : {res['lo']:.0f} à {hi} s"

    st.markdown(
        f"<div class='banner {cls}' style='--c:{col}'><div><div class='b1'><span class='dot'></span>{state}</div>"
        f"<div class='b2'>{msg} Santé {health:.0f}/100 | mode {last['mode']} | niveau embarqué {lvl}</div>"
        f"<div class='b3'>Dernière mesure : ms={int(last['ms'])} (T+{t_now:.1f} s)"
        f"{' | données simulées' if src == SRC_DEMO else ''}</div></div>{wave_svg(float(last['ratio']))}</div>",
        unsafe_allow_html=True)

    lc = LV.get(lvl, T["tele"])
    k = st.columns(5)
    k[0].markdown(kpi_gauge("Machine Health", health, col, "0 = panne, 100 = neuf",
                            "Ratio de vibration (70 %) + score Isolation Forest (30 %), lissé."),
                  unsafe_allow_html=True)
    k[1].markdown(kpi("Remaining Useful Life", rul_txt, col, rul_sub,
                      "Projection linéaire de la santé jusqu'au seuil de panne : estimation, pas une garantie."),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("Vibration", f"{last['std_g']:.3f} g", lc, trend(v["std_g"], 10, False, " g", "{:+.3f}"),
                      "Écart-type de l'accélération sur la dernière fenêtre."), unsafe_allow_html=True)
    k[3].markdown(kpi("Dominant Frequency", f"{last['freq_hz']:.1f} Hz" if c["has_freq"] else "n/d", T["tele"],
                      trend(v["freq_hz"], 10, True, " Hz") if c["has_freq"] else "absente (ancien format CSV)",
                      "Fréquence principale trouvée par la DFT."), unsafe_allow_html=True)
    k[4].markdown(kpi("Active Alerts", f"{n_active}", T["bad"] if n_active else T["good"],
                      "non acquittées et non résolues", "Alertes issues des changements d'état."),
                  unsafe_allow_html=True)

    section("JUMEAU NUMÉRIQUE ET SANTÉ PRÉDICTIVE")
    left, right = st.columns([1, 1.7])
    with left:
        st.markdown("<div class='ptitle'>Digital Twin</div>", unsafe_allow_html=True)
        twin(float(last["freq_hz"]) if c["has_freq"] else 25.0, float(last["ratio"]), lc)
        st.markdown(
            f"<span class='ov' style='--c:{lc}'>Vibration<b>{last['std_g']:.3f} g</b></span>"
            f"<span class='ov' style='--c:{T['tele']}'>Fréquence<b>{last['freq_hz']:.1f} Hz</b></span>"
            f"<span class='ov' style='--c:{T['muted']}'>Vitesse<b>n/d</b></span>"
            "<div class='cap'>Rotation animée proportionnelle à la fréquence dominante. Aucun tachymètre RPM "
            "n'est mesuré.</div>", unsafe_allow_html=True)
    with right:
        rng_label = st.radio("Plage", list(RANGES), horizontal=True, label_visibility="collapsed", key="range")
        rng = RANGES[rng_label]
        vv = v if rng is None else v[v["t_s"] >= t_now - rng]
        fig = go.Figure()
        for y0, y1, key in [(0, 40, "bad"), (40, 70, "warn"), (70, 100, "good")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=rgba(T[key], .06), line_width=0)
        fig.add_trace(go.Scatter(x=vv["t_s"], y=vv["health"], mode="lines", name="Santé (mesurée)",
                                 line=dict(color=T["tele"], width=3, shape="spline", smoothing=0.6),
                                 fill="tozeroy", fillcolor=rgba(T["tele"], .10)))
        chg = vv[vv["niveau"] != vv["niveau"].shift()]
        fig.add_trace(go.Scatter(x=chg["t_s"], y=chg["health"], mode="markers", name="Changement d'état",
                                 marker=dict(size=11, color=[LV.get(x, T["tele"]) for x in chg["niveau"]],
                                             line=dict(color=T["txt"], width=1)),
                                 text=chg["niveau"], hovertemplate="%{text}<br>t=%{x:.1f} s<extra></extra>"))
        t_ml = first_sustained(vv["ml_anomalie"], vv["t_s"])
        if pd.notna(t_ml):
            fig.add_vline(x=t_ml, line_dash="dot", line_color=T["brand"], annotation_text="1re anomalie IA",
                          annotation_font_color=T["brand"])
        fig.add_hline(y=threshold, line_dash="dot", line_color=T["bad"], annotation_text="seuil de panne",
                      annotation_font_color=T["bad"])
        x0, xmax = float(vv["t_s"].min()), t_end
        if res and np.isfinite(res["rul"]) and res["rul"] > 0:
            hi_c = res["hi"] if np.isfinite(res["hi"]) else res["rul"] * 2.2
            fig.add_trace(go.Scatter(
                x=[res["t_last"], res["t_last"] + res["lo"], res["t_last"] + hi_c, res["t_last"]],
                y=[res["now"], threshold, threshold, res["now"]], fill="toself", mode="none",
                fillcolor=rgba(T["bad"], .14), name="Incertitude 95 % (estimée)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[res["t_last"], res["t_last"] + res["rul"]], y=[res["now"], threshold],
                                     mode="lines", name="Projection (estimation)",
                                     line=dict(dash="dash", color=T["bad"], width=2)))
            xmax = max(t_end, res["t_last"] + hi_c)
        fig.update_xaxes(range=[x0, xmax * 1.02], title="temps (s)")
        fig.update_yaxes(range=[0, 105], title="santé")
        show(style(fig, 450, "Santé de la machine : historique et projection"), "health")
        if rng and t_now < rng:
            st.caption(f"Données disponibles : {t_now:.0f} s seulement ; la plage {rng_label} est tronquée.")

    section("CONTRÔLE MACHINE (SIMULATION)")
    st.caption("Contrôles de démonstration : ils changent le scénario simulé de la source Démonstration. "
               "Ils ne commandent aucun ESP32 ni aucune machine réelle.")
    cols = st.columns(4)
    for col_, name in zip(cols, SIM_MODES):
        if col_.button(name, key=f"sim_{name}", width="stretch", disabled=src != SRC_DEMO,
                       type="primary" if (ss.sim == name and src == SRC_DEMO) else "secondary"):
            ss.sim = name
            st.rerun()
    if src != SRC_DEMO:
        st.caption("Désactivé : sélectionne la source Démonstration pour changer de scénario.")

    section("ALERTES RÉCENTES")
    st.markdown(alerts_html(ev, 5), unsafe_allow_html=True)


def page_telemetry(c):
    v = c["v"]
    section("SIGNAL ET SEUILS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["t_s"], y=v["std_g"], mode="lines", name="Vibration",
                             line=dict(color=rgba(T["muted"], .6), width=1.5)))
    fig.add_trace(go.Scatter(x=v["t_s"], y=v["std_g"], mode="markers", name="Niveau d'alerte",
                             marker=dict(size=6, color=[LV.get(x, T["tele"]) for x in v["niveau"]])))
    fig.add_hline(y=c["base"] * 2.5, line_dash="dot", line_color=T["warn"], annotation_text="attention",
                  annotation_font_color=T["warn"])
    fig.add_hline(y=c["base"] * 6, line_dash="dot", line_color=T["bad"], annotation_text="critique",
                  annotation_font_color=T["bad"])
    fig.update_xaxes(range=[0, c["t_end"] * 1.02], title="temps (s)")
    fig.update_yaxes(title="écart-type (g)")
    show(style(fig, 330, "Vibration et seuils d'alerte"), "signal")

    section("TABLE DE TÉLÉMÉTRIE")
    f1, f2, f3 = st.columns([2, 2, 1])
    levels = f1.multiselect("Sévérité", list(RANK), default=list(RANK))
    modes = sorted(v["mode"].unique())
    sel_modes = f2.multiselect("Mode machine", modes, default=modes)
    nrows = f3.number_input("Lignes max", 10, 2000, 200, 10)
    tab = v[v["niveau"].isin(levels) & v["mode"].isin(sel_modes)].iloc[::-1]  # plus récentes d'abord
    if tab.empty:
        st.info("Aucune mesure ne correspond aux filtres (ou aucune donnée reçue).")
        return
    if c["stale"]:
        st.warning(f"Données obsolètes : aucune nouvelle mesure depuis plus de {STALE_S:.0f} s.")
    cols_show = ["ms", "mode", "std_g", "p2p_g", "ratio", "freq_hz", "niveau"]
    out = tab[cols_show].head(int(nrows))
    fmt = {"std_g": "{:.4f}", "p2p_g": "{:.4f}", "ratio": "{:.2f}", "freq_hz": "{:.1f}"}
    st.dataframe(style_levels(out, "niveau").format(fmt), width="stretch", height=380, hide_index=True)
    st.caption("Clique sur un en-tête de colonne pour trier. Les plus récentes sont affichées en premier.")
    st.download_button("Exporter la sélection (CSV)", tab[cols_show].to_csv(index=False).encode("utf-8"),
                       "sentinel_telemetrie.csv", "text/csv", key="dl_tel")


def page_ai(c):
    v, d = c["v"], c["d"]
    last, has_freq = c["last"], c["has_freq"]
    section("SPECTROGRAMME ET ESPACE DES CARACTÉRISTIQUES")
    a, b = st.columns(2)
    with a:
        if has_freq:
            fig = px.density_heatmap(v, x="t_s", y="freq_hz", z="std_g", histfunc="avg", nbinsx=60, nbinsy=34,
                                     color_continuous_scale="Viridis")
            fig.update_xaxes(range=[0, c["t_end"] * 1.02], title="temps (s)")
            fig.update_yaxes(range=[0, 100], title="fréquence (Hz)")
            fig.update_coloraxes(colorbar=dict(title="std (g)", thickness=10))
            show(style(fig, 380, "Spectrogramme (fréquence dominante × amplitude)"), "spectro")
            st.caption("Construit à partir des caractéristiques calculées sur l'ESP32 (fréquence dominante et écart-type "
                       "par fenêtre), pas à partir des échantillons bruts d'accélération.")
        else:
            st.info("Pas de fréquence dans ce fichier (ancien format). Mets à jour le sketch Wokwi.")
    with b:
        axis3d = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=T["grid"], color=T["muted"])
        fig = go.Figure()
        if has_freq:
            for lvl_name in RANK:
                s = v[v["niveau"] == lvl_name]
                if s.empty:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=s["freq_hz"], y=s["std_g"], z=s["p2p_g"], mode="markers", name=lvl_name,
                    marker=dict(size=3.5, color=LV[lvl_name], opacity=0.85),
                    hovertemplate="freq %{x:.1f} Hz<br>std %{y:.4f} g<br>p2p %{z:.4f} g<extra>" + lvl_name + "</extra>"))
            fig.add_trace(go.Scatter3d(x=[last["freq_hz"]], y=[last["std_g"]], z=[last["p2p_g"]], mode="markers",
                                       marker=dict(size=9, color=T["txt"], line=dict(color=T["brand"], width=3)),
                                       name="Maintenant"))
            fig.update_layout(scene=dict(xaxis_title="freq (Hz)", yaxis_title="std (g)", zaxis_title="p2p (g)",
                                         xaxis=axis3d, yaxis=axis3d, zaxis=axis3d))
        else:
            fig = px.scatter(v, x="std_g", y="p2p_g", color="niveau", color_discrete_map=LV)
        show(style(fig, 380, "Espace des caractéristiques"), "feat3d")

    section("SEUIL EMBARQUÉ ET ISOLATION FOREST")
    t_thr = first_sustained(v["seuil_anomalie"], v["t_s"])
    t_ml = first_sustained(v["ml_anomalie"], v["t_s"])
    m = st.columns(3)
    m[0].markdown(kpi("Première alerte seuil", f"{t_thr:.1f} s" if pd.notna(t_thr) else "aucune", T["warn"],
                      "3 mesures consécutives"), unsafe_allow_html=True)
    m[1].markdown(kpi("Première anomalie IA", f"{t_ml:.1f} s" if pd.notna(t_ml) else "aucune", T["brand"],
                      "3 mesures consécutives"), unsafe_allow_html=True)
    if pd.notna(t_thr) and pd.notna(t_ml):
        gap = t_thr - t_ml
        txt = f"{abs(gap):.1f} s"
        sub = ("l'IA détecte avant le seuil" if gap > 0 else
               "le seuil détecte avant l'IA" if gap < 0 else "détection simultanée")
    else:
        txt, sub = "n/d", "comparaison impossible : une méthode n'a pas encore détecté"
    m[2].markdown(kpi("Avance de détection", txt, T["tele"], sub), unsafe_allow_html=True)
    st.caption("Seuil embarqué : règle fixe sur le ratio de vibration (attention ≥ 2,5, critique ≥ 6), calculée sur "
               "l'ESP32. Isolation Forest : score statistique d'écart par rapport au comportement normal appris ; il "
               "signale qu'une mesure est atypique, pas la cause physique du défaut.")

    a, b, c3 = st.columns([1.4, 1.2, 1])
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v["t_s"], y=v["score"], mode="lines", name="Score IA",
                                 line=dict(color=T["brand"], width=2)))
        bad = v[v["ml_anomalie"]]
        fig.add_trace(go.Scatter(x=bad["t_s"], y=bad["score"], mode="markers", name="Anomalie IA",
                                 marker=dict(color=T["bad"], size=6)))
        fig.add_hline(y=0, line_dash="dot", line_color=T["muted"])
        fig.update_xaxes(range=[0, c["t_end"] * 1.02], title="temps (s)")
        fig.update_yaxes(title="score (bas = atypique)")
        show(style(fig, 340, "Score Isolation Forest"), "score")
    with b:
        shares = v["niveau"].value_counts()
        fig = go.Figure(go.Pie(labels=shares.index, values=shares.values, hole=0.72, sort=False,
                               marker=dict(colors=[LV.get(x, T["tele"]) for x in shares.index],
                                           line=dict(color=T["bg"], width=3)), textinfo="percent"))
        fig.add_annotation(text=f"{len(v)}<br>mesures", showarrow=False, font=dict(size=16, color=T["txt"]))
        show(style(fig, 340, "Répartition des niveaux"), "pie")
    with c3:
        show(gauge(float(last["score"]), [-0.3, 0.3], "Score actuel", T["brand"],
                   [{"range": [-0.3, 0], "color": rgba(T["bad"], .2)},
                    {"range": [0, 0.3], "color": rgba(T["good"], .2)}]), "g_score")

    section("SYNTHÈSE DIAGNOSTIQUE")
    feats = c["feats"]
    ref = d[d["mode"] == "NORMALE"]
    if len(ref) < 10:
        ref = d.head(max(10, len(d) // 5))
    rows = []
    for f_ in feats:
        mu, sd = float(ref[f_].mean()), float(ref[f_].std() or np.nan)
        z = (float(last[f_]) - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        rows.append({"Caractéristique": f_, "Actuel": round(float(last[f_]), 4), "Référence (moy.)": round(mu, 4),
                     "Écart (σ)": round(z, 1) if np.isfinite(z) else None})
    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with right:
        flagged = bool(last["ml_anomalie"])
        st.markdown(
            f"<div class='panel'><div class='ptitle'>État du modèle</div>"
            f"<div class='cap'>Isolation Forest, 200 arbres, contamination {contam:.2f}<br>"
            f"Entraîné sur {c['n_train']} mesures de référence (mode NORMALE si disponible)<br>"
            f"Score actuel : {float(last['score']):+.3f}</div></div>", unsafe_allow_html=True)
        st.write("")
        worst = max(rows, key=lambda r_: abs(r_["Écart (σ)"] or 0))
        st.markdown(
            f"<div class='cap'>{'La dernière mesure est jugée atypique' if flagged else 'La dernière mesure reste dans la zone apprise'}"
            f" ; l'écart le plus marqué concerne <b>{worst['Caractéristique']}</b> "
            f"({worst['Écart (σ)'] if worst['Écart (σ)'] is not None else 'n/d'} σ par rapport à la référence). "
            "Ce résultat est statistique et n'identifie pas la cause physique.</div>", unsafe_allow_html=True)


def page_events(c):
    ev = c["ev"]
    section("CENTRE D'ALERTES")
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Sévérité", list(RANK), default=list(RANK), key="ev_sev")
    stt = f2.multiselect("Statut", list(STATUS_COL), default=list(STATUS_COL), key="ev_stt")
    t_max = max(1.0, round(c["t_end"], 1))
    tr = st.slider("Plage de temps (s)", 0.0, t_max, (0.0, t_max), key="ev_t")
    sel = ev[ev["Sévérité"].isin(sev) & ev["Statut"].isin(stt) & ev["temps (s)"].between(tr[0], tr[1])]
    n_active = int((ev["Statut"] == "Active").sum())
    b1, b2 = st.columns([1, 3])
    if b1.button(f"Acquitter les {n_active} alertes actives", disabled=n_active == 0, width="stretch"):
        ss.acked |= set(ev.loc[ev["Statut"] == "Active", "id"].astype(int))
        st.rerun()
    b2.caption("Une alerte = un changement d'état réel ; une condition persistante ne génère pas de doublon.")
    if sel.empty:
        st.info("Aucun événement à afficher.")
        return
    out = sel.drop(columns=["id"])
    st.dataframe(style_levels(out, "Sévérité", "Statut"), width="stretch", height=420, hide_index=True)
    st.download_button("Exporter le journal (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "sentinel_evenements.csv", "text/csv", key="dl_ev")


def page_api(c):
    section("API ET BASE DE DONNÉES")
    if not is_api:
        st.info("Source actuelle : " + src + ". Choisis « API (SQLite) » dans la barre latérale pour interroger le backend FastAPI.")
    st.markdown(f"<div class='panel' style='text-align:left'><div class='ptitle'>Backend</div>"
                f"<div class='cap'>URL : {html.escape(url)} (variable d'environnement SENTINEL_API_URL)<br>"
                f"Mesures chargées : {c['n']}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.dataframe(pd.DataFrame([
        ["POST", "/measure", "Ajoute une mesure"], ["POST", "/measures/batch", "Ajoute un lot de mesures"],
        ["GET", "/measures?limit=2000", "Lit les dernières mesures"], ["GET", "/summary", "Résumé de la base"],
        ["DELETE", "/measures", "Vide la base"]], columns=["Méthode", "Route", "Rôle"]),
        width="stretch", hide_index=True)
    if not is_api:
        return
    section("RÉSUMÉ")
    try:
        r = requests.get(f"{url}/summary", timeout=3)
        r.raise_for_status()
        st.json(r.json())
    except Exception as e:
        st.warning(f"GET /summary indisponible : {e}")
    section("MAINTENANCE DE LA BASE")
    ok = st.checkbox("Je confirme vouloir supprimer toutes les mesures (action irréversible)", key="del_ok")
    if st.button("DELETE /measures", disabled=not ok):
        try:
            requests.delete(f"{url}/measures", timeout=5).raise_for_status()
            ss.pos, ss.acked = 12, set()
            st.success("Base vidée.")
        except Exception as e:
            st.error(f"Échec : {e}")


def page_settings(c):
    section("PRÉFÉRENCES")
    st.markdown(f"<div class='cap'>Thème actif : <b>{theme}</b>. Le thème est conservé dans l'URL de la page "
                "(ajoute-la à tes favoris pour le retrouver).</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ["Source", src], ["Intervalle de rafraîchissement", f"{interval:.1f} s"],
        ["Mesures par rafraîchissement", "n/a (API)" if is_api else step],
        ["Contamination Isolation Forest", contam], ["Seuil de panne (santé)", threshold],
        ["Fenêtre de tendance", f"{window_s} s"]], columns=["Paramètre", "Valeur"]),
        width="stretch", hide_index=True)
    st.caption("Ces paramètres se modifient dans la barre latérale.")
    if st.button("Réinitialiser les préférences"):
        st.query_params.clear()
        st.rerun()
    section("FORMAT CSV ATTENDU")
    st.code("ms,mode,std_g,p2p_g,ratio,freq_hz,niveau", language="text")
    st.caption("L'ancien format à 6 colonnes (sans freq_hz) reste accepté.")


# Vue principale (se rafraîchit seule)

def live_view():
    r, api_ok, stale = raw, True, False
    if is_api:
        fresh = fetch_api(url)
        api_ok = fresh is not None
        if api_ok:
            r = fresh
            mx = int(r["ms"].max())
            if mx != ss.api_last_ms:
                ss.api_last_ms, ss.api_last_change = mx, time.time()
        stale = api_ok and (time.time() - ss.api_last_change) > STALE_S
    d, has_freq, base, feats, n_train = analyse(r, contam)
    n = len(d)
    if is_api:
        ss.pos = n
    elif playing and ss.pos < n:
        ss.pos = min(n, ss.pos + step)
    pos = min(max(ss.pos, 1), n)
    v = d.iloc[:pos]
    last = v.iloc[-1]
    lvl = last["niveau"]
    res = estimate_rul(v, threshold, window_s)
    t_now, t_end = float(last["t_s"]), float(d["t_s"].max())
    ev = build_events(v, ss.acked)

    if ss.last_level is not None and RANK.get(lvl, 0) > RANK.get(ss.last_level, 0):
        st.toast(f"Niveau {lvl} atteint à T+{t_now:.0f} s")
    ss.last_level = lvl

    ctx = dict(v=v, d=d, last=last, has_freq=has_freq, base=base, feats=feats, n_train=n_train, n=n, pos=pos,
               health=float(last["health"]), res=res, lvl=lvl, t_now=t_now, t_end=t_end, ev=ev,
               api_ok=api_ok, stale=stale, finished=pos >= n and not is_api)
    header(ctx)
    {"Overview": page_overview, "Live Telemetry": page_telemetry, "AI Diagnostics": page_ai,
     "Event History": page_events, "API & Database": page_api, "Settings": page_settings}[page](ctx)
    st.markdown(f"<div class='foot'>SentinelEdge | source : {src} | {n} mesures | thème {theme}</div>",
                unsafe_allow_html=True)


st.fragment(run_every=interval if (playing or is_api) else None)(live_view)()

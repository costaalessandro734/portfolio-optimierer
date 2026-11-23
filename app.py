"""Portfolio-Optimierer – Streamlit-Version mit automatischem Kursdownload.

Start lokal:  streamlit run app.py
"""
import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio-Optimierer", layout="wide")

# =====================================================================
# Info-Texte (erscheinen beim ?-Symbol neben jedem Element)
# =====================================================================
INFO = {
    "ticker": "Ticker wie bei Yahoo Finance, getrennt durch Komma, z. B. AAPL, SAP.DE, 6033.KL. "
              "Die Kurse werden automatisch geladen (Yahoo, ersatzweise stooq) und 24 h zwischengespeichert.",
    "bench": "Vergleichsindex, z. B. ^GSPC (S&P 500), ^GDAXI (DAX), ^KLSE. Wird nicht gekauft, sondern nur "
             "für Style Analysis und Vergleich genutzt. Leer lassen, wenn du keinen brauchst.",
    "zeitraum": "Zeitraum der historischen Daten. Mehr Jahre = stabilere Schätzungen, aber ältere Daten "
                "sagen weniger über heute aus. Gerechnet wird mit Monatsrenditen (letzter Kurs je Monat).",
    "rf": "Jährliche Rendite einer sicheren Anlage (Staatsanleihe, Tagesgeld). Wird in der Sharpe Ratio "
          "abgezogen und verschiebt damit, welches Portfolio Max-Sharpe ist.",
    "pv": "Dein Vermögen in diesem Portfolio. Damit werden Umschichtung und Simulation in Euro umgerechnet.",
    "mc": "So viele Portfolios mit zufälligen Gewichten (innerhalb deiner Grenzen) werden erzeugt. "
          "Als Punktwolke zeigen sie, welche Kombinationen aus Rendite und Risiko möglich sind.",
    "split": "Auf dem ersten Teil der Daten werden die Gewichte geschätzt, auf dem Rest getestet, ohne dass die "
             "Optimierung diese Monate kannte. Zeigt, ob die Strategie ohne Blick in die Zukunft funktioniert hätte.",
    "win": "Länge des rollierenden Fensters der Style Analysis. Kürzer reagiert schneller, ist aber unruhiger.",
    "hor": "Wie viele Jahre in die Zukunft simuliert wird (Bootstrap aus historischen Monatsrenditen).",
    "upload": "Notlösung, falls ein Ticker nicht geladen werden kann: CSV oder als HTML gespeicherte Tabelle mit "
              "Datum und Kurs. Der Dateiname wird zum Namen; beginnt er mit ^, wird die Datei zum Benchmark.",
    "portfolio": "Aktuell %: dein heutiger Anteil (wird automatisch auf 100 % normiert). Min/Max %: Grenzen "
                 "für das Gewicht in allen Optimierungen. Summe aller Min ≤ 100 % ≤ Summe aller Max. "
                 "Leerverkäufe sind nicht vorgesehen.",
    "ziel": "Max-Sharpe: höchste Überrendite pro Risikoeinheit, reagiert stark auf Schätzfehler. Min-Varianz: "
            "geringste Schwankung, ignoriert erwartete Renditen und ist dadurch robuster. Gleichgewichtet: jedes "
            "Asset 1/n, ganz ohne Schätzung. Benchmark-Stil: Mischung, die den Benchmark am besten nachbildet.",
    "ret": "Durchschnittliche Monatsrendite × 12, historisch. Keine Prognose.",
    "vol": "Standardabweichung der Monatsrenditen × √12. Misst die Schwankung.",
    "sh": "(Rendite − risikofreier Zins) ÷ Volatilität: Überrendite pro Einheit Risiko. Höher ist besser.",
    "mdd": "Größter Verlust von einem Höchststand bis zum folgenden Tiefpunkt im Zeitraum.",
    "rebal": "Unterschied zwischen aktuellen und Zielgewichten. pp = Prozentpunkte, Betrag = Änderung × "
             "Portfoliowert. Unter 0,5 pp: halten. Transaktionskosten und Steuern sind nicht berücksichtigt.",
    "front": "Jeder Punkt ist ein Zufallsportfolio (rechts = mehr Risiko, oben = mehr Rendite). Die Linie ist die "
             "Efficient Frontier: für jedes Risiko die höchste erreichbare Rendite unter deinen Grenzen. Liegt dein "
             "Portfolio deutlich darunter, wäre bei gleichem Risiko mehr Rendite möglich gewesen.",
    "style": "Returns-based Style Analysis nach Sharpe: Benchmark-Renditen werden auf die Asset-Renditen regressiert, "
             "Gewichte summieren zu 100 % und liegen in deinen Grenzen. R² zeigt, wie viel der Index-Schwankung die "
             "Mischung erklärt (1 = perfekt).",
    "risk": "Kennzahlen über den ganzen Zeitraum. VaR 95 %: Monatsverlust, der in 95 % der Monate nicht überschritten "
            "wurde. CVaR 95 %: durchschnittlicher Verlust in den schlechtesten 5 % der Monate. Achtung: Die optimierten "
            "Gewichte wurden auf denselben Daten geschätzt, sehen hier also zu gut aus.",
    "bt": "Gewichte nur mit den Trainingsmonaten berechnet, dann auf den Testzeitraum angewendet (monatlich "
          "rebalanciert). Start = 100. Der ehrlichste Test, ob eine Strategie wirklich besser ist.",
    "boot": "2000 Zukunftspfade: Jeder Monat ist ein zufällig gezogener historischer Monat (alle Assets gemeinsam, "
            "Korrelationen bleiben). 5 % = schlechtes, Median = mittleres, 95 % = sehr gutes Szenario. "
            "P(Verlust) = Anteil der Pfade unter dem Startwert. Annahme: Die Zukunft ähnelt der Vergangenheit.",
}

# =====================================================================
# Daten laden
# =====================================================================
def to_monthly(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = s[s > 0]
    return s.groupby(s.index.to_period("M")).last()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_yahoo(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return to_monthly(s)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_stooq(ticker, start, end):
    df = pd.read_csv(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d")
    if "Close" not in df.columns:
        return None
    s = df.set_index(pd.to_datetime(df["Date"]))["Close"]
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return to_monthly(s) if len(s) else None


def load(ticker, start, end):
    for name, fn in (("Yahoo", load_yahoo), ("stooq", load_stooq)):
        try:
            s = fn(ticker, start, end)
            if s is not None and len(s) >= 13:
                return s, name
        except Exception:
            pass
    return None, None


def parse_upload(file) -> pd.Series:
    raw = file.getvalue()
    if file.name.lower().endswith((".html", ".htm")):
        df = max(pd.read_html(io.BytesIO(raw)), key=len)
    else:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    cols = [c.lower() for c in cols]
    di = next((i for i, c in enumerate(cols) if "date" in c or "datum" in c or c == "index"), 0)
    pi = next((i for i, c in enumerate(cols) if c.startswith("adj") or c.endswith("adjusted")), None)
    if pi is None:
        pi = next((i for i, c in enumerate(cols) if any(k in c for k in ("close", "schluss", "kurs", "price"))), 1)
    d = df.iloc[:, di].astype(str)
    dates = pd.to_datetime(d, errors="coerce", dayfirst=d.str.contains(r"^\d{1,2}\.\d{1,2}\.\d{4}").any(), format="mixed")
    p = df.iloc[:, pi].astype(str).str.strip()
    if p.str.match(r"^-?\d{1,3}(\.\d{3})*,\d+$").any():      # deutsches Format 1.234,56
        p = p.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        p = p.str.replace(",", "", regex=False)
    s = pd.Series(pd.to_numeric(p, errors="coerce").values, index=dates)
    s = s[s.index.notna()]
    return to_monthly(s)


# =====================================================================
# Mathe
# =====================================================================
def solve(f, n, lo, hi, extra=()):
    w0 = np.clip(np.full(n, 1 / n), lo, hi)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    r = minimize(f, w0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                 options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(r.x, lo, hi)
    return w / w.sum()


def min_var(mu, S, lo, hi):
    return solve(lambda w: w @ S @ w, len(mu), lo, hi)


def max_sharpe(mu, S, lo, hi, rf):
    return solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ S @ w + 1e-12), len(mu), lo, hi)


def frontier(mu, S, lo, hi, k=40):
    n = len(mu)
    r_min = min_var(mu, S, lo, hi) @ mu
    r_max = solve(lambda w: -(w @ mu), n, lo, hi) @ mu
    pts = []
    for t in np.linspace(r_min, r_max, k):
        w = solve(lambda w: w @ S @ w, n, lo, hi, ({"type": "ineq", "fun": lambda w, t=t: w @ mu - t},))
        pts.append((np.sqrt(w @ S @ w), w @ mu))
    return np.array(pts)


def style_fit(X, y, lo, hi):
    w = solve(lambda w: 1e4 * np.sum((y - X @ w) ** 2), X.shape[1], lo, hi)
    r2 = 1 - np.sum((y - X @ w) ** 2) / np.sum((y - y.mean()) ** 2)
    return w, r2


def proj(v, lo, hi):
    """Projektion auf {Summe = 1, lo ≤ w ≤ hi} (für Zufallsportfolios)."""
    a, b = (v - hi).min(), (v - lo).max()
    for _ in range(50):
        t = (a + b) / 2
        if np.clip(v - t, lo, hi).sum() > 1:
            a = t
        else:
            b = t
    return np.clip(v - (a + b) / 2, lo, hi)


def metrics(rp, rf):
    rp = np.asarray(rp)
    ret, vol = rp.mean() * 12, rp.std(ddof=1) * np.sqrt(12)
    q = max(1, int(0.05 * len(rp)))
    srt = np.sort(rp)
    cum = np.cumprod(1 + rp)
    mdd = -(cum / np.maximum.accumulate(cum) - 1).min()
    return {"Rendite p.a.": ret, "Vol. p.a.": vol, "Sharpe": (ret - rf) / vol,
            "VaR 95 %": -srt[q - 1], "CVaR 95 %": -srt[:q].mean(), "Max. Drawdown": mdd}


def optimise_all(R, lo, hi, rf, cur, y=None):
    mu, S = R.mean().values * 12, R.cov().values * 12
    n = len(mu)
    out = {"Aktuell": cur, "Gleichgewichtet": np.full(n, 1 / n),
           "Min-Varianz": min_var(mu, S, lo, hi), "Max-Sharpe": max_sharpe(mu, S, lo, hi, rf)}
    if y is not None:
        out["Benchmark-Stil"] = style_fit(R.values, y, lo, hi)[0]
    return out, mu, S


# =====================================================================
# Sidebar
# =====================================================================
sb = st.sidebar
sb.header("Eingaben")
tick_txt = sb.text_input("Ticker", "AAPL, MSFT, JNJ, XOM, KO", help=INFO["ticker"])
bench_t = sb.text_input("Benchmark (optional)", "^GSPC", help=INFO["bench"]).strip()
c1, c2 = sb.columns(2)
start = c1.date_input("Von", date(2013, 1, 1), help=INFO["zeitraum"])
end = c2.date_input("Bis", date.today())
rf = sb.number_input("Risikofreier Zins p.a. %", value=2.0, step=0.1, help=INFO["rf"]) / 100
V = sb.number_input("Portfoliowert €", value=10000.0, step=500.0, min_value=0.0, help=INFO["pv"])
n_mc = sb.select_slider("Zufallsportfolios", [1000, 3000, 6000], 3000, help=INFO["mc"])
split = sb.slider("Backtest: Trainingsanteil %", 30, 90, 70, 5, help=INFO["split"])
win = int(sb.number_input("Style-Fenster (Monate)", 12, 120, 36, help=INFO["win"]))
hor = int(sb.number_input("Simulationshorizont (Jahre)", 1, 30, 5, help=INFO["hor"]))
uploads = sb.file_uploader("Notlösung: CSV/HTML hochladen", accept_multiple_files=True, help=INFO["upload"])

st.title("Portfolio-Optimierer")
st.caption("Kurse werden automatisch geladen. Trage deine Gewichte ein: Die App simuliert, optimiert unter "
           "deinen Grenzen und zeigt, was du umschichten müsstest. Keine Anlageberatung.")

# =====================================================================
# Kurse holen
# =====================================================================
tickers = list(dict.fromkeys(t.strip().upper() for t in tick_txt.replace(";", ",").split(",") if t.strip()))
series, sources, failed = {}, {}, []
with st.spinner("Lade Kurse …"):
    for t in tickers + ([bench_t] if bench_t else []):
        s, src = load(t, start, end)
        if s is None:
            failed.append(t)
        else:
            series[t], sources[t] = s, src
bench = bench_t if bench_t in series else None

for f in uploads or []:
    name = f.name.rsplit(".", 1)[0]
    try:
        series[name], sources[name] = parse_upload(f), "Upload"
        if name.startswith("^") and bench is None:
            bench = name
        elif name not in tickers:
            tickers.append(name)
        if name in failed:
            failed.remove(name)
    except Exception as e:
        st.warning(f"{f.name} konnte nicht gelesen werden: {e}")

if failed:
    st.warning("Keine Kurse gefunden für: " + ", ".join(failed) +
               ". Ticker prüfen (Yahoo-Schreibweise) oder als Datei in der Sidebar hochladen.")
assets = [t for t in tickers if t in series and t != bench]
if len(assets) < 2:
    st.info("Mindestens 2 Assets mit Kursdaten nötig.")
    st.stop()

prices = pd.concat({t: series[t] for t in assets + ([bench] if bench else [])}, axis=1).dropna()
R_all = prices.pct_change().dropna()
R = R_all[assets]
y = R_all[bench].values if bench else None
T = len(R)
if T < 25:
    st.error(f"Nur {T} gemeinsame Monate. Für Optimierung und Backtest braucht es mindestens 25.")
    st.stop()
st.caption(f"{T} Monatsrenditen von {R.index[0]} bis {R.index[-1]} · Quellen: " +
           ", ".join(f"{t} ({sources[t]})" for t in assets + ([bench] if bench else [])))

# =====================================================================
# Portfolio-Eingabe
# =====================================================================
st.subheader("Dein Portfolio", help=INFO["portfolio"])
base = pd.DataFrame({"Ticker": assets, "Aktuell %": round(100 / len(assets), 1), "Min %": 0.0, "Max %": 100.0})
ed = st.data_editor(base, key="w_" + "_".join(assets), hide_index=True, disabled=["Ticker"], width="stretch",
                    column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=100.0, step=1.0)
                                   for c in ["Aktuell %", "Min %", "Max %"]})
lo, hi = ed["Min %"].fillna(0).values / 100, ed["Max %"].fillna(100).values / 100
cur = ed["Aktuell %"].fillna(0).clip(lower=0).values
cur = cur / cur.sum() if cur.sum() > 0 else np.full(len(assets), 1 / len(assets))
if lo.sum() > 1.0001 or hi.sum() < 0.9999 or (lo > hi).any():
    st.error("Grenzen unzulässig: Summe aller Min ≤ 100 % ≤ Summe aller Max, und Min ≤ Max.")
    st.stop()

# =====================================================================
# Rechnen
# =====================================================================
W, mu, S = optimise_all(R, lo, hi, rf, cur, y)
M = {k: metrics(R.values @ w, rf) for k, w in W.items()}

target = st.selectbox("Ziel-Strategie", [k for k in W if k != "Aktuell"], index=2, help=INFO["ziel"])
tw, tm, cm = W[target], M[target], M["Aktuell"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rendite p.a.", f"{tm['Rendite p.a.']:.1%}", f"{(tm['Rendite p.a.'] - cm['Rendite p.a.']) * 100:+.1f} pp", help=INFO["ret"])
k2.metric("Volatilität p.a.", f"{tm['Vol. p.a.']:.1%}", f"{(tm['Vol. p.a.'] - cm['Vol. p.a.']) * 100:+.1f} pp", delta_color="inverse", help=INFO["vol"])
k3.metric("Sharpe Ratio", f"{tm['Sharpe']:.2f}", f"{tm['Sharpe'] - cm['Sharpe']:+.2f}", help=INFO["sh"])
k4.metric("Max. Drawdown", f"{tm['Max. Drawdown']:.1%}", f"{(tm['Max. Drawdown'] - cm['Max. Drawdown']) * 100:+.1f} pp", delta_color="inverse", help=INFO["mdd"])
st.caption("Veränderung gegenüber deinem aktuellen Portfolio, historisch und in-sample.")

left, right = st.columns([5, 7])
with left:
    st.subheader("Umschichtung", help=INFO["rebal"])
    d = tw - cur
    rb = pd.DataFrame({"Ticker": assets, "Aktuell %": cur * 100, "Ziel %": tw * 100, "Änderung pp": d * 100,
                       "Betrag €": d * V})
    rb["Aktion"] = np.where(np.abs(d) < 0.005, "halten", np.where(d > 0, "kaufen", "verkaufen"))
    st.dataframe(rb.style.format({"Aktuell %": "{:.1f}", "Ziel %": "{:.1f}", "Änderung pp": "{:+.1f}",
                                  "Betrag €": "{:+,.0f}"}), hide_index=True, width="stretch")

with right:
    st.subheader("Efficient Frontier und Monte-Carlo-Portfolios", help=INFO["front"])
    rng = np.random.default_rng(5)
    Wr = np.array([proj(w, lo, hi) for w in rng.dirichlet(np.ones(len(assets)), n_mc)])
    r_ret, r_vol = Wr @ mu, np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr))
    F = frontier(mu, S, lo, hi)
    fig = go.Figure()
    fig.add_scatter(x=r_vol, y=r_ret, mode="markers", name="Zufallsportfolios",
                    marker=dict(size=4, color=(r_ret - rf) / r_vol, colorscale="Blues", showscale=True,
                                colorbar=dict(title="Sharpe")), hoverinfo="skip")
    fig.add_scatter(x=F[:, 0], y=F[:, 1], mode="lines", name="Efficient Frontier", line=dict(color="black", width=3))
    fig.add_scatter(x=np.sqrt(np.diag(S)), y=mu, mode="markers+text", text=assets, textposition="top center",
                    name="Einzelwerte", marker=dict(symbol="square-open", size=9, color="gray"))
    for k, sym, col in [("Aktuell", "circle", "#e8361b"), ("Min-Varianz", "circle-open", "#1240e0"),
                        ("Max-Sharpe", "diamond", "#1240e0")]:
        w = W[k]
        fig.add_scatter(x=[np.sqrt(w @ S @ w)], y=[w @ mu], mode="markers", name=k,
                        marker=dict(symbol=sym, size=14, color=col, line=dict(width=2, color=col)))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Volatilität p.a.",
                      yaxis_title="Erw. Rendite p.a.", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch")

# =====================================================================
# Tabs: Style, Risiko, Backtest, Simulation
# =====================================================================
t1, t2, t3, t4 = st.tabs(["Style Analysis", "Risiko & Gewichte", "Backtest", "Zukunftssimulation"])

with t1:
    st.subheader("Style Analysis (rollierend)", help=INFO["style"])
    if y is None:
        st.info("Trage in der Sidebar einen Benchmark ein (z. B. ^GSPC), um die Style Analysis zu sehen.")
    else:
        w_all, r2 = style_fit(R.values, y, lo, hi)
        st.write(f"Welche Mischung deiner Assets bildet **{bench}** am besten nach? R² gesamt: **{r2:.2f}**")
        if T >= win + 3:
            rows = [style_fit(R.values[t - win:t], y[t - win:t], lo, hi)[0] for t in range(win, T + 1)]
            roll = pd.DataFrame(rows, columns=assets, index=R.index[win - 1:].astype(str))
            fs = go.Figure([go.Scatter(x=roll.index, y=roll[a], stackgroup="one", name=a) for a in assets])
            fs.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fs, width="stretch")
            last = roll.iloc[-1].values
        else:
            st.caption("Zu wenige Monate für das rollierende Fenster.")
            last = np.full(len(assets), np.nan)
        st.dataframe(pd.DataFrame({"Ticker": assets, "Gesamtzeitraum %": w_all * 100, "Letztes Fenster %": last * 100})
                     .style.format({"Gesamtzeitraum %": "{:.1f}", "Letztes Fenster %": "{:.1f}"}),
                     hide_index=True, width="stretch")

with t2:
    st.subheader("Risikokennzahlen", help=INFO["risk"])
    rk = pd.DataFrame(M).T
    if y is not None:
        rk.loc[f"Benchmark {bench}"] = metrics(y, rf)
    pfmt = {c: "{:.1%}" for c in rk.columns if c != "Sharpe"} | {"Sharpe": "{:.2f}"}
    st.dataframe(rk.style.format(pfmt), width="stretch")
    st.subheader("Gewichte je Strategie")
    st.dataframe((pd.DataFrame(W, index=assets) * 100).style.format("{:.1f} %"), width="stretch")

with t3:
    st.subheader("Out-of-Sample-Backtest", help=INFO["bt"])
    s_ = round(T * split / 100)
    if s_ < 24 or T - s_ < 6:
        st.info("Zu wenige Monate für einen Backtest (mindestens 24 Training + 6 Test).")
    else:
        Wt, _, _ = optimise_all(R.iloc[:s_], lo, hi, rf, cur, None if y is None else y[:s_])
        test = R.iloc[s_:]
        st.caption(f"Training {R.index[0]}–{R.index[s_ - 1]} ({s_} Monate), Test {R.index[s_]}–{R.index[-1]} ({T - s_} Monate).")
        curves = {k: np.cumprod(1 + test.values @ w) * 100 for k, w in Wt.items()}
        bm = {k: metrics(test.values @ w, rf) for k, w in Wt.items()}
        if y is not None:
            curves["Benchmark"] = np.cumprod(1 + y[s_:]) * 100
            bm["Benchmark"] = metrics(y[s_:], rf)
        fb = go.Figure([go.Scatter(x=test.index.astype(str), y=v, name=k) for k, v in curves.items()])
        fb.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Wert (Start = 100)")
        st.plotly_chart(fb, width="stretch")
        st.dataframe(pd.DataFrame(bm).T[["Rendite p.a.", "Vol. p.a.", "Sharpe", "Max. Drawdown"]]
                     .style.format({"Rendite p.a.": "{:.1%}", "Vol. p.a.": "{:.1%}", "Sharpe": "{:.2f}",
                                    "Max. Drawdown": "{:.1%}"}), width="stretch")

with t4:
    st.subheader(f"Bootstrap: Endwert nach {hor} Jahren (Start {V:,.0f} €)", help=INFO["boot"])
    idx = np.random.default_rng(11).integers(0, T, (2000, hor * 12))
    sim = {}
    for k, w in W.items():
        end_v = np.prod(1 + (R.values @ w)[idx], axis=1) * V
        sim[k] = {"5 %": np.percentile(end_v, 5), "Median": np.median(end_v),
                  "95 %": np.percentile(end_v, 95), "P(Verlust)": (end_v < V).mean()}
    st.dataframe(pd.DataFrame(sim).T.style.format({"5 %": "{:,.0f} €", "Median": "{:,.0f} €",
                                                   "95 %": "{:,.0f} €", "P(Verlust)": "{:.0%}"}), width="stretch")

st.caption("Alle Optimierungen basieren auf Vergangenheitsdaten. Max-Sharpe reagiert stark auf Schätzfehler und "
           "liefert oft extreme Gewichte; vergleiche immer den Out-of-Sample-Backtest. Keine Anlageberatung.")

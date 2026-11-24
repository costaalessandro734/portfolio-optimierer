"""Portfolio-Check – einsteigerfreundlicher Portfolio-Optimierer mit Seitenmenü (Streamlit).

Start lokal:  streamlit run app.py
"""
import io
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio-Check", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# Größeres Interface
# =====================================================================
st.markdown("""
<style>
html { font-size: 18px; }
.block-container { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1 { font-size: 2.6rem !important; }
h2 { font-size: 1.9rem !important; }
h3 { font-size: 1.45rem !important; }
[data-testid="stMetricValue"] { font-size: 2.3rem; }
[data-testid="stMetricLabel"] p { font-size: 1.05rem; }
[data-testid="stSidebarNav"] a span, [data-testid="stSidebarNavLink"] span { font-size: 1.05rem; }
button[data-baseweb="tab"] p { font-size: 1.1rem; }
.intro { font-size: 1.15rem; opacity: .85; max-width: 70ch; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Deutsche Zahlenformate
# =====================================================================
def de(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def sg(x, d=1):
    return ("+" if x > 0 else "") + de(x, d)


def eur(x):
    return de(x, 0) + " €"


def pc(x, d=1):
    return de(x * 100, d) + " %"


# =====================================================================
# Zustand (bleibt beim Seitenwechsel erhalten)
# =====================================================================
POPULAR = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN", "Alphabet (Google)": "GOOGL",
    "Meta (Facebook)": "META", "Tesla": "TSLA", "Coca-Cola": "KO", "Johnson & Johnson": "JNJ",
    "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE", "Deutsche Telekom": "DTE.DE",
    "Mercedes-Benz": "MBG.DE", "iShares MSCI World ETF": "EUNL.DE", "iShares S&P 500 ETF": "SXR8.DE",
    "Gold (ETF)": "GLD",
}
NAME = {v: k for k, v in POPULAR.items()}
ss = st.session_state
ss.setdefault("names", {})        # Ticker -> Name aus der Suche
ss.setdefault("amounts", {})      # Ticker -> Betrag in €
ss.setdefault("bounds", {})       # Ticker -> (Min, Max) in %
ss.setdefault("imp", 0)
ss.setdefault("prefer_de", True)
ss.setdefault("replace_sel", True)
ss.setdefault("sel", [POPULAR[n] for n in ["Apple", "Microsoft", "iShares MSCI World ETF", "Allianz", "Coca-Cola"]])
for k in ("sel", "bulk", "prefer_de", "replace_sel", "q"):      # Widgets auf anderen Seiten nicht vergessen
    if k in ss:
        ss[k] = ss[k]


def label(t):
    return NAME.get(t) or ss.names.get(t) or t


GOALS = {
    "Möglichst ruhig (wenig Schwankung)": "Min-Varianz",
    "Bestes Verhältnis von Gewinn zu Risiko": "Max-Sharpe",
    "Einfach gleichmäßig verteilen": "Gleichgewichtet",
    "Benchmark nachbilden (Style Analysis)": "Benchmark-Stil",
}

GLOSSAR = {
    "Rendite / Ø Gewinn pro Jahr": "Wie viel das Portfolio im Durchschnitt pro Jahr gewachsen ist.",
    "Schwankung (Volatilität)": "Wie stark der Wert hoch und runter geht. 15 % heißt grob: In einem normalen Jahr kann "
                                "es leicht 15 % besser oder schlechter laufen als im Durchschnitt.",
    "Schlimmster Absturz (Max. Drawdown)": "Der größte Verlust vom Höchststand bis zum Tiefpunkt. 30 % heißt: Aus "
                                           "10.000 € wurden zwischenzeitlich 7.000 €.",
    "Sharpe Ratio / Gewinn pro Risiko": "Gewinn pro Einheit Schwankung. Je höher, desto besser wird das Risiko „bezahlt“.",
    "Korrelation": "Wie ähnlich sich zwei Wertpapiere bewegen, von −1 (gegenläufig) bis +1 (immer gleich). Niedrige "
                   "Werte sind gut für die Streuung.",
    "Risikobeitrag": "Welcher Anteil der Schwankung deines Portfolios von einem Wertpapier kommt.",
    "Backtest / Realitäts-Check": "Der Vorschlag wird nur mit älteren Daten berechnet und mit neueren Daten geprüft, "
                                  "die er nicht kannte. So sieht man, ob er wirklich geholfen hätte.",
    "Simulation (Bootstrap)": "Aus den echten Monaten der Vergangenheit werden tausende mögliche Zukünfte "
                              "zusammengewürfelt. Das zeigt eine Spanne, keine Vorhersage.",
    "Efficient Frontier": "Für jedes Risiko die höchste Rendite, die mit deinen Wertpapieren möglich gewesen wäre.",
    "Style Analysis": "Welche Mischung deiner Wertpapiere einen Index (z. B. den DAX) am besten nachbildet.",
    "VaR / CVaR": "Value at Risk: Monatsverlust, der in 19 von 20 Monaten nicht überschritten wurde. CVaR: "
                  "durchschnittlicher Verlust in den übrigen, schlechtesten Monaten.",
    "Ticker / ISIN": "Ticker ist das Börsenkürzel bei Yahoo (z. B. AAPL, SAP.DE). Die ISIN ist die weltweit "
                     "eindeutige Nummer eines Wertpapiers (z. B. US0378331005) – steht in deinem Depot.",
}

# =====================================================================
# Daten laden & Suche
# =====================================================================
def to_monthly(s):
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
    for fn in (load_yahoo, load_stooq):
        try:
            s = fn(ticker, start, end)
            if s is not None and len(s) >= 13:
                return s
        except Exception:
            pass
    return None


def parse_upload(file):
    raw = file.getvalue()
    if file.name.lower().endswith((".html", ".htm")):
        df = max(pd.read_html(io.BytesIO(raw)), key=len)
    else:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    cols = [(" ".join(map(str, c)) if isinstance(c, tuple) else str(c)).lower() for c in df.columns]
    di = next((i for i, c in enumerate(cols) if "date" in c or "datum" in c or c == "index"), 0)
    pi = next((i for i, c in enumerate(cols) if c.startswith("adj") or c.endswith("adjusted")), None)
    if pi is None:
        pi = next((i for i, c in enumerate(cols) if any(k in c for k in ("close", "schluss", "kurs", "price"))), 1)
    d = df.iloc[:, di].astype(str)
    dates = pd.to_datetime(d, errors="coerce", dayfirst=d.str.contains(r"^\d{1,2}\.\d{1,2}\.\d{4}").any(), format="mixed")
    p = df.iloc[:, pi].astype(str).str.strip()
    if p.str.match(r"^-?\d{1,3}(\.\d{3})*,\d+$").any():
        p = p.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        p = p.str.replace(",", "", regex=False)
    s = pd.Series(pd.to_numeric(p, errors="coerce").values, index=dates)
    return to_monthly(s[s.index.notna()])


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def search(q):
    try:
        quotes = yf.Search(q, max_results=10, news_count=0, lists_count=0, recommended=0, raise_errors=False).quotes
    except Exception:
        return []
    out = []
    for x in quotes or []:
        if x.get("symbol") and x.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "FUTURE"):
            out.append({"sym": x["symbol"], "name": x.get("shortname") or x.get("longname") or x["symbol"],
                        "ex": x.get("exchDisp") or x.get("exchange", ""), "typ": x.get("typeDisp") or x.get("quoteType", "")})
    return out


def add_ticker(sym, name):
    ss.names[sym] = name
    if sym not in ss.sel:
        ss.sel = ss.sel + [sym]


def parse_amount(s):
    s = s.replace("€", "").replace(" ", "").strip()
    if not s:
        return None
    if "," in s or re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def import_list():
    ok, bad = [], []
    if ss.replace_sel:
        ss.sel, ss.amounts = [], {}
    for line in ss.bulk.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ";").split(";")]
        res = search(parts[0])
        hit = next((x for x in res if x["sym"].endswith(".DE")), None) if ss.prefer_de else None
        hit = hit or (res[0] if res else None)
        if hit is None:
            bad.append(parts[0])
            continue
        add_ticker(hit["sym"], hit["name"])
        amt = parse_amount(parts[1]) if len(parts) > 1 else None
        if amt is not None:
            ss.amounts[hit["sym"]] = amt
        ok.append(f"{parts[0]} → {hit['name']} ({hit['sym']})")
    ss.imp += 1
    ss.import_msg = (ok, bad)


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


def style_fit(X, y, lo, hi):
    w = solve(lambda w: 1e4 * np.sum((y - X @ w) ** 2), X.shape[1], lo, hi)
    return w, 1 - np.sum((y - X @ w) ** 2) / np.sum((y - y.mean()) ** 2)


def proj(v, lo, hi):
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


@st.cache_data(show_spinner="Rechne Strategien, Simulation und Backtest …")
def core(R, y, cur, lo, hi, rf, split, hor, V):
    T = len(R)
    W, mu, S = optimise_all(R, lo, hi, rf, cur, y)
    M = {k: metrics(R.values @ w, rf) for k, w in W.items()}
    idx = np.random.default_rng(11).integers(0, T, (2000, hor * 12))
    SIM, BANDS = {}, {}
    for k, w in W.items():
        paths = V * np.cumprod(1 + (R.values @ w)[idx], axis=1)
        v = paths[:, -1]
        SIM[k] = {"5 %": np.percentile(v, 5), "Median": np.median(v), "95 %": np.percentile(v, 95),
                  "P(Verlust)": (v < V).mean()}
        BANDS[k] = np.percentile(paths, [5, 50, 95], axis=0)
    s_ = round(T * split / 100)
    BT = CURVES = None
    if s_ >= 24 and T - s_ >= 6:
        Wt, _, _ = optimise_all(R.iloc[:s_], lo, hi, rf, cur, None if y is None else y[:s_])
        test = R.iloc[s_:]
        BT = {k: metrics(test.values @ w, rf) for k, w in Wt.items()}
        CURVES = {k: np.cumprod(1 + test.values @ w) * 100 for k, w in Wt.items()}
        if y is not None:
            BT["Benchmark"], CURVES["Benchmark"] = metrics(y[s_:], rf), np.cumprod(1 + y[s_:]) * 100
    return dict(W=W, mu=mu, S=S, M=M, SIM=SIM, BANDS=BANDS, BT=BT, CURVES=CURVES, s_=s_)


@st.cache_data(show_spinner="Berechne Efficient Frontier …")
def frontier_mc(mu, S, lo, hi, n_mc):
    n = len(mu)
    r_min = min_var(mu, S, lo, hi) @ mu
    r_max = solve(lambda w: -(w @ mu), n, lo, hi) @ mu
    pts = []
    for t in np.linspace(r_min, r_max, 40):
        w = solve(lambda w: w @ S @ w, n, lo, hi, ({"type": "ineq", "fun": lambda w, t=t: w @ mu - t},))
        pts.append((np.sqrt(w @ S @ w), w @ mu))
    Wr = np.array([proj(w, lo, hi) for w in np.random.default_rng(5).dirichlet(np.ones(n), n_mc)])
    return np.array(pts), Wr @ mu, np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr))


@st.cache_data(show_spinner="Berechne Style Analysis …")
def style_roll(R, y, lo, hi, win):
    w_all, r2 = style_fit(R.values, y, lo, hi)
    T = len(R)
    rows = [style_fit(R.values[t - win:t], y[t - win:t], lo, hi)[0] for t in range(win, T + 1)] if T >= win + 3 else []
    return w_all, r2, (np.array(rows) if rows else None)


# =====================================================================
# Interpretation in Alltagssprache
# =====================================================================
def explain(lines):
    with st.container(border=True):
        st.markdown("**Was bedeutet das?**\n\n" + "\n\n".join(l for l in lines if l))


def lvl_vol(v):
    return ("sehr ruhig" if v < 0.08 else "eher ruhig" if v < 0.14 else "mittel schwankend" if v < 0.22
            else "stark schwankend")


def lvl_sh(s):
    return ("schwach" if s < 0.3 else "in Ordnung" if s < 0.7 else "gut" if s < 1.2
            else "außergewöhnlich gut – so hohe Werte halten selten dauerhaft")


def reasons(i, up, mu, S, C, target):
    if target == "Gleichgewichtet":
        return "Grund: Bei gleichmäßiger Verteilung bekommt jedes Wertpapier gleich viel"
    vol = np.sqrt(np.diag(S))
    shp, c = mu / vol, (C.sum(1) - 1) / max(1, len(mu) - 1)
    r_vol = (vol[i] < np.median(vol)) if up else (vol[i] > np.median(vol))
    r_sh = (shp[i] > np.median(shp)) if up else (shp[i] < np.median(shp))
    r_c = (c[i] < np.median(c)) if up else (c[i] > np.median(c))
    t_vol = f"schwankt {'weniger' if up else 'stärker'} als deine anderen Werte ({pc(vol[i])} pro Jahr)"
    t_sh = f"{'viel' if up else 'wenig'} Gewinn im Verhältnis zum Risiko (Ø {pc(mu[i])} pro Jahr)"
    t_c = ("bewegt sich oft anders als deine übrigen Werte – das fängt Verluste ab" if up
           else "bewegt sich sehr ähnlich wie deine übrigen Werte – streut das Risiko kaum")
    order = [(r_vol, t_vol), (r_c, t_c), (r_sh, t_sh)] if target == "Min-Varianz" else \
            [(r_sh, t_sh), (r_c, t_c), (r_vol, t_vol)]
    hits = [t for ok, t in order if ok][:2]
    return ("Grund: " + "; ".join(hits)) if hits else "Grund: passt rechnerisch besser in die Mischung"


def head(title, intro, help_txt=None):
    st.title(title, help=help_txt)
    st.markdown(f"<p class='intro'>{intro}</p>", unsafe_allow_html=True)


# =====================================================================
# Seitenleiste: Einstellungen (immer sichtbar)
# =====================================================================
sb = st.sidebar
expert = sb.toggle("Expertenmodus", key="expert",
                   help="Schaltet zusätzliche Seiten (Efficient Frontier, Style Analysis, Strategievergleich) und "
                        "Einstellungen frei, z. B. eigene Grenzen pro Wertpapier und einen Benchmark.")
goal_opts = list(GOALS)[:3] + (list(GOALS)[3:] if expert else [])
if ss.get("goal") not in goal_opts:
    ss.goal = goal_opts[1]
sb.radio("Dein Ziel", goal_opts, key="goal",
         help="Ruhig: geringste Schwankung, schaut nicht auf Gewinne und ist daher am robustesten. Bestes "
              "Verhältnis: maximiert die Sharpe Ratio, reagiert aber stark auf Zufälle in den Vergangenheitsdaten. "
              "Gleichmäßig: jedes Wertpapier gleich viel. Benchmark: Mischung, die einen Index nachbildet.")
years = sb.slider("Jahre zurückschauen", 3, 25, 10, key="years",
                  help="So viele Jahre Kursgeschichte werden ausgewertet. Mehr Jahre = mehr Erfahrung, aber sehr "
                       "alte Daten sagen weniger über heute aus.")
hor = sb.slider("Anlagedauer (Jahre)", 1, 30, 5, key="hor",
                help="Für diesen Zeitraum wird simuliert, was aus deinem Geld werden könnte.")
start, end = date.today() - timedelta(days=365 * years), date.today()
rf, n_mc, split, win, bench_t, uploads = 0.02, 3000, 70, 36, "", None
if expert:
    with sb.expander("Experten-Einstellungen", expanded=True):
        if st.checkbox("Eigenen Zeitraum wählen", key="own_dates"):
            c1, c2 = st.columns(2)
            start = c1.date_input("Von", start, key="d_from")
            end = c2.date_input("Bis", end, key="d_to")
        bench_t = st.text_input("Benchmark-Ticker", key="bench", placeholder="z. B. ^GSPC, ^GDAXI, ^KLSE",
                                help="Vergleichsindex für Style Analysis und Backtest.").strip()
        rf = st.number_input("Risikofreier Zins p.a. %", value=2.0, step=0.1, key="rf",
                             help="Rendite einer sicheren Anlage, wird in der Sharpe Ratio abgezogen.") / 100
        n_mc = st.select_slider("Zufallsportfolios", [1000, 3000, 6000], 3000, key="nmc",
                                help="Anzahl Zufallsportfolios für die Efficient Frontier.")
        split = st.slider("Backtest: Trainingsanteil %", 30, 90, 70, 5, key="split",
                          help="Auf diesem Anteil wird optimiert, auf dem Rest getestet.")
        win = int(st.number_input("Style-Fenster (Monate)", 12, 120, 36, key="win",
                                  help="Länge des rollierenden Fensters der Style Analysis."))
        uploads = st.file_uploader("Kursdateien (CSV/HTML)", accept_multiple_files=True, key="up",
                                   help="Notlösung, falls ein Ticker nicht geladen werden kann. Dateiname = Name; "
                                        "beginnt er mit ^, wird die Datei zum Benchmark.")


# =====================================================================
# Gemeinsame Berechnung für alle Seiten
# =====================================================================
def get_ctx():
    tickers = list(dict.fromkeys(ss.sel))
    series, failed = {}, []
    with st.spinner("Lade Kurse …"):
        for t in tickers + ([bench_t] if bench_t else []):
            s = load(t, start, end)
            if s is None:
                failed.append(t)
            else:
                series[t] = s
    bench = bench_t if bench_t in series else None
    for f in uploads or []:
        name = f.name.rsplit(".", 1)[0]
        try:
            series[name] = parse_upload(f)
            if name.startswith("^") and bench is None:
                bench = name
            elif name not in tickers:
                tickers.append(name)
            failed = [x for x in failed if x != name]
        except Exception as e:
            st.warning(f"{f.name} konnte nicht gelesen werden: {e}")
    assets = [t for t in tickers if t in series and t != bench]
    ctx = dict(failed=failed, assets=assets, bench=bench, series=series)
    if len(assets) < 2:
        return ctx | {"err": "Wähle auf der Seite **Mein Portfolio** mindestens 2 Wertpapiere aus."}
    prices = pd.concat({t: series[t] for t in assets + ([bench] if bench else [])}, axis=1).dropna()
    R_all = prices.pct_change().dropna()
    R, y = R_all[assets], (R_all[bench].values if bench else None)
    T = len(R)
    if T < 25:
        return ctx | {"err": f"Für diese Auswahl gibt es nur {T} gemeinsame Monate Kursgeschichte (mindestens 25 "
                             "nötig). Schau mehr Jahre zurück oder entferne sehr junge Wertpapiere."}
    vals = np.array([float(ss.amounts.get(t, 1000.0)) for t in assets])
    V = vals.sum()
    if V <= 0:
        return ctx | {"err": "Trag auf der Seite **Mein Portfolio** bei mindestens einem Wertpapier einen Betrag ein."}
    cur = vals / V
    dmax = 40.0 if len(assets) >= 3 else 100.0
    if expert:
        lo = np.array([ss.bounds.get(t, (0.0, dmax))[0] for t in assets]) / 100
        hi = np.array([ss.bounds.get(t, (0.0, dmax))[1] for t in assets]) / 100
        if lo.sum() > 1.0001 or hi.sum() < 0.9999 or (lo > hi).any():
            return ctx | {"err": "Grenzen unzulässig: Summe aller Min ≤ 100 % ≤ Summe aller Max, und Min ≤ Max."}
    else:
        lo, hi = np.zeros(len(assets)), np.full(len(assets), dmax / 100)
    target = GOALS[ss.goal]
    if target == "Benchmark-Stil" and y is None:
        st.warning("Für „Benchmark nachbilden“ fehlt ein Benchmark – es wird „Bestes Verhältnis“ verwendet.")
        target = "Max-Sharpe"
    res = core(R, y, cur, lo, hi, rf, split, hor, V)
    return ctx | res | dict(R=R, y=y, T=T, V=V, cur=cur, lo=lo, hi=hi, dmax=dmax, target=target,
                            tw=res["W"][target], names=[label(t) for t in assets], C=R.corr().values,
                            goal_txt=ss.goal if target == GOALS[ss.goal] else "Bestes Verhältnis von Gewinn zu Risiko")


def ready():
    c = get_ctx()
    if c["failed"]:
        st.warning("Für **" + ", ".join(c["failed"]) + "** wurden keine Kurse gefunden – sie werden ausgelassen.")
    if "err" in c:
        st.info(c["err"])
        st.page_link(P_INPUT, label="Zu „Mein Portfolio“", icon=":material/edit:")
        st.stop()
    return c


# =====================================================================
# SEITEN
# =====================================================================
def page_input():
    head("Mein Portfolio", "Trag ein, welche Aktien oder ETFs du hast und wie viel Geld jeweils darin steckt. "
                           "Danach findest du links im Menü alle Auswertungen und Simulationen.")
    st.header("1. Welche Wertpapiere hast du?")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        q = st.text_input("Wertpapier suchen", key="q",
                          placeholder="Name, ISIN oder Ticker – z. B. Rheinmetall, US0378331005",
                          help="Durchsucht alle Aktien, ETFs, Fonds und Indizes bei Yahoo Finance weltweit. Du kannst "
                               "den Namen, die ISIN (steht in deinem Depot) oder den Ticker eingeben.")
        if q and q.strip():
            res = search(q.strip())
            if not res:
                st.caption("Nichts gefunden. Probier einen anderen Namen, die ISIN oder den englischen Firmennamen.")
            else:
                i = st.selectbox("Treffer", range(len(res)),
                                 format_func=lambda i: f"{res[i]['name']} ({res[i]['sym']}) · {res[i]['ex']} · {res[i]['typ']}",
                                 help="Viele Wertpapiere gibt es an mehreren Börsen. Endung .DE = Xetra in Euro, ohne "
                                      "Endung meist USA in Dollar. Für Euro-Anleger ist .DE oft passender, weil dann "
                                      "keine Wechselkursschwankungen eingerechnet werden.")
                st.button("Hinzufügen", on_click=add_ticker, args=(res[i]["sym"], res[i]["name"]), type="primary")
    with c2:
        with st.expander("Ganze Liste auf einmal einfügen (z. B. aus deinem Depot)"):
            st.text_area("Eine Zeile pro Wertpapier: Name, ISIN oder Ticker; Betrag in €", key="bulk", height=140,
                         placeholder="US0378331005; 2500\nRheinmetall; 1.200\niShares Core MSCI World; 5000",
                         help="Der Betrag ist optional. Trenne Name und Betrag mit Semikolon (;) oder kopiere zwei "
                              "Spalten direkt aus Excel. Jede Zeile wird gesucht und der beste Treffer übernommen.")
            st.checkbox("Deutsche Börse (Xetra, Euro) bevorzugen", key="prefer_de",
                        help="Gibt es das Wertpapier an der Xetra, wird diese Notierung genommen (Endung .DE).")
            st.checkbox("Bisherige Auswahl ersetzen", key="replace_sel",
                        help="An: Nur die Wertpapiere aus deiner Liste werden ausgewertet. Aus: Sie kommen dazu.")
            st.button("Übernehmen", on_click=import_list)
            if "import_msg" in ss:
                ok, bad = ss.import_msg
                if ok:
                    st.success("Übernommen:\n" + "\n".join("- " + x for x in ok))
                if bad:
                    st.warning("Nicht gefunden: " + ", ".join(bad))
    opts = list(dict.fromkeys(list(POPULAR.values()) + list(ss.names) + ss.sel))
    st.multiselect("Deine Auswahl", opts, key="sel",
                   format_func=lambda t: f"{label(t)} ({t})" if label(t) != t else t,
                   help="Alle Wertpapiere, die ausgewertet werden. Mit × entfernst du einen Wert. Bekannte Werte "
                        "kannst du hier auch direkt auswählen. Empfehlung: höchstens 20–30 Werte.")

    st.header("2. Wie viel Geld steckt jeweils darin?")
    c = get_ctx()
    if c["failed"]:
        st.warning("Für **" + ", ".join(c["failed"]) + "** wurden keine Kurse gefunden. Prüfe die Auswahl.")
    assets = c["assets"]
    if len(assets) < 1:
        st.info("Füge oben Wertpapiere hinzu.")
        return
    dmax = 40.0 if len(assets) >= 3 else 100.0
    base = pd.DataFrame({"Wertpapier": [label(t) for t in assets], "Ticker": assets,
                         "Wert in €": [float(ss.amounts.get(t, 1000.0)) for t in assets]})
    cfg = {"Wert in €": st.column_config.NumberColumn("Wert in €", min_value=0.0, step=100.0, format="%.0f €",
                                                      help="Aktueller Wert dieser Position. Grobe Werte reichen.")}
    if expert:
        base["Min %"] = [ss.bounds.get(t, (0.0, dmax))[0] for t in assets]
        base["Max %"] = [ss.bounds.get(t, (0.0, dmax))[1] for t in assets]
        for col in ("Min %", "Max %"):
            cfg[col] = st.column_config.NumberColumn(col, min_value=0.0, max_value=100.0, step=1.0,
                                                     help="Grenzen für das Gewicht in allen Optimierungen.")
    ed = st.data_editor(base, key=f"ed_{expert}_{ss.imp}_" + "_".join(assets), hide_index=True,
                        disabled=["Wertpapier", "Ticker"], column_config=cfg, width="stretch")
    for _, row in ed.iterrows():
        ss.amounts[row["Ticker"]] = float(row["Wert in €"] or 0)
        if expert:
            ss.bounds[row["Ticker"]] = (float(row["Min %"] or 0), float(row["Max %"] if row["Max %"] == row["Max %"] else 100))
    if not expert and dmax < 100:
        st.caption(f"Damit nicht alles auf eine Karte gesetzt wird, steckt der Vorschlag höchstens {de(dmax, 0)} % in "
                   "ein einzelnes Wertpapier. Im Expertenmodus kannst du die Grenzen selbst festlegen.")

    st.header("3. Was ist dir wichtiger?")
    st.write(f"Aktuell gewählt: **{ss.goal}**. Ändern kannst du das jederzeit links in der Seitenleiste unter "
             "**Dein Ziel**.")
    c = get_ctx()
    if "err" in c:
        st.info(c["err"])
        return
    st.success(f"Alles bereit: {len(c['assets'])} Wertpapiere, {eur(c['V'])}, {c['T']} Monate Kursgeschichte.")
    st.page_link(P_OVER, label="Weiter zu Übersicht & Vorschlag", icon=":material/arrow_forward:")


def page_overview():
    c = ready()
    R, cur, tw, V, M, W, target = c["R"], c["cur"], c["tw"], c["V"], c["M"], c["W"], c["target"]
    assets, names, mu, S, C, hi = c["assets"], c["names"], c["mu"], c["S"], c["C"], c["hi"]
    tm, cm = M[target], M["Aktuell"]
    head("Übersicht & Vorschlag", f"Ziel: <b>{c['goal_txt']}</b>. Hier siehst du auf einen Blick, wie dein Portfolio "
                                  f"in den letzten {years} Jahren lief und was du ändern könntest.")
    d = tw - cur
    moves = [(names[i], d[i] * V) for i in range(len(assets)) if abs(d[i]) >= 0.005]
    if not moves:
        st.success("Dein Portfolio ist für dieses Ziel schon gut aufgeteilt. Du musst nichts ändern.")
    else:
        buy = [f"**{n}** (+{eur(x)})" for n, x in moves if x > 0]
        sell = [f"**{n}** (−{eur(-x)})" for n, x in moves if x < 0]
        st.info("**Vorschlag:** " + ("Mehr kaufen: " + ", ".join(buy) + ". " if buy else "") +
                ("Weniger halten: " + ", ".join(sell) + "." if sell else ""))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ø Gewinn pro Jahr", pc(tm["Rendite p.a."]), f"{sg((tm['Rendite p.a.'] - cm['Rendite p.a.']) * 100)} Prozentpunkte",
              help="Durchschnittliches Wachstum pro Jahr mit dem Vorschlag. Darunter: Unterschied zu deinem jetzigen Portfolio.")
    k2.metric("Schwankung", pc(tm["Vol. p.a."]), f"{sg((tm['Vol. p.a.'] - cm['Vol. p.a.']) * 100)} Prozentpunkte",
              delta_color="inverse", help="Wie stark der Wert hoch und runter geht. Weniger ist ruhiger.")
    k3.metric("Schlimmster Absturz", "−" + pc(tm["Max. Drawdown"]),
              f"{sg((tm['Max. Drawdown'] - cm['Max. Drawdown']) * 100)} Prozentpunkte", delta_color="inverse",
              help="Größter Verlust vom Höchststand bis zum Tiefpunkt. Weniger ist besser.")
    k4.metric("Gewinn pro Risiko", de(tm["Sharpe"], 2), sg(tm["Sharpe"] - cm["Sharpe"], 2),
              help="Sharpe Ratio: Wie gut das eingegangene Risiko „bezahlt“ wurde. Höher ist besser.")
    st.caption(f"Große Zahl = mit Vorschlag. Dein jetziges Portfolio: Ø {pc(cm['Rendite p.a.'])} pro Jahr, Schwankung "
               f"{pc(cm['Vol. p.a.'])}, schlimmster Absturz −{pc(cm['Max. Drawdown'])}. Historisch und in-sample.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Was du ändern müsstest",
                     help=f"Beträge in Euro, bezogen auf dein Gesamtvermögen von {eur(V)}. Änderungen unter 0,5 % "
                          "werden als „behalten“ gezeigt. Gebühren und Steuern sind nicht eingerechnet.")
        act = [("behalten" if abs(x) < 0.005 else (f"+{eur(x * V)} kaufen" if x > 0 else f"−{eur(-x * V)} verkaufen"))
               for x in d]
        st.dataframe(pd.DataFrame({"Wertpapier": names, "Jetzt": [eur(x * V) for x in cur],
                                   "Vorschlag": [eur(w * V) for w in tw], "Was tun?": act}),
                     hide_index=True, width="stretch", height=min(600, 42 + 36 * len(assets)))
    with right:
        st.subheader("Aufteilung jetzt und mit Vorschlag", help="Anteil jedes Wertpapiers an deinem Geld.")
        fig = go.Figure()
        fig.add_bar(y=names, x=cur, orientation="h", name="Jetzt", marker_color="#b8c4e8")
        fig.add_bar(y=names, x=tw, orientation="h", name="Vorschlag", marker_color="#1240e0")
        fig.update_layout(barmode="group", height=80 + 60 * len(assets), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08), yaxis_autorange="reversed",
                          font=dict(size=15))
        st.plotly_chart(fig, width="stretch")

    avg_c = round((C.sum() - len(assets)) / max(1, len(assets) * (len(assets) - 1)), 2) + 0.0
    big = int(np.argmax(cur))
    lines = [
        f"**Dein jetziges Portfolio** ist {lvl_vol(cm['Vol. p.a.'])} (Schwankung {pc(cm['Vol. p.a.'])}). Es ist im "
        f"Schnitt um {pc(cm['Rendite p.a.'])} pro Jahr gewachsen. Der schlimmste Einbruch lag bei "
        f"−{pc(cm['Max. Drawdown'])}: Aus {eur(V)} wären zwischenzeitlich {eur(V * (1 - cm['Max. Drawdown']))} geworden. "
        f"Das Verhältnis von Gewinn zu Risiko war {lvl_sh(cm['Sharpe'])}.",
        (f"**Streuung:** Deine Wertpapiere bewegten sich stark gleichzeitig (Ø Korrelation {de(avg_c, 2)}). Fällt eins, "
         "fallen die anderen meist mit. Werte aus anderen Branchen, Ländern oder Anlageklassen würden mehr bringen.")
        if avg_c > 0.6 else
        (f"**Streuung:** Deine Wertpapiere liefen oft unterschiedlich (Ø Korrelation {de(avg_c, 2)}). Das ist gut: "
         "Verluste des einen werden teilweise von den anderen ausgeglichen.") if avg_c < 0.3 else
        (f"**Streuung:** Deine Wertpapiere hängen mittelstark zusammen (Ø Korrelation {de(avg_c, 2)}). Die Streuung "
         "hilft etwas, aber in Krisen fallen sie wahrscheinlich gemeinsam."),
        (f"**Klumpenrisiko:** {pc(cur[big], 0)} deines Geldes stecken in {names[big]}. Läuft dieser eine Wert "
         "schlecht, zieht er das ganze Portfolio mit.") if cur[big] > 0.4 else "",
    ]
    if moves:
        dr, dv = tm["Rendite p.a."] - cm["Rendite p.a."], tm["Vol. p.a."] - cm["Vol. p.a."]
        trade = ("Der Vorschlag wäre in der Vergangenheit **klar besser** gewesen: mehr Gewinn bei weniger Schwankung."
                 if dr >= 0 and dv <= 0 else
                 "Der Vorschlag bringt **mehr Gewinn, aber auch mehr Schwankung**. Du tauschst Ruhe gegen Rendite – "
                 "das lohnt sich nur, wenn du Einbrüche gelassen aussitzen kannst." if dr >= 0 else
                 "Der Vorschlag ist **ruhiger, bringt dafür etwas weniger Gewinn**. Sinnvoll, wenn dir Verluste "
                 "schlaflose Nächte bereiten." if dv <= 0 else
                 "Der Vorschlag wäre hier eher schlechter – das kommt selten vor; prüfe Eingaben und Grenzen.")
        ups = sorted([i for i in range(len(assets)) if d[i] >= 0.005], key=lambda i: -d[i])[:3]
        downs = sorted([i for i in range(len(assets)) if d[i] <= -0.005], key=lambda i: d[i])[:3]
        why = [f"- **{names[i]}** bekommt mehr ({pc(cur[i], 0)} → {pc(tw[i], 0)}). {reasons(i, True, mu, S, C, target)}."
               for i in ups] + \
              [f"- **{names[i]}** bekommt weniger ({pc(cur[i], 0)} → {pc(tw[i], 0)}). {reasons(i, False, mu, S, C, target)}."
               for i in downs]
        cap_i = [i for i in range(len(assets)) if tw[i] >= hi[i] - 0.001 and hi[i] < 1]
        lines += [trade, "**Warum diese Änderungen?**\n" + "\n".join(why),
                  (f"Bei {', '.join(names[i] for i in cap_i)} stößt der Vorschlag an die Obergrenze von "
                   f"{pc(hi[cap_i[0]], 0)}. Ohne Grenze würde das Modell noch mehr hineinstecken – ein Zeichen, dass es "
                   "stark auf die gute Vergangenheit dieses Werts setzt. Genau davor schützt die Grenze.")
                  if cap_i and target == "Max-Sharpe" else ""]
    explain(lines)
    st.subheader("Weiter zu den Simulationen")
    cols = st.columns(4)
    for col, (pg, txt) in zip(cols, [(P_GROWTH, "Wie sich dein Geld entwickelt hätte"),
                                     (P_FUTURE, "Was in Zukunft passieren könnte"),
                                     (P_CHECK, "Hätte der Vorschlag wirklich geholfen?"),
                                     (P_RISK, "Woher dein Risiko kommt")]):
        with col.container(border=True):
            st.page_link(pg)
            st.caption(txt)


def page_growth():
    c = ready()
    R, cur, tw, V = c["R"], c["cur"], c["tw"], c["V"]
    head("Wertentwicklung", f"So hätte sich dein Geld in den letzten {c['T']} Monaten entwickelt – mit deiner "
                            "jetzigen Aufteilung (rot) und mit dem Vorschlag (blau).")
    dates = R.index.to_timestamp()
    rp_c, rp_t = R.values @ cur, R.values @ tw
    vc, vt = V * np.cumprod(1 + rp_c), V * np.cumprod(1 + rp_t)
    t1, t2, t3 = st.tabs(["Vermögen", "Rückschläge", "Jahr für Jahr"])
    with t1:
        st.subheader(f"Was aus {eur(V)} geworden wäre",
                     help="Angenommen, du hättest am Anfang des Zeitraums so angelegt und jeden Monat auf die gleiche "
                          "Aufteilung zurückgesetzt. Achtung: Der Vorschlag kennt diese Vergangenheit bereits und "
                          "sieht hier deshalb besonders gut aus – ehrlicher ist der Realitäts-Check.")
        f1 = go.Figure()
        f1.add_scatter(x=dates, y=vc, name="Jetziges Portfolio", line=dict(color="#e8361b", width=3))
        f1.add_scatter(x=dates, y=vt, name="Vorschlag", line=dict(color="#1240e0", width=3))
        f1.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Wert in €", separators=",.",
                         legend=dict(orientation="h", y=1.08), hovermode="x unified", font=dict(size=15))
        st.plotly_chart(f1, width="stretch")
        a, b = st.columns(2)
        a.metric("Jetziges Portfolio am Ende", eur(vc[-1]), f"{sg((vc[-1] / V - 1) * 100, 0)} %",
                 help="Endwert und Gesamtgewinn über den ganzen Zeitraum.")
        b.metric("Vorschlag am Ende", eur(vt[-1]), f"{sg((vt[-1] / V - 1) * 100, 0)} %",
                 help="Endwert und Gesamtgewinn über den ganzen Zeitraum – im Nachhinein berechnet.")
        explain([f"Mit deinem jetzigen Portfolio wären aus {eur(V)} **{eur(vc[-1])}** geworden, mit dem Vorschlag "
                 f"**{eur(vt[-1])}**.",
                 "Vorsicht beim Vergleich: Der Vorschlag wurde mit genau diesen Kursen berechnet. Er weiß also schon, "
                 "welche Werte gut liefen. Ob er auch ohne dieses Wissen besser gewesen wäre, zeigt der Realitäts-Check."])
    with t2:
        st.subheader("Wie tief ging es zwischendurch runter?",
                     help="Abstand zum bisherigen Höchststand. 0 % = gerade auf Rekordhoch, −20 % = 20 % unter dem "
                          "letzten Hoch. Je tiefer und länger die Täler, desto härter war es, durchzuhalten.")
        f2 = go.Figure()
        for v, n, col in [(vc, "Jetziges Portfolio", "#e8361b"), (vt, "Vorschlag", "#1240e0")]:
            f2.add_scatter(x=dates, y=v / np.maximum.accumulate(v) - 1, name=n, line=dict(color=col), fill="tozeroy",
                           opacity=0.5)
        f2.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                         legend=dict(orientation="h", y=1.08), hovermode="x unified", font=dict(size=15))
        st.plotly_chart(f2, width="stretch")
        dd = vc / np.maximum.accumulate(vc) - 1
        rec = 0
        low = int(np.argmin(dd))
        after = np.where(dd[low:] >= -1e-9)[0]
        rec_txt = (f"Danach dauerte es {after[0]} Monate, bis der alte Höchststand wieder erreicht war."
                   if len(after) else "Der alte Höchststand ist bis heute nicht wieder erreicht.")
        explain([f"Der tiefste Punkt deines jetzigen Portfolios war im **{c['R'].index[low]}** mit "
                 f"**−{pc(-dd[low])}** unter dem vorherigen Hoch. {rec_txt}",
                 f"Insgesamt lag es in {int((dd < -0.1).sum())} von {c['T']} Monaten mehr als 10 % unter seinem "
                 "Höchststand. Frag dich ehrlich: Hättest du in so einer Phase verkauft? Wenn ja, ist ein ruhigeres "
                 "Portfolio besser für dich."])
    with t3:
        st.subheader("Gewinn oder Verlust in jedem Kalenderjahr",
                     help="Erstes und letztes Jahr können unvollständig sein. So siehst du, wie oft es Verlustjahre gab "
                          "und wie schlimm sie waren.")
        yc = pd.Series(rp_c, index=R.index).add(1).groupby(R.index.year).prod() - 1
        yt = pd.Series(rp_t, index=R.index).add(1).groupby(R.index.year).prod() - 1
        f3 = go.Figure()
        f3.add_bar(x=yc.index.astype(str), y=yc.values, name="Jetziges Portfolio", marker_color="#e8361b")
        f3.add_bar(x=yt.index.astype(str), y=yt.values, name="Vorschlag", marker_color="#1240e0")
        f3.update_layout(barmode="group", height=460, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                         legend=dict(orientation="h", y=1.08), font=dict(size=15))
        st.plotly_chart(f3, width="stretch")
        explain([f"Mit deinem jetzigen Portfolio gab es in **{int((yc > 0).sum())} von {len(yc)} Jahren** Gewinn. "
                 f"Bestes Jahr: {yc.idxmax()} mit {pc(yc.max())}, schlechtestes: {yc.idxmin()} mit {pc(yc.min())}.",
                 "Verlustjahre sind bei Aktien normal. Entscheidend ist, ob du sie aussitzen kannst, ohne zu verkaufen."])


def page_assets():
    c = ready()
    R, cur, names, assets = c["R"], c["cur"], c["names"], c["assets"]
    head("Meine Wertpapiere", "Jedes Wertpapier einzeln betrachtet: Welche treiben dein Portfolio an, welche bremsen es?")
    yr = (1 + R).groupby(R.index.year).prod() - 1
    rows, shp = [], []
    for i, t in enumerate(assets):
        m = metrics(R[t].values, rf)
        shp.append(m["Sharpe"])
        rows.append({"Wertpapier": names[i], "Ticker": t, "Dein Anteil": pc(cur[i], 0), "Ø Gewinn/Jahr": pc(m["Rendite p.a."]),
                     "Schwankung": pc(m["Vol. p.a."]), "Einschätzung": lvl_vol(m["Vol. p.a."]),
                     "Schlimmster Absturz": "−" + pc(m["Max. Drawdown"], 0),
                     "Bestes Jahr": pc(yr[t].max(), 0), "Schlechtestes Jahr": pc(yr[t].min(), 0),
                     "Gewinn pro Risiko": de(m["Sharpe"], 2)})
    H = st.column_config.TextColumn
    st.subheader("Kennzahlen im Überblick", help="Alle Werte beziehen sich auf den ausgewerteten Zeitraum.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(700, 42 + 36 * len(rows)),
                 column_config={
                     "Dein Anteil": H(help="Wie viel Prozent deines Geldes heute in diesem Wertpapier stecken."),
                     "Ø Gewinn/Jahr": H(help="Durchschnittliches Wachstum pro Jahr im ausgewerteten Zeitraum."),
                     "Schwankung": H(help="Wie stark der Kurs hoch und runter ging (Volatilität). Mehr = unruhiger."),
                     "Einschätzung": H(help="Einordnung der Schwankung: unter 8 % sehr ruhig, bis 14 % eher ruhig, "
                                            "bis 22 % mittel, darüber stark schwankend."),
                     "Schlimmster Absturz": H(help="Größter Verlust vom Hoch bis zum Tief. −50 % = zwischenzeitlich halbiert."),
                     "Bestes Jahr": H(help="Kalenderjahr mit dem höchsten Gewinn. Erstes/letztes Jahr evtl. unvollständig."),
                     "Schlechtestes Jahr": H(help="Kalenderjahr mit dem größten Verlust."),
                     "Gewinn pro Risiko": H(help="Sharpe Ratio: Gewinn im Verhältnis zur Schwankung. Über 1 sehr gut, "
                                                 "unter 0,3 schwach."),
                 })
    st.subheader("Gewinn und Schwankung im Vergleich",
                 help="Jeder Punkt ist ein Wertpapier. Rechts = schwankt stärker, oben = mehr Gewinn. Ideal ist oben "
                      "links: viel Gewinn bei wenig Schwankung. Die Punktgröße zeigt deinen Anteil.")
    vol, mu = np.sqrt(np.diag(c["S"])), c["mu"]
    fg = go.Figure(go.Scatter(x=vol, y=mu, mode="markers+text", text=names, textposition="top center",
                              marker=dict(size=14 + 60 * cur, color="#1240e0", opacity=0.75)))
    fg.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Schwankung pro Jahr",
                     yaxis_title="Ø Gewinn pro Jahr", xaxis_tickformat=".0%", yaxis_tickformat=".0%", font=dict(size=15))
    st.plotly_chart(fg, width="stretch")
    best, worst = int(np.argmax(shp)), int(np.argmin(shp))
    explain([f"Bestes Verhältnis von Gewinn zu Risiko: **{names[best]}** (Sharpe {de(shp[best], 2)}). Schwächstes: "
             f"**{names[worst]}** (Sharpe {de(shp[worst], 2)}).",
             f"Am ruhigsten war **{names[int(np.argmin(vol))]}** ({pc(vol.min())} Schwankung), am wildesten "
             f"**{names[int(np.argmax(vol))]}** ({pc(vol.max())}).",
             "Ein schwacher Einzelwert kann trotzdem sinnvoll sein, wenn er sich anders bewegt als der Rest und so "
             "Verluste abfedert – das siehst du auf der Seite **Risiko & Streuung**."])


def page_risk():
    c = ready()
    cur, names, S, C, V = c["cur"], c["names"], c["S"], c["C"], c["V"]
    head("Risiko & Streuung", "Woher kommt die Schwankung deines Portfolios, und sichern sich deine Wertpapiere "
                              "gegenseitig ab?")
    t1, t2 = st.tabs(["Wer bringt das Risiko?", "Zusammenhänge"])
    with t1:
        st.subheader("Anteil am Geld oder Anteil am Risiko?",
                     help="Ein Wertpapier kann 20 % deines Geldes ausmachen, aber 40 % deines Risikos verursachen – "
                          "wenn es stark schwankt und mit den anderen zusammen fällt (Risikobeitrag).")
        rc = cur * (S @ cur) / (cur @ S @ cur)
        f4 = go.Figure()
        f4.add_bar(y=names, x=cur, orientation="h", name="Anteil am Geld", marker_color="#b8c4e8")
        f4.add_bar(y=names, x=rc, orientation="h", name="Anteil am Risiko", marker_color="#e8361b")
        f4.update_layout(barmode="group", height=80 + 60 * len(names), margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08), yaxis_autorange="reversed",
                         font=dict(size=15))
        st.plotly_chart(f4, width="stretch")
        j, k = int(np.argmax(rc - cur)), int(np.argmin(rc - cur))
        explain([(f"**{names[j]}** macht {pc(cur[j], 0)} deines Geldes aus, verursacht aber **{pc(rc[j], 0)} des "
                  "Risikos**. Dieser Wert ist der größte Unruhestifter in deinem Portfolio.") if rc[j] - cur[j] > 0.05
                 else "Geld und Risiko sind bei dir ungefähr gleich verteilt – das ist gut.",
                 (f"**{names[k]}** dagegen beruhigt dein Portfolio: {pc(cur[k], 0)} des Geldes, aber nur {pc(rc[k], 0)} "
                  "des Risikos.") if cur[k] - rc[k] > 0.05 else ""])
    with t2:
        st.subheader("Wie ähnlich bewegen sich deine Wertpapiere?",
                      help="Korrelation von −1 bis +1. +1: bewegen sich immer gleich (keine Absicherung). 0: unabhängig. "
                           "Negativ: gegenläufig (gute Absicherung). Für Streuung sind niedrige Werte gut.")
        f5 = go.Figure(go.Heatmap(z=C, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                                  text=[[de(v, 2) for v in row] for row in C], texttemplate="%{text}",
                                  colorbar=dict(title="Korrelation")))
        f5.update_layout(height=160 + 60 * len(names), margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed",
                         font=dict(size=15))
        st.plotly_chart(f5, width="stretch")
        iu = np.triu_indices(len(names), 1)
        k_hi, k_lo = int(np.argmax(C[iu])), int(np.argmin(C[iu]))
        explain([f"Am ähnlichsten: **{names[iu[0][k_hi]]}** und **{names[iu[1][k_hi]]}** ({de(C[iu][k_hi], 2)}). "
                 "Beide gleichzeitig zu halten, streut das Risiko kaum.",
                 f"Am unterschiedlichsten: **{names[iu[0][k_lo]]}** und **{names[iu[1][k_lo]]}** ({de(C[iu][k_lo], 2)}) – "
                 "dieses Paar sichert sich am besten gegenseitig ab.",
                 "Rot = bewegen sich ähnlich, Blau = bewegen sich unterschiedlich. Ein Portfolio mit viel Blau ist "
                 "besser gestreut."])
    m = c["M"]["Aktuell"]
    st.subheader("Wie viel kann in einem schlechten Monat verloren gehen?",
                 help="Value at Risk (VaR) und Conditional VaR (CVaR) – historisch, auf einen Monat bezogen.")
    a, b = st.columns(2)
    a.metric("Schlechter Monat (1 von 20)", "−" + eur(m["VaR 95 %"] * V), help="VaR 95 %: In etwa 1 von 20 Monaten "
             "war der Verlust größer als dieser Betrag.")
    b.metric("Ø in den schlimmsten Monaten", "−" + eur(m["CVaR 95 %"] * V), help="CVaR 95 %: Durchschnittlicher "
             "Verlust in den schlechtesten 5 % der Monate.")
    explain([f"In etwa **1 von 20 Monaten** hat dein jetziges Portfolio mehr als **{eur(m['VaR 95 %'] * V)}** verloren "
             f"({pc(m['VaR 95 %'])}). In diesen schlechten Monaten waren es im Schnitt {eur(m['CVaR 95 %'] * V)}.",
             "Stell dir vor, dieser Betrag ist nächsten Monat weg. Wenn dich das nicht nervös macht, passt das "
             "Risiko zu dir."])


def page_future():
    c = ready()
    V, SIM, BANDS, target = c["V"], c["SIM"], c["BANDS"], c["target"]
    head("Zukunftssimulation", f"Was könnte in {hor} Jahren aus {eur(V)} werden? Die App würfelt 2000 mögliche "
                               "Zukünfte aus den echten Monaten der Vergangenheit zusammen.",
         "Bootstrap-Simulation: Jeder simulierte Monat ist ein zufällig gezogener historischer Monat (alle Wertpapiere "
         "gemeinsam, Zusammenhänge bleiben erhalten). Annahme: Die Zukunft ähnelt der Vergangenheit.")
    a, b = SIM["Aktuell"], SIM[target]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mittlerer Fall (Vorschlag)", eur(b["Median"]), f"{sg((b['Median'] - a['Median']) / V * 100, 0)} % ggü. jetzt",
              help="Median: Die Hälfte der Simulationen endet darüber, die Hälfte darunter.")
    k2.metric("Schlechter Fall (1 von 20)", eur(b["5 %"]), help="Nur 5 % der Simulationen enden noch schlechter.")
    k3.metric("Guter Fall (1 von 20)", eur(b["95 %"]), help="Nur 5 % der Simulationen enden noch besser.")
    k4.metric("Risiko, im Minus zu enden", pc(b["P(Verlust)"], 0), f"{sg((b['P(Verlust)'] - a['P(Verlust)']) * 100, 0)} Prozentpunkte",
              delta_color="inverse", help="Anteil der Simulationen, die unter deinem Startbetrag enden.")
    st.subheader("Mögliche Wege deines Geldes",
                 help="Linie = mittlerer Fall. Fläche = Bereich, in dem 9 von 10 simulierten Zukünften landen.")
    xm = np.arange(1, hor * 12 + 1) / 12
    f6 = go.Figure()
    for k, col, rgba, nm in [("Aktuell", "#e8361b", "rgba(232,54,27,0.15)", "Jetziges Portfolio"),
                             (target, "#1240e0", "rgba(18,64,224,0.15)", "Vorschlag")]:
        p5, p50, p95 = BANDS[k]
        f6.add_scatter(x=np.r_[xm, xm[::-1]], y=np.r_[p95, p5[::-1]], fill="toself", fillcolor=rgba,
                       line=dict(width=0), name=nm + " (9 von 10 Fällen)", hoverinfo="skip")
        f6.add_scatter(x=xm, y=p50, name=nm + " (mittlerer Fall)", line=dict(color=col, width=3))
    f6.add_hline(y=V, line_dash="dot", line_color="gray", annotation_text="Startwert")
    f6.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Jahre ab heute",
                     yaxis_title="Wert in €", separators=",.", legend=dict(orientation="h", y=-0.15), font=dict(size=15))
    st.plotly_chart(f6, width="stretch")
    st.dataframe(pd.DataFrame({"": ["Jetziges Portfolio", "Vorschlag"],
                               "Schlechter Fall": [eur(a["5 %"]), eur(b["5 %"])],
                               "Mittlerer Fall": [eur(a["Median"]), eur(b["Median"])],
                               "Guter Fall": [eur(a["95 %"]), eur(b["95 %"])],
                               "Risiko im Minus": [pc(a["P(Verlust)"], 0), pc(b["P(Verlust)"], 0)]}),
                 hide_index=True, width="stretch")
    cagr = (b["Median"] / V) ** (1 / hor) - 1
    wa, wb = a["95 %"] - a["5 %"], b["95 %"] - b["5 %"]
    explain([
        f"Im mittleren Fall würden aus {eur(V)} mit dem Vorschlag etwa **{eur(b['Median'])}** – das entspricht rund "
        f"{pc(cagr)} pro Jahr. Im schlechten Fall wären es {eur(b['5 %'])}, im guten Fall {eur(b['95 %'])}.",
        "Die Spanne ist groß, und das ist bei Aktien normal: Niemand weiß, welche Jahre kommen. Je länger du anlegst, "
        "desto kleiner wird meist das Risiko, am Ende im Minus zu sein. Probier links die Anlagedauer aus." if hor < 10
        else "Bei so einem langen Zeitraum ist ein Verlust am Ende eher unwahrscheinlich – zwischendurch kann es "
             "trotzdem deutlich nach unten gehen.",
        (f"Mit dem Vorschlag wird die Spanne **enger** ({eur(wb)} statt {eur(wa)}): Das Ergebnis ist besser planbar."
         if wb < wa * 0.95 else
         f"Mit dem Vorschlag wird die Spanne **breiter** ({eur(wb)} statt {eur(wa)}): Mehr Chance, aber auch mehr Risiko."
         if wb > wa * 1.05 else "Die Spanne bleibt mit dem Vorschlag ungefähr gleich."),
        "Achtung: Die Simulation setzt voraus, dass die Zukunft ähnlich verläuft wie die ausgewerteten Jahre. Waren das "
        "besonders gute Börsenjahre, sind die Zahlen zu optimistisch.",
    ])


def page_check():
    c = ready()
    R, BT, CURVES, s_, T, target = c["R"], c["BT"], c["CURVES"], c["s_"], c["T"], c["target"]
    head("Realitäts-Check", "Hätte der Vorschlag auch geholfen, wenn man die Zukunft nicht gekannt hätte?",
         f"Backtest: Der Vorschlag wird nur mit den älteren {split} % der Daten berechnet und dann auf den neueren "
         "Monaten geprüft, die er nicht kannte.")
    if BT is None or target not in BT:
        st.info("Zu wenig Kursgeschichte für diesen Test (mindestens 24 Monate Training + 6 Monate Test). "
                "Schau links mehr Jahre zurück.")
        return
    bt_t, bt_c = BT[target], BT["Aktuell"]
    per = f"{R.index[s_]} bis {R.index[-1]}"
    if bt_t["Sharpe"] > bt_c["Sharpe"] + 0.05:
        st.success(f"**Ja.** Im Testzeitraum ({per}) hätte der Vorschlag ein besseres Verhältnis von Gewinn zu Risiko "
                   f"gebracht ({de(bt_t['Sharpe'], 2)} statt {de(bt_c['Sharpe'], 2)}).")
    elif bt_t["Sharpe"] >= bt_c["Sharpe"] - 0.05:
        st.warning(f"**Kaum Unterschied.** Im Testzeitraum ({per}) war der Vorschlag ungefähr gleich gut wie dein "
                   "jetziges Portfolio. Eine Umschichtung lohnt sich nach Gebühren wahrscheinlich nicht.")
    else:
        st.error(f"**Nein.** Im Testzeitraum ({per}) wäre dein jetziges Portfolio besser gewesen (Gewinn pro Risiko "
                 f"{de(bt_c['Sharpe'], 2)} statt {de(bt_t['Sharpe'], 2)}). Der Vorschlag passt vor allem zur "
                 "Vergangenheit – sei vorsichtig damit.")
    st.subheader("Verlauf im Testzeitraum", help="Start = 100. Monatlich auf die jeweilige Aufteilung zurückgesetzt.")
    show = ["Aktuell", target] + (["Benchmark"] if "Benchmark" in CURVES else [])
    colors = {"Aktuell": "#e8361b", target: "#1240e0", "Benchmark": "#888"}
    fb = go.Figure([go.Scatter(x=R.index[s_:].to_timestamp(), y=CURVES[k], line=dict(color=colors[k], width=3),
                               name={"Aktuell": "Jetziges Portfolio"}.get(k, "Vorschlag" if k == target else k))
                    for k in show])
    fb.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Wert (Start = 100)",
                     legend=dict(orientation="h", y=1.08), hovermode="x unified", font=dict(size=15))
    st.plotly_chart(fb, width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Ø Gewinn pro Jahr (Test)", pc(bt_t["Rendite p.a."]), f"{sg((bt_t['Rendite p.a.'] - bt_c['Rendite p.a.']) * 100)} Prozentpunkte",
              help="Vorschlag im Testzeitraum; darunter der Unterschied zu deinem jetzigen Portfolio.")
    k2.metric("Schwankung (Test)", pc(bt_t["Vol. p.a."]), f"{sg((bt_t['Vol. p.a.'] - bt_c['Vol. p.a.']) * 100)} Prozentpunkte",
              delta_color="inverse", help="Weniger ist ruhiger.")
    k3.metric("Schlimmster Absturz (Test)", "−" + pc(bt_t["Max. Drawdown"]),
              f"{sg((bt_t['Max. Drawdown'] - bt_c['Max. Drawdown']) * 100)} Prozentpunkte", delta_color="inverse",
              help="Größter Verlust vom Hoch bis zum Tief im Testzeitraum.")
    explain([
        "Warum dieser Test so wichtig ist: Jede Optimierung findet im Nachhinein die „perfekte“ Mischung – das ist leicht. "
        "Die echte Frage ist, ob sie auch ohne Wissen über die Zukunft funktioniert hätte.",
        "Ein **Ja** heißt nicht, dass es weiter klappt, ist aber ein gutes Zeichen. Ein **Nein** ist eine Warnung: Der "
        "Vorschlag beruht dann vor allem auf Zufällen der Vergangenheit. In dem Fall ist „Möglichst ruhig“ oder "
        "„Einfach gleichmäßig verteilen“ oft die vernünftigere Wahl – probier es links unter **Dein Ziel** aus.",
        f"Der Testzeitraum umfasst nur {T - s_} Monate – das ist kurz. Ein einzelnes gutes oder schlechtes Jahr kann das "
        "Ergebnis stark beeinflussen." if T - s_ < 48 else "",
    ])


def page_frontier():
    c = ready()
    mu, S, lo, hi, W, cur, names = c["mu"], c["S"], c["lo"], c["hi"], c["W"], c["cur"], c["names"]
    head("Efficient Frontier", "Alle möglichen Mischungen deiner Wertpapiere: Wo liegt dein Portfolio, und was "
                               "wäre bei gleichem Risiko möglich gewesen?",
         "Punkte = Zufallsportfolios innerhalb deiner Grenzen. Linie = für jedes Risiko die höchste erreichbare "
         "Rendite (Markowitz).")
    F, r_ret, r_vol = frontier_mc(mu, S, lo, hi, n_mc)
    fg = go.Figure()
    fg.add_scatter(x=r_vol, y=r_ret, mode="markers", name="Zufallsportfolios", hoverinfo="skip",
                   marker=dict(size=5, color=(r_ret - rf) / r_vol, colorscale="Blues", showscale=True,
                               colorbar=dict(title="Sharpe")))
    fg.add_scatter(x=F[:, 0], y=F[:, 1], mode="lines", name="Efficient Frontier", line=dict(color="black", width=3))
    fg.add_scatter(x=np.sqrt(np.diag(S)), y=mu, mode="markers+text", text=names, textposition="top center",
                   name="Einzelwerte", marker=dict(symbol="square-open", size=11, color="gray"))
    for k, sym, col in [("Aktuell", "circle", "#e8361b"), ("Min-Varianz", "circle-open", "#1240e0"),
                        ("Max-Sharpe", "diamond", "#1240e0")]:
        w = W[k]
        fg.add_scatter(x=[np.sqrt(w @ S @ w)], y=[w @ mu], mode="markers", name=k,
                       marker=dict(symbol=sym, size=18, color=col, line=dict(width=2, color=col)))
    fg.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Volatilität p.a.",
                     yaxis_title="Erw. Rendite p.a.", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                     legend=dict(orientation="h", y=-0.12), font=dict(size=15))
    st.plotly_chart(fg, width="stretch")
    cv, cr = np.sqrt(cur @ S @ cur), cur @ mu
    gap = (np.interp(cv, F[:, 0], F[:, 1]) - cr if F[0, 0] <= cv <= F[-1, 0]
           else F[-1, 1] - cr if cv > F[-1, 0] else 0)
    mv_ = W["Min-Varianz"]
    explain([
        f"Dein Portfolio (roter Punkt) liegt bei {pc(cv)} Schwankung und {pc(cr)} erwarteter Rendite.",
        (f"Bei gleicher Schwankung wären bis zu **{de(gap * 100)} Prozentpunkte mehr Rendite pro Jahr** möglich gewesen – "
         "so weit liegt dein Punkt unter der Linie.") if gap > 0.005 else
        "Dein Portfolio liegt praktisch auf der Efficient Frontier – es war bereits effizient aufgeteilt.",
        f"Das ruhigste mögliche Portfolio (Min-Varianz, offener Kreis) hätte nur {pc(np.sqrt(mv_ @ S @ mv_))} geschwankt.",
        "Wichtig: Die Linie ist im Nachhinein berechnet. Vor allem der obere Teil beruht auf geschätzten Renditen, die "
        "sich selten so wiederholen.",
    ])


def page_style():
    c = ready()
    R, y, lo, hi, names, bench, T = c["R"], c["y"], c["lo"], c["hi"], c["names"], c["bench"], c["T"]
    head("Style Analysis", "Welche Mischung deiner Wertpapiere bildet einen Index am besten nach – und wie verändert "
                           "sich das über die Zeit?",
         "Returns-based Style Analysis nach Sharpe: Die Index-Renditen werden auf die Renditen deiner Wertpapiere "
         "regressiert. Gewichte summieren zu 100 % und liegen in deinen Grenzen (Min/Max).")
    if y is None:
        st.info("Trag links unter **Experten-Einstellungen** einen Benchmark-Ticker ein (z. B. ^GSPC, ^GDAXI, ^KLSE).")
        return
    w_all, r2, roll = style_roll(R, y, lo, hi, win)
    k1, k2 = st.columns(2)
    k1.metric("R² (Gesamtzeitraum)", de(r2, 2), help="Anteil der Index-Schwankung, den die Mischung erklärt. 1 = perfekt.")
    k2.metric("Fenster", f"{win} Monate", help="Länge des rollierenden Fensters (links einstellbar).")
    if roll is not None:
        st.subheader("Gewichte über die Zeit", help="Jeder Punkt auf der x-Achse ist das Ende eines Fensters.")
        idx = R.index[win - 1:].to_timestamp()
        fs = go.Figure([go.Scatter(x=idx, y=roll[:, j], stackgroup="one", name=n) for j, n in enumerate(names)])
        fs.update_layout(height=520, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10), font=dict(size=15))
        st.plotly_chart(fs, width="stretch")
    last = roll[-1] if roll is not None else np.full(len(names), np.nan)
    st.dataframe(pd.DataFrame({"Wertpapier": names, "Gesamtzeitraum": [pc(x) for x in w_all],
                               "Letztes Fenster": [pc(x) if x == x else "–" for x in last]}),
                 hide_index=True, width="stretch")
    top = np.argsort(-w_all)[:2]
    explain([
        f"Den Index {bench} bildet man mit deinen Wertpapieren am besten vor allem mit "
        + " und ".join(f"{names[i]} ({pc(w_all[i], 0)})" for i in top) + " nach.",
        f"R² = {de(r2, 2)}: " + ("Die Mischung erklärt den Index sehr gut." if r2 > 0.8 else
                                 "Die Mischung erklärt den Index nur teilweise." if r2 > 0.5 else
                                 "Die Mischung erklärt den Index kaum – deine Werte haben wenig mit ihm zu tun."),
        "Starke Sprünge im Diagramm heißen: Der Index verhielt sich mal wie die eine, mal wie die andere Gruppe deiner "
        "Werte." if roll is not None else "Zu wenige Monate für das rollierende Fenster.",
    ])


def page_compare():
    c = ready()
    W, M, BT, SIM, names, V, y, bench, T, s_, R = (c[k] for k in ("W", "M", "BT", "SIM", "names", "V", "y", "bench",
                                                                   "T", "s_", "R"))
    head("Strategievergleich", "Alle Strategien nebeneinander: Gewichte, Kennzahlen, Backtest und Simulation.")
    pfmt = lambda cols: {col: (lambda x: de(x, 2)) if col == "Sharpe" else pc for col in cols}
    t1, t2, t3, t4 = st.tabs(["Kennzahlen (in-sample)", "Gewichte", "Backtest", "Simulation"])
    with t1:
        st.subheader("Risikokennzahlen", help="VaR 95 %: Monatsverlust, der in 95 % der Monate nicht überschritten "
                                              "wurde. CVaR 95 %: Ø Verlust in den schlechtesten 5 % der Monate.")
        rk = pd.DataFrame(M).T
        if y is not None:
            rk.loc[f"Benchmark {bench}"] = metrics(y, rf)
        st.dataframe(rk.style.format(pfmt(rk.columns)), width="stretch")
        explain(["Die optimierten Strategien sehen hier besser aus, als sie in Wirklichkeit wären, weil sie auf genau "
                 "diesen Daten berechnet wurden (in-sample). Vergleiche deshalb den Backtest-Reiter."])
    with t2:
        st.subheader("Gewichte je Strategie", help="Anteil jedes Wertpapiers in jeder Strategie.")
        Wd = pd.DataFrame(W, index=names)
        st.dataframe(Wd.style.format(pc), width="stretch")
        fw = go.Figure([go.Bar(x=list(W), y=Wd.loc[n].values, name=n) for n in names])
        fw.update_layout(barmode="stack", height=460, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10),
                         font=dict(size=15))
        st.plotly_chart(fw, width="stretch")
    with t3:
        if BT is None:
            st.info("Zu wenige Monate (mindestens 24 Training + 6 Test).")
        else:
            st.caption(f"Training {R.index[0]}–{R.index[s_ - 1]}, Test {R.index[s_]}–{R.index[-1]}, monatlich "
                       "rebalanciert, Start = 100.")
            fb = go.Figure([go.Scatter(x=R.index[s_:].to_timestamp(), y=v, name=k) for k, v in c["CURVES"].items()])
            fb.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=15), hovermode="x unified")
            st.plotly_chart(fb, width="stretch")
            bt = pd.DataFrame(BT).T[["Rendite p.a.", "Vol. p.a.", "Sharpe", "Max. Drawdown"]]
            st.dataframe(bt.style.format(pfmt(bt.columns)), width="stretch")
            best = bt["Sharpe"].idxmax()
            txt = {"Aktuell": "Dein jetziges Portfolio war im Test am besten – Umschichten hätte nicht geholfen.",
                   "Gleichgewichtet": "Gleichmäßig verteilen war am besten – typisch, weil es nichts schätzen muss.",
                   "Min-Varianz": "Das ruhigste Portfolio war am besten – häufig so, weil es keine Renditen schätzt.",
                   "Benchmark": "Der Index selbst war am besten – ein breiter ETF wäre die einfachste Lösung gewesen."}
            explain([f"Im Testzeitraum hatte **{best}** das beste Verhältnis von Gewinn zu Risiko (Sharpe "
                     f"{de(bt.loc[best, 'Sharpe'], 2)}).",
                     txt.get(best, "Dass eine renditebasierte Strategie vorne liegt, kann am Testzeitraum liegen.")])
    with t4:
        st.subheader(f"Bootstrap: Endwert nach {hor} Jahren (Start {eur(V)})")
        sm = pd.DataFrame(SIM).T
        st.dataframe(sm.style.format({"5 %": eur, "Median": eur, "95 %": eur, "P(Verlust)": lambda x: pc(x, 0)}),
                     width="stretch")


def page_help():
    head("Begriffe & Hilfe", "Alle Fachbegriffe der App einfach erklärt – und wie du die App am besten benutzt.")
    st.subheader("So gehst du vor")
    st.markdown("1. **Mein Portfolio:** Wertpapiere suchen oder als Liste einfügen, Beträge eintragen.\n"
                "2. **Dein Ziel** links wählen: ruhig, bestes Verhältnis oder gleichmäßig.\n"
                "3. **Übersicht & Vorschlag** lesen – dort steht, was du ändern könntest und warum.\n"
                "4. Im **Realitäts-Check** prüfen, ob der Vorschlag auch ohne Blick in die Zukunft geholfen hätte.\n"
                "5. Wer mehr will: **Expertenmodus** einschalten.")
    st.subheader("Begriffe")
    cols = st.columns(2)
    for i, (k, v) in enumerate(GLOSSAR.items()):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{k}**  \n{v}")
    st.subheader("Grenzen der App")
    st.markdown("- Alle Zahlen beruhen auf vergangenen Kursen. Was früher gut lief, muss nicht so weiterlaufen.\n"
                "- Gebühren, Steuern und Wechselkurse (bei Werten in Fremdwährung) sind nicht gesondert berücksichtigt.\n"
                "- Kurse kommen von Yahoo Finance (ersatzweise stooq) und können Lücken haben.\n"
                "- **Keine Anlageberatung.**")


# =====================================================================
# Menü
# =====================================================================
P_INPUT = st.Page(page_input, title="Mein Portfolio", icon=":material/edit:", url_path="portfolio", default=True)
P_OVER = st.Page(page_overview, title="Übersicht & Vorschlag", icon=":material/dashboard:", url_path="uebersicht")
P_GROWTH = st.Page(page_growth, title="Wertentwicklung", icon=":material/show_chart:", url_path="wertentwicklung")
P_ASSETS = st.Page(page_assets, title="Meine Wertpapiere", icon=":material/list_alt:", url_path="wertpapiere")
P_RISK = st.Page(page_risk, title="Risiko & Streuung", icon=":material/donut_small:", url_path="risiko")
P_FUTURE = st.Page(page_future, title="Zukunftssimulation", icon=":material/auto_graph:", url_path="zukunft")
P_CHECK = st.Page(page_check, title="Realitäts-Check", icon=":material/fact_check:", url_path="realitaets-check")
P_FRONT = st.Page(page_frontier, title="Efficient Frontier", icon=":material/scatter_plot:", url_path="frontier")
P_STYLE = st.Page(page_style, title="Style Analysis", icon=":material/stacked_line_chart:", url_path="style")
P_COMP = st.Page(page_compare, title="Strategievergleich", icon=":material/table_chart:", url_path="vergleich")
P_HELP = st.Page(page_help, title="Begriffe & Hilfe", icon=":material/help:", url_path="hilfe")

menu = {"Start": [P_INPUT, P_OVER],
        "Simulationen & Analysen": [P_GROWTH, P_ASSETS, P_RISK, P_FUTURE, P_CHECK]}
if expert:
    menu["Experten"] = [P_FRONT, P_STYLE, P_COMP]
menu["Hilfe"] = [P_HELP]
pg = st.navigation(menu)
pg.run()
st.caption("Alle Zahlen beruhen auf vergangenen Kursen. Gebühren und Steuern sind nicht berücksichtigt. "
           "Keine Anlageberatung.")

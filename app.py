"""Portfolio-Check – einsteigerfreundlicher Portfolio-Optimierer (Streamlit).

Start lokal:  streamlit run app.py
"""
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio-Check", layout="wide")

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
# Bekannte Wertpapiere (Name -> Yahoo-Ticker)
# =====================================================================
POPULAR = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN", "Alphabet (Google)": "GOOGL",
    "Meta (Facebook)": "META", "Tesla": "TSLA", "Coca-Cola": "KO", "Johnson & Johnson": "JNJ",
    "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE", "Deutsche Telekom": "DTE.DE",
    "Mercedes-Benz": "MBG.DE", "iShares MSCI World ETF": "EUNL.DE", "iShares S&P 500 ETF": "SXR8.DE",
    "Gold (ETF)": "GLD",
}
NAME = {v: k for k, v in POPULAR.items()}
label = lambda t: NAME.get(t, t)

GOALS = {
    "Möglichst ruhig (wenig Schwankung)": "Min-Varianz",
    "Bestes Verhältnis von Gewinn zu Risiko": "Max-Sharpe",
    "Einfach gleichmäßig verteilen": "Gleichgewichtet",
}

GLOSSAR = """
**Rendite / Gewinn pro Jahr** – wie viel das Portfolio im Durchschnitt pro Jahr gewachsen ist.

**Schwankung (Volatilität)** – wie stark der Wert hoch und runter geht. 15 % heißt grob: In einem normalen Jahr kann
es leicht 15 % besser oder schlechter laufen als im Durchschnitt.

**Schlimmster Absturz (Max. Drawdown)** – der größte Verlust vom Höchststand bis zum Tiefpunkt. 30 % heißt: Aus
10.000 € wurden zwischenzeitlich 7.000 €.

**Sharpe Ratio** – Gewinn pro Einheit Risiko. Je höher, desto besser „bezahlt“ wird das Risiko.

**Backtest** – Wir tun so, als stünden wir in der Vergangenheit: Die Vorschläge werden nur mit älteren Daten
berechnet und dann mit den neueren Daten geprüft. So sieht man, ob sie wirklich geholfen hätten.

**Simulation** – Aus den echten Monaten der Vergangenheit werden tausende mögliche Zukünfte zusammengewürfelt.
Das zeigt eine Spanne, keine Vorhersage.

**Ticker** – das Kürzel eines Wertpapiers bei Yahoo Finance, z. B. AAPL für Apple oder SAP.DE für SAP an der
Börse Xetra.
"""

# =====================================================================
# Daten laden
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
    for name, fn in (("Yahoo", load_yahoo), ("stooq", load_stooq)):
        try:
            s = fn(ticker, start, end)
            if s is not None and len(s) >= 13:
                return s, name
        except Exception:
            pass
    return None, None


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
    """Begründet, warum Wertpapier i mehr (up=True) oder weniger Gewicht bekommt."""
    vol = np.sqrt(np.diag(S))
    shp = mu / vol
    c = (C.sum(1) - 1) / max(1, len(mu) - 1)
    r_vol = (vol[i] < np.median(vol)) if up else (vol[i] > np.median(vol))
    r_sh = (shp[i] > np.median(shp)) if up else (shp[i] < np.median(shp))
    r_c = (c[i] < np.median(c)) if up else (c[i] > np.median(c))
    txt_vol = (f"schwankt weniger als deine anderen Werte ({pc(vol[i])} pro Jahr)" if up
               else f"schwankt stärker als deine anderen Werte ({pc(vol[i])} pro Jahr)")
    txt_sh = (f"viel Gewinn im Verhältnis zum Risiko (Ø {pc(mu[i])} pro Jahr)" if up
              else f"wenig Gewinn im Verhältnis zum Risiko (Ø {pc(mu[i])} pro Jahr)")
    txt_c = ("bewegt sich oft anders als deine übrigen Werte – das fängt Verluste ab" if up
             else "bewegt sich sehr ähnlich wie deine übrigen Werte – streut das Risiko kaum")
    order = [(r_vol, txt_vol), (r_c, txt_c), (r_sh, txt_sh)] if target == "Min-Varianz" else \
            [(r_sh, txt_sh), (r_c, txt_c), (r_vol, txt_vol)]
    hits = [t for ok, t in order if ok][:2]
    if target == "Gleichgewichtet":
        return "Grund: Bei gleichmäßiger Verteilung bekommt jedes Wertpapier gleich viel"
    return ("Grund: " + "; ".join(hits)) if hits else "Grund: passt rechnerisch besser in die Mischung"


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


# =====================================================================
# Sidebar
# =====================================================================
sb = st.sidebar
expert = sb.toggle("Expertenmodus", False,
                   help="Zeigt alle Einstellungen und Analysen: Efficient Frontier, Style Analysis, "
                        "Risikokennzahlen, eigene Grenzen pro Wertpapier.")
years = sb.slider("Wie viele Jahre zurückschauen?", 3, 25, 10,
                  help="So viele Jahre Kursgeschichte werden ausgewertet. Mehr Jahre = mehr Erfahrung, "
                       "aber sehr alte Daten sagen weniger über heute aus.")
hor = sb.slider("Für wie viele Jahre willst du anlegen?", 1, 30, 5,
                help="Für diesen Zeitraum wird simuliert, was aus deinem Geld werden könnte.")
start, end = date.today() - timedelta(days=365 * years), date.today()
rf, n_mc, split, win, bench_t, uploads = 0.02, 3000, 70, 36, "", None
if expert:
    sb.divider()
    sb.subheader("Experten-Einstellungen")
    if sb.checkbox("Eigenen Zeitraum wählen"):
        c1, c2 = sb.columns(2)
        start, end = c1.date_input("Von", start), c2.date_input("Bis", end)
    bench_t = sb.text_input("Benchmark-Ticker", "", placeholder="z. B. ^GSPC, ^GDAXI, ^KLSE",
                            help="Vergleichsindex für Style Analysis und Backtest.").strip()
    rf = sb.number_input("Risikofreier Zins p.a. %", value=2.0, step=0.1,
                         help="Rendite einer sicheren Anlage, wird in der Sharpe Ratio abgezogen.") / 100
    n_mc = sb.select_slider("Zufallsportfolios", [1000, 3000, 6000], 3000)
    split = sb.slider("Backtest: Trainingsanteil %", 30, 90, 70, 5,
                      help="Auf diesem Anteil wird optimiert, auf dem Rest getestet.")
    win = int(sb.number_input("Style-Fenster (Monate)", 12, 120, 36))
    uploads = sb.file_uploader("Kursdateien hochladen (CSV/HTML)", accept_multiple_files=True,
                               help="Notlösung, falls ein Ticker nicht geladen werden kann. Dateiname = Name; "
                                    "beginnt er mit ^, wird die Datei zum Benchmark.")

# =====================================================================
# Kopf
# =====================================================================
st.title("Portfolio-Check")
st.write("Trag ein, welche Aktien oder ETFs du hast und wie viel Geld jeweils darin steckt. Die App prüft "
         "mit echten Kursen der letzten Jahre, wie dein Portfolio gelaufen ist, und schlägt vor, wie du es "
         "besser aufteilen könntest.")
with st.expander("Begriffe einfach erklärt"):
    st.markdown(GLOSSAR)

# ---------- Schritt 1 ----------
st.header("1. Welche Wertpapiere hast du?")
c1, c2 = st.columns([3, 2])
picked = c1.multiselect("Aus der Liste wählen", list(POPULAR),
                        default=["Apple", "Microsoft", "iShares MSCI World ETF", "Allianz", "Coca-Cola"],
                        help="Bekannte Aktien und ETFs. Fehlt deine? Trag sie rechts als Ticker ein.")
extra = c2.text_input("Weitere (Ticker, mit Komma getrennt)", "", placeholder="z. B. NOVO-B.CO, BAS.DE",
                      help="Den Ticker findest du auf finance.yahoo.com: Namen ins Suchfeld eingeben, das Kürzel "
                           "steht neben dem Namen. Deutsche Börse Xetra endet auf .DE.")
tickers = [POPULAR[p] for p in picked] + [t.strip().upper() for t in extra.replace(";", ",").split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))

series, failed = {}, []
with st.spinner("Lade Kurse …"):
    for t in tickers + ([bench_t] if bench_t else []):
        s, _ = load(t, start, end)
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

if failed:
    st.warning("Für **" + ", ".join(failed) + "** wurden keine Kurse gefunden. Prüfe die Schreibweise des Tickers "
               "auf finance.yahoo.com. Die übrigen Wertpapiere werden trotzdem ausgewertet.")
assets = [t for t in tickers if t in series and t != bench]
if len(assets) < 2:
    st.info("Wähle mindestens **2 Wertpapiere** aus, damit verglichen und optimiert werden kann.")
    st.stop()

prices = pd.concat({t: series[t] for t in assets + ([bench] if bench else [])}, axis=1).dropna()
R_all = prices.pct_change().dropna()
R, y, T = R_all[assets], (R_all[bench].values if bench else None), len(R_all)
if T < 25:
    st.error(f"Für diese Auswahl gibt es nur {T} gemeinsame Monate Kursgeschichte. Es braucht mindestens 25. "
             "Wähle mehr Jahre oder entferne sehr junge Wertpapiere.")
    st.stop()
young = [label(t) for t in assets if len(series[t]) < years * 12 - 3]
if young and not expert:
    st.caption("Hinweis: " + ", ".join(young) + " gibt es noch nicht so lange. Ausgewertet werden nur die "
               f"gemeinsamen {T} Monate.")

# ---------- Schritt 2 ----------
st.header("2. Wie viel Geld steckt jeweils darin?")
st.write("Trag den aktuellen Wert in Euro ein. Grobe Werte reichen.")
dmax = 40.0 if len(assets) >= 3 else 100.0
base = pd.DataFrame({"Wertpapier": [label(t) for t in assets], "Ticker": assets, "Wert in €": 1000.0})
cfg = {"Wert in €": st.column_config.NumberColumn("Wert in €", min_value=0.0, step=100.0, format="%.0f €")}
if expert:
    base["Min %"], base["Max %"] = 0.0, dmax
    for c in ("Min %", "Max %"):
        cfg[c] = st.column_config.NumberColumn(c, min_value=0.0, max_value=100.0, step=1.0,
                                               help="Grenzen für das Gewicht in allen Optimierungen.")
ed = st.data_editor(base, key=f"ed_{expert}_" + "_".join(assets), hide_index=True, disabled=["Wertpapier", "Ticker"],
                    column_config=cfg, width="stretch")
vals = ed["Wert in €"].fillna(0).clip(lower=0).values
V = vals.sum()
if V <= 0:
    st.info("Trag bei mindestens einem Wertpapier einen Betrag ein.")
    st.stop()
cur = vals / V
if expert:
    lo, hi = ed["Min %"].fillna(0).values / 100, ed["Max %"].fillna(100).values / 100
    if lo.sum() > 1.0001 or hi.sum() < 0.9999 or (lo > hi).any():
        st.error("Grenzen unzulässig: Summe aller Min ≤ 100 % ≤ Summe aller Max, und Min ≤ Max.")
        st.stop()
else:
    lo, hi = np.zeros(len(assets)), np.full(len(assets), dmax / 100)
    if dmax < 100:
        st.caption(f"Damit nicht alles auf eine Karte gesetzt wird, steckt der Vorschlag höchstens {de(dmax, 0)} % "
                   "in ein einzelnes Wertpapier.")

# ---------- Schritt 3 ----------
st.header("3. Was ist dir wichtiger?")
goal_txt = st.radio("Ziel", list(GOALS) + (["Benchmark nachbilden (Style Analysis)"] if expert and bench else []),
                    index=1, label_visibility="collapsed",
                    help="Ruhig: geringste Schwankung, schaut nicht auf Gewinne und ist daher am robustesten. "
                         "Bestes Verhältnis: maximiert die Sharpe Ratio, reagiert aber stark auf Zufälle in "
                         "den Vergangenheitsdaten. Gleichmäßig: jedes Wertpapier gleich viel, ganz ohne Rechnung.")
target = GOALS.get(goal_txt, "Benchmark-Stil")

# =====================================================================
# Rechnen
# =====================================================================
W, mu, S = optimise_all(R, lo, hi, rf, cur, y)
M = {k: metrics(R.values @ w, rf) for k, w in W.items()}
tw, tm, cm = W[target], M[target], M["Aktuell"]

idx = np.random.default_rng(11).integers(0, T, (2000, hor * 12))
SIM = {}
for k, w in W.items():
    v = np.prod(1 + (R.values @ w)[idx], axis=1) * V
    SIM[k] = {"5 %": np.percentile(v, 5), "Median": np.median(v), "95 %": np.percentile(v, 95), "P(Verlust)": (v < V).mean()}

s_ = round(T * split / 100)
BT = None
if s_ >= 24 and T - s_ >= 6:
    Wt, _, _ = optimise_all(R.iloc[:s_], lo, hi, rf, cur, None if y is None else y[:s_])
    test = R.iloc[s_:]
    BT = {k: metrics(test.values @ w, rf) for k, w in Wt.items()}
    CURVES = {k: np.cumprod(1 + test.values @ w) * 100 for k, w in Wt.items()}
    if y is not None:
        BT["Benchmark"], CURVES["Benchmark"] = metrics(y[s_:], rf), np.cumprod(1 + y[s_:]) * 100

# =====================================================================
# Ergebnis – einfach
# =====================================================================
st.header("Ergebnis")
d = tw - cur
moves = [(label(t), d[i] * V) for i, t in enumerate(assets) if abs(d[i]) >= 0.005]
if not moves:
    st.success(f"Dein Portfolio ist für das Ziel „{goal_txt}“ schon gut aufgeteilt. Du musst nichts ändern.")
else:
    buy = [f"**{n}** (+{eur(x)})" for n, x in moves if x > 0]
    sell = [f"**{n}** (−{eur(-x)})" for n, x in moves if x < 0]
    st.info("**Vorschlag:** " + ("Mehr kaufen: " + ", ".join(buy) + ". " if buy else "") +
            ("Weniger halten: " + ", ".join(sell) + "." if sell else ""))

left, right = st.columns([5, 6])
with left:
    st.subheader("Was du ändern müsstest",
                 help="Beträge in Euro, bezogen auf dein Gesamtvermögen von " + eur(V) +
                      ". Kleine Änderungen unter 0,5 % werden als „behalten“ gezeigt. Gebühren und Steuern "
                      "sind nicht eingerechnet.")
    act = [("behalten" if abs(x) < 0.005 else (f"+{eur(x * V)} kaufen" if x > 0 else f"−{eur(-x * V)} verkaufen"))
           for x in d]
    st.dataframe(pd.DataFrame({"Wertpapier": [label(t) for t in assets], "Jetzt": [eur(c * V) for c in cur],
                               "Vorschlag": [eur(w * V) for w in tw], "Was tun?": act}),
                 hide_index=True, width="stretch")
with right:
    fig = go.Figure()
    names = [label(t) for t in assets]
    fig.add_bar(y=names, x=cur, orientation="h", name="Jetzt", marker_color="#b8c4e8")
    fig.add_bar(y=names, x=tw, orientation="h", name="Vorschlag", marker_color="#1240e0")
    fig.update_layout(barmode="group", height=60 + 55 * len(assets), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08), yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")

st.subheader(f"Vorher und nachher – so wäre es in den letzten {years} Jahren gelaufen")
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
st.caption(f"Dein jetziges Portfolio: Ø {pc(cm['Rendite p.a.'])} pro Jahr, Schwankung {pc(cm['Vol. p.a.'])}, "
           f"schlimmster Absturz −{pc(cm['Max. Drawdown'])}.")

C = R.corr().values
avg_c = round((C.sum() - len(assets)) / max(1, len(assets) * (len(assets) - 1)), 2) + 0.0
big = int(np.argmax(cur))
lines = [
    f"**Dein jetziges Portfolio** ist {lvl_vol(cm['Vol. p.a.'])} (Schwankung {pc(cm['Vol. p.a.'])}). In den letzten "
    f"{years} Jahren ist es im Schnitt um {pc(cm['Rendite p.a.'])} pro Jahr gewachsen. Der schlimmste Einbruch lag bei "
    f"−{pc(cm['Max. Drawdown'])}: Aus {eur(V)} wären zwischenzeitlich {eur(V * (1 - cm['Max. Drawdown']))} geworden. "
    f"Das Verhältnis von Gewinn zu Risiko war {lvl_sh(cm['Sharpe'])}.",
    ("**Streuung:** Deine Wertpapiere bewegten sich stark gleichzeitig (Ø Korrelation " + de(avg_c, 2) + "). Wenn eins "
     "fällt, fallen die anderen meist mit – die Streuung schützt dich kaum. Wertpapiere aus anderen Branchen, Ländern "
     "oder Anlageklassen würden mehr bringen.") if avg_c > 0.6 else
    ("**Streuung:** Deine Wertpapiere liefen oft unterschiedlich (Ø Korrelation " + de(avg_c, 2) + "). Das ist gut: "
     "Verluste des einen werden teilweise von den anderen ausgeglichen.") if avg_c < 0.3 else
    ("**Streuung:** Deine Wertpapiere hängen mittelstark zusammen (Ø Korrelation " + de(avg_c, 2) + "). Die Streuung "
     "hilft etwas, aber in Krisen fallen sie wahrscheinlich gemeinsam."),
    (f"**Klumpenrisiko:** {pc(cur[big], 0)} deines Geldes stecken in {label(assets[big])}. Läuft dieser eine Wert "
     "schlecht, zieht er das ganze Portfolio mit.") if cur[big] > 0.4 else "",
]
dr, dv = tm["Rendite p.a."] - cm["Rendite p.a."], tm["Vol. p.a."] - cm["Vol. p.a."]
if moves:
    if dr >= 0 and dv <= 0:
        trade = "Der Vorschlag wäre in der Vergangenheit **klar besser** gewesen: mehr Gewinn bei weniger Schwankung."
    elif dr >= 0:
        trade = ("Der Vorschlag bringt **mehr Gewinn, aber auch mehr Schwankung**. Du tauschst also Ruhe gegen Rendite – "
                 "das lohnt sich nur, wenn du Einbrüche gelassen aussitzen kannst.")
    elif dv <= 0:
        trade = ("Der Vorschlag ist **ruhiger, bringt dafür etwas weniger Gewinn**. Sinnvoll, wenn dir Verluste "
                 "schlaflose Nächte bereiten.")
    else:
        trade = "Der Vorschlag wäre hier eher schlechter – das kommt selten vor; prüfe deine Eingaben und Grenzen."
    ups = sorted([i for i in range(len(assets)) if d[i] >= 0.005], key=lambda i: -d[i])[:2]
    downs = sorted([i for i in range(len(assets)) if d[i] <= -0.005], key=lambda i: d[i])[:2]
    why = [f"- **{label(assets[i])}** bekommt mehr ({pc(cur[i], 0)} → {pc(tw[i], 0)}). "
           f"{reasons(i, True, mu, S, C, target)}." for i in ups] + \
          [f"- **{label(assets[i])}** bekommt weniger ({pc(cur[i], 0)} → {pc(tw[i], 0)}). "
           f"{reasons(i, False, mu, S, C, target)}." for i in downs]
    cap_i = [i for i in range(len(assets)) if tw[i] >= hi[i] - 0.001 and hi[i] < 1]
    capped = [label(assets[i]) for i in cap_i]
    lines += [trade, "**Warum diese Änderungen?**\n" + "\n".join(why),
              (f"Bei {', '.join(capped)} stößt der Vorschlag an die Obergrenze von {pc(hi[cap_i[0]], 0)}. "
               "Ohne Grenze würde das Modell noch mehr hineinstecken – ein Zeichen, dass es stark auf die gute "
               "Vergangenheit dieses Werts setzt. Genau davor schützt die Grenze.") if capped and target == "Max-Sharpe" else ""]
explain(lines)

st.subheader(f"Was könnte in {hor} Jahren aus {eur(V)} werden?",
             help="Simulation: Aus den echten Monaten der Vergangenheit werden 2000 mögliche Zukünfte "
                  "zusammengewürfelt. Die Spanne zeigt, wo 9 von 10 Fällen landen. Keine Vorhersage: "
                  "Die Zukunft kann ganz anders laufen.")
a, b = SIM["Aktuell"], SIM[target]
c1, c2 = st.columns(2)
c1.markdown(f"**Jetziges Portfolio**  \nwahrscheinlich etwa **{eur(a['Median'])}**  \n"
            f"in 9 von 10 Fällen zwischen {eur(a['5 %'])} und {eur(a['95 %'])}  \n"
            f"Risiko, am Ende weniger als eingesetzt zu haben: {pc(a['P(Verlust)'], 0)}")
c2.markdown(f"**Mit dem Vorschlag**  \nwahrscheinlich etwa **{eur(b['Median'])}**  \n"
            f"in 9 von 10 Fällen zwischen {eur(b['5 %'])} und {eur(b['95 %'])}  \n"
            f"Risiko, am Ende weniger als eingesetzt zu haben: {pc(b['P(Verlust)'], 0)}")
cagr = (b["Median"] / V) ** (1 / hor) - 1
wa, wb = a["95 %"] - a["5 %"], b["95 %"] - b["5 %"]
explain([
    f"Im mittleren Fall würden aus {eur(V)} mit dem Vorschlag etwa **{eur(b['Median'])}** – das entspricht rund "
    f"{pc(cagr)} pro Jahr. Im schlechten Fall (nur 1 von 20 Simulationen ist noch schlechter) wären es "
    f"{eur(b['5 %'])}, im guten Fall {eur(b['95 %'])}.",
    "Die Spanne ist groß, und das ist bei Aktien normal: Niemand weiß, welche Jahre kommen. Je länger du anlegst, "
    "desto kleiner wird meist das Risiko, am Ende im Minus zu sein." if hor < 10 else
    "Bei so einem langen Zeitraum ist ein Verlust am Ende eher unwahrscheinlich – zwischendurch kann es trotzdem "
    "deutlich nach unten gehen.",
    (f"Mit dem Vorschlag wird die Spanne **enger** ({eur(wb)} statt {eur(wa)}): Das Ergebnis ist besser planbar."
     if wb < wa * 0.95 else
     f"Mit dem Vorschlag wird die Spanne **breiter** ({eur(wb)} statt {eur(wa)}): Mehr Chance, aber auch mehr Risiko."
     if wb > wa * 1.05 else "Die Spanne bleibt mit dem Vorschlag ungefähr gleich."),
    "Achtung: Die Simulation setzt voraus, dass die Zukunft ähnlich verläuft wie die ausgewerteten Jahre. Waren das "
    "besonders gute Börsenjahre, sind die Zahlen zu optimistisch.",
])

st.subheader("Hätte der Vorschlag wirklich geholfen?",
             help="Backtest: Der Vorschlag wird nur mit den älteren 70 % der Daten berechnet und dann auf den "
                  "neueren 30 % geprüft, die er nicht kannte. Das ist der ehrlichste Test.")
if BT is None or target not in BT:
    st.caption("Zu wenig Kursgeschichte für diesen Test.")
else:
    bt_t, bt_c = BT[target], BT["Aktuell"]
    per = f"{R.index[s_]} bis {R.index[-1]}"
    if bt_t["Sharpe"] > bt_c["Sharpe"] + 0.05:
        st.success(f"**Ja.** Im Testzeitraum ({per}) hätte der Vorschlag ein besseres Verhältnis von Gewinn zu Risiko "
                   f"gebracht ({de(bt_t['Sharpe'], 2)} statt {de(bt_c['Sharpe'], 2)}), Ø {pc(bt_t['Rendite p.a.'])} "
                   f"statt {pc(bt_c['Rendite p.a.'])} pro Jahr.")
    elif bt_t["Sharpe"] >= bt_c["Sharpe"] - 0.05:
        st.warning(f"**Kaum Unterschied.** Im Testzeitraum ({per}) war der Vorschlag ungefähr gleich gut wie dein "
                   "jetziges Portfolio. Eine Umschichtung lohnt sich nach Gebühren wahrscheinlich nicht.")
    else:
        st.error(f"**Nein.** Im Testzeitraum ({per}) wäre dein jetziges Portfolio besser gewesen "
                 f"(Gewinn pro Risiko {de(bt_c['Sharpe'], 2)} statt {de(bt_t['Sharpe'], 2)}). Der Vorschlag passt "
                 "vor allem zur Vergangenheit – sei vorsichtig damit.")
    explain([
        "Warum dieser Test so wichtig ist: Jede Optimierung findet im Nachhinein die „perfekte“ Mischung – das ist "
        "leicht. Die echte Frage ist, ob sie auch ohne Wissen über die Zukunft funktioniert hätte.",
        "Ein **Ja** heißt nicht, dass es weiter klappt, ist aber ein gutes Zeichen. Ein **Nein** ist eine Warnung: "
        "Der Vorschlag beruht dann vor allem auf Zufällen der Vergangenheit. In dem Fall ist „Möglichst ruhig“ oder "
        "„Einfach gleichmäßig verteilen“ oft die vernünftigere Wahl.",
        f"Der Testzeitraum umfasst nur {T - s_} Monate – das ist kurz. Ein einzelnes gutes oder schlechtes Jahr kann "
        "das Ergebnis stark beeinflussen." if T - s_ < 48 else "",
    ])

st.caption("Wichtig: Alle Zahlen beruhen auf vergangenen Kursen. Was früher gut lief, muss nicht so weiterlaufen. "
           "Gebühren und Steuern sind nicht berücksichtigt. Das ist keine Anlageberatung.")

# =====================================================================
# Expertenbereich
# =====================================================================
if not expert:
    st.caption("Mehr Details (Efficient Frontier, Risikokennzahlen, Style Analysis): **Expertenmodus** in der Seitenleiste einschalten.")
    st.stop()

st.divider()
st.header("Für Fortgeschrittene")
t0, t1, t2, t3, t4 = st.tabs(["Efficient Frontier", "Style Analysis", "Risiko & Gewichte", "Backtest", "Simulation"])

with t0:
    st.subheader("Efficient Frontier und Monte-Carlo-Portfolios",
                 help="Punkte = Zufallsportfolios innerhalb deiner Grenzen. Linie = für jedes Risiko die höchste "
                      "erreichbare Rendite. Liegt dein Portfolio deutlich darunter, war es ineffizient.")
    rng = np.random.default_rng(5)
    Wr = np.array([proj(w, lo, hi) for w in rng.dirichlet(np.ones(len(assets)), n_mc)])
    r_ret, r_vol = Wr @ mu, np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr))
    F = frontier(mu, S, lo, hi)
    fg = go.Figure()
    fg.add_scatter(x=r_vol, y=r_ret, mode="markers", name="Zufallsportfolios", hoverinfo="skip",
                   marker=dict(size=4, color=(r_ret - rf) / r_vol, colorscale="Blues", showscale=True,
                               colorbar=dict(title="Sharpe")))
    fg.add_scatter(x=F[:, 0], y=F[:, 1], mode="lines", name="Efficient Frontier", line=dict(color="black", width=3))
    fg.add_scatter(x=np.sqrt(np.diag(S)), y=mu, mode="markers+text", text=[label(t) for t in assets],
                   textposition="top center", name="Einzelwerte", marker=dict(symbol="square-open", size=9, color="gray"))
    for k, sym, col in [("Aktuell", "circle", "#e8361b"), ("Min-Varianz", "circle-open", "#1240e0"),
                        ("Max-Sharpe", "diamond", "#1240e0")]:
        w = W[k]
        fg.add_scatter(x=[np.sqrt(w @ S @ w)], y=[w @ mu], mode="markers", name=k,
                       marker=dict(symbol=sym, size=14, color=col, line=dict(width=2, color=col)))
    fg.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Volatilität p.a.",
                     yaxis_title="Erw. Rendite p.a.", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                     legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fg, width="stretch")
    cv, cr = np.sqrt(cur @ S @ cur), cur @ mu
    gap = np.interp(cv, F[:, 0], F[:, 1]) - cr if F[0, 0] <= cv <= F[-1, 0] else F[-1, 1] - cr if cv > F[-1, 0] else 0
    mv_ = W["Min-Varianz"]
    explain([
        f"Dein Portfolio (roter Punkt) liegt bei {pc(cv)} Schwankung und {pc(cr)} erwarteter Rendite.",
        (f"Bei gleicher Schwankung wären bis zu **{de(gap * 100)} Prozentpunkte mehr Rendite pro Jahr** möglich "
         "gewesen – so weit liegt dein Punkt unter der Linie. Das ist der „verschenkte“ Teil.") if gap > 0.005 else
        "Dein Portfolio liegt praktisch auf der Efficient Frontier – es war bereits effizient aufgeteilt.",
        f"Das ruhigste mögliche Portfolio (Min-Varianz, offener Kreis) hätte nur {pc(np.sqrt(mv_ @ S @ mv_))} "
        "geschwankt.",
        "Wichtig: Die Linie ist im Nachhinein berechnet. Vor allem der obere Teil (hohe Rendite) beruht auf "
        "geschätzten Renditen, die sich selten so wiederholen.",
    ])

with t1:
    st.subheader("Style Analysis (rollierend)",
                 help="Returns-based Style Analysis nach Sharpe: Benchmark-Renditen werden auf die Asset-Renditen "
                      "regressiert (Gewichte summieren zu 100 %, innerhalb deiner Grenzen).")
    if y is None:
        st.info("Trag in der Seitenleiste einen Benchmark-Ticker ein (z. B. ^GSPC).")
    else:
        w_all, r2 = style_fit(R.values, y, lo, hi)
        st.write(f"Mischung, die **{bench}** am besten nachbildet. R² gesamt: **{de(r2, 2)}**")
        last = np.full(len(assets), np.nan)
        if T >= win + 3:
            rows = [style_fit(R.values[t - win:t], y[t - win:t], lo, hi)[0] for t in range(win, T + 1)]
            roll = pd.DataFrame(rows, columns=[label(t) for t in assets], index=R.index[win - 1:].astype(str))
            fs = go.Figure([go.Scatter(x=roll.index, y=roll[c], stackgroup="one", name=c) for c in roll])
            fs.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fs, width="stretch")
            last = roll.iloc[-1].values
        st.dataframe(pd.DataFrame({"Wertpapier": [label(t) for t in assets], "Gesamtzeitraum": [pc(x) for x in w_all],
                                   "Letztes Fenster": [pc(x) if x == x else "–" for x in last]}),
                     hide_index=True, width="stretch")
        top = np.argsort(-w_all)[:2]
        explain([
            f"Den Index {bench} bildet man mit deinen Wertpapieren am besten vor allem mit "
            + " und ".join(f"{label(assets[i])} ({pc(w_all[i], 0)})" for i in top) + " nach.",
            f"R² = {de(r2, 2)}: " + ("Die Mischung erklärt den Index sehr gut." if r2 > 0.8 else
                                     "Die Mischung erklärt den Index nur teilweise – deine Werte decken ihn nicht ganz ab."
                                     if r2 > 0.5 else "Die Mischung erklärt den Index kaum – deine Werte haben wenig "
                                     "mit ihm zu tun."),
            "Im Diagramm siehst du, wie sich der „Stil“ über die Zeit verschiebt. Starke Sprünge heißen: Der Index "
            "verhielt sich mal wie die eine, mal wie die andere Gruppe deiner Werte." if T >= win + 3 else "",
        ])

with t2:
    st.subheader("Risikokennzahlen (in-sample)",
                 help="VaR 95 %: Monatsverlust, der in 95 % der Monate nicht überschritten wurde. CVaR 95 %: "
                      "Ø Verlust in den schlechtesten 5 % der Monate.")
    rk = pd.DataFrame(M).T
    if y is not None:
        rk.loc[f"Benchmark {bench}"] = metrics(y, rf)
    st.dataframe(rk.style.format({c: (lambda x: de(x, 2)) if c == "Sharpe" else pc for c in rk.columns}), width="stretch")
    explain([
        f"**VaR 95 %** deines jetzigen Portfolios: {pc(cm['VaR 95 %'])}. Übersetzt: In etwa 1 von 20 Monaten verlierst du "
        f"mehr als **{eur(cm['VaR 95 %'] * V)}**. In diesen schlechten Monaten waren es im Schnitt "
        f"{eur(cm['CVaR 95 %'] * V)} (CVaR).",
        "Die optimierten Strategien sehen in dieser Tabelle besser aus, als sie in Wirklichkeit wären, weil sie auf "
        "genau diesen Daten berechnet wurden (in-sample). Vergleiche deshalb den Backtest.",
    ])
    st.subheader("Gewichte je Strategie")
    st.dataframe(pd.DataFrame(W, index=[label(t) for t in assets]).style.format(pc), width="stretch")

with t3:
    st.subheader("Out-of-Sample-Backtest")
    if BT is None:
        st.info("Zu wenige Monate (mindestens 24 Training + 6 Test).")
    else:
        st.caption(f"Training {R.index[0]}–{R.index[s_ - 1]}, Test {R.index[s_]}–{R.index[-1]}, monatlich rebalanciert, Start = 100.")
        fb = go.Figure([go.Scatter(x=R.index[s_:].astype(str), y=v, name=k) for k, v in CURVES.items()])
        fb.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fb, width="stretch")
        bt = pd.DataFrame(BT).T[["Rendite p.a.", "Vol. p.a.", "Sharpe", "Max. Drawdown"]]
        st.dataframe(bt.style.format({c: (lambda x: de(x, 2)) if c == "Sharpe" else pc for c in bt.columns}), width="stretch")
        best = bt["Sharpe"].idxmax()
        best_txt = {"Aktuell": "Dein jetziges Portfolio war im Test am besten – Umschichten hätte hier nicht geholfen.",
                    "Gleichgewichtet": "Gleichmäßig verteilen war am besten. Das ist typisch: Es braucht keine Schätzung "
                                       "und kann sich deshalb nicht verschätzen.",
                    "Min-Varianz": "Das ruhigste Portfolio war am besten. Das ist häufig so: Es schätzt keine Renditen "
                                   "und ist dadurch robuster als Max-Sharpe.",
                    "Benchmark": "Der Index selbst war am besten – ein breiter ETF wäre die einfachste Lösung gewesen."}
        explain([f"Im Testzeitraum hatte **{best}** das beste Verhältnis von Gewinn zu Risiko (Sharpe "
                 f"{de(bt.loc[best, 'Sharpe'], 2)}).",
                 best_txt.get(best, "Dass eine renditebasierte Strategie vorne liegt, kann am Testzeitraum liegen – "
                                    "bei einem anderen Zeitraum kann es anders aussehen.")])

with t4:
    st.subheader(f"Bootstrap: Endwert nach {hor} Jahren (Start {eur(V)})")
    sm = pd.DataFrame(SIM).T
    st.dataframe(sm.style.format({"5 %": eur, "Median": eur, "95 %": eur, "P(Verlust)": lambda x: pc(x, 0)}), width="stretch")

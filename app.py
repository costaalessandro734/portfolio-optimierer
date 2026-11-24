"""Portfolio-Check – einsteigerfreundlicher Portfolio-Optimierer mit Seitenmenü (Streamlit).

Start lokal:  streamlit run app.py
"""
import io
import json
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
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def sg(x, d=1):
    """Mit Vorzeichen; ASCII-Minus, damit Streamlit-Kennzahlen die Richtung (Pfeil/Farbe) richtig erkennen."""
    return ("+" if x > 0 else "") + de(x, d).replace("−", "-")


def eur(x):
    return de(x, 0) + " €"


def eur2(x):
    return "–" if x is None or x != x else de(x, 2) + " €"


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
ss.setdefault("tr_map", {})       # ISIN -> Yahoo-Ticker (Trade-Republic-Import)
ss.setdefault("pos", {})          # Ticker -> {"qty", "cost", "date"}
ss.setdefault("input_mode", "Einfach: nur Betrag in €")
ss.setdefault("tax", 26.4)
ss.setdefault("newmoney", 0.0)
ss.setdefault("nosell", False)
for k, v in {"sp_rate": 200.0, "sp_years": 20, "sp_goal": 100000.0, "sp_dyn": 0, "sp_infl": False,
             "wd_years": 25, "wd_infl": True, "rep_stress": True,
             "sp_port": "Mein jetziges Portfolio", "wd_port": "Mein jetziges Portfolio"}.items():
    ss.setdefault(k, v)
for k in ("sel", "bulk", "prefer_de", "replace_sel", "q", "input_mode", "tax", "newmoney", "nosell", "sp_rate",
          "sp_years", "sp_goal", "sp_dyn", "sp_infl", "sp_start", "wd_rate", "wd_years", "wd_infl", "wd_start",
          "rep_stress", "sp_port", "wd_port", "bm_choice", "stress_pick"):      # Widgets auf anderen Seiten nicht vergessen
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
    "KGV (Kurs-Gewinn-Verhältnis)": "Aktienkurs geteilt durch Gewinn pro Aktie. Wie viele Jahresgewinne du für das "
                                    "Unternehmen bezahlst. Grob: unter 12 günstig, 12–25 üblich, über 25 teuer.",
    "Börsenwert (Marktkapitalisierung)": "Was ein Unternehmen an der Börse insgesamt wert ist: Kurs × Anzahl Aktien.",
    "Dividendenrendite": "Jährliche Ausschüttung an die Aktionäre im Verhältnis zum Aktienkurs.",
    "Gewinnmarge": "Wie viel vom Umsatz als Gewinn übrig bleibt.",
    "Beta": "Wie stark eine Aktie im Vergleich zum Gesamtmarkt schwankt. 1 = wie der Markt.",
    "Anleihen": "Kredite an Staaten oder Unternehmen, die Zinsen zahlen. Schwanken meist weniger als Aktien.",
    "ETF": "Ein börsengehandelter Fonds, der viele Wertpapiere bündelt (z. B. alle großen Firmen der Welt).",
    "Ø Kaufkurs (Einstandskurs)": "Was du im Durchschnitt pro Stück bezahlt hast – bei mehreren Käufen der "
                                  "Mittelwert. Steht in deinem Depot.",
    "Realisierter Gewinn": "Ein Gewinn wird erst „echt“, wenn du verkaufst. Erst dann fällt auch Steuer an.",
    "Rebalancing": "Das Depot wieder auf die gewünschte Aufteilung bringen – durch Verkaufen oder steuerschonend nur "
                   "durch gezieltes Nachkaufen mit neuem Geld.",
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
# Aktuelle Kurse, Wechselkurse, Depot
# =====================================================================
EUR_SUFFIX = (".DE", ".F", ".PA", ".AS", ".MC", ".MI", ".BR", ".VI", ".HE", ".LS", ".IR")


@st.cache_data(ttl=900, show_spinner=False)
def get_quote(t):
    """Letzter Kurs (nicht dividendenbereinigt) und Währung."""
    price = None
    try:
        df = yf.download(t, period="5d", auto_adjust=False, progress=False)
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        price = float(s.iloc[-1]) if len(s) else None
    except Exception:
        pass
    try:
        cur_ = yf.Ticker(t).info.get("currency")
    except Exception:
        cur_ = None
    return price, cur_ or ("EUR" if t.upper().endswith(EUR_SUFFIX) else "USD")


@st.cache_data(ttl=3600, show_spinner=False)
def get_fx(cur_):
    """Euro pro 1 Einheit der Kurswährung (z. B. USD → 0,92)."""
    if cur_ in (None, "EUR"):
        return 1.0
    div = 1.0
    if cur_ in ("GBp", "GBX"):
        cur_, div = "GBP", 100.0
    try:
        df = yf.download(f"EUR{cur_.upper()}=X", period="5d", progress=False)
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return 1.0 / (float(s.iloc[-1]) * div) if len(s) else None
    except Exception:
        return None


def depot(assets):
    """Tabelle mit Stück, Kaufkurs, aktuellem Wert und Gewinn/Verlust je Position (alles in €)."""
    rows = []
    for t in assets:
        p_ = ss.pos.get(t, {})
        qty, cost, dt = float(p_.get("qty") or 0), float(p_.get("cost") or 0), p_.get("date")
        price, cur_ = get_quote(t)
        fx = get_fx(cur_)
        pe = price * fx if price is not None and fx is not None else None
        value = qty * pe if pe is not None else 0.0
        inv = qty * cost
        gain = value - inv if inv > 0 and pe is not None else None
        yrs = (date.today() - dt).days / 365.25 if dt else None
        pa = ((value / inv) ** (1 / yrs) - 1) if gain is not None and yrs and yrs >= 0.5 and value > 0 else None
        rows.append({"t": t, "qty": qty, "cost": cost, "date": dt, "price": price, "cur": cur_, "fx": fx,
                     "price_eur": pe, "value": value, "invested": inv, "gain": gain,
                     "gain_pct": gain / inv if gain is not None else None, "years": yrs, "pa": pa})
    return pd.DataFrame(rows).set_index("t")


def plan(cur_v, tw, new_money, no_sell):
    """Euro-Beträge pro Position, um auf die Zielgewichte zu kommen (optional nur durch Nachkaufen)."""
    total = cur_v.sum() + new_money
    want = tw * total
    if not no_sell:
        return want - cur_v
    gap = np.maximum(0, want - cur_v)
    return new_money * gap / gap.sum() if gap.sum() > 0 else tw * new_money


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
# Speichern & Laden
# =====================================================================
SAVE_KEYS = ("sel", "names", "amounts", "bounds", "input_mode", "goal", "years", "hor", "tax", "expert", "bench",
             "newmoney", "nosell", "sp_start", "sp_rate", "sp_years", "sp_goal", "sp_dyn", "sp_infl", "wd_start", "wd_rate", "wd_years", "wd_infl", "bm_choice")


def state_to_json():
    data = {k: ss.get(k) for k in SAVE_KEYS if k in ss}
    data["pos"] = {t: {"qty": v.get("qty"), "cost": v.get("cost"),
                       "date": v["date"].isoformat() if v.get("date") else None} for t, v in ss.pos.items()}
    data["bounds"] = {t: list(v) for t, v in ss.bounds.items()}
    data["app"], data["version"], data["saved"] = "Portfolio-Check", 1, date.today().isoformat()
    return json.dumps(data, ensure_ascii=False, indent=1)


def apply_load():
    f = ss.get("loadfile")
    if f is None:
        return
    try:
        data = json.loads(f.getvalue().decode("utf-8"))
        if data.get("app") != "Portfolio-Check":
            raise ValueError("Das ist keine gespeicherte Portfolio-Check-Datei.")
        for k in ("sel", "names", "amounts", "input_mode", "tax", "expert", "bench", "years", "hor",
                  "newmoney", "nosell", "sp_start", "sp_rate", "sp_years", "sp_goal", "sp_dyn", "sp_infl", "wd_start", "wd_rate", "wd_years", "wd_infl", "bm_choice"):
            if k in data and data[k] is not None:
                ss[k] = data[k]
        if data.get("goal") in GOALS:
            ss.goal = data["goal"]
        ss.bounds = {t: tuple(v) for t, v in (data.get("bounds") or {}).items()}
        ss.pos = {t: {"qty": v.get("qty") or 0.0, "cost": v.get("cost") or 0.0,
                      "date": date.fromisoformat(v["date"]) if v.get("date") else None}
                  for t, v in (data.get("pos") or {}).items()}
        ss.imp += 1
        ss.load_msg = ("ok", f"Depot vom {data.get('saved', '?')} geladen: {len(ss.sel)} Wertpapiere.")
    except Exception as e:
        ss.load_msg = ("err", f"Datei konnte nicht geladen werden: {e}")


# =====================================================================
# Seitenleiste: Einstellungen (immer sichtbar)
# =====================================================================
sb = st.sidebar
if "load_msg" in ss:
    kind, msg = ss.pop("load_msg")
    (sb.success if kind == "ok" else sb.error)(msg)
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
    young = []
    while True:
        prices = pd.concat({t: series[t] for t in assets + ([bench] if bench else [])}, axis=1).dropna()
        if len(prices) >= 37 or len(assets) <= 2:
            break
        yt = min(assets, key=lambda t: len(series[t]))      # jüngstes Wertpapier auslassen
        if len(series[yt]) >= len(prices) + 12 or len(assets) - 1 < 2:
            break
        young.append(yt)
        assets = [t for t in assets if t != yt]
    ctx["assets"], ctx["young"] = assets, young
    R_all = prices.pct_change().dropna()
    R, y = R_all[assets], (R_all[bench].values if bench else None)
    T = len(R)
    if T < 25:
        return ctx | {"err": f"Für diese Auswahl gibt es nur {T} gemeinsame Monate Kursgeschichte (mindestens 25 "
                             "nötig). Schau mehr Jahre zurück oder entferne sehr junge Wertpapiere."}
    dep = None
    if ss.input_mode.startswith("Genau"):
        dep = depot(assets)
        vals = dep["value"].values.astype(float)
    else:
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
    return ctx | res | dict(dep=dep, R=R, y=y, T=T, V=V, cur=cur, lo=lo, hi=hi, dmax=dmax, target=target,
                            tw=res["W"][target], names=[label(t) for t in assets], C=R.corr().values,
                            goal_txt=ss.goal if target == GOALS[ss.goal] else "Bestes Verhältnis von Gewinn zu Risiko")


def ready():
    c = get_ctx()
    if c["failed"]:
        st.warning("Für **" + ", ".join(c["failed"]) + "** wurden keine Kurse gefunden – sie werden ausgelassen.")
    if c.get("young"):
        st.info("Noch zu junge Kurshistorie, deshalb in dieser Analyse ausgelassen: **" +
                ", ".join(label(t) for t in c["young"]) + "**. Sonst blieben zu wenige gemeinsame Monate für "
                "aussagekräftige Ergebnisse. Ihren Wert siehst du weiterhin auf „Mein Depot“.", icon=":material/info:")
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
    st.radio("Wie willst du dein Depot eingeben?", ["Einfach: nur Betrag in €", "Genau: Stückzahl & Kaufkurs"],
             key="input_mode", horizontal=True,
             help="Einfach: Du trägst nur den heutigen Wert in Euro ein. Genau: Du trägst ein, wie viele Stück du "
                  "hast und zu welchem Durchschnittskurs du gekauft hast. Dann holt die App den aktuellen Kurs, "
                  "berechnet deinen Gewinn oder Verlust seit Kauf und kann Steuern bei Verkäufen abschätzen.")
    genau = ss.input_mode.startswith("Genau")
    NC = st.column_config.NumberColumn
    if genau:
        dep = depot(assets)
        base = pd.DataFrame({
            "Wertpapier": [label(t) for t in assets], "Ticker": assets,
            "Stück": [dep.loc[t, "qty"] for t in assets],
            "Ø Kaufkurs in €": [dep.loc[t, "cost"] for t in assets],
            "Gekauft am": [pd.to_datetime(dep.loc[t, "date"]) if dep.loc[t, "date"] else pd.NaT for t in assets],
            "Kurs heute in €": [dep.loc[t, "price_eur"] for t in assets],
            "Wert in €": [dep.loc[t, "value"] for t in assets],
            "Gewinn/Verlust €": [dep.loc[t, "gain"] for t in assets],
            "Gewinn/Verlust %": [(dep.loc[t, "gain_pct"] * 100) if dep.loc[t, "gain_pct"] is not None else None
                                 for t in assets]})
        cfg = {"Stück": NC(min_value=0.0, step=1.0, format="%.4g",
                           help="Wie viele Anteile du besitzt. Bruchstücke (z. B. 3,25 aus Sparplänen) sind möglich."),
               "Ø Kaufkurs in €": NC(min_value=0.0, step=0.01, format="%.2f €",
                                     help="Dein durchschnittlicher Einstandskurs pro Stück in Euro, so wie er in deinem "
                                          "Depot steht (inkl. Kaufgebühren, wenn dein Broker sie einrechnet)."),
               "Gekauft am": st.column_config.DateColumn(format="DD.MM.YYYY",
                                                         help="Optional. Bei mehreren Käufen das Datum des ersten Kaufs. "
                                                              "Damit berechnet die App den Gewinn pro Jahr."),
               "Kurs heute in €": NC(format="%.2f €", help="Letzter Kurs von Yahoo, bei Fremdwährung in Euro "
                                                           "umgerechnet. Wird alle 15 Minuten aktualisiert."),
               "Wert in €": NC(format="%.0f €", help="Stück × Kurs heute."),
               "Gewinn/Verlust €": NC(format="%+.0f €", help="Wert heute minus investiertes Geld (Stück × Kaufkurs)."),
               "Gewinn/Verlust %": NC(format="%+.1f %%", help="Gewinn oder Verlust im Verhältnis zum investierten Geld.")}
        disabled = ["Wertpapier", "Ticker", "Kurs heute in €", "Wert in €", "Gewinn/Verlust €", "Gewinn/Verlust %"]
    else:
        base = pd.DataFrame({"Wertpapier": [label(t) for t in assets], "Ticker": assets,
                             "Wert in €": [float(ss.amounts.get(t, 1000.0)) for t in assets]})
        cfg = {"Wert in €": NC("Wert in €", min_value=0.0, step=100.0, format="%.0f €",
                               help="Aktueller Wert dieser Position. Grobe Werte reichen.")}
        disabled = ["Wertpapier", "Ticker"]
    if expert:
        base["Min %"] = [ss.bounds.get(t, (0.0, dmax))[0] for t in assets]
        base["Max %"] = [ss.bounds.get(t, (0.0, dmax))[1] for t in assets]
        for col in ("Min %", "Max %"):
            cfg[col] = NC(col, min_value=0.0, max_value=100.0, step=1.0,
                          help="Grenzen für das Gewicht in allen Optimierungen.")
    ed = st.data_editor(base, key=f"ed_{expert}_{genau}_{ss.imp}_" + "_".join(assets), hide_index=True,
                        disabled=disabled, column_config=cfg, width="stretch")
    changed = False
    for _, row in ed.iterrows():
        t = row["Ticker"]
        if genau:
            dt = row["Gekauft am"]
            new = {"qty": float(row["Stück"]) if row["Stück"] == row["Stück"] and row["Stück"] is not None else 0.0,
                   "cost": float(row["Ø Kaufkurs in €"]) if row["Ø Kaufkurs in €"] == row["Ø Kaufkurs in €"]
                   and row["Ø Kaufkurs in €"] is not None else 0.0,
                   "date": None if dt is None or pd.isna(dt) else pd.Timestamp(dt).date()}
            if ss.pos.get(t) != new:
                ss.pos[t], changed = new, True
        else:
            ss.amounts[t] = float(row["Wert in €"] or 0)
        if expert:
            ss.bounds[t] = (float(row["Min %"] or 0), float(row["Max %"] if row["Max %"] == row["Max %"] else 100))
    if changed:
        st.rerun()
    if genau:
        dep = depot(assets)
        miss = [label(t) for t in assets if dep.loc[t, "price_eur"] is None]
        if miss:
            st.warning("Für " + ", ".join(miss) + " konnte kein aktueller Kurs oder Wechselkurs geladen werden.")
        inv, val = dep["invested"].sum(), dep["value"].sum()
        nocost = [label(t) for t in assets if dep.loc[t, "qty"] > 0 and dep.loc[t, "cost"] <= 0]
        if nocost:
            st.caption("Ohne Kaufkurs (zählt nicht zum Gewinn/Verlust): " + ", ".join(nocost))
        if inv > 0:
            g = float(dep["gain"].dropna().sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Depotwert heute", eur(val), help="Summe aller Positionen zum aktuellen Kurs.")
            m2.metric("Investiert", eur(inv), help="Summe aus Stück × Ø Kaufkurs.")
            m3.metric("Gewinn / Verlust", ("+" if g >= 0 else "−") + eur(abs(g)), f"{sg(g / inv * 100)} %",
                      help="Wert heute minus investiertes Geld. Details auf der Seite „Mein Depot“.")
        elif val == 0:
            st.info("Trag bei jedem Wertpapier ein, wie viele **Stück** du hast und deinen **Ø Kaufkurs**.")
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



def page_depot():
    head("Mein Depot", "Wie stehen deine Positionen seit dem Kauf? Gewinne, Verluste und wie stark jede Position "
                       "dein Depot prägt.",
         "Kurse von Yahoo Finance (alle 15 Minuten aktualisiert), Fremdwährungen zum aktuellen Wechselkurs in Euro "
         "umgerechnet. Dein Kaufkurs wird so übernommen, wie du ihn eingetragen hast.")
    if not ss.input_mode.startswith("Genau"):
        st.info("Für diese Seite brauchst du die **genaue Eingabe** (Stückzahl & Kaufkurs). Stell sie auf "
                "**Mein Portfolio** unter Schritt 2 um.")
        st.page_link(P_INPUT, label="Zu „Mein Portfolio“", icon=":material/edit:")
        return
    assets = [t for t in dict.fromkeys(ss.sel)]
    with st.spinner("Lade aktuelle Kurse …"):
        dep = depot(assets)
    dep = dep[(dep["qty"] > 0)]
    if dep.empty:
        st.info("Trag auf **Mein Portfolio** bei deinen Wertpapieren Stückzahl und Ø Kaufkurs ein.")
        st.page_link(P_INPUT, label="Zu „Mein Portfolio“", icon=":material/edit:")
        return
    val, inv = dep["value"].sum(), dep.loc[dep["invested"] > 0, "invested"].sum()
    known = dep[dep["gain"].notna()]
    g = known["gain"].sum()
    nocost = [label(t) for t in dep.index if dep.loc[t, "cost"] <= 0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Depotwert heute", eur(val), help="Summe aller Positionen zum aktuellen Kurs in Euro.")
    k2.metric("Investiert", eur(inv), help="Summe aus Stück × Ø Kaufkurs.")
    k3.metric("Gewinn / Verlust", ("+" if g >= 0 else "−") + eur(abs(g)), f"{sg(g / inv * 100) if inv else '–'} %",
              help="Wert heute minus investiertes Geld, über alle Positionen mit Kaufkurs.")
    k4.metric("Positionen im Plus", f"{int((known['gain'] > 0).sum())} von {len(known)}",
              help="Wie viele deiner Positionen aktuell über dem Kaufkurs stehen.")

    if nocost:
        st.caption("Ohne Kaufkurs (zählt nicht zu Gewinn/Verlust): " + ", ".join(nocost) + ". Trag den Ø Kaufkurs auf "
                   "„Mein Portfolio“ nach.")
    names = [label(t) for t in dep.index]
    rows = []
    for t, n in zip(dep.index, names):
        r_ = dep.loc[t]
        rows.append({"Wertpapier": n, "Stück": de(r_["qty"], 2), "Ø Kaufkurs": eur2(r_["cost"]),
                     "Kurs heute": eur2(r_["price_eur"]), "Wert": eur(r_["value"]), "Anteil": pc(r_["value"] / val, 0),
                     "Investiert": eur(r_["invested"]) if r_["invested"] else "–",
                     "Gewinn/Verlust": (("+" if r_["gain"] >= 0 else "−") + eur(abs(r_["gain"]))) if r_["gain"] is not None and r_["gain"] == r_["gain"] else "–",
                     "in %": (sg(r_["gain_pct"] * 100) + " %") if r_["gain_pct"] is not None and r_["gain_pct"] == r_["gain_pct"] else "–",
                     "Gekauft": r_["date"].strftime("%d.%m.%Y") if r_["date"] else "–",
                     "pro Jahr": (sg(r_["pa"] * 100) + " %") if r_["pa"] is not None and r_["pa"] == r_["pa"] else "–"})
    TC = st.column_config.TextColumn
    st.subheader("Deine Positionen", help="Alle Beträge in Euro.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(700, 42 + 36 * len(rows)),
                 column_config={"Anteil": TC(help="Anteil am gesamten Depotwert."),
                                "Gewinn/Verlust": TC(help="Wert heute minus investiertes Geld. Noch nicht realisiert – "
                                                          "erst beim Verkauf wird daraus ein echter Gewinn oder Verlust."),
                                "pro Jahr": TC(help="Durchschnittlicher Gewinn pro Jahr seit dem Kaufdatum (nur wenn "
                                                    "ein Datum eingetragen ist und der Kauf mindestens ein halbes Jahr "
                                                    "zurückliegt).")})
    if len(known):
        st.subheader("Gewinn und Verlust je Position", help="Grün = im Plus, Rot = im Minus seit Kauf.")
        kk = known.sort_values("gain")
        fg = go.Figure(go.Bar(y=[label(t) for t in kk.index], x=kk["gain"], orientation="h",
                              marker_color=["#0b7a3b" if x >= 0 else "#e8361b" for x in kk["gain"]],
                              text=[("+" if x >= 0 else "−") + eur(abs(x)) for x in kk["gain"]], textposition="outside"))
        fg.update_layout(height=80 + 50 * len(kk), margin=dict(l=10, r=40, t=10, b=10), xaxis_title="€",
                         separators=",.", font=dict(size=15))
        st.plotly_chart(fg, width="stretch")
    st.subheader("Wie dein Geld verteilt ist", help="Größe der Kachel = Wert der Position. Farbe = Gewinn oder Verlust "
                                                    "in Prozent seit Kauf.")
    gp = [float(x * 100) if x is not None and x == x else 0.0 for x in dep["gain_pct"]]
    tm = go.Figure(go.Treemap(labels=names, parents=[""] * len(names), values=dep["value"].values,
                              marker=dict(colors=gp, colorscale=[[0, "#e8361b"], [0.5, "#f1f1f1"], [1, "#0b7a3b"]],
                                          cmid=0, showscale=True, colorbar=dict(title="% seit Kauf")),
                              texttemplate="<b>%{label}</b><br>%{value:,.0f} €", textfont=dict(size=16)))
    tm.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), separators=",.")
    st.plotly_chart(tm, width="stretch")

    lines = []
    if len(known):
        best, worst = known["gain_pct"].astype(float).idxmax(), known["gain_pct"].astype(float).idxmin()
        lines.append(f"Insgesamt liegst du **{('im Plus' if g >= 0 else 'im Minus')}: {('+' if g >= 0 else '−')}{eur(abs(g))}** "
                     f"({sg(g / inv * 100)} %) auf {eur(inv)} investiertes Geld.")
        lines.append(f"Bester Wert: **{label(best)}** mit {sg(known.loc[best, 'gain_pct'] * 100)} %. "
                     f"Schwächster Wert: **{label(worst)}** mit {sg(known.loc[worst, 'gain_pct'] * 100)} %.")
    top = dep["value"].idxmax()
    if dep.loc[top, "value"] / val > 0.3:
        gpct = dep.loc[top, "gain_pct"]
        lines.append(f"**{label(top)}** macht inzwischen **{pc(dep.loc[top, 'value'] / val, 0)}** deines Depots aus"
                     + (" – vor allem, weil der Wert stark gestiegen ist." if gpct is not None and gpct == gpct and gpct > 0.5 else ".")
                     + " Gewinner wachsen mit der Zeit zu großen Klumpen. Auf **Übersicht & Vorschlag** siehst du, ob "
                       "und wie du das ausgleichen könntest.")
    losers = known[known["gain_pct"].astype(float) < -0.2]
    if len(losers):
        lines.append("Positionen mit über 20 % Verlust: " + ", ".join(label(t) for t in losers.index) + ". Frag dich "
                     "ehrlich: Würdest du diese Aktie **heute** kaufen? Wenn nicht, ist Festhalten nur, um den Verlust "
                     "nicht „wahr“ zu machen, ein bekannter Denkfehler. Verkaufte Verluste lassen sich außerdem mit "
                     "Gewinnen steuerlich verrechnen.")
    lines.append("Solange du nicht verkaufst, sind Gewinne und Verluste nur auf dem Papier. Steuern fallen erst beim "
                 "Verkauf an – das berücksichtigt die Umschichtung auf **Übersicht & Vorschlag**.")
    explain(lines)


def page_overview():
    c = ready()
    R, cur, tw, V, M, W, target = c["R"], c["cur"], c["tw"], c["V"], c["M"], c["W"], c["target"]
    assets, names, mu, S, C, hi = c["assets"], c["names"], c["mu"], c["S"], c["C"], c["hi"]
    tm, cm = M[target], M["Aktuell"]
    head("Übersicht & Vorschlag", f"Ziel: <b>{c['goal_txt']}</b>. Hier siehst du auf einen Blick, wie dein Portfolio "
                                  f"in den letzten {years} Jahren lief und was du ändern könntest.")
    d = tw - cur
    dep = c["dep"]
    with st.container(border=True):
        st.markdown("**Wie willst du umschichten?**")
        o1, o2, o3 = st.columns(3)
        newm = o1.number_input("Neues Geld zum Investieren (€)", min_value=0.0, step=100.0, key="newmoney",
                               help="Willst du zusätzliches Geld anlegen? Dann wird es so verteilt, dass dein Depot "
                                    "näher an den Vorschlag rückt.")
        nosell = o2.toggle("Nichts verkaufen, nur nachkaufen", key="nosell",
                           help="Statt Gewinner zu verkaufen (kostet Steuern und Gebühren), wird nur das neue Geld "
                                "gezielt in die Werte gesteckt, die laut Vorschlag zu klein sind. So nähert sich dein "
                                "Depot dem Ziel langsamer, aber steuerschonend.")
        taxr = o3.number_input("Steuersatz auf Gewinne %", min_value=0.0, max_value=60.0, step=0.1, key="tax",
                               help="Für die Schätzung der Steuer bei Verkäufen. Deutschland: ca. 26,4 % "
                                    "(Abgeltungsteuer + Soli, ggf. + Kirchensteuer), Österreich 27,5 %, Italien 26 %. "
                                    "Freibeträge und Verlusttöpfe werden nicht berücksichtigt.") / 100
        if nosell and newm <= 0:
            st.caption("Für „nur nachkaufen“ trag links ein, wie viel neues Geld du investieren willst.")
    cur_v = cur * V
    d_eur = plan(cur_v, tw, newm, nosell)
    after = cur_v + d_eur
    thr = 0.005 * max(V + newm, 1)
    moves = [(names[i], d_eur[i]) for i in range(len(assets)) if abs(d_eur[i]) >= thr]
    if not moves:
        st.success("Dein Portfolio ist für dieses Ziel schon gut aufgeteilt. Du musst nichts ändern.")
    else:
        buy = [f"**{n}** (+{eur(x)})" for n, x in moves if x > 0]
        sell = [f"**{n}** (−{eur(-x)})" for n, x in moves if x < 0]
        st.info("**Vorschlag:** " + ("Kaufen: " + ", ".join(buy) + ". " if buy else "") +
                ("Verkaufen: " + ", ".join(sell) + "." if sell else ""))
    realized = np.zeros(len(assets))
    if dep is not None:
        for i, t in enumerate(assets):
            g, v = dep.loc[t, "gain"], dep.loc[t, "value"]
            if d_eur[i] < -thr and g is not None and v > 0:
                realized[i] = min(1.0, -d_eur[i] / v) * g
    tax_total = max(0.0, realized.sum()) * taxr

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
                     help=f"Beträge in Euro, bezogen auf dein Depot von {eur(V)}" + (f" plus {eur(newm)} neues Geld"
                          if newm else "") + ". Änderungen unter 0,5 % werden als „behalten“ gezeigt. Gebühren sind "
                          "nicht eingerechnet." + (" Stückzahlen basieren auf dem aktuellen Kurs." if dep is not None else ""))
        act = [("behalten" if abs(x) < thr else (f"+{eur(x)} kaufen" if x > 0 else f"−{eur(-x)} verkaufen"))
               for x in d_eur]
        tab = {"Wertpapier": names, "Jetzt": [eur(x) for x in cur_v], "Nachher": [eur(x) for x in after], "Was tun?": act}
        TC = st.column_config.TextColumn
        tcfg = {"Nachher": TC(help="Wert der Position, wenn du den Vorschlag umsetzt.")}
        if dep is not None:
            pe = [dep.loc[t, "price_eur"] for t in assets]
            tab["Stück"] = [("–" if abs(x) < thr or not p_ else ("+" if x > 0 else "−") + de(abs(x) / p_, 2))
                            for x, p_ in zip(d_eur, pe)]
            tab["Gewinn realisiert"] = [(("+" if x > 0 else "−") + eur(abs(x))) if x else "–" for x in realized]
            tab["Steuer ca."] = [eur(max(0, x) * taxr) if x > 0 else "–" for x in realized]
            tcfg |= {"Stück": TC(help="Wie viele Anteile du kaufen (+) oder verkaufen (−) müsstest, zum aktuellen Kurs."),
                     "Gewinn realisiert": TC(help="Der Teil deines Gewinns (oder Verlusts), den du durch den Verkauf "
                                                  "„mitnimmst“. Nur realisierte Gewinne werden besteuert."),
                     "Steuer ca.": TC(help="Geschätzte Steuer auf den realisierten Gewinn mit deinem Steuersatz – "
                                           "ohne Freibetrag und ohne Verrechnung mit anderen Verlusten.")}
        st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch", column_config=tcfg,
                     height=min(600, 42 + 36 * len(assets)))
        if dep is not None and (realized != 0).any():
            s1, s2 = st.columns(2)
            s1.metric("Gewinne, die du realisierst", ("+" if realized.sum() >= 0 else "−") + eur(abs(realized.sum())),
                      help="Summe aller realisierten Gewinne und Verluste aus den Verkäufen. Verluste verringern die Steuer.")
            s2.metric("Geschätzte Steuer", eur(tax_total), help="Steuersatz × realisierte Gewinne (nach Verrechnung "
                                                                 "mit realisierten Verlusten), ohne Freibeträge.")
    with right:
        st.subheader("Aufteilung jetzt und nachher", help="Anteil jedes Wertpapiers an deinem Geld. „Ziel“ ist die "
                                                          "ideale Aufteilung; „Nachher“ das, was du mit dieser "
                                                          "Umschichtung erreichst (bei „nur nachkaufen“ oft noch nicht "
                                                          "ganz das Ziel).")
        fig = go.Figure()
        fig.add_bar(y=names, x=cur, orientation="h", name="Jetzt", marker_color="#b8c4e8")
        fig.add_bar(y=names, x=after / max(after.sum(), 1e-9), orientation="h", name="Nachher", marker_color="#1240e0")
        if nosell:
            fig.add_scatter(y=names, x=tw, mode="markers", name="Ziel", marker=dict(symbol="line-ns-open", size=22,
                                                                                    color="#e8361b", line=dict(width=3)))
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
    if dep is not None and moves:
        if tax_total > 1:
            lines.append(f"**Steuern:** Die Umsetzung würde etwa **{eur(tax_total)}** Steuern kosten "
                         f"({pc(tax_total / max(V, 1))} deines Depots). Dieser Betrag ist sofort weg – der Vorschlag muss "
                         "ihn erst wieder einspielen. Mit **„Nichts verkaufen, nur nachkaufen“** vermeidest du das und "
                         "näherst dich dem Ziel Schritt für Schritt mit neuem Geld.")
        elif realized.sum() < -1:
            lines.append(f"**Steuern:** Die Verkäufe würden insgesamt **Verluste von {eur(-realized.sum())}** realisieren. "
                         "Die kannst du mit anderen Gewinnen verrechnen – das kann die Steuer an anderer Stelle senken.")
    if nosell and newm > 0:
        gap_left = np.abs(after / max(after.sum(), 1e-9) - tw).sum() / 2
        lines.append(f"**Nur nachkaufen:** Mit {eur(newm)} neuem Geld kommst du dem Ziel näher; es bleibt noch eine "
                     f"Abweichung von etwa {pc(gap_left, 0)} des Depots. Mit weiterem Geld (z. B. Sparplan) schließt sich "
                     "die Lücke nach und nach.")
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
# Unternehmensdaten, Empfehlungen, Nachrichten – Daten
# =====================================================================
INFO_KEYS = ["shortName", "longName", "quoteType", "sector", "industry", "country", "currency", "financialCurrency",
             "marketCap", "trailingPE", "forwardPE", "priceToBook", "totalRevenue", "netIncomeToCommon",
             "profitMargins", "revenueGrowth", "earningsGrowth", "dividendRate", "trailingAnnualDividendYield",
             "currentPrice", "regularMarketPrice", "previousClose", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
             "longBusinessSummary", "category", "totalAssets", "fundFamily", "netExpenseRatio",
             "annualReportExpenseRatio", "website", "fullTimeEmployees", "recommendationKey",
             "numberOfAnalystOpinions", "targetMeanPrice"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_info(t):
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    return {k: info.get(k) for k in INFO_KEYS}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_fin(t):
    try:
        df = yf.Ticker(t).income_stmt
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    out = {}
    for key, rows in {"Umsatz": ["Total Revenue", "TotalRevenue"],
                      "Gewinn": ["Net Income", "NetIncome", "Net Income Common Stockholders"]}.items():
        row = next((df.loc[r] for r in rows if r in df.index), None)
        if row is not None:
            out[key] = pd.to_numeric(row, errors="coerce")
    if not out:
        return None
    res = pd.DataFrame(out).dropna(how="all").sort_index()
    res.index = [str(getattr(i, "year", i)) for i in res.index]
    return res


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(t, n=10):
    try:
        items = yf.Ticker(t).news or []
    except Exception:
        items = []
    out = []
    for it in items[:n]:
        c = it.get("content", it) if isinstance(it, dict) else {}
        title = c.get("title")
        url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
               or c.get("link"))
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else None
        prov = prov or c.get("publisher") or ""
        ts = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
        ts = (pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float))
              else pd.to_datetime(ts, utc=True, errors="coerce"))
        summ = re.sub(r"<[^>]+>", "", c.get("summary") or c.get("description") or "").strip()
        if title and url:
            out.append({"title": title, "url": url, "prov": prov, "ts": ts, "summary": summ[:320], "t": t})
    return out


def fmt_big(x, cur=""):
    if x is None or x != x:
        return "–"
    a = abs(x)
    s = (de(x / 1e12, 2) + " Bio." if a >= 1e12 else de(x / 1e9, 1) + " Mrd." if a >= 1e9
         else de(x / 1e6, 0) + " Mio." if a >= 1e6 else de(x, 0))
    return f"{s} {cur}".strip()


def num(x, d=1):
    return "–" if x is None or x != x else de(x, d)


def pct_or(x, d=1):
    return "–" if x is None or x != x else pc(x, d)


def div_yield(i):
    p = i.get("currentPrice") or i.get("regularMarketPrice") or i.get("previousClose")
    if i.get("dividendRate") and p:
        return i["dividendRate"] / p
    return i.get("trailingAnnualDividendYield")


SECTOR_DE = {"Technology": "Technologie", "Healthcare": "Gesundheit", "Financial Services": "Finanzen",
             "Consumer Defensive": "Basiskonsum", "Consumer Cyclical": "Zyklischer Konsum", "Energy": "Energie",
             "Industrials": "Industrie", "Utilities": "Versorger", "Basic Materials": "Rohstoffe & Chemie",
             "Communication Services": "Kommunikation", "Real Estate": "Immobilien"}

COUNTRY_DE = {"United States": "USA", "Germany": "Deutschland", "United Kingdom": "Großbritannien",
              "Switzerland": "Schweiz", "France": "Frankreich", "Netherlands": "Niederlande", "Denmark": "Dänemark",
              "Japan": "Japan", "China": "China", "Taiwan": "Taiwan", "Ireland": "Irland", "Italy": "Italien",
              "Spain": "Spanien", "Canada": "Kanada", "Sweden": "Schweden", "Malaysia": "Malaysia",
              "South Korea": "Südkorea", "Austria": "Österreich", "Belgium": "Belgien", "Norway": "Norwegen"}


def country(i):
    return COUNTRY_DE.get(i.get("country"), i.get("country"))


PEERS = {
    "Technology": [("MSFT", "Microsoft"), ("AAPL", "Apple"), ("NVDA", "Nvidia"), ("SAP.DE", "SAP"), ("ASML.AS", "ASML")],
    "Healthcare": [("JNJ", "Johnson & Johnson"), ("NOVO-B.CO", "Novo Nordisk"), ("ROG.SW", "Roche"), ("UNH", "UnitedHealth")],
    "Financial Services": [("JPM", "JPMorgan"), ("ALV.DE", "Allianz"), ("MUV2.DE", "Munich Re"), ("V", "Visa")],
    "Consumer Defensive": [("KO", "Coca-Cola"), ("PG", "Procter & Gamble"), ("NESN.SW", "Nestlé"), ("PEP", "PepsiCo")],
    "Consumer Cyclical": [("AMZN", "Amazon"), ("MC.PA", "LVMH"), ("HD", "Home Depot"), ("MCD", "McDonald's")],
    "Energy": [("XOM", "ExxonMobil"), ("SHEL.L", "Shell"), ("TTE.PA", "TotalEnergies")],
    "Industrials": [("SIE.DE", "Siemens"), ("AIR.PA", "Airbus"), ("CAT", "Caterpillar"), ("HON", "Honeywell")],
    "Utilities": [("NEE", "NextEra Energy"), ("RWE.DE", "RWE"), ("EOAN.DE", "E.ON"), ("IBE.MC", "Iberdrola")],
    "Basic Materials": [("LIN", "Linde"), ("BAS.DE", "BASF"), ("RIO.L", "Rio Tinto")],
    "Communication Services": [("GOOGL", "Alphabet"), ("META", "Meta"), ("DTE.DE", "Deutsche Telekom"), ("NFLX", "Netflix")],
    "Real Estate": [("VNA.DE", "Vonovia"), ("PLD", "Prologis"), ("O", "Realty Income")],
}
CAND = {
    "bonds": [("EUNH.DE", "iShares Core € Govt Bond (Euro-Staatsanleihen)"),
              ("IEF", "iShares 7-10 Year Treasury (US-Staatsanleihen, USD)"),
              ("AGG", "iShares Core US Aggregate Bond (USD)")],
    "gold": [("4GLD.DE", "Xetra-Gold (physisch besichert, EUR)"), ("GLD", "SPDR Gold Shares (USD)")],
    "broad": [("EUNL.DE", "iShares Core MSCI World"), ("IS3N.DE", "iShares Core MSCI Emerging Markets IMI"),
              ("VT", "Vanguard Total World Stock (USD)")],
    "defensive": [("XLP", "Basiskonsum-Sektor-ETF (USD)"), ("XLU", "Versorger-Sektor-ETF (USD)"),
                  ("XLV", "Gesundheits-Sektor-ETF (USD)")],
    "intl": [("EXSA.DE", "iShares STOXX Europe 600"), ("IS3N.DE", "iShares Core MSCI Emerging Markets IMI"),
             ("EUNL.DE", "iShares Core MSCI World")],
}


def classify(t, i):
    name = f"{i.get('shortName') or ''} {i.get('longName') or ''} {i.get('category') or ''}".lower()
    qt = (i.get("quoteType") or "").upper()
    if any(k in name for k in ("bond", "treasury", "govt", "anleihe", "aggregate", "fixed income", "rentenfonds")):
        return "Anleihen"
    if any(k in name for k in ("gold", "silver", "silber", "commodity", "rohstoff")) or t in ("GLD", "4GLD.DE"):
        return "Gold & Rohstoffe"
    if qt == "CRYPTOCURRENCY":
        return "Krypto"
    if qt in ("ETF", "MUTUALFUND"):
        return "Aktien-ETF / Fonds"
    if (i.get("sector") or "") == "Real Estate":
        return "Immobilien"
    return "Einzelaktie"


def test_candidate(sym, R, rp_c, rf):
    """Rechnet: Was passiert, wenn 10 % des Portfolios in sym gehen?"""
    s = load(sym, start, end)
    if s is None:
        return None
    r = s.pct_change().dropna().reindex(R.index).dropna()
    if len(r) < 24:
        return None
    base = pd.Series(rp_c, index=R.index).loc[r.index].values
    mix = 0.9 * base + 0.1 * r.values
    m0, m1 = metrics(base, rf), metrics(mix, rf)
    return {"corr": float(np.corrcoef(base, r.values)[0, 1]), "m0": m0, "m1": m1, "n": len(r),
            "own": metrics(r.values, rf)}


def add_test(sym, name):
    add_ticker(sym, name)
    ss.amounts[sym] = 0.0
    ss.added_msg = name


# =====================================================================
# Seite: Unternehmensdaten
# =====================================================================
def page_info():
    c = ready()
    assets, names = c["assets"], c["names"]
    head("Unternehmensdaten", "Kennzahlen zu deinen Aktien und ETFs: Bewertung (KGV), Umsatz, Gewinn, Dividende und "
                              "mehr. Daten von Yahoo Finance.",
         "Die Zahlen kommen von Yahoo Finance und werden einmal pro Tag aktualisiert. Beträge sind in der Währung des "
         "Unternehmens angegeben. Bei ETFs gibt es keine Unternehmenskennzahlen, dafür Kosten und Fondsgröße.")
    with st.spinner("Lade Unternehmensdaten …"):
        infos = {t: get_info(t) for t in assets}
    if not any(v.get("quoteType") or v.get("shortName") for v in infos.values()):
        st.warning("Yahoo Finance liefert gerade keine Unternehmensdaten (vermutlich zu viele Anfragen). Versuch es in "
                   "ein paar Minuten noch einmal – die Analyse unten ist bis dahin unvollständig.")
    H = st.column_config.TextColumn
    rows = []
    for t, n in zip(assets, names):
        i = infos[t]
        rows.append({"Wertpapier": n, "Typ": classify(t, i), "Sektor": SECTOR_DE.get(i.get("sector"), i.get("sector") or "–"),
                     "Land": country(i) or "–", "Börsenwert": fmt_big(i.get("marketCap"), i.get("currency") or ""),
                     "KGV": num(i.get("trailingPE")), "KGV erwartet": num(i.get("forwardPE")),
                     "Dividende": pct_or(div_yield(i)), "Gewinnmarge": pct_or(i.get("profitMargins")),
                     "Umsatzwachstum": pct_or(i.get("revenueGrowth"))})
    st.subheader("Alle Werte im Überblick", help="Klick auf eine Spaltenüberschrift sortiert die Tabelle.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(700, 42 + 36 * len(rows)),
                 column_config={
                     "Typ": H(help="Einzelaktie, Aktien-ETF/Fonds, Anleihen, Gold & Rohstoffe usw."),
                     "Börsenwert": H(help="Marktkapitalisierung: Was das ganze Unternehmen an der Börse wert ist "
                                          "(Aktienkurs × Anzahl Aktien)."),
                     "KGV": H(help="Kurs-Gewinn-Verhältnis: Aktienkurs geteilt durch Gewinn pro Aktie der letzten 12 "
                                   "Monate. Wie viele Jahresgewinne du für das Unternehmen bezahlst. Grob: unter 12 "
                                   "günstig, 12–25 üblich, über 25 teuer – je nach Branche sehr verschieden."),
                     "KGV erwartet": H(help="Wie das KGV, aber mit dem erwarteten Gewinn der nächsten 12 Monate "
                                            "(Analystenschätzung). Niedriger als das normale KGV = Gewinne sollen steigen."),
                     "Dividende": H(help="Dividendenrendite: jährliche Ausschüttung im Verhältnis zum Aktienkurs."),
                     "Gewinnmarge": H(help="Wie viel vom Umsatz als Gewinn übrig bleibt. 20 % = von 100 € Umsatz "
                                           "bleiben 20 € Gewinn."),
                     "Umsatzwachstum": H(help="Veränderung des Umsatzes im letzten Quartal gegenüber dem Vorjahresquartal."),
                 })
    st.divider()
    k = st.selectbox("Wertpapier im Detail ansehen", range(len(assets)), format_func=lambda j: names[j],
                     help="Wähle ein Wertpapier, um alle Details zu sehen.")
    t, i = assets[k], infos[assets[k]]
    cur_ = i.get("currency") or ""
    fcur = i.get("financialCurrency") or cur_
    price = i.get("currentPrice") or i.get("regularMarketPrice") or i.get("previousClose")
    st.header(f"{names[k]} ({t})")
    st.caption(" · ".join(x for x in [classify(t, i), SECTOR_DE.get(i.get("sector"), i.get("sector")),
                                      i.get("industry"), country(i)] if x))
    is_fund = classify(t, i) in ("Aktien-ETF / Fonds", "Anleihen") or (i.get("quoteType") or "") in ("ETF", "MUTUALFUND")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Kurs", f"{num(price, 2)} {cur_}", help="Letzter verfügbarer Kurs.")
    if is_fund:
        ter = i.get("netExpenseRatio") or i.get("annualReportExpenseRatio")
        a2.metric("Fondsvolumen", fmt_big(i.get("totalAssets"), cur_), help="Wie viel Geld insgesamt im Fonds steckt. "
                  "Größere Fonds sind meist günstiger und werden seltener geschlossen.")
        a3.metric("Laufende Kosten", "–" if ter is None else (de(ter, 2) + " %" if ter > 0.05 else pc(ter, 2)),
                  help="TER: Kosten pro Jahr, die vom Fondsvermögen abgezogen werden. Unter 0,3 % ist günstig.")
        a4.metric("Kategorie", i.get("category") or "–", help="Anlageschwerpunkt laut Yahoo/Morningstar.")
    else:
        a2.metric("Börsenwert", fmt_big(i.get("marketCap"), cur_), help="Marktkapitalisierung: Kurs × Anzahl Aktien.")
        a3.metric("KGV", num(i.get("trailingPE")), help="Kurs-Gewinn-Verhältnis (letzte 12 Monate).")
        a4.metric("Dividendenrendite", pct_or(div_yield(i)), help="Jährliche Ausschüttung im Verhältnis zum Kurs.")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Umsatz (12 Monate)", fmt_big(i.get("totalRevenue"), fcur), help="Gesamte Verkaufserlöse der letzten 12 Monate.")
        b2.metric("Gewinn (12 Monate)", fmt_big(i.get("netIncomeToCommon"), fcur), help="Nettogewinn nach Steuern, "
                  "der den Aktionären zusteht.")
        b3.metric("Gewinnmarge", pct_or(i.get("profitMargins")), help="Gewinn geteilt durch Umsatz.")
        b4.metric("Beta", num(i.get("beta"), 2), help="Wie stark die Aktie im Vergleich zum Gesamtmarkt schwankt. "
                  "1 = wie der Markt, 1,5 = 50 % stärker, 0,5 = halb so stark.")
    lo52, hi52 = i.get("fiftyTwoWeekLow"), i.get("fiftyTwoWeekHigh")
    if lo52 and hi52 and price and hi52 > lo52:
        pos = (price - lo52) / (hi52 - lo52)
        st.markdown(f"**52-Wochen-Spanne:** {num(lo52, 2)} – {num(hi52, 2)} {cur_}")
        st.progress(min(1.0, max(0.0, pos)), text=f"Aktueller Kurs liegt bei {pc(pos, 0)} der Jahresspanne "
                                                   "(0 % = Jahrestief, 100 % = Jahreshoch)")
    fin = None if is_fund else get_fin(t)
    if fin is not None and len(fin):
        st.subheader("Umsatz und Gewinn der letzten Jahre", help=f"Geschäftsjahre laut Jahresabschluss, in {fcur}.")
        fg = go.Figure()
        if "Umsatz" in fin:
            fg.add_bar(x=fin.index, y=fin["Umsatz"], name="Umsatz", marker_color="#b8c4e8")
        if "Gewinn" in fin:
            fg.add_bar(x=fin.index, y=fin["Gewinn"], name="Gewinn", marker_color="#1240e0")
        fg.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=10), separators=",.",
                         legend=dict(orientation="h", y=1.08), font=dict(size=15))
        st.plotly_chart(fg, width="stretch")
    # Interpretation
    lines = []
    if is_fund:
        ter = i.get("netExpenseRatio") or i.get("annualReportExpenseRatio")
        if ter is not None:
            terp = ter / 100 if ter > 0.05 else ter
            lines.append(f"Die laufenden Kosten liegen bei {de(terp * 100, 2)} % pro Jahr – "
                         + ("günstig." if terp < 0.003 else "im üblichen Bereich." if terp < 0.006 else
                            "eher teuer. Vergleichbare ETFs gibt es oft günstiger."))
        lines.append("Ein ETF bündelt viele Wertpapiere. Er ist dadurch schon in sich gestreut – Kennzahlen wie KGV "
                     "oder Umsatz beziehen sich bei ETFs nicht auf ein einzelnes Unternehmen.")
    else:
        pe, fpe = i.get("trailingPE"), i.get("forwardPE")
        if pe is None:
            lines.append("Es gibt kein KGV – meist, weil das Unternehmen zuletzt **Verlust** gemacht hat. Dann ist die "
                         "Aktie besonders schwer zu bewerten.")
        else:
            lines.append(f"**KGV {num(pe)}:** Du zahlst etwa das {num(pe, 0)}-Fache des Jahresgewinns. " +
                         ("Das ist niedrig – entweder günstig, oder der Markt erwartet Probleme." if pe < 12 else
                          "Das ist ein üblicher Wert." if pe < 25 else
                          "Das ist hoch – der Markt erwartet starkes Wachstum. Bleibt es aus, kann der Kurs stark fallen."
                          if pe < 40 else
                          "Das ist sehr hoch – hier ist viel Zukunftsfantasie eingepreist."))
        if pe and fpe:
            lines.append("Das erwartete KGV ist niedriger: Analysten rechnen mit **steigenden Gewinnen**." if fpe < pe * 0.95
                         else "Das erwartete KGV ist höher: Analysten rechnen mit **sinkenden Gewinnen**." if fpe > pe * 1.05
                         else "Analysten erwarten ungefähr gleichbleibende Gewinne.")
        pm = i.get("profitMargins")
        if pm is not None:
            lines.append(f"**Gewinnmarge {pc(pm, 0)}:** " + ("Sehr profitabel." if pm > 0.2 else
                         "Solide profitabel." if pm > 0.08 else "Eher geringe Marge – typisch für Handel, Industrie "
                         "oder Unternehmen unter Druck." if pm > 0 else "Das Unternehmen macht derzeit Verlust."))
        b = i.get("beta")
        if b is not None:
            lines.append(f"**Beta {num(b, 2)}:** " + ("Schwankt deutlich stärker als der Gesamtmarkt." if b > 1.3 else
                         "Schwankt ähnlich wie der Markt." if b > 0.8 else "Schwankt weniger als der Markt – eher defensiv."))
        dy = div_yield(i)
        if dy:
            lines.append(f"**Dividende {pc(dy)}:** " + ("Hohe Ausschüttung – prüfe, ob sie durch Gewinne gedeckt ist."
                         if dy > 0.05 else "Solide Ausschüttung." if dy > 0.02 else "Kleine Ausschüttung – das "
                         "Unternehmen investiert seine Gewinne eher in Wachstum."))
        if i.get("recommendationKey") and i.get("numberOfAnalystOpinions"):
            rk = {"strong_buy": "stark kaufen", "buy": "kaufen", "hold": "halten", "underperform": "unterdurchschnittlich",
                  "sell": "verkaufen"}.get(i["recommendationKey"], i["recommendationKey"])
            lines.append(f"Analysten ({i['numberOfAnalystOpinions']}) sagen im Schnitt: **{rk}**"
                         + (f", Kursziel {num(i.get('targetMeanPrice'), 2)} {cur_}." if i.get("targetMeanPrice") else ".")
                         + " Analystenmeinungen liegen oft daneben – nimm sie als eine Stimme von vielen.")
    explain(lines or ["Für dieses Wertpapier liefert Yahoo keine Kennzahlen."])
    if i.get("longBusinessSummary"):
        with st.expander("Was macht das Unternehmen? (Beschreibung von Yahoo, Englisch)"):
            st.write(i["longBusinessSummary"])
            if i.get("website"):
                st.markdown(f"[Zur Website]({i['website']})")


# =====================================================================
# Seite: Portfolio verbessern
# =====================================================================
def page_improve():
    c = ready()
    R, cur, names, assets, V, C = c["R"], c["cur"], c["names"], c["assets"], c["V"], c["C"]
    head("Portfolio verbessern", "Die App prüft, wie breit dein Portfolio aufgestellt ist, und zeigt Ideen für "
                                 "Ergänzungen oder Alternativen – jeweils mit deinen echten Daten durchgerechnet.",
         "Die genannten Wertpapiere sind bekannte Beispiele, keine Kaufempfehlung. Prüfe vor einem Kauf immer "
         "Kosten, Börsenplatz und ob es zu dir passt.")
    if "added_msg" in ss:
        st.success(f"**{ss.added_msg}** wurde mit 0 € zu deiner Auswahl hinzugefügt. Auf **Übersicht & Vorschlag** "
                   "siehst du jetzt, ob der Optimierer ihm Gewicht gibt.")
        del ss.added_msg
    with st.spinner("Analysiere deine Wertpapiere …"):
        infos = {t: get_info(t) for t in assets}
    if not any(v.get("quoteType") or v.get("shortName") for v in infos.values()):
        st.warning("Yahoo Finance liefert gerade keine Unternehmensdaten (vermutlich zu viele Anfragen). Versuch es in "
                   "ein paar Minuten noch einmal – die Analyse unten ist bis dahin unvollständig.")
    cls = [classify(t, infos[t]) for t in assets]
    sec = [SECTOR_DE.get(infos[t].get("sector"), infos[t].get("sector")) if cls[j] in ("Einzelaktie", "Immobilien")
           else ("Breit gestreut (ETF)" if cls[j] == "Aktien-ETF / Fonds" else cls[j]) for j, t in enumerate(assets)]
    sec = [s or "Unbekannt" for s in sec]
    ctry = [(country(infos[t]) or "Unbekannt") if cls[j] == "Einzelaktie" else "ETF / weltweit"
            for j, t in enumerate(assets)]

    st.subheader("So ist dein Geld verteilt", help="Aufteilung nach Anlageklasse, Sektor (bei Einzelaktien) und "
                                                   "Land des Unternehmens. ETFs zählen als breit gestreut.")
    cols = st.columns(3)
    for col, (title, keys) in zip(cols, [("Anlageklasse", cls), ("Sektor", sec), ("Land", ctry)]):
        s = pd.Series(cur, index=keys).groupby(level=0).sum().sort_values(ascending=False)
        fig = go.Figure(go.Pie(labels=s.index, values=s.values, hole=0.55, textinfo="percent",
                               sort=False))
        fig.update_layout(title=dict(text=title, x=0.5), height=360, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.1), font=dict(size=14), separators=",.")
        col.plotly_chart(fig, width="stretch")

    share = lambda arr, val: float(sum(w for w, a in zip(cur, arr) if a == val))
    bonds, gold = share(cls, "Anleihen"), share(cls, "Gold & Rohstoffe")
    broad = share(cls, "Aktien-ETF / Fonds")
    stock_sec = pd.Series(cur, index=sec)[[x in ("Einzelaktie", "Immobilien") and s != "Unbekannt"
                                           for x, s in zip(cls, sec)]].groupby(level=0).sum()
    top_sec, top_sec_w = (stock_sec.idxmax(), float(stock_sec.max())) if len(stock_sec) else (None, 0.0)
    stock_c = pd.Series(cur, index=ctry)[[x == "Einzelaktie" and k != "Unbekannt" for x, k in zip(cls, ctry)]].groupby(level=0).sum()
    top_c, top_c_w = (stock_c.idxmax(), float(stock_c.max())) if len(stock_c) else (None, 0.0)
    avg_c = round((C.sum() - len(assets)) / max(1, len(assets) * (len(assets) - 1)), 2) + 0.0

    checks = []   # (ok, titel, text, kandidaten-schlüssel)
    checks.append((bonds > 0.05, "Anleihen",
                   f"{pc(bonds, 0)} in Anleihen." if bonds > 0.05 else
                   "Du hast **keine oder kaum Anleihen**. Anleihen (vor allem Staatsanleihen) schwanken weniger als "
                   "Aktien und fallen in Aktienkrisen oft weniger stark – sie wirken wie ein Stoßdämpfer. Ausnahme: "
                   "2022 fielen Aktien und Anleihen gleichzeitig.", "bonds"))
    checks.append((gold > 0.02, "Gold",
                   f"{pc(gold, 0)} in Gold/Rohstoffen." if gold > 0.02 else
                   "Du hast **kein Gold**. Gold bringt keine Zinsen, bewegt sich aber oft unabhängig von Aktien und "
                   "war in Krisen häufig gefragt. Viele Anleger beimischen 5–10 %.", "gold"))
    sec_ok = top_sec_w <= 0.4
    checks.append((sec_ok, "Branchen-Streuung",
                   "Kein Sektor dominiert deine Einzelaktien." if sec_ok else
                   f"**{pc(top_sec_w, 0)} deines Geldes stecken im Sektor {top_sec}.** Läuft diese Branche schlecht "
                   "(z. B. Tech 2022: −30 % und mehr), trifft es fast dein ganzes Portfolio."
                   + (" Gerade Tech-Aktien sind hoch bewertet und schwanken stark." if top_sec == "Technologie" else ""),
                   "defensive"))
    c_ok = top_c_w <= 0.7 or broad > 0.3
    checks.append((c_ok, "Länder-Streuung",
                   "Deine Einzelaktien sind auf mehrere Länder verteilt oder du hast breite ETFs." if c_ok else
                   f"**{pc(top_c_w, 0)} deines Geldes stecken in Unternehmen aus {top_c}.** Läuft dieses Land "
                   "schlecht, gibt es keinen Ausgleich.", "intl"))
    n_ok = len(assets) >= 8 or broad > 0.3
    checks.append((n_ok, "Anzahl & Breite",
                   "Du hast genug Werte oder einen breiten ETF als Basis." if n_ok else
                   (f"Du hast nur **{len(assets)} Wertpapiere**, und dein breiter ETF macht nur {pc(broad, 0)} aus. "
                    if broad > 0 else f"Du hast nur **{len(assets)} Wertpapiere** und keinen breiten ETF. ")
                   + "Ein Welt-ETF als Basis (z. B. 30–60 %) streut auf über 1.000 Unternehmen.", "broad"))
    checks.append((avg_c <= 0.6, "Gegenseitige Absicherung",
                   f"Deine Werte laufen ausreichend unterschiedlich (Ø Korrelation {de(avg_c, 2)})." if avg_c <= 0.6 else
                   f"Deine Werte bewegen sich **stark im Gleichschritt** (Ø Korrelation {de(avg_c, 2)}). In einem "
                   "Crash fallen sie wahrscheinlich gemeinsam.", "bonds" if bonds <= 0.05 else "gold"))

    st.subheader("Portfolio-Check", help="Sechs Prüfpunkte für ein robustes Portfolio. Grün = in Ordnung, "
                                         "Gelb = hier könntest du nachbessern.")
    for ok, title, text, _ in checks:
        (st.success if ok else st.warning)(f"**{title}:** {text}", icon=":material/check_circle:" if ok else ":material/warning:")

    todo = list(dict.fromkeys(key for ok, _, _, key in checks if not ok))
    rp_c = R.values @ cur
    if todo:
        st.subheader("Ideen zum Ergänzen – durchgerechnet",
                     help="Für jede Idee rechnet die App mit deinen echten Kursdaten: Was wäre passiert, wenn 10 % deines "
                          "Portfolios in diesem Wertpapier gesteckt hätten? Nur der gemeinsame Zeitraum zählt.")
        titles = {"bonds": "Anleihen als Stoßdämpfer", "gold": "Gold als Krisenschutz",
                  "defensive": "Defensive Branchen gegen den Sektor-Klumpen", "intl": "Andere Länder und Regionen",
                  "broad": "Breiter Welt-ETF als Basis"}
        held = set(assets)
        for key in todo:
            st.markdown(f"#### {titles[key]}")
            rows, btns = [], []
            for sym, nm in CAND[key]:
                if sym in held:
                    continue
                with st.spinner(f"Teste {nm} …"):
                    res = test_candidate(sym, R, rp_c, rf)
                if res is None:
                    rows.append({"Idee": nm, "Ticker": sym, "Zusammenhang mit deinem Portfolio": "keine Daten",
                                 "Schwankung mit 10 %": "–", "Schlimmster Absturz mit 10 %": "–", "Gewinn pro Risiko mit 10 %": "–"})
                    continue
                m0, m1 = res["m0"], res["m1"]
                rows.append({"Idee": nm, "Ticker": sym, "Zusammenhang mit deinem Portfolio": de(res["corr"], 2),
                             "Schwankung mit 10 %": f"{pc(m0['Vol. p.a.'])} → {pc(m1['Vol. p.a.'])}",
                             "Schlimmster Absturz mit 10 %": f"−{pc(m0['Max. Drawdown'], 0)} → −{pc(m1['Max. Drawdown'], 0)}",
                             "Gewinn pro Risiko mit 10 %": f"{de(m0['Sharpe'], 2)} → {de(m1['Sharpe'], 2)}"})
                btns.append((sym, nm, m1["Vol. p.a."] < m0["Vol. p.a."]))
            H = st.column_config.TextColumn
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Zusammenhang mit deinem Portfolio": H(help="Korrelation von −1 bis +1. Je niedriger, desto besser "
                                                            "sichert die Idee dein bisheriges Portfolio ab."),
                "Schwankung mit 10 %": H(help="Schwankung deines Portfolios vorher → mit 10 % Beimischung."),
                "Schlimmster Absturz mit 10 %": H(help="Größter Einbruch vorher → mit 10 % Beimischung."),
                "Gewinn pro Risiko mit 10 %": H(help="Sharpe Ratio vorher → mit 10 % Beimischung. Höher ist besser."),
            })
            bc = st.columns(max(1, len(btns)))
            for col, (sym, nm, calmer) in zip(bc, btns):
                col.button(f"{nm.split(' (')[0]} testen", key=f"add_{key}_{sym}", on_click=add_test, args=(sym, nm),
                           help="Fügt den Wert mit 0 € zu deiner Auswahl hinzu. Der Optimierer entscheidet dann, "
                                "wie viel er ihm geben würde.")
        explain(["„Zusammenhang“ nahe 0 oder negativ heißt: Die Idee bewegt sich anders als dein Portfolio und kann "
                 "Verluste abfedern. Genau das ist der Sinn einer Beimischung.",
                 "Achte vor allem darauf, ob **Schwankung und schlimmster Absturz sinken**. Ein kleiner Verlust an "
                 "Rendite ist oft ein fairer Preis für deutlich mehr Ruhe.",
                 "Über „testen“ kommt der Wert mit 0 € in deine Auswahl. Auf **Übersicht & Vorschlag** und im "
                 "**Realitäts-Check** siehst du dann, wie viel der Optimierer hineinstecken würde – und ob es "
                 "ohne Blick in die Zukunft geholfen hätte."])
    else:
        st.success("Dein Portfolio besteht alle Prüfpunkte. Du bist breit aufgestellt.")

    # Ähnliche Alternativen
    stocks = [j for j, t in enumerate(assets) if cls[j] == "Einzelaktie" and infos[t].get("sector") in PEERS]
    st.subheader("Ähnliche Alternativen zu deinen Aktien",
                 help="Für jede Einzelaktie bekannte Unternehmen aus derselben Branche, im selben Zeitraum verglichen. "
                      "Ein besserer Wert in der Vergangenheit ist keine Garantie für die Zukunft.")
    if not stocks:
        st.caption("Keine Einzelaktien mit bekannter Branche in deiner Auswahl.")
    for j in stocks:
        t = assets[j]
        sector = infos[t]["sector"]
        own = metrics(R[t].values, rf)
        with st.expander(f"{names[j]} – Branche {SECTOR_DE.get(sector, sector)} (Gewinn pro Risiko {de(own['Sharpe'], 2)})",
                         expanded=own["Sharpe"] < 0.3):
            rows, better = [{"Wertpapier": f"{names[j]} (deins)", "Ø Gewinn/Jahr": pc(own["Rendite p.a."]),
                             "Schwankung": pc(own["Vol. p.a."]), "Schlimmster Absturz": "−" + pc(own["Max. Drawdown"], 0),
                             "Gewinn pro Risiko": de(own["Sharpe"], 2), "Ähnlichkeit": "–"}], []
            for sym, nm in PEERS[sector]:
                if sym in assets:
                    continue
                s = load(sym, start, end)
                if s is None:
                    continue
                r = s.pct_change().dropna().reindex(R.index).dropna()
                if len(r) < 24:
                    continue
                m = metrics(r.values, rf)
                corr = float(np.corrcoef(R[t].loc[r.index].values, r.values)[0, 1])
                rows.append({"Wertpapier": nm, "Ø Gewinn/Jahr": pc(m["Rendite p.a."]), "Schwankung": pc(m["Vol. p.a."]),
                             "Schlimmster Absturz": "−" + pc(m["Max. Drawdown"], 0), "Gewinn pro Risiko": de(m["Sharpe"], 2),
                             "Ähnlichkeit": de(corr, 2)})
                if m["Sharpe"] > own["Sharpe"] + 0.15:
                    better.append((sym, nm, m))
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Ähnlichkeit": st.column_config.TextColumn(help="Korrelation mit deiner Aktie. Nahe 1 = verhält sich "
                                                                "sehr ähnlich, wäre also ein echter Ersatz.")})
            if better:
                st.markdown("**Historisch besseres Verhältnis von Gewinn zu Risiko:** " +
                            ", ".join(f"{nm} ({de(m['Sharpe'], 2)})" for _, nm, m in better))
                bc = st.columns(len(better))
                for col, (sym, nm, _) in zip(bc, better):
                    col.button(f"{nm} testen", key=f"peer_{t}_{sym}", on_click=add_test, args=(sym, nm))
            else:
                st.caption(f"Keine der Vergleichsaktien war im selben Zeitraum klar besser als {names[j]}.")
    if stocks:
        explain(["Die Vergleichsaktien stammen aus derselben Branche. Eine hohe **Ähnlichkeit** heißt, sie verhalten "
                 "sich ähnlich wie deine Aktie und wären ein echter Ersatz. Eine niedrige Ähnlichkeit heißt, sie würden "
                 "zusätzlich streuen.",
                 "Wichtig: Dass eine Aktie in der Vergangenheit besser lief, sagt wenig über die Zukunft. Tausche nicht "
                 "nur wegen dieser Tabelle – schau dir auch die **Unternehmensdaten** (KGV, Wachstum) an."])


# =====================================================================
# Seite: Nachrichten
# =====================================================================
def news_cards(items, show_t=True):
    if not items:
        st.caption("Keine Nachrichten gefunden.")
        return
    for it in items:
        with st.container(border=True):
            ttl = it["title"].replace("[", "(").replace("]", ")")
            st.markdown(f"##### [{ttl}]({it['url']})")
            meta = [it["prov"] or "Quelle unbekannt"]
            if it["ts"] is not None and it["ts"] == it["ts"]:
                meta.append(it["ts"].tz_convert("Europe/Berlin").strftime("%d.%m.%Y, %H:%M Uhr"))
            if show_t:
                meta.append(label(it["t"]))
            st.caption(" · ".join(meta))
            if it["summary"]:
                st.write(it["summary"] + ("…" if len(it["summary"]) >= 320 else ""))


def merge_news(tickers, n):
    seen, out = set(), []
    for t in tickers:
        for it in get_news(t, n):
            if it["url"] not in seen:
                seen.add(it["url"])
                out.append(it)
    epoch = pd.Timestamp(0, tz="UTC")
    return sorted(out, key=lambda x: x["ts"] if x["ts"] is not None and x["ts"] == x["ts"] else epoch, reverse=True)


def page_news():
    head("Nachrichten", "Aktuelle Börsennachrichten – allgemein und zu deinen Wertpapieren. Die Meldungen kommen "
                        "über Yahoo Finance und sind meist auf Englisch.",
         "Die Nachrichten werden alle 30 Minuten neu geladen. Ein Klick auf die Überschrift öffnet den Artikel beim "
         "Anbieter.")
    t1, t2 = st.tabs(["Markt allgemein", "Zu deinen Wertpapieren"])
    with t1:
        with st.spinner("Lade Marktnachrichten …"):
            items = merge_news(["^GSPC", "^GDAXI", "^IXIC", "^STOXX50E"], 8)
        news_cards(items[:25], show_t=False)
    with t2:
        sel = list(dict.fromkeys(ss.sel))
        pick = st.multiselect("Nur Nachrichten zu", sel, default=sel, format_func=label, key="news_pick",
                              help="Wähle, zu welchen deiner Wertpapiere du Nachrichten sehen willst.")
        with st.spinner("Lade Nachrichten zu deinen Wertpapieren …"):
            items = merge_news(pick, 6)
        news_cards(items[:40])
    st.caption("Nachrichten sind keine Handlungsempfehlung. Kurzfristige Meldungen sagen wenig über die langfristige "
               "Entwicklung eines Unternehmens.")


# =====================================================================
# Planen & Testen – gemeinsame Helfer
# =====================================================================
LONG_START = date(1999, 1, 1)
CRISES = [("Dotcom-Crash", "2000-03", "2002-09", "Platzen der Internet-Blase: Technologie-Aktien verloren teils über 80 %."),
          ("Finanzkrise", "2007-10", "2009-02", "US-Immobilienblase und Lehman-Pleite: weltweiter Bankencrash."),
          ("Euro-Schuldenkrise", "2011-04", "2011-09", "Angst um Griechenland, Italien und den Euro."),
          ("Corona-Crash", "2020-01", "2020-03", "Pandemie-Schock: der schnellste Absturz der Geschichte."),
          ("Zinswende 2022", "2021-12", "2022-09", "Hohe Inflation, stark steigende Zinsen: Aktien UND Anleihen fielen.")]
BENCHES = {"MSCI World (iShares, EUNL.DE)": "EUNL.DE", "S&P 500 (iShares, SXR8.DE)": "SXR8.DE",
           "DAX (^GDAXI)": "^GDAXI", "Gold (Xetra-Gold, 4GLD.DE)": "4GLD.DE"}


def long_returns(tickers):
    """Monatsrenditen mit möglichst langer Historie (ab 1999), Lücken bleiben NaN."""
    ser = {}
    for t in tickers:
        s = load(t, LONG_START, date.today())
        if s is not None:
            ser[t] = s
    if not ser:
        return pd.DataFrame()
    return pd.concat(ser, axis=1).sort_index().pct_change(fill_method=None)


def boot_paths(rp, months, n=2000, seed=21):
    rp = np.asarray(rp)
    return rp[np.random.default_rng(seed).integers(0, len(rp), (n, months))]


def port_choice(c, key):
    opts = {"Mein jetziges Portfolio": c["cur"], "Der Vorschlag": c["tw"]}
    ch = st.radio("Mit welcher Aufteilung rechnen?", list(opts), key=key, horizontal=True,
                  help="Die Simulation zieht zufällig echte Monatsrenditen aus dem ausgewerteten Zeitraum – entweder "
                       "mit deiner heutigen Aufteilung oder mit der vorgeschlagenen.")
    return ch, c["R"].values @ opts[ch]


# =====================================================================
# Seite: Krisen-Stresstest
# =====================================================================
def page_stress():
    c = ready()
    assets, names, cur, tw, V = c["assets"], c["names"], c["cur"], c["tw"], c["V"]
    head("Krisen-Stresstest", "Wie hätte dein heutiges Depot die großen Börsencrashs der letzten 25 Jahre überstanden? "
                              "Die App spielt deine Aufteilung mit den echten Kursen von damals durch.",
         "Gerechnet wird mit Monatsdaten und deiner heutigen Aufteilung (monatlich auf diese Gewichte zurückgesetzt). "
         "Wertpapiere, die es damals noch nicht gab, werden weggelassen; die übrigen werden hochgerechnet. Innerhalb eines "
         "Monats können die echten Tiefs noch tiefer gewesen sein.")
    with st.spinner("Lade lange Kurshistorie (einmalig etwas langsamer) …"):
        RL = long_returns(assets + ["^GSPC"])
    if RL.empty:
        st.warning("Es konnte keine lange Kurshistorie geladen werden.")
        return
    rows, paths = [], {}
    for name, a, b, desc in CRISES:
        win_ = RL.loc[(RL.index > pd.Period(a, "M")) & (RL.index <= pd.Period(b, "M"))]
        if win_.empty:
            continue
        avail = [i for i, t in enumerate(assets) if t in win_ and win_[t].notna().all()]
        res = {"Krise": name, "Zeitraum": f"{pd.Period(a, 'M').strftime('%m/%Y')} – {pd.Period(b, 'M').strftime('%m/%Y')}",
               "desc": desc}
        for lab, w in (("Dein Depot", cur), ("Vorschlag", tw)):
            ww = np.array([w[i] for i in avail])
            cov = ww.sum()
            if not avail or cov < 0.3:
                res[lab], res[lab + " cov"] = None, cov
                continue
            rp = win_[[assets[i] for i in avail]].values @ (ww / cov)
            path = np.cumprod(1 + rp)
            res[lab], res[lab + " cov"], res[lab + " dd"] = path[-1] - 1, cov, (path / np.maximum.accumulate(path) - 1).min()
            # Erholung: wie viele Monate nach dem Tief bis zum alten Stand?
            after = RL.loc[RL.index > pd.Period(b, "M"), [assets[i] for i in avail]].dropna()
            lvl, rec = path[-1], None
            for k_, r_ in enumerate(after.values @ (ww / cov)):
                lvl *= 1 + r_
                if lvl >= 1:
                    rec = k_ + 1
                    break
            res[lab + " rec"] = rec
            if lab == "Dein Depot":
                paths[name] = (win_.index, path)
        g = win_["^GSPC"] if "^GSPC" in win_ else None
        res["S&P 500"] = (np.prod(1 + g.values) - 1) if g is not None and g.notna().all() else None
        rows.append(res)
    if not rows:
        st.info("Für keine der Krisen gibt es genug Kursdaten zu deinen Wertpapieren.")
        return
    st.subheader("Verlust in jeder Krise", help="Wie viel dein Depot vom Beginn bis zum Tiefpunkt der Krise verloren "
                                                "hätte. Zum Vergleich der US-Aktienmarkt (S&P 500).")
    fg = go.Figure()
    for lab, col in (("Dein Depot", "#e8361b"), ("Vorschlag", "#1240e0"), ("S&P 500", "#999")):
        fg.add_bar(x=[r["Krise"] for r in rows], y=[r.get(lab) for r in rows], name=lab, marker_color=col,
                   text=[pc(r[lab], 0) if r.get(lab) is not None else "keine Daten" for r in rows], textposition="outside")
    fg.update_layout(barmode="group", height=480, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                     legend=dict(orientation="h", y=1.1), font=dict(size=15), separators=",.")
    st.plotly_chart(fg, width="stretch")
    TC = st.column_config.TextColumn
    st.dataframe(pd.DataFrame([{
        "Krise": r["Krise"], "Zeitraum": r["Zeitraum"],
        "Dein Depot": pc(r["Dein Depot"], 0) if r["Dein Depot"] is not None else "keine Daten",
        "in €": (("−" if r["Dein Depot"] < 0 else "+") + eur(abs(r["Dein Depot"]) * V)) if r["Dein Depot"] is not None else "–",
        "Erholung": (f"{r['Dein Depot rec']} Monate" if r.get("Dein Depot rec") else "nicht erreicht")
        if r["Dein Depot"] is not None else "–",
        "Abgedeckt": pc(r["Dein Depot cov"], 0),
        "Vorschlag": pc(r["Vorschlag"], 0) if r["Vorschlag"] is not None else "–",
        "S&P 500": pc(r["S&P 500"], 0) if r["S&P 500"] is not None else "–"} for r in rows]),
        hide_index=True, width="stretch", column_config={
            "in €": TC(help="So viel hätte dein heutiges Depot (in heutiger Größe) in dieser Krise verloren."),
            "Erholung": TC(help="Monate nach dem Tiefpunkt, bis der Wert vor der Krise wieder erreicht war."),
            "Abgedeckt": TC(help="Welcher Anteil deines Depots damals schon existierte und eingerechnet ist. Unter "
                                 "100 % ist das Ergebnis eine Hochrechnung.")})
    st.subheader("Verlauf in einer Krise", help="Wert deines Depots während der gewählten Krise, Start = heutiger Depotwert.")
    pick = st.selectbox("Krise auswählen", list(paths), key="stress_pick") if paths else None
    if pick:
        idx, path = paths[pick]
        fp = go.Figure(go.Scatter(x=idx.to_timestamp(), y=V * np.r_[path], mode="lines+markers",
                                  line=dict(color="#e8361b", width=3)))
        fp.add_hline(y=V, line_dash="dot", line_color="gray", annotation_text="Wert vor der Krise")
        fp.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€", separators=",.",
                         font=dict(size=15))
        st.plotly_chart(fp, width="stretch")
        st.caption(next(r["desc"] for r in rows if r["Krise"] == pick))
    valid = [r for r in rows if r["Dein Depot"] is not None]
    lines = []
    if valid:
        worst = min(valid, key=lambda r: r["Dein Depot"])
        lines.append(f"Am härtesten hätte dich die **{worst['Krise']}** getroffen: **{pc(worst['Dein Depot'], 0)}**, also "
                     f"etwa **{eur(-worst['Dein Depot'] * V)}** weniger bei deinem heutigen Depot."
                     + (f" Bis der alte Stand wieder erreicht war, dauerte es {worst['Dein Depot rec']} Monate."
                        if worst.get("Dein Depot rec") else " Der alte Stand wurde in den Daten nicht wieder erreicht."))
        cmp_ = [r for r in valid if r["S&P 500"] is not None]
        if cmp_:
            better = sum(r["Dein Depot"] > r["S&P 500"] for r in cmp_)
            lines.append(f"In {better} von {len(cmp_)} Krisen hat dein Depot weniger verloren als der US-Aktienmarkt.")
        vb = [r for r in valid if r["Vorschlag"] is not None]
        if vb:
            avg_d = np.mean([r["Vorschlag"] - r["Dein Depot"] for r in vb])
            lines.append("Der Vorschlag hätte in den Krisen im Schnitt **" + ("weniger" if avg_d > 0 else "mehr") +
                         f"** verloren ({sg(avg_d * 100, 0)} Prozentpunkte).")
        if any(r["Dein Depot cov"] < 0.95 for r in valid):
            lines.append("Achtung: Bei manchen Krisen gab es einige deiner Wertpapiere noch nicht (siehe „Abgedeckt“). "
                         "Dort ist das Ergebnis eine Hochrechnung mit den übrigen Werten.")
    lines.append("Die wichtigste Frage: **Hättest du in so einer Phase durchgehalten, ohne zu verkaufen?** Wer im Crash "
                 "verkauft, macht Buchverluste endgültig. Wenn dich die Zahlen erschrecken, ist ein ruhigeres Portfolio "
                 "(z. B. mit Anleihen) besser für dich.")
    explain(lines)


# =====================================================================
# Seite: Vergleich mit dem Markt
# =====================================================================
def page_vs_market():
    c = ready()
    assets, names, cur, R, V, T = c["assets"], c["names"], c["cur"], c["R"], c["V"], c["T"]
    head("Vergleich mit dem Markt", "Hättest du mit einem einfachen ETF mehr oder weniger verdient? Die ehrlichste Frage "
                                    "für jeden Anleger, der einzelne Aktien auswählt.",
         "Der Vergleichsindex wird mit Kursen inklusive Dividenden gerechnet (thesaurierende ETFs). Deine Positionen "
         "seit Kauf werden ohne erhaltene Dividenden gerechnet – bei Dividendenaktien ist dein Ergebnis also etwas besser, "
         "als hier steht.")
    bname = st.selectbox("Vergleichen mit", list(BENCHES), key="bm_choice",
                         help="MSCI World: über 1.500 große Unternehmen aus Industrieländern – der Standard-ETF vieler "
                              "Anleger. S&P 500: die 500 größten US-Unternehmen. DAX: 40 deutsche Unternehmen.")
    bt_ = BENCHES[bname]
    with st.spinner("Lade Vergleichsindex …"):
        bl = load(bt_, LONG_START, date.today())
    if bl is None:
        st.warning(f"Für {bname} konnten keine Kurse geladen werden.")
        return
    dep = c["dep"]
    st.header("Seit deinen Käufen")
    rows = []
    if dep is not None:
        for t, n in zip(assets, names):
            d_, inv, val = dep.loc[t, "date"], dep.loc[t, "invested"], dep.loc[t, "value"]
            if not d_ or inv <= 0:
                continue
            per = pd.Period(d_, "M")
            if per not in bl.index:
                continue
            mult = float(bl.iloc[-1] / bl.loc[per])
            rows.append({"t": t, "n": n, "date": d_, "inv": inv, "val": val, "bench": inv * mult})
    if not rows:
        st.info("Trag auf **Mein Portfolio** (genaue Eingabe) bei deinen Positionen das **Kaufdatum** und den **Kaufkurs** "
                "ein. Dann siehst du hier für jede Position, ob ein ETF besser gewesen wäre.")
    else:
        df_ = pd.DataFrame(rows)
        tot_v, tot_b, tot_i = df_["val"].sum(), df_["bench"].sum(), df_["inv"].sum()
        k1, k2, k3 = st.columns(3)
        k1.metric("Deine Positionen heute", eur(tot_v), f"{sg((tot_v / tot_i - 1) * 100)} %",
                  help="Wert heute der Positionen mit Kaufdatum, Veränderung seit Kauf.")
        k2.metric(f"Mit {bname.split(' (')[0]} wäre es", eur(tot_b), f"{sg((tot_b / tot_i - 1) * 100)} %",
                  help="Wenn du dasselbe Geld zum selben Zeitpunkt in den Vergleichsindex gesteckt hättest.")
        k3.metric("Unterschied", ("+" if tot_v >= tot_b else "−") + eur(abs(tot_v - tot_b)),
                  help="Positiv: Deine Auswahl war besser als der Index. Negativ: Der Index wäre besser gewesen.")
        st.dataframe(pd.DataFrame({
            "Wertpapier": df_["n"], "Gekauft": [d.strftime("%d.%m.%Y") for d in df_["date"]],
            "Investiert": [eur(x) for x in df_["inv"]], "Wert heute": [eur(x) for x in df_["val"]],
            f"Mit {bname.split(' (')[0]}": [eur(x) for x in df_["bench"]],
            "Unterschied": [("+" if v >= b else "−") + eur(abs(v - b)) for v, b in zip(df_["val"], df_["bench"])],
            "Besser?": ["deine Wahl" if v >= b else "der Index" for v, b in zip(df_["val"], df_["bench"])]}),
            hide_index=True, width="stretch")
        fg = go.Figure()
        fg.add_bar(x=df_["n"], y=df_["val"], name="Deine Position", marker_color="#1240e0")
        fg.add_bar(x=df_["n"], y=df_["bench"], name=f"Mit {bname.split(' (')[0]}", marker_color="#b8c4e8")
        fg.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€",
                         separators=",.", legend=dict(orientation="h", y=1.08), font=dict(size=15))
        st.plotly_chart(fg, width="stretch")
        wins = int((df_["val"] >= df_["bench"]).sum())
        explain([f"Mit deinen Positionen hast du **{('mehr' if tot_v >= tot_b else 'weniger')}** verdient als mit dem "
                 f"{bname.split(' (')[0]}: Unterschied **{('+' if tot_v >= tot_b else '−')}{eur(abs(tot_v - tot_b))}**.",
                 f"{wins} von {len(df_)} Positionen haben den Index geschlagen.",
                 "Zur Einordnung: Die meisten Profi-Fonds schaffen es über 10 Jahre nicht, den Markt zu schlagen. Wenn "
                 "der Index bei dir vorne liegt, bist du in guter Gesellschaft – und ein ETF als Basis könnte dir "
                 "Arbeit und Risiko sparen."])
    st.header(f"Im ausgewerteten Zeitraum ({years} Jahre)")
    br = bl.pct_change().dropna().reindex(R.index)
    ok_ = br.notna()
    if ok_.sum() < 24:
        st.info("Für diesen Zeitraum gibt es zu wenige Kurse des Vergleichsindex.")
        return
    rp = (R.values @ cur)[ok_.values]
    brv = br[ok_].values
    idx = R.index[ok_.values].to_timestamp()
    f2 = go.Figure()
    f2.add_scatter(x=idx, y=V * np.cumprod(1 + rp), name="Dein Depot (heutige Aufteilung)", line=dict(color="#e8361b", width=3))
    f2.add_scatter(x=idx, y=V * np.cumprod(1 + brv), name=bname.split(" (")[0], line=dict(color="#555", width=3))
    f2.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€", separators=",.",
                     legend=dict(orientation="h", y=1.08), hovermode="x unified", font=dict(size=15))
    st.plotly_chart(f2, width="stretch")
    m_p, m_b = metrics(rp, rf), metrics(brv, rf)
    st.dataframe(pd.DataFrame({"": ["Dein Depot", bname.split(" (")[0]],
                               "Ø Gewinn/Jahr": [pc(m_p["Rendite p.a."]), pc(m_b["Rendite p.a."])],
                               "Schwankung": [pc(m_p["Vol. p.a."]), pc(m_b["Vol. p.a."])],
                               "Schlimmster Absturz": ["−" + pc(m_p["Max. Drawdown"]), "−" + pc(m_b["Max. Drawdown"])],
                               "Gewinn pro Risiko": [de(m_p["Sharpe"], 2), de(m_b["Sharpe"], 2)]}),
                 hide_index=True, width="stretch")
    corr = float(np.corrcoef(rp, brv)[0, 1])
    explain([f"Mit deiner heutigen Aufteilung wärst du im Schnitt **{pc(m_p['Rendite p.a.'])}** pro Jahr gewachsen, der "
             f"{bname.split(' (')[0]} **{pc(m_b['Rendite p.a.'])}** – bei {pc(m_p['Vol. p.a.'])} gegenüber "
             f"{pc(m_b['Vol. p.a.'])} Schwankung.",
             "Dein Depot hatte " + ("ein **besseres**" if m_p["Sharpe"] > m_b["Sharpe"] + 0.05 else "ein **schlechteres**"
                                    if m_p["Sharpe"] < m_b["Sharpe"] - 0.05 else "ein ähnliches")
             + f" Verhältnis von Gewinn zu Risiko ({de(m_p['Sharpe'], 2)} gegenüber {de(m_b['Sharpe'], 2)}).",
             f"Ähnlichkeit zum Index: {de(corr, 2)}. " + ("Dein Depot verhält sich fast wie der Index – dann bietet ein "
                                                          "günstiger ETF dasselbe mit weniger Aufwand." if corr > 0.85 else
                                                          "Dein Depot verhält sich deutlich anders als der Index.")])


# =====================================================================
# Seite: Sparplan-Rechner
# =====================================================================
def fan(x, P, name, col, rgba):
    p5, p50, p95 = np.percentile(P, [5, 50, 95], axis=0)
    return [go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[p95, p5[::-1]], fill="toself", fillcolor=rgba, line=dict(width=0),
                       name=name + " (9 von 10 Fällen)", hoverinfo="skip"),
            go.Scatter(x=x, y=p50, name=name + " (mittlerer Fall)", line=dict(color=col, width=3))]


def page_savings():
    c = ready()
    head("Sparplan-Rechner", "Wie viel wird aus deinem Depot, wenn du jeden Monat etwas dazu sparst – und wann "
                             "erreichst du dein Ziel?",
         "2000 Simulationen mit echten Monatsrenditen deiner Aufteilung (Bootstrap). Die Rate wird jeweils am "
         "Monatsanfang investiert.")
    ch, rp = port_choice(c, "sp_port")
    if ss.get("tr"):
        tr = ss.tr
        plans = detect_plans(tr["df"])
        act = plans[plans["aktiv"]] if len(plans) else plans
        ns = net_saving_12m(tr["df"])
        tot_tr = max(0.0, tr_total(tr))
        with st.container(border=True):
            st.markdown("**Aus deinem Trade-Republic-Export übernehmen**")
            b1, b2, b3 = st.columns(3)
            if len(act):
                b1.button(f"Aktive Sparpläne: {eur(act['monatlich'].sum())}/Monat", width="stretch",
                          on_click=set_ss, kwargs={"sp_rate": float(round(act["monatlich"].sum(), 0))},
                          help="Summe deiner aktiven Sparpläne, umgerechnet auf einen Monat: " +
                               ", ".join(f"{n} {eur2(m)}" for n, m in zip(act["name"], act["monatlich"])))
            else:
                b1.button("Keine aktiven Sparpläne gefunden", disabled=True, width="stretch",
                          help="Im Export wurden keine laufenden Sparpläne erkannt.")
            b2.button(f"Ø Einzahlung: {eur(max(ns, 0))}/Monat", width="stretch", disabled=ns <= 0,
                      on_click=set_ss, kwargs={"sp_rate": float(round(max(ns, 0), 0))},
                      help="Deine durchschnittliche Netto-Einzahlung der letzten 12 Monate (Einzahlungen minus "
                           "Überweisungen raus).")
            b3.button(f"Startkapital: {eur(tot_tr)}", width="stretch",
                      on_click=set_ss, kwargs={"sp_start": float(round(tot_tr, -1))},
                      help="Dein Gesamtvermögen bei Trade Republic (Wertpapiere + Krypto + Cash).")
    a1, a2, a3 = st.columns(3)
    ss.setdefault("sp_start", float(round(c["V"], -1)))
    start_v = a1.number_input("Startkapital (€)", min_value=0.0, step=500.0, key="sp_start",
                              help="Womit du startest – voreingestellt ist dein heutiger Depotwert.")
    rate = a2.number_input("Monatliche Sparrate (€)", min_value=0.0, step=25.0, key="sp_rate",
                           help="Wie viel du jeden Monat dazu investierst.")
    yrs = a3.slider("Wie viele Jahre?", 1, 40, key="sp_years", help="Wie lange du sparst.")
    b1, b2, b3 = st.columns(3)
    goal_v = b1.number_input("Dein Ziel (€)", min_value=0.0, step=5000.0, key="sp_goal",
                             help="Welche Summe willst du erreichen? Die App rechnet aus, wie wahrscheinlich das ist.")
    dyn = b2.slider("Sparrate steigt jedes Jahr um (%)", 0, 10, key="sp_dyn",
                    help="Dynamik: Wenn dein Einkommen steigt, kannst du die Rate jährlich erhöhen.") / 100
    infl = b3.toggle("Inflation abziehen (2 % pro Jahr)", key="sp_infl",
                     help="Zeigt, was das Geld in heutiger Kaufkraft wert wäre. 100.000 € in 20 Jahren kaufen bei 2 % "
                          "Inflation nur so viel wie heute etwa 67.000 €.")
    m = yrs * 12
    R_ = boot_paths(rp, m)
    rates = rate * (1 + dyn) ** (np.arange(m) // 12)
    vals = np.empty_like(R_)
    v = np.full(R_.shape[0], start_v)
    for k in range(m):
        v = (v + rates[k]) * (1 + R_[:, k])
        vals[:, k] = v
    paid = start_v + np.cumsum(rates)
    if infl:
        defl = (1.02) ** (np.arange(1, m + 1) / 12)
        vals, paid = vals / defl, paid / defl
    x = np.arange(1, m + 1) / 12
    end_ = vals[:, -1]
    med, p5, p95 = np.median(end_), np.percentile(end_, 5), np.percentile(end_, 95)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eingezahlt", eur(paid[-1]), help="Startkapital plus alle Sparraten" + (" (in heutiger Kaufkraft)" if infl else "") + ".")
    k2.metric("Mittlerer Fall", eur(med), f"+{eur(med - paid[-1])} Ertrag" if med > paid[-1] else None,
              help="Median aller Simulationen. Ertrag = Endwert minus Eingezahltes.")
    k3.metric("Schlechter Fall (1 von 20)", eur(p5), help="Nur 5 % der Simulationen enden noch schlechter.")
    prob = (end_ >= goal_v).mean() if goal_v > 0 else None
    k4.metric("Ziel erreicht", pc(prob, 0) if prob is not None else "–",
              help="Anteil der Simulationen, die am Ende mindestens dein Ziel erreichen.")
    st.subheader("Mögliche Entwicklung", help="Linie = mittlerer Fall, Fläche = Bereich von 9 von 10 Simulationen, "
                                              "gestrichelt = eingezahltes Geld.")
    fg = go.Figure(fan(x, vals, "Depot", "#1240e0", "rgba(18,64,224,0.15)"))
    fg.add_scatter(x=x, y=paid, name="Eingezahlt", line=dict(color="#555", dash="dash", width=2))
    if goal_v > 0:
        fg.add_hline(y=goal_v, line_color="#0b7a3b", line_dash="dot", annotation_text="Dein Ziel")
    fg.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Jahre", yaxis_title="€",
                     separators=",.", legend=dict(orientation="h", y=-0.15), font=dict(size=15))
    st.plotly_chart(fg, width="stretch")
    lines = [f"Mit {eur(start_v)} Start und {eur(rate)} im Monat" + (f" (jährlich +{de(dyn * 100, 0)} %)" if dyn else "")
             + f" zahlst du in {yrs} Jahren insgesamt **{eur(paid[-1])}** ein. Im mittleren Fall wird daraus "
             f"**{eur(med)}** – in 9 von 10 Fällen zwischen {eur(p5)} und {eur(p95)}."]
    if med > paid[-1]:
        lines.append(f"Davon sind etwa **{pc((med - paid[-1]) / med, 0)} Erträge** (Zinseszins-Effekt): Dein Geld "
                     "verdient Geld, und dieses Geld verdient wieder Geld. Je länger, desto stärker.")
    if goal_v > 0:
        hit = (vals >= goal_v).any(axis=1)
        first = np.where(hit, (vals >= goal_v).argmax(axis=1) + 1, np.nan)
        when = np.nanmedian(first) / 12 if hit.mean() >= 0.5 else None
        lines.append(f"Dein Ziel von {eur(goal_v)} erreichst du in **{pc(prob, 0)} der Fälle**"
                     + (f", im mittleren Fall nach etwa **{de(when, 1)} Jahren**." if when else ".")
                     + ("" if prob >= 0.7 else " Um sicherer ans Ziel zu kommen: Rate erhöhen, länger sparen oder das "
                                               "Ziel anpassen."))
    lines.append("Grundlage ist die Vergangenheit deiner Aufteilung (" + ch.lower() + "). Lief diese besonders gut, sind "
                 "die Zahlen zu optimistisch – rechne lieber mit dem schlechten Fall.")
    explain(lines)
    with st.container(border=True):
        st.subheader("Und danach? Für die Rente", help="Grobe Faustregel: 3,5 % des Vermögens pro Jahr gelten als "
                                                      "Entnahme, die über ca. 30 Jahre meist reicht. Genauer rechnet der "
                                                      "Entnahmeplan.")
        r1, r2 = st.columns(2)
        r1.metric("Mögliche Entnahme (mittlerer Fall)", eur(med * 0.035 / 12) + " / Monat",
                  help=f"3,5 % von {eur(med)} pro Jahr, geteilt durch 12.")
        r2.metric("Vorsichtig (schlechter Fall)", eur(p5 * 0.035 / 12) + " / Monat",
                  help=f"3,5 % von {eur(p5)} pro Jahr, geteilt durch 12.")
        st.button("Mittleren Endwert in den Entnahmeplan übernehmen", icon=":material/payments:",
                  on_click=set_ss, kwargs={"wd_start": float(round(med, -2)), "wd_rate": float(round(med * 0.035 / 12, -1))},
                  help="Setzt im Entnahmeplan das Startkapital auf den mittleren Endwert und die Entnahme auf 3,5 % "
                       "pro Jahr. Danach links im Menü „Entnahmeplan“ öffnen.")
        st.page_link(P_WD, label="Zum Entnahmeplan", icon=":material/arrow_forward:")


# =====================================================================
# Seite: Entnahmeplan
# =====================================================================
def page_withdraw():
    c = ready()
    head("Entnahmeplan", "Wie lange reicht dein Geld, wenn du jeden Monat etwas entnimmst – zum Beispiel als "
                         "Zusatzrente?",
         "2000 Simulationen mit echten Monatsrenditen deiner Aufteilung. Die Entnahme erfolgt am Monatsanfang. Steuern "
         "auf Entnahmen sind nicht berücksichtigt.")
    ch, rp = port_choice(c, "wd_port")
    if ss.get("tr"):
        tt = max(0.0, tr_total(ss.tr))
        st.button(f"Startkapital = Trade-Republic-Vermögen ({eur(tt)})", on_click=set_ss,
                  kwargs={"wd_start": float(round(tt, -1))}, icon=":material/account_balance_wallet:",
                  help="Übernimmt dein Gesamtvermögen aus dem Export (Wertpapiere + Krypto + Cash).")
    a1, a2, a3 = st.columns(3)
    ss.setdefault("wd_start", float(round(c["V"], -1)))
    ss.setdefault("wd_rate", float(max(10, round(c["V"] * 0.035 / 12, -1))))
    start_v = a1.number_input("Startkapital (€)", min_value=0.0, step=1000.0, key="wd_start",
                              help="Voreingestellt ist dein heutiger Depotwert.")
    wd = a2.number_input("Monatliche Entnahme (€)", min_value=0.0, step=50.0, key="wd_rate",
                         help="Wie viel du jeden Monat herausnimmst.")
    yrs = a3.slider("Wie lange soll es reichen? (Jahre)", 5, 50, key="wd_years")
    infl = st.toggle("Entnahme jedes Jahr um 2 % erhöhen (Inflationsausgleich)", key="wd_infl",
                     help="Damit deine Entnahme ihre Kaufkraft behält, steigt sie jährlich um 2 %.")
    m = yrs * 12
    R_ = boot_paths(rp, m, seed=33)

    def run(w):
        v = np.full(R_.shape[0], start_v)
        out = np.empty_like(R_)
        alive = np.full(R_.shape[0], m + 1)
        for k in range(m):
            wk = w * (1.02 ** (k // 12) if infl else 1)
            v = np.maximum(v - wk, 0) * (1 + R_[:, k])
            out[:, k] = v
            alive = np.where((v <= 0) & (alive > m), k + 1, alive)
        return out, alive

    vals, alive = run(wd)
    succ = (alive > m).mean()
    lo_, hi_ = 0.0, max(start_v / 12, 1.0)
    for _ in range(25):
        mid = (lo_ + hi_) / 2
        if (run(mid)[1] > m).mean() >= 0.9:
            lo_ = mid
        else:
            hi_ = mid
    safe = lo_
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Reicht {yrs} Jahre", pc(succ, 0), help="Anteil der Simulationen, in denen das Geld bis zum Ende reicht.")
    k2.metric("Sichere Entnahme", eur(safe) + " / Monat",
              help=f"Höchste monatliche Entnahme, bei der das Geld in 9 von 10 Simulationen {yrs} Jahre reicht.")
    k3.metric("Rest im mittleren Fall", eur(np.median(vals[:, -1])), help=f"Was nach {yrs} Jahren im Median übrig ist.")
    k4.metric("Entnommen gesamt", eur(sum(wd * (1.02 ** (k // 12) if infl else 1) for k in range(m))),
              help="Summe aller geplanten Entnahmen.")
    x = np.arange(1, m + 1) / 12
    st.subheader("Wie sich dein Kapital entwickeln könnte", help="Linie = mittlerer Fall, Fläche = 9 von 10 Simulationen.")
    fg = go.Figure(fan(x, vals, "Kapital", "#1240e0", "rgba(18,64,224,0.15)"))
    fg.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Jahre", yaxis_title="€",
                     separators=",.", legend=dict(orientation="h", y=-0.15), font=dict(size=15))
    st.plotly_chart(fg, width="stretch")
    lines = [f"Bei {eur(wd)} pro Monat" + (" (jährlich +2 %)" if infl else "") + f" reicht dein Geld in **{pc(succ, 0)} "
             f"der Fälle** volle {yrs} Jahre.",
             f"Die **sichere Entnahme** liegt bei etwa **{eur(safe)} pro Monat** – damit reicht es in 9 von 10 "
             f"Simulationen. Das sind {pc(safe * 12 / max(start_v, 1))} des Startkapitals pro Jahr."]
    if succ < 0.9:
        early = alive[alive <= m]
        if len(early):
            lines.append(f"In den schlechten Fällen ist das Geld im Schnitt nach **{de(np.median(early) / 12, 1)} Jahren** "
                         "aufgebraucht.")
    lines.append("**Reihenfolge-Risiko:** Ein Crash in den ersten Jahren der Entnahme ist viel gefährlicher als später, weil "
                 "du dann bei niedrigen Kursen verkaufen musst. Viele halten deshalb 1–2 Jahresentnahmen als Bargeld "
                 "oder Anleihen bereit.")
    explain(lines)


# =====================================================================
# Seite: Dividenden
# =====================================================================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_divs(t):
    try:
        s = yf.Ticker(t).dividends
    except Exception:
        return None
    if s is None or len(s) == 0:
        return None
    s = pd.Series(pd.to_numeric(s, errors="coerce").values, index=pd.to_datetime(s.index, utc=True)).dropna()
    return s[s.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=372)]


MONTHS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def page_div():
    c = ready()
    assets, names, cur, V, dep = c["assets"], c["names"], c["cur"], c["V"], c["dep"]
    head("Dividenden", "Wie viel Geld zahlen dir deine Wertpapiere jedes Jahr aus – und in welchen Monaten kommt es?",
         "Grundlage sind die Ausschüttungen der letzten 12 Monate pro Stück (von Yahoo), umgerechnet in Euro. "
         "Künftige Dividenden können höher, niedriger oder ganz ausfallen.")
    taxr = ss.tax / 100
    rows, cal = [], np.zeros((len(assets), 12))
    with st.spinner("Lade Dividenden …"):
        for i, t in enumerate(assets):
            price, cur_ = get_quote(t)
            fx = get_fx(cur_) or 0
            pe = price * fx if price else None
            shares = (dep.loc[t, "qty"] if dep is not None else (cur[i] * V / pe if pe else 0))
            dv = get_divs(t)
            per_share = float(dv.sum()) if dv is not None else 0.0
            if dv is not None:
                for ts, amt in dv.items():
                    cal[i, ts.month - 1] += amt * shares * fx
            year_eur = per_share * shares * fx
            yld = per_share * fx / pe if pe else None
            yoc = (per_share * fx / dep.loc[t, "cost"]) if dep is not None and dep.loc[t, "cost"] > 0 else None
            rows.append({"n": names[i], "year": year_eur, "yld": yld, "yoc": yoc, "count": 0 if dv is None else len(dv),
                         "months": "" if dv is None else ", ".join(sorted({MONTHS[d.month - 1] for d in dv.index},
                                                                          key=MONTHS.index))})
    tot = sum(r["year"] for r in rows)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dividenden pro Jahr (brutto)", eur(tot), help="Summe aller Ausschüttungen der letzten 12 Monate, "
                                                             "hochgerechnet auf deine heutigen Stückzahlen.")
    k2.metric("Nach Steuern ca.", eur(tot * (1 - taxr)), help=f"Mit deinem Steuersatz von {de(ss.tax, 1)} % (einstellbar "
                                                              "auf Übersicht & Vorschlag), ohne Freibetrag.")
    k3.metric("Ø pro Monat (netto)", eur(tot * (1 - taxr) / 12), help="Netto-Dividenden geteilt durch 12.")
    k4.metric("Dividendenrendite Depot", pc(tot / V if V else 0), help="Jährliche Dividenden im Verhältnis zum Depotwert.")
    st.subheader("Dividenden-Kalender", help="In welchen Monaten in den letzten 12 Monaten ausgezahlt wurde – als "
                                             "Erwartung für die nächsten 12 Monate.")
    fg = go.Figure([go.Bar(x=MONTHS, y=cal[i], name=names[i]) for i in range(len(assets)) if cal[i].sum() > 0])
    fg.update_layout(barmode="stack", height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€ (brutto)",
                     separators=",.", legend=dict(orientation="h", y=-0.15), font=dict(size=15))
    st.plotly_chart(fg, width="stretch")
    TC = st.column_config.TextColumn
    st.dataframe(pd.DataFrame({"Wertpapier": [r["n"] for r in rows], "Pro Jahr": [eur(r["year"]) for r in rows],
                               "Rendite": [pct_or(r["yld"]) for r in rows],
                               "Auf deinen Kaufkurs": [pct_or(r["yoc"]) for r in rows],
                               "Zahlungen": [r["count"] for r in rows], "Monate": [r["months"] or "–" for r in rows]}),
                 hide_index=True, width="stretch", column_config={
                     "Rendite": TC(help="Dividende pro Jahr geteilt durch den heutigen Kurs."),
                     "Auf deinen Kaufkurs": TC(help="Dividende pro Jahr geteilt durch deinen Kaufkurs (Yield on Cost). "
                                                    "Nur bei genauer Eingabe.")})
    payers = [r for r in rows if r["year"] > 0]
    lines = [f"Deine Wertpapiere schütten zusammen etwa **{eur(tot)} pro Jahr** aus, nach Steuern rund "
             f"**{eur(tot * (1 - taxr))}** – im Schnitt {eur(tot * (1 - taxr) / 12)} pro Monat."]
    if payers:
        top = max(payers, key=lambda r: r["year"])
        lines.append(f"Größter Zahler: **{top['n']}** mit {eur(top['year'])} pro Jahr.")
        empty = [MONTHS[k] for k in range(12) if cal[:, k].sum() == 0]
        if empty and len(empty) < 12:
            lines.append("Monate ohne Ausschüttung: " + ", ".join(empty) + ".")
    nopay = [r["n"] for r in rows if r["year"] == 0]
    if nopay:
        lines.append("Keine Dividende: " + ", ".join(nopay) + ". Bei thesaurierenden ETFs ist das normal – sie legen die "
                     "Erträge automatisch wieder an, das steckt im Kurs.")
    lines.append("Wichtig: Dividenden sind kein Geschenk. Am Tag der Ausschüttung sinkt der Kurs ungefähr um den "
                 "ausgezahlten Betrag. Entscheidend ist der Gesamtertrag aus Kursgewinn **und** Dividende.")
    explain(lines)


# =====================================================================
# Seite: PDF-Bericht
# =====================================================================
def crisis_losses(assets, w):
    RL = long_returns(assets)
    out = []
    for name, a, b, _ in CRISES:
        if RL.empty:
            break
        win_ = RL.loc[(RL.index > pd.Period(a, "M")) & (RL.index <= pd.Period(b, "M"))]
        avail = [i for i, t in enumerate(assets) if t in win_ and len(win_) and win_[t].notna().all()]
        cov = float(sum(w[i] for i in avail))
        if not avail or cov < 0.3:
            out.append((name, None, cov))
            continue
        ww = np.array([w[i] for i in avail]) / cov
        out.append((name, float(np.prod(1 + win_[[assets[i] for i in avail]].values @ ww) - 1), cov))
    return out


def build_pdf(c, with_stress):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    fdir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    pdf = FPDF()
    pdf.add_font("DV", "", os.path.join(fdir, "DejaVuSans.ttf"))
    pdf.add_font("DV", "B", os.path.join(fdir, "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    BLUE = (18, 64, 224)

    def h1(t):
        pdf.set_font("DV", "B", 20)
        pdf.set_text_color(*BLUE)
        pdf.multi_cell(0, 10, t, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    def h2(t, need=40):
        if pdf.get_y() > pdf.h - 15 - need:      # Überschrift nicht allein am Seitenende
            pdf.add_page()
        pdf.ln(3)
        pdf.set_font("DV", "B", 13)
        pdf.set_text_color(*BLUE)
        pdf.multi_cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    def p(t, size=10):
        pdf.set_font("DV", "", size)
        pdf.multi_cell(0, 5.5, t, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def table(head_, rows_, widths=None):
        pdf.set_font("DV", "", 9)
        with pdf.table(col_widths=widths, text_align="LEFT", line_height=6,
                       headings_style=__import__("fpdf").FontFace(emphasis="BOLD", fill_color=(230, 235, 250))) as tb:
            r_ = tb.row()
            for x in head_:
                r_.cell(str(x))
            for row in rows_:
                r_ = tb.row()
                for x in row:
                    r_.cell(str(x))
        pdf.ln(2)

    def fig_to_pdf(fig, h=70):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pdf.image(buf, w=180, h=h, keep_aspect_ratio=True)
        pdf.ln(2)

    assets, names, cur, tw, V, M, target = c["assets"], c["names"], c["cur"], c["tw"], c["V"], c["M"], c["target"]
    tm, cm = M[target], M["Aktuell"]
    h1("Portfolio-Check – Bericht")
    p(f"Erstellt am {date.today().strftime('%d.%m.%Y')} · Ziel: {c['goal_txt']} · ausgewertet: {c['R'].index[0]} bis "
      f"{c['R'].index[-1]} ({c['T']} Monate) · Depotwert {eur(V)}", 9)
    h2("Auf einen Blick")
    p(f"Dein Portfolio ist {lvl_vol(cm['Vol. p.a.'])} (Schwankung {pc(cm['Vol. p.a.'])}). Es ist im Schnitt um "
      f"{pc(cm['Rendite p.a.'])} pro Jahr gewachsen; der schlimmste Einbruch lag bei −{pc(cm['Max. Drawdown'])}. "
      f"Das Verhältnis von Gewinn zu Risiko war {lvl_sh(cm['Sharpe'])}.")
    table(["Kennzahl", "Jetziges Portfolio", "Vorschlag"],
          [["Ø Gewinn pro Jahr", pc(cm["Rendite p.a."]), pc(tm["Rendite p.a."])],
           ["Schwankung", pc(cm["Vol. p.a."]), pc(tm["Vol. p.a."])],
           ["Schlimmster Absturz", "−" + pc(cm["Max. Drawdown"]), "−" + pc(tm["Max. Drawdown"])],
           ["Gewinn pro Risiko (Sharpe)", de(cm["Sharpe"], 2), de(tm["Sharpe"], 2)]], (60, 50, 50))
    dep = c["dep"]
    if dep is not None and dep["qty"].sum() > 0:
        h2("Dein Depot")
        rows_ = []
        for t, n in zip(assets, names):
            r_ = dep.loc[t]
            if r_["qty"] <= 0:
                continue
            g = r_["gain"]
            rows_.append([n, de(r_["qty"], 2), eur2(r_["cost"]) if r_["cost"] else "–", eur2(r_["price_eur"]),
                          eur(r_["value"]), ("+" if g >= 0 else "−") + eur(abs(g)) if g is not None and g == g else "–",
                          (sg(r_["gain_pct"] * 100) + " %") if r_["gain_pct"] is not None and r_["gain_pct"] == r_["gain_pct"] else "–"])
        table(["Wertpapier", "Stück", "Ø Kaufkurs", "Kurs", "Wert", "Gewinn/Verlust", "in %"], rows_,
              (50, 18, 24, 24, 24, 26, 18))
    h2(f"Vorschlag zur Umschichtung ({c['goal_txt']})")
    cur_v = cur * V
    newm, nosell = float(ss.get("newmoney") or 0), bool(ss.get("nosell"))
    d_eur = plan(cur_v, tw, newm, nosell)
    thr = 0.005 * max(V + newm, 1)
    table(["Wertpapier", "Jetzt", "Nachher", "Was tun?"],
          [[n, eur(a), eur(a + d), "behalten" if abs(d) < thr else (f"+{eur(d)} kaufen" if d > 0 else f"−{eur(-d)} verkaufen")]
           for n, a, d in zip(names, cur_v, d_eur)], (60, 35, 35, 50))
    if newm or nosell:
        p(f"Mit {eur(newm)} neuem Geld" + (", ohne Verkäufe (nur nachkaufen)." if nosell else "."), 9)
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(names)))
    yy = np.arange(len(names))
    ax.barh(yy + 0.2, cur * 100, 0.4, label="Jetzt", color="#b8c4e8")
    ax.barh(yy - 0.2, tw * 100, 0.4, label="Vorschlag", color="#1240e0")
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlabel("Anteil in %")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig_to_pdf(fig, h=min(110, 20 + 9 * len(names)))
    h2("So hätte sich dein Geld entwickelt", need=95)
    R = c["R"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(R.index.to_timestamp(), V * np.cumprod(1 + R.values @ cur), color="#e8361b", lw=2, label="Jetziges Portfolio")
    ax.plot(R.index.to_timestamp(), V * np.cumprod(1 + R.values @ tw), color="#1240e0", lw=2, label="Vorschlag")
    ax.set_ylabel("€")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig_to_pdf(fig, h=75)
    p("Hinweis: Der Vorschlag ist mit genau diesen Kursen berechnet und sieht hier deshalb besonders gut aus.", 8)
    h2("Realitäts-Check (Backtest)")
    BT = c["BT"]
    if BT is None or target not in BT:
        p("Zu wenig Kursgeschichte für den Test.")
    else:
        a, b = BT["Aktuell"]["Sharpe"], BT[target]["Sharpe"]
        p(("Ja – " if b > a + 0.05 else "Kaum Unterschied – " if b >= a - 0.05 else "Nein – ")
          + f"im Testzeitraum ({R.index[c['s_']]} bis {R.index[-1]}), den der Vorschlag nicht kannte, lag das Verhältnis "
            f"von Gewinn zu Risiko bei {de(b, 2)} (Vorschlag) gegenüber {de(a, 2)} (jetziges Portfolio).")
    h2(f"Zukunftssimulation: {hor} Jahre")
    S_ = c["SIM"]
    table(["", "Schlechter Fall (5 %)", "Mittlerer Fall", "Guter Fall (95 %)", "Risiko im Minus"],
          [[lab, eur(S_[k]["5 %"]), eur(S_[k]["Median"]), eur(S_[k]["95 %"]), pc(S_[k]["P(Verlust)"], 0)]
           for lab, k in (("Jetziges Portfolio", "Aktuell"), ("Vorschlag", target))], (40, 36, 34, 36, 34))
    if with_stress:
        h2("Krisen-Stresstest")
        cl = crisis_losses(assets, cur)
        table(["Krise", "Verlust deines Depots", "in €", "Abgedeckt"],
              [[n, pc(l, 0) if l is not None else "keine Daten", ("−" + eur(-l * V)) if l is not None and l < 0 else "–",
                pc(cv, 0)] for n, l, cv in cl], (60, 45, 40, 35))
    h2("Wichtige Hinweise")
    p("Alle Zahlen beruhen auf vergangenen Kursen (Yahoo Finance). Was früher gut lief, muss nicht so weiterlaufen. "
      "Gebühren, Steuern und Wechselkurseffekte sind nur teilweise berücksichtigt. Dieser Bericht ist keine "
      "Anlageberatung.", 9)
    return bytes(pdf.output())


def page_report():
    head("PDF-Bericht", "Die wichtigsten Ergebnisse als PDF – zum Ausdrucken, Abspeichern oder Weiterschicken.",
         "Der Bericht nutzt deine aktuellen Eingaben und Einstellungen (Ziel, Zeitraum, neues Geld, nur nachkaufen).")
    c = ready()
    st.markdown("**Inhalt:** Auf einen Blick · Dein Depot (bei genauer Eingabe) · Vorschlag zur Umschichtung mit "
                "Grafik · Wertentwicklung · Realitäts-Check · Zukunftssimulation · optional Krisen-Stresstest · Hinweise")
    ws = st.checkbox("Krisen-Stresstest einbeziehen", key="rep_stress",
                     help="Lädt lange Kurshistorien – beim ersten Mal dauert der Bericht dann etwas länger.")
    if st.button("Bericht erstellen", type="primary", icon=":material/picture_as_pdf:"):
        with st.spinner("Erstelle Bericht …"):
            try:
                ss.pdf = build_pdf(c, ws)
                ss.pdf_err = None
            except Exception as e:
                ss.pdf, ss.pdf_err = None, str(e)
    if ss.get("pdf_err"):
        st.error(f"Der Bericht konnte nicht erstellt werden: {ss.pdf_err}")
    if ss.get("pdf"):
        st.success("Fertig! Dein Bericht ist bereit.")
        st.download_button("PDF herunterladen", ss.pdf, file_name=f"portfolio-bericht-{date.today().isoformat()}.pdf",
                           mime="application/pdf", icon=":material/download:", on_click="ignore", type="primary")
        st.caption("Ändern sich deine Eingaben, erstell den Bericht einfach neu.")


# =====================================================================
# Trade-Republic-Transaktionsexport auswerten
# =====================================================================
INFLOW = {"CUSTOMER_INBOUND", "CUSTOMER_INPAYMENT", "TRANSFER_INBOUND", "TRANSFER_INSTANT_INBOUND"}
OUTFLOW = {"TRANSFER_OUTBOUND", "TRANSFER_INSTANT_OUTBOUND"}
REQUIRED = {"datetime", "category", "type", "asset_class", "name", "symbol", "shares", "amount", "fee", "tax"}


def read_tr_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Das sieht nicht nach einem Trade-Republic-Transaktionsexport aus (es fehlen die Spalten "
                         + ", ".join(sorted(missing)) + ").")
    for c in ("shares", "price", "amount", "fee", "tax"):
        df[c] = pd.to_numeric(df[c].replace("", "0"), errors="coerce").fillna(0.0)
    df["ts"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce", format="mixed")
    df["day"] = pd.to_datetime(df.get("date", df["datetime"]).str[:10], errors="coerce")
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def analyse_tr(df: pd.DataFrame) -> dict:
    pos = {}         # symbol -> dict(qty, cost, name, cls, first)
    realized = []    # Verkäufe
    warn = []

    def p(sym, row):
        if sym not in pos:
            pos[sym] = {"qty": 0.0, "cost": 0.0, "name": row["name"].strip(), "cls": row["asset_class"] or "STOCK",
                        "first": row["day"], "fees": 0.0}
        return pos[sym]

    # Stornierte Verkäufe samt Storno entfernen
    canc = df[df["type"] == "SELL_CANCELLED"]
    drop = set(canc.index)
    for _, cr in canc.iterrows():
        m = df[(df["type"] == "SELL") & (df["symbol"] == cr["symbol"]) & ((df["shares"] + cr["shares"]).abs() < 1e-9)
               & (~df.index.isin(drop))]
        if len(m):
            drop.add(m.index[0])
    df = df.drop(index=list(drop))

    # Kapitalmaßnahmen nach Zeitpunkt gruppieren (Fusion / Umtausch: alte ISIN raus, neue rein)
    pair = {}
    for ts, g in df[df["category"] == "CORPORATE_ACTION"].groupby("ts"):
        neg, posr = g[g["shares"] < 0], g[g["shares"] > 0]
        if g["type"].iloc[0] in ("MERGER", "SHARE_EXCHANGE") and len(neg) == 1 and len(posr) == 1:
            for k in g.index:
                pair[k] = (neg.index[0], posr.index[0])
    done = set()
    for i, r in df.iterrows():          # streng chronologisch
        typ, sym, sh = r["type"], r["symbol"], r["shares"]
        if r["category"] == "TRADING" and typ in ("BUY", "SELL"):
            q = p(sym, r)
            if typ == "BUY":
                if q["qty"] <= 1e-9:
                    q["first"] = r["day"]
                q["qty"] += sh
                q["cost"] += -r["amount"] - r["fee"]          # Betrag + Gebühr (fee ist negativ)
                q["fees"] += -r["fee"]
            else:
                n = -sh
                avg = q["cost"] / q["qty"] if q["qty"] > 1e-12 else 0.0
                cost_out = avg * min(n, max(q["qty"], 0.0))
                if q["qty"] + 1e-6 < n:
                    warn.append(f"{q['name']}: Verkauf von {n:g} Stück, laut Export waren nur {q['qty']:g} im Bestand.")
                proceeds = r["amount"] + r["fee"]
                realized.append({"day": r["day"], "symbol": sym, "name": q["name"], "cls": q["cls"], "shares": n,
                                 "proceeds": proceeds, "cost": cost_out, "gain": proceeds - cost_out, "tax": r["tax"],
                                 "first": q["first"], "closed": q["qty"] - n < 1e-6})
                q["qty"] -= n
                q["cost"] -= cost_out
                if q["qty"] < 1e-9:
                    q["qty"], q["cost"] = 0.0, 0.0
        elif r["category"] == "CORPORATE_ACTION":
            if i in pair:
                if i in done:
                    continue
                ni, pi = pair[i]
                done |= {ni, pi}
                neg, posr = df.loc[ni], df.loc[pi]
                old, new = p(neg["symbol"], neg), p(posr["symbol"], posr)
                frac = min(1.0, -neg["shares"] / old["qty"]) if old["qty"] > 0 else 0.0
                moved = old["cost"] * frac
                if new["qty"] <= 1e-9:
                    new["first"] = old["first"]
                new["qty"] += posr["shares"]
                new["cost"] += moved
                new["cls"] = old["cls"] or new["cls"]
                old["qty"] += neg["shares"]
                old["cost"] -= moved
                if old["qty"] < 1e-9:
                    old["qty"], old["cost"] = 0.0, 0.0
            elif typ == "SPLIT":
                q = p(sym, r)
                if sh > 0 and q["qty"] > 0:
                    q["qty"] = sh                               # neue Gesamtstückzahl, Kosten bleiben gleich
            elif typ in ("CAPITAL_INCR_CASH", "STOCK_DIVIDEND", "SPINOFF", "BONUS"):
                q = p(sym, r)
                q["qty"] += sh                                  # Bezugsrechte/Gratisaktien: Kosten 0
            else:
                q = p(sym, r)
                q["qty"] += sh
                warn.append(f"Unbekannte Kapitalmaßnahme {typ} bei {r['name']} – Stückzahl übernommen, Kosten nicht.")

    def bucket(row):
        t_, c_ = row["type"], row["category"]
        if c_ == "TRADING":
            return ("Aktiengeschäft", "Käufe (inkl. Gebühren)" if row["shares"] > 0 else "Verkäufe (netto)")
        if c_ == "CORPORATE_ACTION":
            return ("Aktiengeschäft", "Kapitalmaßnahmen")
        if t_ == "DIVIDEND":
            return ("Aktiengeschäft", "Dividenden (netto)")
        if t_ == "STOCKPERK":
            return ("Aktiengeschäft", "Boni (Stockperks)")
        if t_ in ("TAX_OPTIMIZATION", "SEC_ACCOUNT"):
            return ("Aktiengeschäft", "Steuer-Korrekturen")
        if t_ in INFLOW:
            return ("Konto", "Einzahlungen")
        if t_ in OUTFLOW:
            return ("Konto", "Überweisungen raus")
        if t_.startswith("CARD"):
            return ("Konto", "Kartenzahlungen" if t_ == "CARD_TRANSACTION" else "Kartengebühren & -erstattungen")
        if t_ == "GIFT":
            return ("Konto", "Geschenke")
        if t_ == "INTEREST_PAYMENT":
            return ("Konto", "Zinsen (netto)")
        return ("Konto", "Sonstiges")
    bb = df.apply(bucket, axis=1, result_type="expand")
    df = df.assign(bereich=bb[0], topf=bb[1], effekt=(df["amount"] + df["fee"] + df["tax"]).where(
        df["category"] != "CORPORATE_ACTION", 0.0))

    hold = pd.DataFrame([{"symbol": s, **v} for s, v in pos.items() if v["qty"] > 1e-6])
    real = pd.DataFrame(realized)
    cash = df[df["category"] == "CASH"]
    div = cash[cash["type"] == "DIVIDEND"]
    summary = {
        "einzahlungen": float(cash.loc[cash["type"].isin(INFLOW), "amount"].sum()),
        "auszahlungen": float(-cash.loc[cash["type"].isin(OUTFLOW), "amount"].sum()),
        "karte": float(-cash.loc[cash["type"] == "CARD_TRANSACTION", "amount"].sum()),
        "geschenke": float(cash.loc[cash["type"] == "GIFT", "amount"].sum()),
        "boni": float(cash.loc[cash["type"] == "STOCKPERK", "amount"].sum()),
        "dividenden_brutto": float(div["amount"].sum()),
        "dividenden_steuer": float(div["tax"].sum()),
        "zinsen_brutto": float(cash.loc[cash["type"] == "INTEREST_PAYMENT", "amount"].sum()),
        "zinsen_steuer": float(cash.loc[cash["type"] == "INTEREST_PAYMENT", "tax"].sum()),
        "steuer_optimierung": float(cash.loc[cash["type"] == "TAX_OPTIMIZATION", ["amount", "tax"]].sum().sum()),
        "gebuehren": float(df["fee"].sum()),
        "steuern_gesamt": float(df["tax"].sum()),
        "kontostand": float((df["amount"] + df["fee"] + df["tax"]).where(df["category"] != "CORPORATE_ACTION", 0).sum()),
        "start": df["day"].min(), "ende": df["day"].max(), "n": len(df),
        "aktien_cash": float(df.loc[df["bereich"] == "Aktiengeschäft", "effekt"].sum()),
        "konto_cash": float(df.loc[df["bereich"] == "Konto", "effekt"].sum()),
    }
    return {"hold": hold, "real": real, "sum": summary, "div": div, "warn": warn, "df": df}


# =====================================================================
# Seite: Trade-Republic-Import
# =====================================================================
@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False)
def isin_to_ticker(sym, name, cls):
    if cls == "CRYPTO":
        return f"{sym}-EUR"
    res = search(sym) or search(name)
    return res[0]["sym"] if res else None


def tr_load():
    f = ss.get("tr_file")
    if f is None:
        return
    try:
        ss.tr = analyse_tr(read_tr_csv(f.getvalue()))
        ss.tr_err = None
        ss.tr_map = {}
    except Exception as e:
        ss.tr, ss.tr_err = None, str(e)


def tr_valuation(tr):
    rows = []
    for _, h_ in tr["hold"].iterrows():
        sym, cls = h_["symbol"], h_["cls"]
        tk = ss.tr_map.get(sym)
        if tk is None:
            tk = isin_to_ticker(sym, h_["name"], cls) or ""
            ss.tr_map[sym] = tk
        price, cur_ = get_quote(tk) if tk else (None, None)
        fx = get_fx(cur_) if price is not None else None
        pe = price * fx if price is not None and fx is not None else None
        val = h_["qty"] * pe if pe is not None else None
        rows.append({"symbol": sym, "ticker": tk, "name": h_["name"], "krypto": cls == "CRYPTO",
                     "typ": "Krypto" if cls == "CRYPTO" else ("ETF/Fonds" if cls == "FUND" else "Aktie"),
                     "qty": h_["qty"], "cost": h_["cost"], "avg": h_["cost"] / h_["qty"], "first": h_["first"],
                     "price": pe, "value": val, "gain": (val - h_["cost"]) if val is not None else None})
    return pd.DataFrame(rows)


def tr_apply(v, with_crypto):
    use = v[v["price"].notna() & (with_crypto | ~v["krypto"])]
    ss.sel = list(dict.fromkeys(use["ticker"]))
    for tk, g in use.groupby("ticker", sort=False):       # gleiche Ticker (z. B. zwei ISINs) zusammenfassen
        qty, cost = float(g["qty"].sum()), float(g["cost"].sum())
        first = g["first"].min()
        ss.names[tk] = g["name"].iloc[0]
        ss.pos[tk] = {"qty": qty, "cost": cost / qty if qty else 0.0,
                      "date": first.date() if pd.notna(first) else None}
    ss.input_mode = "Genau: Stückzahl & Kaufkurs"
    ss.imp += 1
    ss.tr_applied = len(use)


def pos_table(d):
    return pd.DataFrame({
        "Name": d["name"], "Typ": d["typ"], "Stück": [de(x, 4 if x < 1 else 2) for x in d["qty"]],
        "Ø Kaufkurs": [eur2(x) for x in d["avg"]], "Kurs heute": [eur2(x) for x in d["price"]],
        "Wert": [eur(x) if x == x and x is not None else "kein Kurs" for x in d["value"]],
        "Investiert": [eur(x) for x in d["cost"]],
        "Gewinn/Verlust": [(("+" if g >= 0 else "−") + eur(abs(g))) if g is not None and g == g else "–" for g in d["gain"]],
        "in %": [(sg(g / c * 100) + " %") if g is not None and g == g and c > 0 else "–" for g, c in zip(d["gain"], d["cost"])],
        "Seit": [x.strftime("%m/%Y") if pd.notna(x) else "–" for x in d["first"]]})


def page_tr():
    head("Trade-Republic-Import", "Lade deinen Transaktionsexport hoch: Die App berechnet Bestand, Einstandskurse, "
                                  "Gewinne, Dividenden und dein Gesamtvermögen – Wertpapiere und Krypto getrennt.",
         "So bekommst du die Datei: Trade Republic App → Profil → Einstellungen/Dokumente → „Transaktionsexport“ "
         "(CSV). Die Datei wird nur in deiner Sitzung verarbeitet und nirgends gespeichert. Kontodaten (IBAN, "
         "Empfänger) werden nicht angezeigt.")
    st.file_uploader("Transaktionsexport (CSV) hochladen", type=["csv"], key="tr_file", on_change=tr_load,
                     help="Die CSV-Datei aus der Trade Republic App mit allen Transaktionen seit Kontoeröffnung.")
    if ss.get("tr_err"):
        st.error(ss.tr_err)
    tr = ss.get("tr")
    if not tr:
        st.info("Noch keine Datei geladen.")
        return
    S_ = tr["sum"]
    with st.spinner("Suche Kurse zu deinen Positionen (beim ersten Mal etwas langsamer) …"):
        v = tr_valuation(tr)
    for w in tr["warn"]:
        st.warning(w)
    st.caption(f"{S_['n']} Buchungen von {S_['start'].strftime('%d.%m.%Y')} bis {S_['ende'].strftime('%d.%m.%Y')}.")
    sec, cry = v[~v["krypto"]], v[v["krypto"]]
    val_sec = sec["value"].fillna(sec["cost"]).sum()
    val_cry = cry["value"].fillna(cry["cost"]).sum()
    cash = S_["kontostand"]
    total = val_sec + val_cry + cash
    net_in = S_["einzahlungen"] - S_["auszahlungen"] - S_["karte"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gesamtvermögen", eur(total), help="Wertpapiere + Krypto + Guthaben auf dem Verrechnungskonto.")
    k2.metric("Wertpapiere", eur(val_sec), help="Aktien, ETFs und Fonds zum aktuellen Kurs.")
    k3.metric("Krypto", eur(val_cry), help="Kryptowährungen zum aktuellen Kurs in Euro.")
    k4.metric("Guthaben (Cash)", eur(cash), help="Berechneter Kontostand aus allen Buchungen der Datei.")
    gt = total - net_in
    inv_res = val_sec + val_cry + S_["aktien_cash"]          # Wert heute + alles, was das Aktiengeschäft ein-/ausgezahlt hat
    paid_in = -tr["df"].loc[tr["df"]["topf"] == "Käufe (inkl. Gebühren)", "effekt"].sum()
    st.markdown("##### Aktiengeschäft")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Ergebnis Aktiengeschäft", ("+" if inv_res >= 0 else "−") + eur(abs(inv_res)),
              f"{sg(inv_res / paid_in * 100) if paid_in > 0 else '–'} % auf alle Käufe",
              help="Nur dein Investieren: Wert der offenen Positionen + Verkaufserlöse + Dividenden + Boni − alle "
                   "Käufe (inkl. Gebühren) ± Steuern. Kartenzahlungen, Überweisungen und Zinsen sind NICHT enthalten.")
    real = tr["real"]
    rg = real["gain"].sum() if len(real) else 0.0
    k6.metric("Realisierte Gewinne", ("+" if rg >= 0 else "−") + eur(abs(rg)),
              help="Gewinne/Verluste aus allen Verkäufen, nach Durchschnittskosten-Methode (inkl. Gebühren).")
    divn = S_["dividenden_brutto"] + S_["dividenden_steuer"]
    k7.metric("Dividenden (netto)", eur(divn), help=f"Dividenden {eur(S_['dividenden_brutto'])} brutto, abzüglich "
              "einbehaltener Steuern.")
    tdf = tr["df"]
    trade_fees = -tdf.loc[tdf["bereich"] == "Aktiengeschäft", "fee"].sum()
    trade_tax = -tdf.loc[tdf["bereich"] == "Aktiengeschäft", "tax"].sum()
    k8.metric("Gebühren + Steuern (Handel)", "−" + eur(trade_fees + trade_tax),
              help=f"Ordergebühren {eur(trade_fees)}, Steuern auf Verkäufe/Dividenden netto {eur(trade_tax)}.")
    kt = tdf[tdf["bereich"] == "Konto"]
    with st.container(border=True):
        st.markdown("##### Konto & Zahlungen (getrennt vom Aktiengeschäft)")
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("Eingezahlt", eur(kt.loc[kt["topf"] == "Einzahlungen", "effekt"].sum()),
                  help="Alle Überweisungen auf dein Trade-Republic-Konto.")
        z2.metric("Überwiesen raus", "−" + eur(-kt.loc[kt["topf"] == "Überweisungen raus", "effekt"].sum()),
                  help="Alle Überweisungen von Trade Republic auf andere Konten.")
        z3.metric("Kartenzahlungen", "−" + eur(-kt.loc[kt["topf"].str.startswith("Karten"), "effekt"].sum()),
                  help="Ausgaben mit der Trade-Republic-Karte inkl. Kartengebühren.")
        z4.metric("Zinsen (netto)", eur(kt.loc[kt["topf"] == "Zinsen (netto)", "effekt"].sum()),
                  help="Zinsen auf dein Guthaben nach Steuern.")
        st.page_link(P_KONTO, label="Alle Zahlungen im Detail", icon=":material/account_balance:")

    cA, cB = st.columns([3, 2], gap="large")
    for col, d, title, colr in ((cA, sec, "Wertpapiere (Aktien, ETFs, Fonds)", "#1240e0"), (cB, cry, "Krypto", "#f7931a")):
        with col.container(border=True):
            st.subheader(title, help="Unrealisierter Gewinn/Verlust = Wert heute minus Einstandskosten (Durchschnitts"
                                     "kosten inkl. Kaufgebühren) der noch gehaltenen Stücke.")
            if d.empty:
                st.caption("Keine offenen Positionen.")
                continue
            vv, cc = d["value"].fillna(d["cost"]).sum(), d["cost"].sum()
            gg = d["gain"].dropna().sum()
            rr = real[(real["cls"] == "CRYPTO") == (title == "Krypto")] if len(real) else real
            rgain = float(rr["gain"].sum()) if len(rr) else 0.0
            m1, m2 = st.columns(2)
            m1.metric("Wert heute", eur(vv), help="Offene Positionen zum aktuellen Kurs.")
            m2.metric("Investiert (offen)", eur(cc), help="Einstandskosten der noch gehaltenen Stücke.")
            m3, m4 = st.columns(2)
            m3.metric("Unrealisiert", ("+" if gg >= 0 else "−") + eur(abs(gg)), f"{sg(gg / cc * 100) if cc else '–'} %",
                      help="Gewinn/Verlust der noch gehaltenen Positionen – nur auf dem Papier.")
            m4.metric("Realisiert", ("+" if rgain >= 0 else "−") + eur(abs(rgain)),
                      help="Gewinne und Verluste aus allen bisherigen Verkäufen in diesem Bereich. Details auf der Seite "
                           "„Realisierte Gewinne“.")
            st.caption(f"{len(d)} {'Position' if len(d) == 1 else 'Positionen'} offen · {len(rr)} Verkäufe bisher")
    st.subheader("Wie dein Vermögen verteilt ist")
    fd = go.Figure(go.Pie(labels=["Wertpapiere", "Krypto", "Cash"], values=[val_sec, val_cry, max(cash, 0)], hole=0.55,
                          marker=dict(colors=["#1240e0", "#f7931a", "#9aa7c7"]), textinfo="label+percent", sort=False))
    fd.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, font=dict(size=15), separators=",.")
    st.plotly_chart(fd, width="stretch")

    st.page_link(P_REAL, label="Alle realisierten Gewinne & Verluste im Detail", icon=":material/receipt_long:")
    t1, t2, t3, t4, t6, t5 = st.tabs(["Wertpapiere", "Krypto", "Verkäufe", "Dividenden & Zinsen", "Sparpläne",
                                      "Zuordnung der Kurse"])
    with t6:
        plans = detect_plans(tr["df"])
        ns = net_saving_12m(tr["df"])
        st.metric("Ø Netto-Einzahlung pro Monat (letzte 12 Monate)", eur(ns),
                  help="Einzahlungen minus Überweisungen raus, geteilt durch 12. Zeigt, wie viel du tatsächlich "
                       "monatlich zur Seite legst – auch ohne festen Sparplan.")
        if plans.empty:
            st.info("In deinem Export wurden keine Sparpläne gefunden – weder als Sparplan markierte Käufe noch "
                    "regelmäßige Käufe mit gleichem Betrag. Du kannst trotzdem deine Ø Netto-Einzahlung im "
                    "Sparplan-Rechner verwenden.")
        else:
            st.dataframe(pd.DataFrame({
                "Wertpapier": plans["name"], "Art": np.where(plans["krypto"], "Krypto", "Wertpapier"),
                "Betrag": [eur2(x) for x in plans["betrag"]], "Rhythmus": plans["rhythmus"],
                "≈ pro Monat": [eur2(x) for x in plans["monatlich"]], "Ausführungen": plans["ausfuehrungen"],
                "Seit": [d.strftime("%m/%Y") for d in plans["seit"]], "Status": np.where(plans["aktiv"], "aktiv", "beendet"),
                "Quelle": plans["quelle"]}), hide_index=True, width="stretch",
                column_config={"Quelle": st.column_config.TextColumn(
                    help="„markiert“: im Export als Sparplan gekennzeichnet. „erkannt“: regelmäßige Käufe mit ähnlichem "
                         "Betrag – prüf, ob das wirklich ein Sparplan ist.")})
        st.page_link(P_SAVE, label="Im Sparplan-Rechner verwenden", icon=":material/savings:")
    for tab, d in ((t1, sec), (t2, cry)):
        with tab:
            if d.empty:
                st.caption("Keine offenen Positionen.")
                continue
            d = d.sort_values("value", ascending=False, na_position="last")
            st.dataframe(pos_table(d), hide_index=True, width="stretch", height=min(800, 42 + 36 * len(d)))
            dk = d[d["gain"].notna()].sort_values("gain")
            if len(dk):
                fg = go.Figure(go.Bar(y=dk["name"], x=dk["gain"], orientation="h",
                                      marker_color=["#0b7a3b" if x >= 0 else "#e8361b" for x in dk["gain"]],
                                      text=[("+" if x >= 0 else "−") + eur(abs(x)) for x in dk["gain"]], textposition="outside"))
                fg.update_layout(height=80 + 32 * len(dk), margin=dict(l=10, r=50, t=10, b=10), xaxis_title="€",
                                 separators=",.", font=dict(size=14))
                st.plotly_chart(fg, width="stretch")
            miss = d[d["price"].isna()]
            if len(miss):
                st.caption("Ohne aktuellen Kurs (zum Einstandswert gerechnet): " + ", ".join(miss["name"]) +
                           ". Den Ticker kannst du im Reiter „Zuordnung der Kurse“ korrigieren.")
    with t3:
        if not len(real):
            st.caption("Keine Verkäufe.")
        else:
            rr = real.copy()
            rr["Jahr"] = rr["day"].dt.year
            rr["Art"] = np.where(rr["cls"] == "CRYPTO", "Krypto", "Wertpapiere")
            pv = rr.pivot_table(index="Jahr", columns="Art", values="gain", aggfunc="sum").fillna(0)
            fy = go.Figure([go.Bar(x=pv.index.astype(str), y=pv[c], name=c,
                                   marker_color="#f7931a" if c == "Krypto" else "#1240e0") for c in pv.columns])
            fy.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€",
                             separators=",.", legend=dict(orientation="h", y=1.1), font=dict(size=15))
            st.subheader("Realisierte Gewinne pro Jahr", help="Durchschnittskosten-Methode. Das Finanzamt rechnet bei "
                                                              "Aktien mit FIFO – einzelne Werte können abweichen.")
            st.plotly_chart(fy, width="stretch")
            st.dataframe(pd.DataFrame({"Datum": rr["day"].dt.strftime("%d.%m.%Y"), "Name": rr["name"], "Art": rr["Art"],
                                       "Stück": [de(x, 4) for x in rr["shares"]], "Erlös": [eur2(x) for x in rr["proceeds"]],
                                       "Einstand": [eur2(x) for x in rr["cost"]],
                                       "Gewinn/Verlust": [("+" if x >= 0 else "−") + eur2(abs(x)) for x in rr["gain"]]}
                                      ).iloc[::-1], hide_index=True, width="stretch", height=420)
    with t4:
        dv = tr["div"].copy()
        if len(dv):
            dv["Jahr"] = dv["day"].dt.year
            yy = dv.groupby("Jahr")[["amount", "tax"]].sum()
            fz = go.Figure([go.Bar(x=yy.index.astype(str), y=yy["amount"], name="Dividenden brutto", marker_color="#1240e0"),
                            go.Bar(x=yy.index.astype(str), y=yy["amount"] + yy["tax"], name="netto", marker_color="#b8c4e8")])
            fz.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€",
                             separators=",.", legend=dict(orientation="h", y=1.1), font=dict(size=15))
            st.subheader("Dividenden pro Jahr")
            st.plotly_chart(fz, width="stretch")
            top = dv.groupby("name")["amount"].sum().sort_values(ascending=False).head(10)
            st.dataframe(pd.DataFrame({"Name": top.index, "Dividenden gesamt": [eur2(x) for x in top.values]}),
                         hide_index=True, width="stretch")
        st.caption(f"Zinsen auf Guthaben insgesamt: {eur2(S_['zinsen_brutto'])} brutto · Boni (Stockperks): "
                   f"{eur2(S_['boni'])}")
    with t5:
        st.caption("Jede Position braucht einen Yahoo-Ticker, damit der aktuelle Kurs geladen werden kann. Die App "
                   "sucht ihn automatisch über die ISIN. Ist er falsch oder fehlt er, trag ihn hier ein (z. B. AAPL, "
                   "SAP.DE, BTC-EUR).")
        mp = pd.DataFrame({"Name": v["name"], "ISIN / Symbol": v["symbol"], "Yahoo-Ticker": v["ticker"],
                           "Kurs gefunden": ["ja" if p_ is not None and p_ == p_ else "nein" for p_ in v["price"]]})
        ed = st.data_editor(mp, hide_index=True, width="stretch", disabled=["Name", "ISIN / Symbol", "Kurs gefunden"],
                            key="tr_map_ed")
        if st.button("Zuordnung übernehmen", icon=":material/sync:"):
            for _, r_ in ed.iterrows():
                ss.tr_map[r_["ISIN / Symbol"]] = (r_["Yahoo-Ticker"] or "").strip().upper()
            st.rerun()

    st.divider()
    st.subheader("In die Analyse übernehmen", help="Übernimmt Stückzahlen, Einstandskurse und Kaufdaten in „Mein "
                                                   "Portfolio“ (genaue Eingabe). Danach funktionieren alle Seiten – "
                                                   "Übersicht, Simulationen, Stresstest – mit deinem echten Depot.")
    wc = st.checkbox("Krypto in Optimierung und Simulationen einbeziehen", True, key="tr_crypto",
                     help="Krypto schwankt sehr stark. Aus: Nur Wertpapiere werden analysiert; Krypto bleibt hier "
                          "in der Vermögensübersicht sichtbar.")
    if st.button("Depot übernehmen", type="primary", icon=":material/move_down:"):
        tr_apply(v, wc)
    if ss.get("tr_applied"):
        st.success(f"{ss.tr_applied} Positionen übernommen. Schau dir jetzt **Mein Depot** und **Übersicht & "
                   "Vorschlag** an. Tipp: Mit „Depot speichern“ (Seitenleiste) sicherst du den Stand.")
        st.page_link(P_DEPOT, label="Zu „Mein Depot“", icon=":material/arrow_forward:")
    lines = [f"Dein Gesamtvermögen bei Trade Republic beträgt rund **{eur(total)}**: {pc(val_sec / total if total else 0, 0)} "
             f"Wertpapiere, {pc(val_cry / total if total else 0, 0)} Krypto, {pc(cash / total if total else 0, 0)} Cash.",
             f"Dein **Aktiengeschäft** hat bisher **{('+' if inv_res >= 0 else '−')}{eur(abs(inv_res))}** gebracht "
             f"(offene Positionen, Verkäufe, Dividenden und Boni gegen alle Käufe gerechnet) – unabhängig davon, was du "
             "mit Karte bezahlt oder überwiesen hast.",
             f"Über das Konto sind netto **{eur(S_['konto_cash'])}** ins Investieren geflossen (Einzahlungen minus "
             "Überweisungen raus und Kartenzahlungen, plus Zinsen). Diese Bewegungen findest du getrennt auf "
             "**Konto & Zahlungen**."]
    if len(sec):
        best = sec.loc[sec["gain"].fillna(-1e18).idxmax()]
        worst = sec.loc[sec["gain"].fillna(1e18).idxmin()]
        lines.append(f"Größter Gewinner: **{best['name']}** ({('+' if best['gain'] >= 0 else '−')}{eur(abs(best['gain']))}). "
                     f"Größter Verlierer: **{worst['name']}** ({('+' if worst['gain'] >= 0 else '−')}{eur(abs(worst['gain']))}).")
        small = sec[sec["value"].fillna(sec["cost"]) < 0.01 * max(total, 1)]
        if len(small) >= 5:
            lines.append(f"**{len(small)} Positionen** machen jeweils weniger als 1 % deines Vermögens aus. Viele Mini-"
                         "Positionen bringen kaum Streuung, kosten aber Überblick – ein breiter ETF streut einfacher.")
    lines.append("Hinweis: Gewinne werden nach der Durchschnittskosten-Methode berechnet (wie in der Trade-Republic-App). "
                 "Für die Steuer gilt bei Aktien FIFO; bei Krypto sind Gewinne in Deutschland nach einem Jahr Haltedauer "
                 "steuerfrei – die App rechnet das nicht heraus.")
    explain(lines)




# =====================================================================
# Seite: Realisierte Gewinne & Verluste
# =====================================================================
def real_block(rr, title, color):
    """Kennzahlen + Tabelle für einen Bereich (Wertpapiere oder Krypto)."""
    win, los = rr[rr["gain"] > 0], rr[rr["gain"] < 0]
    tot = rr["gain"].sum()
    with st.container(border=True):
        st.subheader(title)
        a, b = st.columns(2)
        a.metric("Realisiert gesamt", ("+" if tot >= 0 else "−") + eur(abs(tot)),
                 f"{sg(tot / rr['cost'].sum() * 100) if rr['cost'].sum() else '–'} % auf den Einstand",
                 help="Summe aller Gewinne und Verluste aus Verkäufen.")
        b.metric("Trefferquote", pc(len(win) / len(rr), 0) if len(rr) else "–",
                 help="Anteil der Verkäufe mit Gewinn.")
        c, d = st.columns(2)
        c.metric("Summe Gewinne", "+" + eur(win["gain"].sum()), help=f"{len(win)} Verkäufe mit Gewinn.")
        d.metric("Summe Verluste", ("−" + eur(abs(los["gain"].sum()))) if len(los) else "0 €",
                 help=f"{len(los)} Verkäufe mit Verlust.")


def page_real():
    head("Realisierte Gewinne & Verluste", "Was du mit deinen Verkäufen tatsächlich verdient oder verloren hast – "
                                          "Wertpapiere und Krypto getrennt.",
         "Berechnet aus deinem Trade-Republic-Export mit der Durchschnittskosten-Methode (inkl. Kauf- und "
         "Verkaufsgebühren). Realisiert heißt: Der Gewinn oder Verlust ist durch den Verkauf „echt“ geworden.")
    tr = ss.get("tr")
    if not tr:
        st.info("Lade zuerst deinen Transaktionsexport auf der Seite **Trade-Republic-Import** hoch.")
        st.page_link(P_TR, label="Zum Trade-Republic-Import", icon=":material/upload_file:")
        return
    real = tr["real"].copy()
    if not len(real):
        st.info("In deinem Export gibt es noch keine Verkäufe.")
        return
    real["Art"] = np.where(real["cls"] == "CRYPTO", "Krypto", "Wertpapiere")
    real["tage"] = (real["day"] - real["first"]).dt.days
    real["pct"] = real["gain"] / real["cost"].where(real["cost"] > 0)
    tot = real["gain"].sum()
    S_ = tr["sum"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Realisiert insgesamt", ("+" if tot >= 0 else "−") + eur(abs(tot)),
              help="Alle Verkäufe seit Kontoeröffnung, Wertpapiere und Krypto zusammen.")
    k2.metric("Anzahl Verkäufe", f"{len(real)}", help="Teilverkäufe zählen einzeln.")
    k3.metric("Bester Verkauf", "+" + eur(real["gain"].max()), real.loc[real["gain"].idxmax(), "name"],
              delta_color="off")
    k4.metric("Schlechtester Verkauf", ("+" if real["gain"].min() >= 0 else "−") + eur(abs(real["gain"].min())),
              real.loc[real["gain"].idxmin(), "name"], delta_color="off")
    cA, cB = st.columns(2, gap="large")
    with cA:
        real_block(real[real["Art"] == "Wertpapiere"], "Wertpapiere", "#1240e0")
    with cB:
        if (real["Art"] == "Krypto").any():
            real_block(real[real["Art"] == "Krypto"], "Krypto", "#f7931a")
        else:
            st.container(border=True).caption("Keine Krypto-Verkäufe.")

    t1, t2, t3, t4, t5 = st.tabs(["Nach Wertpapier", "Pro Jahr", "Verlauf", "Alle Verkäufe", "Gesamtergebnis"])
    with t1:
        g = real.groupby(["name", "Art"]).agg(verk=("gain", "size"), erl=("proceeds", "sum"), ein=("cost", "sum"),
                                              gew=("gain", "sum"), closed=("closed", "max"),
                                              tage=("tage", "mean")).reset_index().sort_values("gew")
        st.subheader("Gewinn/Verlust je Wertpapier", help="Alle Verkäufe eines Wertpapiers zusammengefasst.")
        fg = go.Figure(go.Bar(y=g["name"], x=g["gew"], orientation="h",
                              marker_color=["#0b7a3b" if x >= 0 else "#e8361b" for x in g["gew"]],
                              text=[("+" if x >= 0 else "−") + eur(abs(x)) for x in g["gew"]], textposition="outside"))
        fg.update_layout(height=80 + 30 * len(g), margin=dict(l=10, r=60, t=10, b=10), xaxis_title="€",
                         separators=",.", font=dict(size=14))
        st.plotly_chart(fg, width="stretch")
        held = set(tr["hold"]["symbol"]) if len(tr["hold"]) else set()
        names_held = set(tr["hold"]["name"]) if len(tr["hold"]) else set()
        TC = st.column_config.TextColumn
        st.dataframe(pd.DataFrame({
            "Wertpapier": g["name"], "Art": g["Art"], "Verkäufe": g["verk"],
            "Erlös": [eur2(x) for x in g["erl"]], "Einstand": [eur2(x) for x in g["ein"]],
            "Gewinn/Verlust": [("+" if x >= 0 else "−") + eur2(abs(x)) for x in g["gew"]],
            "in %": [(("+" if x > 0 else "") + de(x / e * 100) + " %") if e > 0 else "–" for x, e in zip(g["gew"], g["ein"])],
            "Ø Haltedauer": [f"{de(t / 30.44, 0)} Monate" if t == t else "–" for t in g["tage"]],
            "Status": ["noch im Depot" if n in names_held else "komplett verkauft" for n in g["name"]]}).iloc[::-1],
            hide_index=True, width="stretch", height=min(800, 42 + 36 * len(g)),
            column_config={"Ø Haltedauer": TC(help="Durchschnittliche Zeit vom ersten Kauf bis zum Verkauf."),
                           "Status": TC(help="Ob du noch Stücke dieses Wertpapiers hältst.")})
    with t2:
        real["Jahr"] = real["day"].dt.year
        pv = real.pivot_table(index="Jahr", columns="Art", values="gain", aggfunc="sum").fillna(0)
        st.subheader("Realisiert pro Jahr", help="Wichtig für die Steuer: Gewinne und Verluste werden pro Kalenderjahr "
                                                 "verrechnet. In Deutschland sind 1.000 € pro Jahr (Sparer-Pauschbetrag) "
                                                 "steuerfrei, wenn ein Freistellungsauftrag vorliegt.")
        fy = go.Figure([go.Bar(x=pv.index.astype(str), y=pv[c], name=c,
                               marker_color="#f7931a" if c == "Krypto" else "#1240e0",
                               text=[("+" if v_ >= 0 else "−") + eur(abs(v_)) for v_ in pv[c]], textposition="outside")
                        for c in pv.columns])
        fy.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="€",
                         separators=",.", legend=dict(orientation="h", y=1.12), font=dict(size=15))
        st.plotly_chart(fy, width="stretch")
        yt = real.groupby("Jahr").agg(g=("gain", lambda s: s[s > 0].sum()), l=("gain", lambda s: s[s < 0].sum()),
                                      n=("gain", "size"), t=("tax", "sum"))
        st.dataframe(pd.DataFrame({"Jahr": yt.index.astype(str), "Verkäufe": yt["n"],
                                   "Gewinne": ["+" + eur2(x) for x in yt["g"]], "Verluste": ["−" + eur2(-x) for x in yt["l"]],
                                   "Saldo": [("+" if x >= 0 else "−") + eur2(abs(x)) for x in yt["g"] + yt["l"]],
                                   "Steuer bei Verkauf": [eur2(-x) if x else "–" for x in yt["t"]]}),
                     hide_index=True, width="stretch")
    with t3:
        rs = real.sort_values("day")
        st.subheader("Realisierte Gewinne über die Zeit", help="Aufsummiert: Wie sich dein realisiertes Ergebnis mit "
                                                              "jedem Verkauf entwickelt hat.")
        fc = go.Figure()
        for art, col in (("Wertpapiere", "#1240e0"), ("Krypto", "#f7931a")):
            s_ = rs[rs["Art"] == art]
            if len(s_):
                fc.add_scatter(x=s_["day"], y=s_["gain"].cumsum(), name=art, mode="lines+markers", line=dict(color=col, width=3),
                               line_shape="hv", text=s_["name"], hovertemplate="%{text}<br>%{x|%d.%m.%Y}: %{y:,.0f} €")
        fc.add_scatter(x=rs["day"], y=rs["gain"].cumsum(), name="Gesamt", line=dict(color="#555", width=2, dash="dash"),
                       line_shape="hv")
        fc.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€", separators=",.",
                         legend=dict(orientation="h", y=1.08), font=dict(size=15))
        st.plotly_chart(fc, width="stretch")
    with t4:
        rr = real.sort_values("day", ascending=False)
        st.dataframe(pd.DataFrame({
            "Datum": rr["day"].dt.strftime("%d.%m.%Y"), "Wertpapier": rr["name"], "Art": rr["Art"],
            "Stück": [de(x, 4) for x in rr["shares"]], "Erlös": [eur2(x) for x in rr["proceeds"]],
            "Einstand": [eur2(x) for x in rr["cost"]],
            "Gewinn/Verlust": [("+" if x >= 0 else "−") + eur2(abs(x)) for x in rr["gain"]],
            "in %": [(("+" if x > 0 else "") + de(x * 100) + " %") if x == x else "–" for x in rr["pct"]],
            "Gehalten": [f"{int(t)} Tage" if t == t else "–" for t in rr["tage"]],
            "Krypto > 1 Jahr": [("ja" if t > 365 else "nein") if a_ == "Krypto" and t == t else "" for t, a_ in
                                zip(rr["tage"], rr["Art"])]}),
            hide_index=True, width="stretch", height=600,
            column_config={"Krypto > 1 Jahr": st.column_config.TextColumn(
                help="In Deutschland sind Krypto-Gewinne nach über einem Jahr Haltedauer steuerfrei. Grobe Angabe ab "
                     "dem ersten Kauf – maßgeblich ist FIFO, prüfe das im Zweifel.")})
    with t5:
        hold = tr["hold"]
        v = tr_valuation(tr) if len(hold) else pd.DataFrame(columns=["gain"])
        unreal = float(v["gain"].dropna().sum()) if len(v) else 0.0
        divn = S_["dividenden_brutto"]
        zins = S_["zinsen_brutto"]
        fees = S_["gebuehren"]
        tax = S_["steuern_gesamt"]
        st.subheader("Woraus sich dein Gesamtergebnis zusammensetzt",
                     help="Realisiert + unrealisiert + Dividenden + Zinsen + Boni, abzüglich Gebühren und Steuern. Die "
                          "Gebühren für Käufe/Verkäufe stecken bereits in den Gewinnen und werden hier nicht doppelt "
                          "abgezogen.")
        parts = [("Realisiert", tot), ("Unrealisiert", unreal), ("Dividenden", divn), ("Zinsen", zins),
                 ("Boni", S_["boni"]), ("Steuern", tax), ("Sonstige Gebühren", S_.get("sonst_gebuehren", 0.0))]
        parts = [(n, x) for n, x in parts if abs(x) > 0.005]
        fw = go.Figure(go.Waterfall(x=[n for n, _ in parts] + ["Gesamt"], y=[x for _, x in parts] + [0],
                                    measure=["relative"] * len(parts) + ["total"],
                                    text=[("+" if x >= 0 else "−") + eur(abs(x)) for _, x in parts] +
                                         [("+" if sum(x for _, x in parts) >= 0 else "−") + eur(abs(sum(x for _, x in parts)))],
                                    textposition="outside", increasing=dict(marker_color="#0b7a3b"),
                                    decreasing=dict(marker_color="#e8361b"), totals=dict(marker_color="#1240e0")))
        fw.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="€", separators=",.",
                         font=dict(size=15))
        st.plotly_chart(fw, width="stretch")
        st.caption(f"Ordergebühren insgesamt: {eur2(-fees)} – bereits in den Gewinnen/Einstandskosten enthalten.")

    win, los = real[real["gain"] > 0], real[real["gain"] < 0]
    lines = [f"Mit **{len(real)} Verkäufen** hast du insgesamt **{('+' if tot >= 0 else '−')}{eur(abs(tot))}** "
             f"realisiert: {eur(win['gain'].sum())} Gewinne aus {len(win)} Verkäufen, {eur(-los['gain'].sum())} Verluste "
             f"aus {len(los)} Verkäufen."]
    if len(win) and len(los):
        aw, al = win["pct"].mean(), los["pct"].mean()
        lines.append(f"Im Schnitt hast du Gewinner mit **+{de(aw * 100, 0)} %** verkauft und Verlierer mit "
                     f"**{de(al * 100, 0)} %**. " + (
                         "Deine Gewinner bringen deutlich mehr, als die Verlierer kosten – gut so."
                         if aw > -al * 1.5 else
                         "Gewinne und Verluste sind ähnlich groß. Ein bekanntes Muster ist, Gewinner zu früh zu "
                         "verkaufen und Verlierer zu lange zu halten (Dispositionseffekt) – prüf, ob das auf dich zutrifft."))
        hw, hl = win["tage"].median(), los["tage"].median()
        if hw == hw and hl == hl:
            lines.append(f"Gewinner hast du im Mittel nach **{de(hw / 30.44, 0)} Monaten** verkauft, Verlierer nach "
                         f"**{de(hl / 30.44, 0)} Monaten**." + (" Verlierer hältst du also länger als Gewinner."
                                                               if hl > hw * 1.3 else ""))
    best = real.loc[real["gain"].idxmax()]
    lines.append(f"Dein bester Verkauf: **{best['name']}** am {best['day'].strftime('%d.%m.%Y')} mit "
                 f"+{eur(best['gain'])}.")
    cr = real[real["Art"] == "Krypto"]
    if len(cr):
        short = cr[cr["tage"] <= 365]
        if len(short):
            lines.append(f"{len(short)} Krypto-Verkäufe lagen innerhalb eines Jahres nach dem ersten Kauf – deren "
                         f"Gewinne ({eur(short['gain'].sum())}) sind in Deutschland grundsätzlich steuerpflichtig "
                         "(Freigrenze 1.000 € pro Jahr für private Veräußerungsgeschäfte).")
    lines.append("Hinweis: Die App rechnet mit Durchschnittskosten wie die Trade-Republic-App. Das Finanzamt nutzt bei "
                 "Aktien FIFO – einzelne Werte können abweichen. Für die Steuer zählen die Jahressteuerbescheinigungen.")
    explain(lines)




# =====================================================================
# Seite: Konto & Zahlungen
# =====================================================================
def page_konto():
    head("Konto & Zahlungen", "Alles, was nicht zum Aktiengeschäft gehört: Einzahlungen, Überweisungen, Kartenzahlungen, "
                              "Geschenke und Zinsen – sauber getrennt von deinen Investments.",
         "Aus deinem Trade-Republic-Export. IBANs und Verwendungszwecke werden nicht angezeigt.")
    tr = ss.get("tr")
    if not tr:
        st.info("Lade zuerst deinen Transaktionsexport auf der Seite **Trade-Republic-Import** hoch.")
        st.page_link(P_TR, label="Zum Trade-Republic-Import", icon=":material/upload_file:")
        return
    df = tr["df"]
    kt = df[df["bereich"] == "Konto"].copy()
    S_ = tr["sum"]
    tot = lambda t_: float(kt.loc[kt["topf"] == t_, "effekt"].sum())
    card = kt[kt["topf"].str.startswith("Karten")]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eingezahlt", eur(tot("Einzahlungen")), f"{int((kt['topf'] == 'Einzahlungen').sum())} Überweisungen",
              delta_color="off", help="Alle Einzahlungen auf dein Trade-Republic-Konto.")
    k2.metric("Überwiesen raus", "−" + eur(-tot("Überweisungen raus")),
              f"{int((kt['topf'] == 'Überweisungen raus').sum())} Überweisungen", delta_color="off",
              help="Alle Überweisungen auf andere Konten.")
    k3.metric("Mit Karte bezahlt", "−" + eur(-card["effekt"].sum()), f"{int((kt['topf'] == 'Kartenzahlungen').sum())} Zahlungen",
              delta_color="off", help="Kartenzahlungen inkl. Kartengebühren.")
    k4.metric("Zinsen (netto)", eur(tot("Zinsen (netto)")), help="Zinsen aufs Guthaben nach Steuern.")

    kt["Monat"] = kt["day"].dt.to_period("M").dt.to_timestamp()
    st.subheader("Geldflüsse pro Monat", help="Grün = Geld kommt aufs Konto, Rot = Überweisungen raus, Orange = Karte.")
    fg = go.Figure()
    for topf, col in (("Einzahlungen", "#0b7a3b"), ("Überweisungen raus", "#e8361b"), ("Kartenzahlungen", "#f7931a"),
                      ("Zinsen (netto)", "#1240e0"), ("Geschenke", "#9aa7c7")):
        s_ = kt[kt["topf"] == topf].groupby("Monat")["effekt"].sum()
        if len(s_):
            fg.add_bar(x=s_.index, y=s_.values, name=topf, marker_color=col)
    fg.update_layout(barmode="relative", height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="€",
                     separators=",.", legend=dict(orientation="h", y=1.1), font=dict(size=15))
    st.plotly_chart(fg, width="stretch")

    t1, t2, t3, t4 = st.tabs(["Kartenzahlungen", "Überweisungen", "Zinsen", "Wie dein Kontostand zustande kommt"])
    with t1:
        cz = kt[kt["topf"] == "Kartenzahlungen"].sort_values("day", ascending=False)
        if not len(cz):
            st.caption("Keine Kartenzahlungen.")
        else:
            st.dataframe(pd.DataFrame({"Datum": cz["day"].dt.strftime("%d.%m.%Y"),
                                       "Händler": [n.strip() or "–" for n in cz["name"]],
                                       "Betrag": [("−" if x < 0 else "+") + eur2(abs(x)) for x in cz["effekt"]]}),
                         hide_index=True, width="stretch")
            top = cz.groupby(cz["name"].str.strip())["effekt"].sum().sort_values().head(10)
            st.caption("Größte Ausgaben nach Händler: " + ", ".join(f"{n or '–'} ({eur(-x)})" for n, x in top.items()))
    with t2:
        tf = kt[kt["topf"].isin(["Einzahlungen", "Überweisungen raus", "Geschenke"])].sort_values("day", ascending=False)
        st.dataframe(pd.DataFrame({"Datum": tf["day"].dt.strftime("%d.%m.%Y"),
                                   "Art": tf["topf"].replace({"Einzahlungen": "Einzahlung", "Überweisungen raus": "Überweisung raus",
                                                              "Geschenke": "Geschenk"}),
                                   "Betrag": [("−" if x < 0 else "+") + eur2(abs(x)) for x in tf["effekt"]]}),
                     hide_index=True, width="stretch", height=480)
        yy = kt[kt["topf"].isin(["Einzahlungen", "Überweisungen raus"])].groupby([kt["day"].dt.year, "topf"])["effekt"].sum().unstack(fill_value=0)
        if len(yy):
            yy["Netto"] = yy.sum(axis=1)
            st.dataframe(pd.DataFrame({"Jahr": yy.index.astype(str), **{c: [eur(x) for x in yy[c]] for c in yy.columns}}),
                         hide_index=True, width="stretch")
    with t3:
        zi = kt[kt["topf"] == "Zinsen (netto)"]
        if len(zi):
            zz = zi.groupby(zi["day"].dt.year)[["amount", "tax"]].sum()
            st.dataframe(pd.DataFrame({"Jahr": zz.index.astype(str), "Zinsen brutto": [eur2(x) for x in zz["amount"]],
                                       "Steuer": [eur2(-x) for x in zz["tax"]],
                                       "Netto": [eur2(x) for x in zz["amount"] + zz["tax"]]}),
                         hide_index=True, width="stretch")
        else:
            st.caption("Keine Zinsen.")
    with t4:
        st.subheader("Vom ersten bis zum letzten Euro", help="Alle Buchungen in Töpfe sortiert. Die Summe ergibt genau "
                                                           "deinen heutigen Kontostand (Cash).")
        g = df.groupby(["bereich", "topf"])["effekt"].sum()
        g = g[g.abs() > 0.004]
        order = [k for k in g.index if k[0] == "Konto"] + [k for k in g.index if k[0] == "Aktiengeschäft"]
        fw = go.Figure(go.Waterfall(
            x=[f"{t_}" for _, t_ in order] + ["Kontostand"], y=[g[k] for k in order] + [0],
            measure=["relative"] * len(order) + ["total"],
            text=[("+" if g[k] >= 0 else "−") + eur(abs(g[k])) for k in order] + [eur(S_["kontostand"])],
            textposition="outside", increasing=dict(marker_color="#0b7a3b"), decreasing=dict(marker_color="#e8361b"),
            totals=dict(marker_color="#1240e0")))
        fw.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="€", separators=",.",
                         font=dict(size=13))
        st.plotly_chart(fw, width="stretch")
        st.dataframe(pd.DataFrame({"Bereich": [b for b, _ in order], "Topf": [t_ for _, t_ in order],
                                   "Summe": [("+" if g[k] >= 0 else "−") + eur2(abs(g[k])) for k in order]}),
                     hide_index=True, width="stretch")
    konto_net = S_["konto_cash"]
    explain([
        f"Du hast **{eur(tot('Einzahlungen'))}** eingezahlt und **{eur(-tot('Überweisungen raus'))}** wieder auf andere "
        f"Konten überwiesen. Mit der Karte hast du **{eur(-card['effekt'].sum())}** ausgegeben.",
        f"Für das Aktiengeschäft standen damit netto {eur(konto_net)} zur Verfügung (inklusive {eur(tot('Zinsen (netto)'))} "
        "Zinsen). Diese Bewegungen beeinflussen dein Anlageergebnis nicht – sie sind nur Geld, das kommt und geht.",
        f"Dein Kontostand von **{eur(S_['kontostand'])}** ergibt sich aus diesen Zahlungen plus allem, was das "
        "Aktiengeschäft ein- und ausgezahlt hat (Reiter „Wie dein Kontostand zustande kommt“).",
    ])




# =====================================================================
# Sparpläne aus dem Trade-Republic-Export erkennen
# =====================================================================
def detect_plans(df):
    """Ausdrücklich markierte Sparpläne + regelmäßige Käufe (gleicher Wert, ähnlicher Betrag, fester Rhythmus)."""
    b = df[(df["category"] == "TRADING") & (df["shares"] > 0)].copy()
    if b.empty:
        return pd.DataFrame()
    desc = b.get("description", pd.Series("", index=b.index)).fillna("")
    b["explicit"] = b["type"].str.contains("SAVING", case=False) | desc.str.contains("savings plan|sparplan", case=False)
    end_ = df["day"].max()
    out = []
    for sym, g in b.groupby("symbol"):
        explicit = bool(g["explicit"].any())
        src = g[g["explicit"]] if explicit else g
        daily = src.groupby(src["day"].dt.normalize())["amount"].sum().abs().sort_index()   # Teilstücke zusammenfassen
        if len(daily) < (2 if explicit else 3):
            continue
        dates = pd.Series(daily.index)
        # längste zusammenhängende Serie am Ende mit regelmäßigem Abstand und ähnlichem Betrag
        k = len(daily) - 1
        while k > 0:
            gap = (dates.iloc[k] - dates.iloc[k - 1]).days
            med = daily.iloc[k - 1:].median()
            if not (5 <= gap <= 40) or abs(daily.iloc[k - 1] - med) > 0.15 * med:
                break
            k -= 1
        run = daily.iloc[k:]
        if len(run) < (2 if explicit else 3):
            continue
        gap = float(pd.Series(run.index).diff().dt.days.dropna().median())
        amt = float(run.iloc[-3:].median())
        rhythm = ("wöchentlich" if gap < 10 else "alle 2 Wochen" if gap < 20 else "monatlich" if gap < 45
                  else "vierteljährlich")
        active = (end_ - run.index[-1]).days <= 1.6 * gap + 5
        out.append({"symbol": sym, "name": g["name"].iloc[0].strip(), "krypto": g["asset_class"].iloc[0] == "CRYPTO",
                    "betrag": amt, "rhythmus": rhythm, "monatlich": amt * 30.44 / gap, "ausfuehrungen": len(run),
                    "seit": run.index[0], "zuletzt": run.index[-1], "aktiv": active,
                    "quelle": "als Sparplan markiert" if explicit else "erkannt (regelmäßige Käufe)"})
    return pd.DataFrame(out)


def net_saving_12m(df):
    """Ø monatliche Netto-Einzahlung (Einzahlungen − Überweisungen raus) der letzten 12 Monate."""
    end_ = df["day"].max()
    last = df[(df["day"] > end_ - pd.Timedelta(days=365)) & df["topf"].isin(["Einzahlungen", "Überweisungen raus"])]
    return float(last["effekt"].sum()) / 12


def tr_total(tr):
    v = tr_valuation(tr)
    return float(v["value"].fillna(v["cost"]).sum() + tr["sum"]["kontostand"]) if len(v) else float(tr["sum"]["kontostand"])


def set_ss(**kw):
    for k, v in kw.items():
        ss[k] = v


# =====================================================================
# Menü
# =====================================================================
P_INPUT = st.Page(page_input, title="Mein Portfolio", icon=":material/edit:", url_path="portfolio", default=True)
P_DEPOT = st.Page(page_depot, title="Mein Depot", icon=":material/account_balance_wallet:", url_path="depot")
P_TR = st.Page(page_tr, title="Trade-Republic-Import", icon=":material/upload_file:", url_path="trade-republic")
P_REAL = st.Page(page_real, title="Realisierte Gewinne", icon=":material/receipt_long:", url_path="realisiert")
P_KONTO = st.Page(page_konto, title="Konto & Zahlungen", icon=":material/account_balance:", url_path="konto")
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
P_INFO = st.Page(page_info, title="Unternehmensdaten", icon=":material/apartment:", url_path="unternehmensdaten")
P_IMPROVE = st.Page(page_improve, title="Portfolio verbessern", icon=":material/lightbulb:", url_path="verbessern")
P_STRESS = st.Page(page_stress, title="Krisen-Stresstest", icon=":material/thunderstorm:", url_path="stresstest")
P_VSM = st.Page(page_vs_market, title="Vergleich mit dem Markt", icon=":material/compare_arrows:", url_path="markt")
P_SAVE = st.Page(page_savings, title="Sparplan-Rechner", icon=":material/savings:", url_path="sparplan")
P_WD = st.Page(page_withdraw, title="Entnahmeplan", icon=":material/payments:", url_path="entnahme")
P_DIV = st.Page(page_div, title="Dividenden", icon=":material/paid:", url_path="dividenden")
P_PDF = st.Page(page_report, title="PDF-Bericht", icon=":material/picture_as_pdf:", url_path="bericht")
P_NEWS = st.Page(page_news, title="Nachrichten", icon=":material/newspaper:", url_path="nachrichten")

menu = {"Start": [P_INPUT, P_TR, P_REAL, P_KONTO, P_DEPOT, P_OVER],
        "Simulationen & Analysen": [P_GROWTH, P_ASSETS, P_RISK, P_FUTURE, P_CHECK],
        "Planen & Testen": [P_STRESS, P_VSM, P_SAVE, P_WD, P_DIV],
        "Infos & Ideen": [P_IMPROVE, P_INFO, P_NEWS]}
if expert:
    menu["Experten"] = [P_FRONT, P_STYLE, P_COMP]
menu["Bericht & Hilfe"] = [P_PDF, P_HELP]
pg = st.navigation(menu)
pg.run()
with st.sidebar.expander("Depot speichern & laden", icon=":material/save:"):
    st.caption("Deine Eingaben werden nicht auf dem Server gespeichert. Sichere sie als Datei und lade sie beim "
               "nächsten Besuch wieder hoch.")
    st.download_button("Depot speichern", state_to_json(), file_name=f"mein-depot-{date.today().isoformat()}.json",
                       mime="application/json", icon=":material/download:", on_click="ignore", width="stretch",
                       help="Lädt eine kleine Datei mit deiner Auswahl, Stückzahlen, Kaufkursen und Einstellungen "
                            "herunter. Sie bleibt nur bei dir.")
    st.file_uploader("Gespeichertes Depot laden", type=["json"], key="loadfile", on_change=apply_load,
                     help="Wähle die Datei, die du mit „Depot speichern“ heruntergeladen hast.")
st.caption("Alle Zahlen beruhen auf vergangenen Kursen. Gebühren und Steuern sind nicht berücksichtigt. "
           "Keine Anlageberatung.")

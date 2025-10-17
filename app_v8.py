
import re
import io
import math
from math import log, sqrt, exp
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict

import asyncio
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from requests_html import HTMLSession, AsyncHTMLSession
import nest_asyncio

# ============================================================
# Config e título
# ============================================================

st.set_page_config(page_title="Strangle Vendido Coberto – v8 (JS render fix + loop)", layout="wide")
st.title("💼 Strangle Vendido Coberto – Sugeridor (v8)")
st.caption("Render JS estável no opções.net (corrigido o event loop do Streamlit) • Saídas automáticas • Ranking prêmio/risco • PoE • Top 3 • Catálogo B3")

# ============================================================
# Utils / Black–Scholes
# ============================================================

SQRT_2 = math.sqrt(2.0)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def yearfrac(start: date, end: date) -> float:
    return max(1e-9, (end - start).days / 365.0)

def bs_d1(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return np.nan
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def bs_d2(d1, sigma, T):
    if np.isnan(d1) or sigma <= 0 or T <= 0:
        return np.nan
    return d1 - sigma * math.sqrt(T)

def call_delta(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T)
    if np.isnan(d1):
        return np.nan
    return norm_cdf(d1)

def put_delta(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T)
    if np.isnan(d1):
        return np.nan
    return norm_cdf(d1) - 1.0

def prob_ITM_call(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    return norm_cdf(d2) if not np.isnan(d2) else np.nan

def prob_ITM_put(S, K, r, sigma, T):
    p_call = prob_ITM_call(S, K, r, sigma, T)
    return (1.0 - p_call) if not np.isnan(p_call) else np.nan

# ============================================================
# Helpers para Options.net e normalização
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

CALL_SERIES = set(list("ABCDEFGHIJKL"))
PUT_SERIES  = set(list("MNOPQRSTUVWX"))

def base_ticker_for_optionsnet(ticker: str) -> str:
    txt = (ticker or "").strip().upper()
    txt = re.sub(r"\.SA$", "", txt)
    txt = re.sub(r"[^A-Z0-9]", "", txt)
    return txt

def optionsnet_url_from_ticker(ticker: str) -> str:
    return f"https://opcoes.net.br/opcoes/bovespa/{base_ticker_for_optionsnet(ticker)}"

def _br_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    if s in ('', 'nan', '-', '—'): return np.nan
    s = s.replace('.', '').replace(',', '.')
    s = re.sub(r'[^0-9.\-eE]', '', s)
    try: return float(s)
    except: return np.nan

def try_read_csv_text(text: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return None

def normalize_chain_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        'symbol': ['symbol','símbolo','ticker','codigo','código','asset','codigo da opcao','código da opção','codigo da opção'],
        'expiration': ['expiration','venc','vencimento','expiração','expiracao','expiry','expiration_date','venc.','vencimento (d/m/a)'],
        'type': ['type','tipo','opção','opcao','option_type','class'],
        'strike': ['strike','preço de exercício','preco de exercicio','exercicio','k','strike_price'],
        'bid': ['bid','compra','melhor compra','oferta de compra'],
        'ask': ['ask','venda','melhor venda','oferta de venda'],
        'last': ['last','último','ultimo','preço','preco','close','último negócio','ultimo negocio','último neg.','ult.'],
        'impliedVol': ['iv','ivol','impliedvol','implied_vol','vol implícita','vol implicita','vol. impl. (%)','vol impl (%)'],
        'delta': ['delta','Δ']
    }
    rename = {}
    lowcols = {c.lower().strip(): c for c in df.columns}
    for target, aliases in colmap.items():
        for a in aliases:
            if a.lower() in lowcols:
                rename[lowcols[a.lower()]] = target
                break
    df = df.rename(columns=rename)

    if 'type' not in df.columns and 'symbol' in df.columns:
        def infer_type(code: str):
            if not isinstance(code, str): return np.nan
            s = re.sub(r'[^A-Z0-9]', '', code.upper().strip())
            m = re.search(r'([A-Z])\d+$', s)
            if not m: return np.nan
            letra = m.group(1)
            if letra in CALL_SERIES: return 'C'
            if letra in PUT_SERIES:  return 'P'
            return np.nan
        df['type'] = df['symbol'].map(infer_type)

    for c in ["strike","bid","ask","last","impliedVol","delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].map(_br_to_float), errors="coerce")

    if "expiration" in df.columns:
        def _parse_date(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    continue
            try:
                return pd.to_datetime(x, dayfirst=True).date()
            except Exception:
                return np.nan
        df["expiration"] = df["expiration"].map(_parse_date)

    for c in ["symbol","expiration","type","strike"]:
        if c not in df.columns:
            df[c] = np.nan

    df["type"] = df["type"].astype(str).str.upper().str.strip().replace({
        'CALL':'C','C':'C','COMPRA':'C','CALLS':'C',
        'PUT':'P','P':'P','VENDA':'P','PUTS':'P'
    })
    return df.dropna(subset=["symbol","type","strike"], how="any")

def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(p).strip() for p in tpl if str(p).strip() not in ('','nan','None')]) for tpl in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    seen, cols = {}, []
    for c in df.columns:
        if c not in seen:
            seen[c]=0; cols.append(c)
        else:
            seen[c]+=1; cols.append(f"{c}_{seen[c]}")
    df.columns = cols
    return df

def _choose_table_from_html(html_text: str) -> Optional[pd.DataFrame]:
    soup = BeautifulSoup(html_text, "html.parser")
    best_tbl, best_rows = None, -1
    for tbl in soup.find_all("table"):
        tbody = tbl.find("tbody")
        nrows = len(tbody.find_all("tr")) if tbody else max(len(tbl.find_all("tr"))-1, 0)
        if nrows > best_rows:
            best_rows, best_tbl = nrows, tbl
    if not best_tbl or best_rows <= 0:
        return None
    table_html = str(best_tbl)
    try:
        df = pd.read_html(io.StringIO(table_html), flavor="bs4", thousands=".", decimal=",", header=0)[0]
        if df.shape[0] == 0:
            raise ValueError("sem linhas")
    except Exception:
        df = pd.read_html(io.StringIO(table_html), flavor="bs4", thousands=".", decimal=",", header=[0,1])[0]
    return _flatten_columns(df)

# ------- NOVO: renderizador estável com event loop explícito -------

async def _fetch_render_async(url: str) -> str:
    asess = AsyncHTMLSession()
    r = await asess.get(url, headers=HEADERS, timeout=40)
    # arender executa JS em Chromium headless (pyppeteer)
    await r.html.arender(timeout=90, sleep=2)
    return r.html.html

def render_with_event_loop(url: str) -> str:
    """
    Garante um event loop válido dentro da thread do Streamlit.
    Usa nest_asyncio para permitir reentrância.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # Permite reuse do loop na thread do ScriptRunner
    nest_asyncio.apply()
    return loop.run_until_complete(_fetch_render_async(url))

@st.cache_data(show_spinner=False)
def fetch_optionsnet(ticker_url: str) -> pd.DataFrame:
    ticker_base = ticker_url.rsplit("/", 1)[-1].strip().upper()
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker_base}"

    # 1) Renderização JS com loop explícito
    try:
        html = render_with_event_loop(url)
        df = _choose_table_from_html(html)
        if df is not None and not df.empty:
            return normalize_chain_columns(df)
    except Exception as e:
        pass  # tenta fallbacks abaixo

    # 2) Fallback: HTML bruto sem render
    try:
        r = requests.get(url, headers=HEADERS, timeout=40)
        tables = pd.read_html(r.text)
        if tables:
            df2 = _flatten_columns(max(tables, key=lambda t: t.shape[0]*t.shape[1]))
            df2 = normalize_chain_columns(df2)
            if not df2.empty:
                return df2
    except Exception:
        pass

    # 3) Rota alternativa
    try:
        alt = f"https://opcoes.net.br/opcoes2/bovespa?ativo={ticker_base}"
        rr = requests.get(alt, headers=HEADERS, timeout=40)
        tables = pd.read_html(rr.text)
        if tables:
            df3 = _flatten_columns(max(tables, key=lambda t: t.shape[0]*t.shape[1]))
            df3 = normalize_chain_columns(df3)
            if not df3.empty:
                return df3
    except Exception:
        pass

    raise RuntimeError("Não foi possível obter a tabela de opções (render + fallbacks).")

# ============================================================
# Yahoo/IV helpers
# ============================================================

@st.cache_data(show_spinner=False)
def load_spot_and_iv_proxy(yahoo_ticker: str):
    try:
        y = yf.Ticker(yahoo_ticker)
        hist = y.history(period="2y")
        if len(hist) >= 30:
            spot_val = float(hist["Close"].iloc[-1])
            rets = hist["Close"].pct_change()
            daily_vol = rets.rolling(20).std()
            hv20_daily_annual = daily_vol * np.sqrt(252)
            return (spot_val,
                    float(hv20_daily_annual.dropna().iloc[-1]),
                    pd.DataFrame({"date": hist.index.date, "iv": hv20_daily_annual.values}).dropna())
    except Exception:
        pass
    return (np.nan, np.nan, None)

def compute_iv_rank_percentile(series_df: Optional[pd.DataFrame], lookback: int) -> Tuple[float, float, float]:
    if series_df is None or series_df.empty:
        return (np.nan, np.nan, np.nan)
    s = series_df.sort_values("date").tail(lookback)["iv"].dropna()
    if len(s) < 5:
        return (np.nan, np.nan, np.nan)
    iv_now = float(s.iloc[-1])
    iv_min, iv_max = float(s.min()), float(s.max())
    iv_rank = (iv_now - iv_min) / max(1e-9, (iv_max - iv_min))
    iv_percentile = float((s <= iv_now).mean())
    return (iv_now, iv_rank, iv_percentile)

# ============================================================
# Lista de tickers + nomes (dadosdemercado)
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_b3_ticker_list() -> pd.DataFrame:
    url = "https://www.dadosdemercado.com.br/acoes"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    if not tables:
        raise RuntimeError("Tabela de ações não encontrada na página.")
    df = max(tables, key=lambda t: t.shape[0] * t.shape[1]).copy()
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "ticker" in cl:
            colmap[c] = "ticker"
        elif "nome" in cl:
            colmap[c] = "name"
    df = df.rename(columns=colmap)
    if "ticker" not in df.columns or "name" not in df.columns:
        df = df.rename(columns={df.columns[0]:"ticker", df.columns[1]:"name"})
    df = df[["ticker","name"]].dropna().drop_duplicates().reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[(df["ticker"].str.len()>=3) & (df["name"].str.len()>0)]
    return df

# ============================================================
# Sidebar – parâmetros GERAIS
# ============================================================

with st.sidebar:
    st.header("Parâmetros gerais")
    st.caption("Use o catálogo para escolher Ticker — Empresa **ou** digite manualmente.")
    selic_input = st.number_input("Taxa livre de risco anual (r)", min_value=0.0, max_value=1.0, step=0.005, value=0.11, format="%.3f", help="Aproxima a SELIC para precificação (anual).")
    st.markdown("---")
    st.subheader("Tickers")
    load_btn = st.button("🔄 Carregar lista oficial (dadosdemercado)")
    if load_btn:
        try:
            b3_df = fetch_b3_ticker_list()
            st.success(f"Carregados {len(b3_df)} tickers da B3.")
            st.session_state["b3_df"] = b3_df
        except Exception as e:
            st.error(f"Falha ao carregar lista: {e}")

    b3_df = st.session_state.get("b3_df")
    tickers_manual = st.text_input("Tickers (manual, separados por vírgula)", value="PETR4.SA, VALE3.SA, ITUB4.SA")

    tickers_selected = []
    if b3_df is not None and not b3_df.empty:
        options = (b3_df["ticker"] + " — " + b3_df["name"]).tolist()
        chosen = st.multiselect("Escolha no catálogo (buscável):", options=options, help="Pesquise por ticker ou nome da empresa.")
        tickers_selected = [c.split(" — ", 1)[0] for c in chosen]

    st.markdown("---")
    st.subheader("Filtros por |Δ| (globais)")
    delta_min = st.number_input("|Δ| mínimo", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f", help="Ex.: 0.10–0.30 tende a ser mais OTM/seguro; prêmios menores.")
    delta_max = st.number_input("|Δ| máximo", min_value=0.00, max_value=1.00, value=0.35, step=0.01, format="%.2f", help="Limite superior de sensibilidade (|Δ|).")

    st.markdown("---")
    st.subheader("Bandas de risco (por perna)")
    risk_selection = st.multiselect("Selecionar", ["Baixo", "Médio", "Alto"], default=["Baixo","Médio","Alto"], help="Classificação pela probabilidade de exercício por perna (CALL e PUT).")
    st.caption("Baixo=0–15% · Médio=15–35% · Alto=35–55% de prob. ITM por perna")

    st.markdown("---")
    st.subheader("IV Rank / Percentil (proxy)")
    lookback_days = st.number_input("Janela (dias corridos)", min_value=60, max_value=500, value=252, step=1, help="Janela usada para IV Rank/Percentil (proxy com HV20).")

    st.markdown("---")
    st.subheader("Preferências de saída (didático)")
    show_exit_help = st.checkbox("Mostrar instruções operacionais de SAÍDA", value=True)
    days_exit_thresh = st.number_input("Janela crítica para saída (dias até o vencimento)", min_value=1, max_value=30, value=10)
    prox_pct = st.number_input("Proximidade ao strike que aciona alerta (%)", min_value=1, max_value=20, value=5) / 100.0
    capture_target = st.number_input("Meta de captura do prêmio para sair (%)", min_value=10, max_value=95, value=70) / 100.0
    st.caption("Regra didática: perto do vencimento + preço encostando no strike → recomprar a perna ameaçada; ou encerrar as duas quando já capturou grande parte do prêmio.")

# determina a lista final de tickers a processar
if tickers_selected:
    tickers = tickers_selected
else:
    tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_manual) if t.strip()]

if not tickers:
    st.info("Informe tickers manualmente ou carregue e selecione no catálogo.")
    st.stop()

# ============================================================
# Didática: bloco explicativo
# ============================================================

with st.expander("📘 Como interpretar (didático)", expanded=True):
    st.markdown("""
- **Strangle vendido coberto** = vender **1 PUT OTM** (com caixa garantido) + **1 CALL OTM** (com ações em carteira).  
- **Objetivo**: **maximizar o prêmio** e manter **baixa probabilidade** de exercício.  
- **probITM_call/put** (por perna): estimativa via **N(d2)** (Black‑Scholes). Fallback: **|Δ|**.  
- **PoE_total**: probabilidade combinada de **qualquer** perna ser exercida ≈ `probITM_put + probITM_call`.  
- **PoE_dentro**: probabilidade do preço **ficar entre os strikes** ≈ `1 − PoE_total`.  
- **Retorno potencial (%)**: `crédito por ação / preço spot`.  
- **Score (prêmio/risco)**: `retorno_pct / (risk_score + 0.01)`.  
""")

# ============================================================
# Pipeline por TICKER (função principal)
# ============================================================

def run_pipeline_for_ticker(tk: str, shares_owned: int, cash_available: float, lot_size: int):
    yahoo_ticker = tk if tk.endswith(".SA") else f"{tk}.SA"
    spot, hv20, iv_series = load_spot_and_iv_proxy(yahoo_ticker)

    # cabeçalho de métricas
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Preço à vista (S)", f"{spot:,.2f}" if not np.isnan(spot) else "—")
    with m2: st.metric("HV20 (σ anual – proxy)", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
    with m3: st.metric("r (anual)", f"{selic_input:.2%}")

    # fonte (options.net) e chain
    url = optionsnet_url_from_ticker(tk)
    st.write(f"🔗 **Fonte**: {url}")

    try:
        chain = fetch_optionsnet(url)
    except Exception as e:
        st.error(f"Erro ao obter dados do options.net: {e}")
        return None

    if chain.empty:
        st.warning("Nenhuma linha válida retornada.")
        return None

    # mid e IV efetiva
    if "bid" in chain.columns and "ask" in chain.columns and chain["bid"].notna().any() and chain["ask"].notna().any():
        chain["mid"] = (chain["bid"].fillna(0) + chain["ask"].fillna(0)) / 2.0
    else:
        if "last" not in chain.columns:
            st.error("Tabela sem bid/ask e sem 'last' — não há como estimar o prêmio.")
            return None
        chain["mid"] = chain["last"].fillna(0)

    if "impliedVol" not in chain.columns:
        chain["impliedVol"] = np.nan
    chain["iv_eff"] = chain["impliedVol"]
    if chain["iv_eff"].isna().all():
        chain["iv_eff"] = hv20

    # vencimentos
    expirations = sorted([d for d in chain["expiration"].dropna().unique().tolist() if isinstance(d, (date, datetime))], key=lambda x: x)
    if not expirations:
        st.error("Nenhum vencimento válido encontrado.")
        return None

    exp_choice = st.selectbox("Vencimento", options=expirations, key=f"exp_{tk}")
    T = yearfrac(date.today(), exp_choice if isinstance(exp_choice, date) else exp_choice.date())
    days_to_exp = (exp_choice - date.today()).days if isinstance(exp_choice, date) else (exp_choice.date() - date.today()).days
    if np.isnan(spot) or spot <= 0 or T <= 0:
        st.error("Não foi possível calcular S ou T para este ticker.")
        return None

    chain = chain[chain["expiration"] == exp_choice].copy()
    if chain.empty:
        st.warning("Nenhuma opção para o vencimento escolhido.")
        return None

    # IV Rank / Percentil (proxy)
    iv_now, iv_rank, iv_pct = compute_iv_rank_percentile(iv_series, int(lookback_days))
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("IV atual (proxy)", f"{iv_now:.2%}" if not np.isnan(iv_now) else "—")
    with c2: st.metric("IV Rank", f"{iv_rank:.0%}" if not np.isnan(iv_rank) else "—")
    with c3: st.metric("IV Percentil", f"{iv_pct:.0%}" if not np.isnan(iv_pct) else "—")

    # Separa calls/puts OTM
    calls = chain[(chain["type"] == "C") & (chain["strike"] >= 0)].copy()
    puts  = chain[(chain["type"] == "P") & (chain["strike"] >= 0)].copy()
    calls["OTM"] = calls["strike"] > spot
    puts["OTM"]  = puts["strike"]  < spot
    calls = calls[calls["OTM"]]
    puts  = puts[puts["OTM"]]

    # garante delta
    for side_df, side in [(calls,"C"), (puts,"P")]:
        if "delta" not in side_df.columns:
            side_df["delta"] = np.nan
        need = side_df["delta"].isna()
        if need.any():
            calc = []
            for _, row in side_df.loc[need].iterrows():
                K = float(row["strike"])
                sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
                sigma = max(sigma, 1e-6)
                d = call_delta(spot, K, selic_input, sigma, T) if side == "C" else put_delta(spot, K, selic_input, sigma, T)
                calc.append(d)
            side_df.loc[need, "delta"] = calc

    # filtro por |Δ|
    def apply_delta_filter(df: pd.DataFrame) -> pd.DataFrame:
        if (delta_min <= 0 and delta_max <= 0) or "delta" not in df.columns:
            return df
        d = df.copy()
        d["abs_delta"] = d["delta"].abs()
        if delta_min > 0:
            d = d[d["abs_delta"] >= delta_min]
        if delta_max > 0:
            d = d[d["abs_delta"] <= delta_max]
        return d

    calls = apply_delta_filter(calls)
    puts  = apply_delta_filter(puts)

    # ProbITM por perna (N(d2), fallback |Δ|)
    def compute_probs(df_side: pd.DataFrame, side: str) -> pd.DataFrame:
        df_side = df_side.copy()
        probs = []
        for _, row in df_side.iterrows():
            K = float(row["strike"])
            sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
            sigma = max(sigma, 1e-6)
            p_itm = prob_ITM_call(spot, K, selic_input, sigma, T) if side=="C" else prob_ITM_put(spot, K, selic_input, sigma, T)
            if np.isnan(p_itm) and not np.isnan(row.get("delta", np.nan)):
                p_itm = abs(float(row["delta"]))
            probs.append(p_itm)
        df_side["prob_ITM"] = probs
        return df_side

    calls = compute_probs(calls, "C")
    puts  = compute_probs(puts,  "P")

    # Cobertura e bandas
    max_qty_call = (shares_owned // lot_size) if lot_size > 0 else 0
    def max_qty_put_for_strike(K_put: float) -> int:
        if lot_size <= 0 or K_put <= 0:
            return 0
        return int(cash_available // (K_put * lot_size))

    risk_bands = {"Baixo":(0.00,0.15),"Médio":(0.15,0.35),"Alto":(0.35,0.55)}
    def label_band(p: float) -> str:
        if np.isnan(p):
            return "Fora"
        for k,(a,b) in risk_bands.items():
            if a <= p <= b:
                return k
        return "Fora"

    calls["band"] = calls["prob_ITM"].apply(label_band)
    puts["band"]  = puts["prob_ITM"].apply(label_band)
    calls = calls[calls["band"].isin(risk_selection)]
    puts  = puts[puts["band"].isin(risk_selection)]

    if calls.empty or puts.empty:
        st.warning("Sem CALLs/PUTs OTM dentro dos filtros/risco. Ajuste |Δ|/bandas.")
        return None

    # Combinações e ranking
    def combine_strangles(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
        combos: List[dict] = []
        for _, c in calls.iterrows():
            for _, p in puts.iterrows():
                Kc, Kp = float(c["strike"]), float(p["strike"])
                if not (Kp < spot < Kc):
                    continue
                mid_credit = float(c["mid"]) + float(p["mid"])
                qty_call_cap = max_qty_call
                qty_put_cap  = max_qty_put_for_strike(Kp)
                qty = min(qty_call_cap if qty_call_cap>0 else 0, qty_put_cap if qty_put_cap>0 else 0)
                if qty < 1:
                    continue
                be_low, be_high = Kp - mid_credit, Kc + mid_credit
                probITM_put = float(p.get("prob_ITM"))
                probITM_call = float(c.get("prob_ITM"))
                poe_total = min(1.0, max(0.0, probITM_put + probITM_call))
                poe_inside = max(0.0, 1.0 - poe_total)
                combos.append({
                    "ticker": tk,
                    "call_symbol": c.get("symbol"), "put_symbol": p.get("symbol"),
                    "K_call": Kc, "K_put": Kp,
                    "band_call": c.get("band"), "band_put": p.get("band"),
                    "probITM_call": probITM_call, "probITM_put": probITM_put,
                    "poe_total": poe_total, "poe_inside": poe_inside,
                    "delta_call": float(c.get("delta", np.nan)), "delta_put": float(p.get("delta", np.nan)),
                    "credit_total_por_contrato": mid_credit,
                    "qty": qty,
                    "credit_total_na_carteira": mid_credit * qty * lot_size,
                    "BE_low": be_low, "BE_high": be_high,
                    "iv_eff_avg": float(np.nanmean([c.get("iv_eff"), p.get("iv_eff")])),
                    "be_range": (be_high - be_low),
                    "lot_size": lot_size,
                    "shares_owned": shares_owned,
                    "cash_available": cash_available,
                    "expiration": exp_choice,
                })
        return pd.DataFrame(combos)

    combo_df = combine_strangles(calls, puts)
    if combo_df.empty:
        st.warning("Não há strangles possíveis respeitando cobertura (ações/caixa).")
        return None

    combo_df["retorno_pct"] = combo_df["credit_total_por_contrato"] / spot
    combo_df["risk_score"] = (combo_df["probITM_call"] + combo_df["probITM_put"]) / 2.0
    combo_df["score_final"] = combo_df["retorno_pct"] / (combo_df["risk_score"] + 0.01)

    def build_exit_guidance(row):
        Kc, Kp = row["K_call"], row["K_put"]
        credit = row["credit_total_por_contrato"]
        near_call = abs(spot - Kc) / Kc <= prox_pct
        near_put  = abs(spot - Kp) / Kp <= prox_pct
        time_critical = days_to_exp <= days_exit_thresh
        target_debit_per_leg = (1 - capture_target) * credit / 2.0
        target_debit_both = (1 - capture_target) * credit
        msg = []
        if time_critical and (near_call or near_put):
            if near_call and near_put:
                msg.append("⚠️ **Ambos strikes sob pressão** perto do vencimento.")
            elif near_call:
                msg.append("⚠️ **CALL ameaçada**: preço próximo de **K_call**.")
            elif near_put:
                msg.append("⚠️ **PUT ameaçada**: preço próximo de **K_put**.")
            msg.append(f"🕐 Faltam **{days_to_exp} d**; regra didática: **zerar o risco** recomprando a perna ameaçada.")
            msg.append("➡️ Ação: **compre de volta** a opção vendida (CALL ou PUT) para sair da operação parcial.")
        else:
            msg.append("✅ **Conforto**: preço razoavelmente distante dos strikes ou ainda há tempo até o vencimento.")
            msg.append("➡️ Ação: **mantenha** e monitore. Considere encerrar se capturar boa parte do prêmio.")
        msg.append(f"💰 Meta didática: **encerrar** quando capturar **{int(capture_target*100)}%** do prêmio.")
        msg.append(f"🔧 Ordem sugestão p/ **zerar ambas**: recomprar por ~ **R$ {target_debit_both:.2f}/ação** (≈ {(1-capture_target):.0%} do crédito).")
        msg.append(f"🔧 Ordem sugestão p/ **zerar só a perna ameaçada**: ~ **R$ {target_debit_per_leg:.2f}/ação** por perna.")
        return "  \n".join(msg), ("⚠️" if (time_critical and (near_call or near_put)) else "✅")

    if show_exit_help:
        exit_texts, alerts = [], []
        for _, r in combo_df.iterrows():
            text, alert = build_exit_guidance(r)
            exit_texts.append(text); alerts.append(alert)
        combo_df["Instrucao_saida"] = exit_texts
        combo_df["Alerta_saida"] = alerts

    def pick_by_band(df, band, n=3):
        sub = df[(df["band_call"]==band) & (df["band_put"]==band)].copy()
        if sub.empty: return sub
        return sub.sort_values(by=["score_final","credit_total_na_carteira","be_range"], ascending=[False,False,False]).head(n)

    final = []
    for band in risk_selection:
        pick = pick_by_band(combo_df, band, n=3)
        if not pick.empty:
            pick.insert(0, "Risco", band)
            final.append(pick)
    if not final:
        st.warning("Nenhuma combinação dentro das bandas escolhidas.")
        return None

    result = pd.concat(final, ignore_index=True)

    with st.expander("📊 Resumo do Ticker (médias e destaques)"):
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Combos sugeridos", f"{len(result):,}")
        with col2: st.metric("Crédito médio/ação", f"R$ {result['credit_total_por_contrato'].mean():.2f}")
        with col3: st.metric("PoE_total médio", f"{result['poe_total'].mean():.0%}")
        with col4: st.metric("Faixa BE média", f"R$ {result['be_range'].mean():.2f}")

    st.markdown("### 🏆 Top 3 Recomendações (melhor prêmio/risco)")
    top3 = combo_df.sort_values(by=["score_final","credit_total_na_carteira","be_range"], ascending=[False,False,False]).head(3)
    if not top3.empty:
        display_cols = ["call_symbol","K_call","put_symbol","K_put","credit_total_por_contrato","poe_total","retorno_pct","score_final","qty","BE_low","BE_high"]
        if show_exit_help:
            display_cols += ["Alerta_saida"]
        st.dataframe(top3[display_cols].rename(columns={
            "call_symbol":"CALL","put_symbol":"PUT","credit_total_por_contrato":"Crédito/ação",
            "poe_total":"PoE_total","retorno_pct":"Retorno %","score_final":"Score"
        }).style.format({"K_call":"%.2f","K_put":"%.2f","Crédito/ação":"R$ %.2f","PoE_total":"{:.0%}","Retorno %":"{:.2%}","Score":"{:.2f}","BE_low":"R$ %.2f","BE_high":"R$ %.2f"}), use_container_width=True)
    else:
        st.info("Sem combinações suficientes para ranquear o Top 3.")

    st.subheader("📌 Sugestões por banda de risco (ranqueadas por prêmio/risco)")
    st.caption("PoE_total = probabilidade de qualquer perna ser exercida; PoE_dentro = prob. do preço ficar entre os strikes no vencimento.")

    show_cols = ["ticker","Risco","call_symbol","K_call","probITM_call","delta_call",
                 "put_symbol","K_put","probITM_put","delta_put",
                 "credit_total_por_contrato","retorno_pct","poe_total","poe_inside",
                 "qty","credit_total_na_carteira","BE_low","BE_high","iv_eff_avg","be_range","expiration","score_final"]
    if show_exit_help:
        show_cols += ["Alerta_saida","Instrucao_saida"]

    fmt = {"K_call":"%.2f","K_put":"%.2f",
           "probITM_call":"{:.0%}","probITM_put":"{:.0%}",
           "delta_call":"{:.2f}","delta_put":"{:.2f}",
           "credit_total_por_contrato":"R$ {:.2f}",
           "retorno_pct":"{:.2%}",
           "poe_total":"{:.0%}","poe_inside":"{:.0%}",
           "credit_total_na_carteira":"R$ {:.2f}",
           "BE_low":"R$ {:.2f}","BE_high":"R$ {:.2f}",
           "iv_eff_avg":"{:.0%}","be_range":"R$ {:.2f}",
           "score_final":"{:.2f}"}

    st.dataframe(
        result[show_cols].rename(columns={
            "call_symbol":"CALL","put_symbol":"PUT","iv_eff_avg":"IV (média)",
            "be_range":"Faixa BE","expiration":"Vencimento","credit_total_por_contrato":"Crédito/ação",
            "retorno_pct":"Retorno %","poe_total":"PoE_total","poe_inside":"PoE_dentro","score_final":"Score",
            "Instrucao_saida":"📘 Instrução de saída", "Alerta_saida":"Alerta"
        }).style.format(fmt),
        use_container_width=True
    )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Exportar sugestões ({tk})", data=csv_bytes,
                       file_name=f"strangles_{base_ticker_for_optionsnet(tk)}_v8.csv",
                       mime="text/csv", key=f"dl_{tk}")

    st.markdown("### 📈 Payoff no Vencimento (P/L por ação)")
    result["id"] = (result["Risco"] + " | " + result["call_symbol"].astype(str) + " & " +
                    result["put_symbol"].astype(str) + " | Kc=" + result["K_call"].round(2).astype(str) +
                    " Kp=" + result["K_put"].round(2).astype(str) +
                    " | crédito≈" + result["credit_total_por_contrato"].round(2).astype(str))
    sel = st.selectbox("Escolha a estrutura", options=result["id"].tolist(), key=f"pick_{tk}")
    row = result[result["id"]==sel].iloc[0]

    Kc, Kp, credit = float(row["K_call"]), float(row["K_put"]), float(row["credit_total_por_contrato"])
    S_grid = np.linspace(max(0.01, Kp - 2*max(1.0, Kp*0.1)), Kc + 2*max(1.0, Kc*0.1), 400)
    payoff = -np.maximum(0.0, S_grid - Kc) - np.maximum(0.0, Kp - S_grid) + credit

    fig = plt.figure()
    plt.plot(S_grid, payoff)
    plt.axhline(0, linestyle="--")
    plt.axvline(Kp, linestyle=":")
    plt.axvline(Kc, linestyle=":")
    plt.title(f"{tk} — Payoff | Kp={Kp:.2f}, Kc={Kc:.2f}, Crédito≈R$ {credit:.2f}/ação")
    plt.xlabel("Preço do ativo no vencimento (S)")
    plt.ylabel("P/L por ação (R$)")
    st.pyplot(fig, use_container_width=True)

    with st.expander("🧭 Quando SAIR da operação? (didático)"):
        st.markdown(f"""
- **Perto do vencimento** (**≤ {days_exit_thresh} dias**) **e** preço **encostando em um strike** (±{int(prox_pct*100)}%):  
  → **Recomprar** a perna ameaçada (CALL se acima de K_call; PUT se abaixo de K_put).  
- **Capturou** cerca de **{int(capture_target*100)}%** do prêmio?  
  → Considere **encerrar** recomprando ambas as pernas.  
- **Rolagem**: se quiser manter a estratégia, **role** a perna pressionada para **próximo vencimento** e **strike mais OTM**.  
- **Disciplina**: defina **ordens limit** para recomprar por volta de **{(1-capture_target):.0%} do crédito** inicial (≈ R$ {(1-capture_target)*credit:.2f}/ação neste setup ilustrado).
""")

    return result

# ============================================================
# MAIN – Abas por ticker + parâmetros específicos por ticker
# ============================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("**Requisitos extras para coleta JS:** `requests-html`, `pyppeteer`, `bs4`, `lxml`, `lxml_html_clean`, `nest_asyncio`")

tabs = st.tabs(tickers)

all_results = []
for tk, tab in zip(tickers, tabs):
    with tab:
        st.subheader(f"Ativo: {tk}")
        colA, colB, colC = st.columns(3)
        with colA:
            shares_owned = st.number_input(f"Ações em carteira ({tk})", min_value=0, step=100, value=0, key=f"shares_{tk}")
        with colB:
            cash_available = st.number_input(f"Caixa disponível (R$) ({tk})", min_value=0.0, step=100.0, value=10000.0, format="%.2f", key=f"cash_{tk}")
        with colC:
            lot_size = st.number_input(f"Tamanho do contrato ({tk})", min_value=1, step=1, value=100, key=f"lot_{tk}")
        res = run_pipeline_for_ticker(tk, shares_owned, cash_available, lot_size)
        if isinstance(res, pd.DataFrame) and not res.empty:
            all_results.append(res)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    st.markdown("---")
    st.subheader("📦 Export combinado (todos os tickers)")
    st.download_button("⬇️ Exportar tudo (CSV)", data=combined.to_csv(index=False).encode("utf-8"),
                       file_name="strangles_sugeridos_all_v8.csv", mime="text/csv")

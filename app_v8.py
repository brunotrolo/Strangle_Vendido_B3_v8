
import re
import io
import math
from math import log, sqrt, exp
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strangle Vendido Coberto – v8a (Hotfix CSAN3)", layout="wide")
st.title("💼 Strangle Vendido Coberto – v8a (Hotfix de Coleta)")
st.caption("Inclui fallbacks: cabeçalhos HTTP, rota /opcoes2, colar HTML e subir CSV quando a grade não carrega.")

# ---------- Utils / Black–Scholes ----------
SQRT_2 = math.sqrt(2.0)
def norm_cdf(x: float) -> float: return 0.5*(1.0+math.erf(x/SQRT_2))
def yearfrac(start: date, end: date) -> float: return max(1e-9,(end-start).days/365.0)
def bs_d1(S,K,r,sigma,T):
    if S<=0 or K<=0 or sigma<=0 or T<=0: return np.nan
    return (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
def bs_d2(d1,sigma,T):
    if np.isnan(d1) or sigma<=0 or T<=0: return np.nan
    return d1 - sigma*math.sqrt(T)
def call_delta(S,K,r,sigma,T):
    d1=bs_d1(S,K,r,sigma,T); return np.nan if np.isnan(d1) else norm_cdf(d1)
def put_delta(S,K,r,sigma,T):
    d1=bs_d1(S,K,r,sigma,T); return np.nan if np.isnan(d1) else norm_cdf(d1)-1.0
def prob_ITM_call(S,K,r,sigma,T):
    d1=bs_d1(S,K,r,sigma,T); d2=bs_d2(d1,sigma,T); return norm_cdf(d2) if not np.isnan(d2) else np.nan
def prob_ITM_put(S,K,r,sigma,T):
    p_call=prob_ITM_call(S,K,r,sigma,T); return (1.0-p_call) if not np.isnan(p_call) else np.nan

# ---------- Normalização ----------
def try_read_csv_text(text: str) -> Optional[pd.DataFrame]:
    try: return pd.read_csv(io.StringIO(text))
    except Exception: return None

def normalize_chain_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap={'symbol':['symbol','símbolo','ticker','codigo','código','asset'],
            'expiration':['expiration','vencimento','expiração','expiracao','expiry','expiration_date'],
            'type':['type','tipo','opção','opcao','option_type'],
            'strike':['strike','preço de exercício','preco de exercicio','exercicio','k','strike_price'],
            'bid':['bid','compra','melhor compra','oferta de compra'],
            'ask':['ask','venda','melhor venda','oferta de venda'],
            'last':['last','último','ultimo','preço','preco','close','último negócio'],
            'impliedVol':['iv','ivol','impliedvol','implied_vol','vol implícita','vol implicita'],
            'delta':['delta','Δ']}
    rename={}; lowcols={c.lower().strip():c for c in df.columns}
    for target,aliases in colmap.items():
        for a in aliases:
            if a.lower() in lowcols: rename[lowcols[a.lower()]]=target; break
    df=df.rename(columns=rename)
    for c in ["symbol","expiration","type","strike","bid","ask"]:
        if c not in df.columns: df[c]=np.nan
    df["type"]=df["type"].astype(str).str.upper().str.strip().replace({'CALL':'C','C':'C','COMPRA':'C','CALLS':'C','PUT':'P','P':'P','VENDA':'P','PUTS':'P'})
    df["expiration"]=pd.to_datetime(df["expiration"],errors="coerce").dt.date
    for c in ["strike","bid","ask","last","impliedVol","delta"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["expiration","type","strike"], how="any")

def base_ticker_for_optionsnet(ticker: str) -> str:
    txt=(ticker or "").strip().upper(); txt=re.sub(r"\.SA$","",txt); txt=re.sub(r"[^A-Z0-9]","",txt); return txt

# ---------- Coleta com fallbacks ----------
UA = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"}

@st.cache_data(show_spinner=False)
def fetch_optionsnet_any(ticker: str) -> pd.DataFrame:
    tk = base_ticker_for_optionsnet(ticker)
    urls = [
        f"https://opcoes.net.br/opcoes/bovespa/{tk}",
        f"https://opcoes.net.br/opcoes2/bovespa?ativo={tk}",
    ]
    sess = requests.Session()
    for url in urls:
        try:
            r = sess.get(url, headers=UA, timeout=25)
            r.raise_for_status()
            # tenta CSV direto
            df = try_read_csv_text(r.text)
            if df is not None and len(df)>0:
                return normalize_chain_columns(df)
            # tenta HTML
            tbls = pd.read_html(r.text)
            if tbls:
                big = max(tbls, key=lambda t: t.shape[0]*t.shape[1])
                nd = normalize_chain_columns(big)
                if not nd.empty:
                    return nd
        except Exception:
            continue
    return pd.DataFrame()

# ---------- Yahoo helpers ----------
@st.cache_data(show_spinner=False)
def load_spot_and_iv_proxy(yahoo_ticker: str):
    try:
        y=yf.Ticker(yahoo_ticker); hist=y.history(period="2y")
        if len(hist)>=30:
            spot=float(hist["Close"].iloc[-1])
            rets=hist["Close"].pct_change(); daily=rets.rolling(20).std()
            hv20 = daily*np.sqrt(252)
            return (spot, float(hv20.dropna().iloc[-1]), pd.DataFrame({"date":hist.index.date,"iv":hv20.values}).dropna())
    except Exception: pass
    return (np.nan, np.nan, None)

def compute_iv_rank_percentile(series_df: Optional[pd.DataFrame], lookback: int) -> Tuple[float,float,float]:
    if series_df is None or series_df.empty: return (np.nan,np.nan,np.nan)
    s = series_df.sort_values("date").tail(lookback)["iv"].dropna()
    if len(s)<5: return (np.nan,np.nan,np.nan)
    iv_now=float(s.iloc[-1]); iv_min=float(s.min()); iv_max=float(s.max())
    iv_rank=(iv_now-iv_min)/max(1e-9,(iv_max-iv_min)); iv_pct=float((s<=iv_now).mean())
    return (iv_now, iv_rank, iv_pct)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Parâmetros gerais")
    r = st.number_input("Taxa livre de risco anual (r)", min_value=0.0, max_value=1.0, step=0.005, value=0.11, format="%.3f")
    st.markdown("---")
    ticker = st.text_input("Ticker (ex.: PETR4.SA ou CSAN3)", value="CSAN3")
    lot = st.number_input("Tamanho do contrato", min_value=1, value=100, step=1)
    shares = st.number_input("Ações em carteira", min_value=0, step=100, value=0)
    cash = st.number_input("Caixa disponível (R$)", min_value=0.0, step=100.0, value=10000.0, format="%.2f")
    st.markdown("---")
    st.subheader("Fallbacks quando a coleta falha")
    paste_html = st.text_area("Cole aqui o HTML da página do options.net (após clicar 'Continuar com dados de fechamento')", height=120)
    upload_csv = st.file_uploader("...ou envie um CSV/HTML exportado da grade", type=["csv","htm","html"])
    st.caption("Se a coleta automática falhar para um ativo (ex.: CSAN3), use um destes fallbacks.")

# ---------- Pipeline simples para demonstrar a coleta com fallback ----------
st.subheader(f"Teste de coleta para: {ticker}")
spot, hv20, iv_series = load_spot_and_iv_proxy(ticker if ticker.endswith(".SA") else f"{ticker}.SA")
st.write(f"Spot: **{spot:.2f}** | HV20: **{hv20:.2%}**")

df = fetch_optionsnet_any(ticker)
source = "Coleta automática (options.net)"
if df.empty and paste_html.strip():
    try:
        tables = pd.read_html(paste_html)
        if tables:
            df = normalize_chain_columns(max(tables, key=lambda t: t.shape[0]*t.shape[1]))
            source = "HTML colado manualmente"
    except Exception as e:
        st.error(f"Falha ao ler HTML colado: {e}")

if df.empty and upload_csv is not None:
    try:
        if upload_csv.name.endswith(".csv"):
            df = pd.read_csv(upload_csv)
        else:
            tables = pd.read_html(upload_csv)
            if tables:
                df = max(tables, key=lambda t: t.shape[0]*t.shape[1])
        df = normalize_chain_columns(df)
        source = f"Arquivo enviado: {upload_csv.name}"
    except Exception as e:
        st.error(f"Falha ao processar arquivo: {e}")

if df.empty:
    st.error("Ainda sem dados. Tente colar o HTML da página ou enviar um CSV.")
    st.stop()

st.success(f"Fonte: {source} — linhas: {len(df)}")
st.dataframe(df.head(20), use_container_width=True)

# A partir daqui você pode plugar o restante do pipeline v8 (filtros, bandas, ranking, payoff...)

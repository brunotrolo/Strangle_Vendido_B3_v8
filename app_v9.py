# app_v9.py
# Streamlit app – Strangle Vendido Coberto (v9)
# Requisitos: streamlit, pandas, numpy, requests, yfinance, python-dateutil

import re
import io
import math
import time
import json
import html
import warnings
from datetime import datetime, date
from dateutil import parser as dtparser

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Config & helpers
# ---------------------------

st.set_page_config(
    page_title="Strangle Vendido Coberto — v9",
    page_icon="💼",
    layout="wide",
)

CONTRACT_SIZE = 100

def format_brl(x):
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"

def brl_to_float(s):
    """Converte strings '1.234,56' -> 1234.56; números retornam direto."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).strip()
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except:
        return np.nan

def ensure_float_series(x):
    """Converte Series/array/escalares para float numpy-friendly."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    return pd.Series(x, dtype="float64")

# ---------------------------
# Black–Scholes (probabilidades)
# ---------------------------

SQRT_2 = math.sqrt(2.0)

def norm_cdf(x):
    # erfc para estabilidade; mas erf é suficiente aqui
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def prob_ITM_call(S, K, r, sigma, T):
    """Probabilidade de CALL terminar ITM (S_T > K) sob BS (risk-neutral)."""
    if any(map(lambda v: (v is None) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))), [S, K, r, sigma, T])):
        return np.nan
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return np.nan
    try:
        d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        return 1.0 - norm_cdf(d2)
    except Exception:
        return np.nan

def prob_ITM_put(S, K, r, sigma, T):
    """Probabilidade de PUT terminar ITM (S_T < K)."""
    p_call = prob_ITM_call(S, K, r, sigma, T)
    if np.isnan(p_call):
        return np.nan
    return 1.0 - p_call

# ---------------------------
# Tickers (dadosdemercado.com.br)
# ---------------------------

@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 min
def fetch_b3_tickers():
    url = "https://www.dadosdemercado.com.br/acoes"
    try:
        html_text = requests.get(url, timeout=15).text
        # A página tem uma tabela com colunas "Código" e "Empresa"
        dfs = pd.read_html(html_text)
        # Procura a tabela que contenha essas duas colunas
        idx = None
        for i, df in enumerate(dfs):
            cols_lower = [str(c).strip().lower() for c in df.columns]
            if any("código" in c or "codigo" in c for c in cols_lower) and any("empresa" in c for c in cols_lower):
                idx = i
                break
        if idx is None:
            raise RuntimeError("Tabela de tickers não encontrada.")

        df = dfs[idx].copy()
        # Normaliza nomes
        cols_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if "código" in cl or "codigo" in cl:
                cols_map[c] = "ticker"
            elif "empresa" in cl:
                cols_map[c] = "empresa"
            else:
                cols_map[c] = c
        df = df.rename(columns=cols_map)
        df = df[["ticker", "empresa"]].dropna()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["empresa"] = df["empresa"].astype(str).str.strip()
        # Remove tickers não padrão (ex: UNITs ou preferir .)
        df = df[~df["ticker"].str.contains("FII|ETF|BDR", case=False, na=False)]
        df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        return df
    except Exception:
        # Fallback mínimo para não quebrar o app
        data = [
            {"ticker": "PETR4", "empresa": "Petrobras PN"},
            {"ticker": "VALE3", "empresa": "Vale ON"},
            {"ticker": "BBDC4", "empresa": "Bradesco PN"},
            {"ticker": "ITUB4", "empresa": "Itaú Unibanco PN"},
            {"ticker": "ABEV3", "empresa": "Ambev ON"},
        ]
        return pd.DataFrame(data)

def build_ticker_options(df_ticks):
    return [f"{row.ticker} — {row.empresa}" for _, row in df_ticks.iterrows()]

def parse_sel_label(label):
    # "BBDC4 — Banco Bradesco"
    if "—" in label:
        return label.split("—", 1)[0].strip().upper()
    return label.strip().upper()

# ---------------------------
# yfinance spot (sempre automático)
# ---------------------------

@st.cache_data(ttl=60*5, show_spinner=False)  # 5 min
def fetch_spot_yf(b3_ticker):
    # regra geral: .SA
    yf_ticker = b3_ticker.upper().strip() + ".SA"
    try:
        t = yf.Ticker(yf_ticker)
        # history tem sido mais confiável do que info em várias situações
        hist = t.history(period="5d")
        if not hist.empty:
            last = float(hist["Close"].dropna().iloc[-1])
            return last
    except Exception:
        pass
    # fallback rápido
    try:
        t = yf.Ticker(yf_ticker)
        info = t.fast_info  # mais leve
        last = float(info["last_price"])
        if last and last > 0:
            return last
    except Exception:
        pass
    return np.nan

# ---------------------------
# Parser da option chain colada
# ---------------------------

def try_parse_date(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    # aceita 17/10/2025 ou 2025-10-17
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            continue
    # tentativa mais flexível
    try:
        return dtparser.parse(s, dayfirst=True).date()
    except:
        return None

def normalize_pasted_table(text):
    """
    Retorna df padronizado com colunas:
    symbol, type('C'/'P'), strike, last, impliedVol, delta, expiration (YYYY-MM-DD)
    """
    txt = text.strip()
    # Tenta TSV (quando copia do site costuma vir tabulado)
    sep = "\t" if "\t" in txt else None
    df_raw = pd.read_csv(io.StringIO(txt), sep=sep, engine="python")
    # Troca vírgula por ponto em colunas numéricas chave
    # Mapeia nomes comuns do opcoes.net
    cols = {c: c for c in df_raw.columns}
    def find_col(targets):
        for c in df_raw.columns:
            cl = str(c).strip().lower()
            for t in targets:
                if t in cl:
                    return c
        return None

    c_symbol = find_col(["ticker", "símbolo", "simbolo"])
    c_type   = find_col(["tipo"])
    c_strike = find_col(["strike"])
    c_last   = find_col(["último", "ultimo", "last"])
    c_iv     = find_col(["vol. impl", "vol impl", "volatilidade", "impl"])
    c_delta  = find_col(["delta"])
    c_exp    = find_col(["vencimento", "expira", "expire", "expiration"])

    sel = {}
    if c_symbol: sel["symbol"] = df_raw[c_symbol].astype(str).str.strip()
    if c_type:   sel["type"]   = df_raw[c_type].astype(str).str.upper().str.strip().map(lambda x: "C" if "C" in x else ("P" if "P" in x else ""))
    if c_strike: sel["strike"] = df_raw[c_strike].map(brl_to_float)
    if c_last:   sel["last"]   = df_raw[c_last].map(brl_to_float)
    if c_iv:     sel["impliedVol"] = pd.to_numeric(df_raw[c_iv].astype(str).str.replace("%","").str.replace(",","."), errors="coerce")/100.0
    if c_delta:  sel["delta"]  = pd.to_numeric(df_raw[c_delta].astype(str).str.replace(",", "."), errors="coerce")
    if c_exp:    sel["expiration"] = df_raw[c_exp].map(try_parse_date).map(lambda d: d.isoformat() if d else None)

    df = pd.DataFrame(sel)
    # Limpeza
    if "type" in df.columns:
        df = df[df["type"].isin(["C","P"])]
    if "strike" in df.columns:
        df = df[~df["strike"].isna()]
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# Core: gera strangles
# ---------------------------

def build_strangles(df_opts, spot, r_annual, hv20_annual, sel_exp_iso):
    # Filtra pelo vencimento
    if "expiration" not in df_opts.columns:
        return pd.DataFrame()
    dfe = df_opts[df_opts["expiration"] == sel_exp_iso].copy()
    if dfe.empty:
        return pd.DataFrame()

    # Preço por ação: usa 'last'; se tiver 'mid' em alguma versão futura, daria prioridade
    dfe["price"] = pd.to_numeric(dfe.get("last", np.nan), errors="coerce")

    # Determina OTM
    calls = dfe[dfe["type"]=="C"].copy()
    puts  = dfe[dfe["type"]=="P"].copy()
    calls["OTM"] = calls["strike"] > spot
    puts["OTM"]  = puts["strike"]  < spot
    calls = calls[calls["OTM"] & ~calls["price"].isna()]
    puts  = puts[ puts["OTM"]  & ~puts["price"].isna()]

    if calls.empty or puts.empty:
        return pd.DataFrame()

    # Parâmetros anuais → para T em anos
    # T = dias úteis aproximados pela diferença de datas no df? Preferimos 252b?
    # Aqui assumimos diferença de calendário: usa a data ISO e hoje.
    try:
        expiry = datetime.fromisoformat(sel_exp_iso).date()
        today  = date.today()
        days   = max((expiry - today).days, 1)
    except Exception:
        days = 30
    T = days / 365.0

    r = max(float(r_annual), 0.0)
    sigma_hv = max(float(hv20_annual), 0.0001)

    # monta todas combinações PUT x CALL (OTM)
    puts = puts.sort_values("strike")
    calls = calls.sort_values("strike")

    rows = []
    for _, rp in puts.iterrows():
        for _, rc in calls.iterrows():
            Kp = float(rp["strike"])
            Kc = float(rc["strike"])
            prem_put  = float(rp["price"])
            prem_call = float(rc["price"])
            cred = prem_put + prem_call

            # IVs por perna (se houver), senão HV20
            iv_p = float(rp.get("impliedVol", np.nan))
            iv_c = float(rc.get("impliedVol", np.nan))
            sp = iv_p if not np.isnan(iv_p) and iv_p>0 else sigma_hv
            sc = iv_c if not np.isnan(iv_c) and iv_c>0 else sigma_hv

            poe_put  = prob_ITM_put(spot, Kp, r, sp, T)   # prob de exercício da PUT
            poe_call = prob_ITM_call(spot, Kc, r, sc, T)  # prob de exercício da CALL

            be_low  = Kp - cred
            be_high = Kc + cred

            # heurística simples prêmio/risco
            risk_proxy = max((poe_put or 0) + (poe_call or 0), 1e-6)
            score = cred / risk_proxy

            rows.append({
                "symbol_put": str(rp.get("symbol","")),
                "symbol_call": str(rc.get("symbol","")),
                "Kp": Kp,
                "Kc": Kc,
                "premio_put": prem_put,
                "premio_call": prem_call,
                "credito": cred,
                "be_low": be_low,
                "be_high": be_high,
                "poe_put": poe_put,
                "poe_call": poe_call,
                "score": score
            })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    # Ordena por melhor score, depois maior crédito
    df_out = df_out.sort_values(["score","credito"], ascending=[False, False]).reset_index(drop=True)
    return df_out

# ---------------------------
# UI
# ---------------------------

# CSS leve para legibilidade
st.markdown("""
<style>
/* títulos mais limpos */
h1, h2, h3 { line-height: 1.2; }
.small-muted {
  opacity: 0.8;
  font-size: 0.9rem;
}
.info-chip {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 8px;
  background: rgba(125, 125, 125, 0.12);
  margin-right: 8px;
  margin-bottom: 6px;
}
.key {
  font-weight: 600;
}
.val {
  font-variant-numeric: tabular-nums;
}
/* evita caixas com fundo branco puro */
.block-highlight {
  background: rgba(0,0,0,0.04);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  padding: 10px 12px;
}
.bigline { font-size: 1.05rem; }
.kpi { font-size: 1.15rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("💼 Strangle Vendido Coberto — v9")
st.caption("Cole a option chain do opcoes.net, escolha o vencimento e veja as sugestões didáticas de strangle coberto.")

# ---------- Sidebar (parâmetros) ----------
with st.sidebar:
    st.subheader("Parâmetros de risco")
    hv20 = st.number_input("HV20 (σ anual – proxy, %)", min_value=1.0, max_value=200.0, value=20.0, step=0.1) / 100.0
    rate = st.number_input("r (anual, %)", min_value=0.0, max_value=50.0, value=11.0, step=0.1) / 100.0

    st.subheader("Lotes")
    lots = st.number_input("Quantidade de strangles (lotes)", min_value=0, max_value=10_000, value=10, step=1)

# ---------- Linha seletores ----------
tickers_df = fetch_b3_tickers()
opts_list = build_ticker_options(tickers_df)

colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown("### Selecione pelo nome da empresa ou ticker")
    sel_label = st.selectbox("", opts_list, index=0, label_visibility="collapsed")
    sel = parse_sel_label(sel_label)

with colB:
    spot = fetch_spot_yf(sel)
    st.markdown("### Strike (preço à vista via yfinance)")
    st.markdown(f"<div class='block-highlight kpi'>R$ {spot:,.2f}</div>".replace(",", "X").replace(".", ",").replace("X","."), unsafe_allow_html=True)

# ---------- Pega option chain colada ----------
st.markdown("### 1) Cole abaixo a option chain do opcoes.net")
pasted = st.text_area(
    "Cole (Ctrl/Cmd+V) a tabela inteira copiada do site.",
    height=180,
    label_visibility="collapsed",
    placeholder="Cole aqui a tabela copiada do opcoes.net…",
)

df_norm = pd.DataFrame()
expiries = []
sel_exp = None

if pasted.strip():
    try:
        df_norm = normalize_pasted_table(pasted)
        if not df_norm.empty:
            expiries = sorted([d for d in df_norm["expiration"].dropna().unique()])
            if expiries:
                st.markdown("### 2) Escolha o vencimento")
                sel_exp = st.selectbox("", expiries, index=0, label_visibility="collapsed")
    except Exception as e:
        st.error("Não consegui entender a tabela colada. Confira se copiou as colunas do site (incluindo 'Vencimento', 'Tipo', 'Strike', 'Último', 'Vol. Impl. (%)', 'Delta').")

# ---------- Calcula sugestões ----------
df_sug = pd.DataFrame()
if not df_norm.empty and sel_exp and (spot is not None) and not np.isnan(spot) and spot>0:
    df_sug = build_strangles(df_norm, spot, rate, hv20, sel_exp)

# ---------- Top 3 em tabela ----------
st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")

if df_sug.empty:
    st.info("Sem combinações OTM viáveis para este vencimento (ou faltam preços/IV). Tente outro vencimento ou cole uma cadeia mais completa.")
else:
    top = df_sug.head(3).copy()

    # colunas legíveis / formatadas
    top_display = pd.DataFrame({
        "PUT": top["symbol_put"].fillna("").values,
        "CALL": top["symbol_call"].fillna("").values,
        "Strike PUT (R$)": top["Kp"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")),
        "Strike CALL (R$)": top["Kc"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")),
        "Prêmio PUT (R$)": top["premio_put"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")),
        "Prêmio CALL (R$)": top["premio_call"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")),
        "Crédito/ação (R$)": top["credito"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")),
        "Break-evens": top.apply(lambda r: f"[{r['be_low']:.2f}, {r['be_high']:.2f}]".replace(".",","), axis=1),
        "PoE PUT (%)": (100*top["poe_put"]).map(lambda x: f"{x:.1f}".replace(".",",")),
        "PoE CALL (%)": (100*top["poe_call"]).map(lambda x: f"{x:.1f}".replace(".",",")),
    })

    st.dataframe(
        top_display,
        use_container_width=True,
        hide_index=True
    )

    # KPI de prêmio total com base na 1ª sugestão e lotes
    rw = top.iloc[0]
    credito_acao = float(rw["credito"])
    premio_total = credito_acao * CONTRACT_SIZE * lots

    col_k1, col_k2, col_k3 = st.columns([1,1,1])
    with col_k1:
        st.markdown("**Contrato (ações por lote)**")
        st.markdown(f"<div class='block-highlight bigline'>{CONTRACT_SIZE}</div>", unsafe_allow_html=True)
    with col_k2:
        st.markdown("**Lotes**")
        st.markdown(f"<div class='block-highlight bigline'>{lots}</div>", unsafe_allow_html=True)
    with col_k3:
        st.markdown("**🎯 Prêmio estimado (R$)**")
        st.markdown(f"<div class='block-highlight bigline'>{format_brl(premio_total)}</div>", unsafe_allow_html=True)

    # Explicações contextualizadas (com base na 1ª linha)
    dias_alerta = 7
    st.markdown("### 📘 O que significa cada item?")
    st.markdown(
        f"""
**Crédito/ação**  
Soma dos prêmios recebidos ao vender **1 PUT** e **1 CALL** (por **ação**).  
*Exemplo deste cenário:* a **PUT** paga **{format_brl(rw['premio_put'])}** e a **CALL** paga **{format_brl(rw['premio_call'])}**, somando **{format_brl(credito_acao)}** por ação.  

**Break-evens (mín–máx)**  
Faixa de preço no vencimento onde o resultado ainda é ≥ 0.  
*Exemplo:* se o ativo oscilar entre **{format_brl(rw['be_low'])} e {format_brl(rw['be_high'])}**, você encerra a operação no zero a zero ou com lucro.  

**Probabilidade de exercício (PUT / CALL)**  
Chance estimada (modelo Black–Scholes) de cada opção terminar **dentro do dinheiro** no vencimento.  
*Exemplos:*  
• **PUT {100*rw['poe_put']:.1f}%** → ~{100*rw['poe_put']:.1f}% de chance de o preço cair abaixo do **Strike da PUT ({format_brl(rw['Kp'])})**.  
• **CALL {100*rw['poe_call']:.1f}%** → ~{100*rw['poe_call']:.1f}% de chance de subir acima do **Strike da CALL ({format_brl(rw['Kc'])})**.  

**Lotes e prêmio total**  
Cada **lote** = vender **1 PUT + 1 CALL** (contrato = {CONTRACT_SIZE} ações).  
Prêmio total = **crédito/ação × contrato × lotes**.  
*Exemplo:* {format_brl(credito_acao)} × {CONTRACT_SIZE} × {lots} = **{format_brl(premio_total)}**.  

**Regras práticas de saída**  
⏳ faltam **≤ {dias_alerta} dias** → acompanhe com mais atenção.  
📈 se **S** encostar no **Strike da CALL ({format_brl(rw['Kc'])})**, recompre a CALL.  
🎯 tente capturar **~75%** do crédito e encerrar.
""".strip()
    )

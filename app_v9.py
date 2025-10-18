# app_v9.py
# -*- coding: utf-8 -*-
import io
import re
import math
import json
import time
import string
import warnings
from datetime import datetime, date
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

# =============== Config Streamlit ===============
st.set_page_config(
    page_title="Strangle Vendido Coberto — v9",
    page_icon="💼",
    layout="wide"
)
warnings.filterwarnings("ignore")

# =============== Utilidades numéricas ===============
SQRT_2 = math.sqrt(2.0)

def norm_cdf(x: float) -> float:
    # Usar math.erf (evita dependência de scipy)
    try:
        return 0.5 * (1.0 + math.erf(x / SQRT_2))
    except Exception:
        # fallback tosco
        return float(pd.Series([x]).apply(lambda y: 0.5 * (1 + math.erf(y / SQRT_2)))[0])

def d1_d2(S, K, r, sigma, T):
    """ Black-Scholes d1 e d2 (sem dividendos) """
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return (np.nan, np.nan)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return (d1, d2)
    except Exception:
        return (np.nan, np.nan)

def prob_ITM_call(S, K, r, sigma, T):
    """Probabilidade de CALL terminar ITM sob RN."""
    _, d2 = d1_d2(S, K, r, sigma, T)
    return norm_cdf(d2) if not np.isnan(d2) else np.nan

def prob_ITM_put(S, K, r, sigma, T):
    """Probabilidade de PUT terminar ITM sob RN."""
    # Put ITM = 1 - P(Call ITM) quando strikes simétricos em RN? Melhor computar direto:
    # P(S_T < K) = N(-d2)
    _, d2 = d1_d2(S, K, r, sigma, T)
    return norm_cdf(-d2) if not np.isnan(d2) else np.nan

def _br_to_float(x):
    """Converte string pt-BR para float (1.234,56 -> 1234.56)."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # remove espaços e NBSP
    s = s.replace("\xa0", " ").replace(" ", "")
    # normaliza sinal de porcentagem
    s = s.replace("%", "")
    # troca separadores
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def to_date(s):
    """Converte 17/10/2025 ou 2025-10-17 para datetime.date."""
    if pd.isna(s):
        return None
    if isinstance(s, (datetime, date)):
        return s.date() if isinstance(s, datetime) else s
    ss = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(ss, fmt).date()
        except Exception:
            pass
    return None

def _abs_delta_band(delta_abs, bands):
    """Retorna rótulo da banda a partir de |Δ|."""
    if pd.isna(delta_abs):
        return "—"
    d = float(delta_abs)
    low_max, mid_max = bands  # (0→low_max], (low_max→mid_max], (>mid_max→alto)
    if d <= low_max:
        return "Baixo"
    elif d <= mid_max:
        return "Médio"
    else:
        return "Alto"

# =============== Dados da B3 (tickers) ===============
B3_FALLBACK = [
    ("PETR4", "Petrobras PN"), ("VALE3", "Vale ON"), ("BBDC4", "Bradesco PN"),
    ("ITUB4", "Itaú Unibanco PN"), ("ABEV3", "Ambev ON"), ("BBAS3", "Banco do Brasil ON"),
    ("WEGE3", "WEG ON"), ("SUZB3", "Suzano ON"), ("JBSS3", "JBS ON"),
    ("ELET3", "Eletrobras ON"), ("ELET6", "Eletrobras PN"), ("PRIO3", "PRIO ON"),
]

@lru_cache(maxsize=1)
def fetch_b3_universe() -> pd.DataFrame:
    """
    Tenta baixar a lista de ações listadas na B3.
    Se falhar, devolve fallback local.
    """
    try:
        # Fonte oficial (pode variar ao longo do tempo). Mantemos duas tentativas.
        urls = [
            "https://sistemaswebb3-listados.b3.com.br/listedCompaniesPage/listedCompaniesPage.do",
            "https://brasilbolsa.github.io/arq/listed/b3_tickers.csv",
        ]
        df_acc = []
        for u in urls:
            try:
                if u.endswith(".csv"):
                    df = pd.read_csv(u, sep=",", encoding="utf-8")
                else:
                    # endpoint HTML/json da B3 costuma precisar de POST; usar fallback se falhar
                    raise Exception("Preferindo fallback para B3.")
                df_acc.append(df)
            except Exception:
                continue
        if df_acc:
            df0 = df_acc[0]
            # tentar achar colunas por heurística
            cols = [c.lower() for c in df0.columns]
            if "ticker" in cols or "symbol" in cols:
                if "ticker" in cols:
                    tcol = df0.columns[cols.index("ticker")]
                else:
                    tcol = df0.columns[cols.index("symbol")]
                # nome
                name_col = None
                for cand in ["name", "company", "empresa", "companyname"]:
                    if cand in cols:
                        name_col = df0.columns[cols.index(cand)]
                        break
                if name_col is None:
                    name_col = tcol
                out = (df0[[tcol, name_col]]
                       .dropna()
                       .rename(columns={tcol: "ticker", name_col: "name"}))
                out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
                out["name"] = out["name"].astype(str).str.strip()
                out = out.drop_duplicates(subset=["ticker"])
                out["label"] = out["ticker"] + " — " + out["name"]
                out = out[ out["ticker"].str.match(r"^[A-Z]{4}\d$") ]  # form. mais comum
                if len(out) >= 50:
                    return out.sort_values("ticker").reset_index(drop=True)
        # fallback local
        fb = pd.DataFrame(B3_FALLBACK, columns=["ticker", "name"])
        fb["label"] = fb["ticker"] + " — " + fb["name"]
        return fb
    except Exception:
        fb = pd.DataFrame(B3_FALLBACK, columns=["ticker", "name"])
        fb["label"] = fb["ticker"] + " — " + fb["name"]
        return fb

# =============== Cotação do Yahoo Finance ===============
def _yf_symbol_b3(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not t.endswith(".SA"):
        t = t + ".SA"
    return t

@lru_cache(maxsize=1024)
def yahoo_price_b3(ticker: str) -> float:
    """ Último preço (regularMarketPrice) via yfinance """
    try:
        import yfinance as yf
    except Exception:
        st.warning("Não encontrei yfinance no ambiente. Instale com: pip install yfinance")
        return np.nan
    try:
        ysym = _yf_symbol_b3(ticker)
        tk = yf.Ticker(ysym)
        inf = tk.fast_info if hasattr(tk, "fast_info") else {}
        px = None
        if inf:
            px = inf.get("last_price") or inf.get("lastPrice") or inf.get("last_trade")
        if px is None:
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                px = float(hist["Close"].iloc[-1])
        return float(px) if px is not None else np.nan
    except Exception:
        return np.nan

# =============== Parsing da option chain colada ===============
EXPECTED_PT_HEADERS = {
    "Ticker": "symbol",
    "Vencimento": "expiration",
    "Tipo": "type",
    "Strike": "strike",
    "Último": "last",
    "Vol. Impl. (%)": "impliedVol",
    "Delta": "delta",
}

def _clean_cols(cols):
    out = []
    for c in cols:
        cc = str(c).strip()
        cc = cc.replace("\xa0", " ")
        cc = re.sub(r"\s+", " ", cc)
        out.append(cc)
    return out

def parse_pasted_table(raw_text: str) -> pd.DataFrame:
    """
    Recebe a tabela colada do opcoes.net e devolve DF padronizado com:
    symbol, type (CALL/PUT), strike, last, mid(=last se não houver), impliedVol (decimal), delta, expiration (date)
    """
    if not raw_text or str(raw_text).strip() == "":
        return pd.DataFrame()

    # Tenta detectar separador: tab ou múltiplos espaços
    txt = raw_text.strip("\n")
    # normaliza cabeçalho quebrado por tabs
    if "\t" in txt:
        df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str, engine="python")
    else:
        # fallback: separador por múltiplos espaços
        df = pd.read_csv(io.StringIO(txt), sep=r"\s{2,}", dtype=str, engine="python")

    if df.empty:
        return pd.DataFrame()

    df.columns = _clean_cols(df.columns)
    # mapeia colunas esperadas
    colmap = {}
    for k_pt, k_std in EXPECTED_PT_HEADERS.items():
        # procura pelo nome exato primeiro
        if k_pt in df.columns:
            colmap[k_pt] = k_std
        else:
            # heurística: contém?
            for c in df.columns:
                if k_pt.lower() in c.lower():
                    colmap[c] = k_std
                    break
    slim = df[list(colmap.keys())].rename(columns=colmap)

    # normalizações
    if "type" in slim.columns:
        slim["type"] = slim["type"].astype(str).str.upper().str.strip()
        slim["type"] = slim["type"].replace({"CALL": "C", "PUT": "P"})
    if "strike" in slim.columns:
        slim["strike"] = slim["strike"].apply(_br_to_float)
    if "last" in slim.columns:
        slim["last"] = slim["last"].apply(_br_to_float)
    # implied vol para decimal anual (ex.: "27,5" -> 0.275)
    if "impliedVol" in slim.columns:
        slim["impliedVol"] = slim["impliedVol"].apply(_br_to_float) / 100.0
    if "delta" in slim.columns:
        slim["delta"] = slim["delta"].apply(_br_to_float)
        # alguns sites trazem delta de put negativo; manter sinal
    if "expiration" in slim.columns:
        slim["expiration"] = slim["expiration"].apply(to_date)

    # colunas faltantes
    if "symbol" not in slim.columns and "Ticker" in df.columns:
        slim["symbol"] = df["Ticker"].astype(str)
    if "last" not in slim.columns:
        slim["last"] = np.nan
    slim["mid"] = slim["last"]  # como fallback

    # drop linhas sem strike/expiration/tipo
    req = ["type", "strike", "expiration"]
    for r in req:
        if r not in slim.columns:
            slim[r] = np.nan
    slim = slim.dropna(subset=["type", "strike", "expiration"])
    # mantém apenas CALL/PUT válidos
    slim = slim[slim["type"].isin(["C", "P"])]

    return slim.reset_index(drop=True)

# =============== Métricas e pairing ===============
def days_to_exp(d: date) -> int:
    return (d - date.today()).days

def T_years(d: date) -> float:
    return max(days_to_exp(d), 0) / 252.0

def make_pairs(df_opts: pd.DataFrame, spot: float, r: float, sigma: float,
               dmin: float, dmax: float) -> pd.DataFrame:
    """
    Gera combinações PUT (OTM) × CALL (OTM) para o mesmo vencimento.
    Retorna DF com colunas: expiration, Kp, Kc, prem_put, prem_call, credit, breakeven_low, breakeven_high, poe_put, poe_call, band_put, band_call
    """
    if df_opts.empty or np.isnan(spot):
        return pd.DataFrame()

    # filtra apenas vencimento válido
    expirations = sorted(df_opts["expiration"].dropna().unique())
    out_rows = []

    for exp in expirations:
        T = T_years(exp)
        if T <= 0:
            continue
        subset = df_opts[df_opts["expiration"] == exp].copy()

        # OTM por lado
        calls = subset[(subset["type"] == "C") & (subset["strike"] > spot)].copy()
        puts  = subset[(subset["type"] == "P") & (subset["strike"] < spot)].copy()

        # filtra por |Δ|
        for side_df in (calls, puts):
            if "delta" in side_df.columns:
                side_df["abs_delta"] = side_df["delta"].abs()
                side_df = side_df[(side_df["abs_delta"] >= dmin) & (side_df["abs_delta"] <= dmax)]
            else:
                side_df["abs_delta"] = np.nan
            # premium: usar 'last'; se faltar, ignora
            side_df["prem"] = side_df["last"].astype(float)
            # limpa NaNs
            side_df = side_df.dropna(subset=["prem", "strike"])
            # sobrescreve nas referências
            if len(side_df) == 0:
                if side_df is calls:
                    calls = side_df
                else:
                    puts = side_df
            else:
                if side_df is calls:
                    calls = side_df
                else:
                    puts = side_df

        if calls.empty or puts.empty:
            continue

        # combinações
        for _, rc in calls.iterrows():
            for _, rp in puts.iterrows():
                Kc = float(rc["strike"]); Kp = float(rp["strike"])
                c_prem = float(rc["prem"]); p_prem = float(rp["prem"])
                credit = (c_prem + p_prem)
                be_low  = Kp - credit
                be_high = Kc + credit

                poe_c = prob_ITM_call(spot, Kc, r, sigma, T)
                poe_p = prob_ITM_put(spot, Kp, r, sigma, T)

                # bandas por |Δ|
                bd_c = _abs_delta_band(rc.get("abs_delta", np.nan), bands=get_bands_tuple())
                bd_p = _abs_delta_band(rp.get("abs_delta", np.nan), bands=get_bands_tuple())

                out_rows.append({
                    "expiration": exp,
                    "Kp": Kp, "Kc": Kc,
                    "prem_put": p_prem, "prem_call": c_prem,
                    "credit": credit,
                    "be_low": be_low, "be_high": be_high,
                    "poe_put": poe_p, "poe_call": poe_c,
                    "band_put": bd_p, "band_call": bd_c,
                    "sym_put": rp.get("symbol", ""),
                    "sym_call": rc.get("symbol", ""),
                    "delta_put": rp.get("delta", np.nan),
                    "delta_call": rc.get("delta", np.nan),
                })

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    # ranking simples: maior crédito / largura, penaliza prob. sair fora (poe_put+poe_call)
    width = (out["Kc"] - out["Kp"]).clip(lower=1e-6)
    score = (out["credit"] / width) * (1 - 0.5 * (out["poe_put"].fillna(0) + out["poe_call"].fillna(0)))
    out["score"] = score
    out = out.sort_values(["expiration", "score"], ascending=[True, False]).reset_index(drop=True)
    return out

# =============== Estado das "bandas" nas configurações (sidebar) ===============
def get_bands_tuple():
    low_max = st.session_state.get("band_low_max", 0.15)
    mid_max = st.session_state.get("band_mid_max", 0.35)
    return (low_max, mid_max)

# =============== UI ===============
st.title("💼 Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)")
st.write("Cole a option chain do **opcoes.net**, escolha o vencimento e veja as **sugestões didáticas** de strangle coberto e a **comparação de estratégias**.")

# ---------- Barra lateral: parâmetros e ajuda ----------
with st.sidebar:
    st.markdown("### ⚙️ Parâmetros")
    # r e HV20 foram movidos para a barra lateral
    r = st.number_input(
        "Taxa livre de risco anual (r)",
        min_value=0.00, max_value=1.00, value=0.11, step=0.01,
        format="%.2f",
        help="Usada no Black–Scholes. No Brasil, aproxime pela SELIC anualizada. Ex.: 0,11 = 11% a.a."
    )
    hv20 = st.number_input(
        "HV20 (σ anual – proxy)",
        min_value=0.00, max_value=1.00, value=0.20, step=0.01,
        format="%.2f",
        help="Volatilidade histórica de 20 dias anualizada (proxy de IV caso falte). Ex.: 0,20 = 20% a.a."
    )

    st.divider()
    st.markdown("#### 🎯 Filtros por |Δ|")
    dmin = st.slider("|Δ| mínimo", 0.00, 0.80, 0.05, 0.01,
                     help="Filtro de ‘moneyness’ por Delta. Vendedores costumam usar |Δ| ~ 0,05–0,35.")
    dmax = st.slider("|Δ| máximo", 0.05, 1.00, 0.35, 0.01)

    st.markdown("#### 🧭 Bandas de risco por perna")
    st.session_state["band_low_max"] = st.number_input(
        "Limite banda Baixo (|Δ| ≤ ...)",
        min_value=0.00, max_value=1.00, value=0.15, step=0.01,
        help="Classificação didática pela probabilidade de exercício (PoE) de cada perna pelos deltas."
    )
    st.session_state["band_mid_max"] = st.number_input(
        "Limite banda Médio (|Δ| ≤ ...)",
        min_value=0.00, max_value=1.00, value=0.35, step=0.01
    )

    st.markdown("#### ⏱️ Janela p/ IV Rank/Percentil (dias)")
    _ = st.number_input(
        "Janela histórica (dias)",
        min_value=20, max_value=252, value=120, step=10,
        help="Compara IV atual vs histórico (usa HV20 como proxy se IV faltar). IV mais alta → melhor para venda de prêmio."
    )

    st.markdown("#### 🚦 Instruções de SAÍDA — Regras práticas")
    st.caption("• Recomprar a perna ameaçada quando o preço encostar no strike perto do vencimento.\n"
               "• Encerrar após capturar 70–80% do crédito.\n"
               "• Com ≤ N dias, mensagens de saída ficam mais proativas.\n"
               "• Considere proximidade ao strike para alertas.")

# ---------- Seleção de ticker com busca (nome/código) ----------
st.markdown("#### 🔎 Escolha um ticker da B3 (pesquise por **nome** ou **código**)")

df_b3 = fetch_b3_universe()
if df_b3 is None or df_b3.empty:
    st.warning("Não consegui carregar a lista de tickers. Usando fallback.")
    df_b3 = pd.DataFrame(B3_FALLBACK, columns=["ticker", "name"])
    df_b3["label"] = df_b3["ticker"] + " — " + df_b3["name"]

labels = df_b3["label"].tolist()
default_idx = 0
try:
    idx_petr4 = list(df_b3["ticker"]).index("PETR4")
    default_idx = idx_petr4
except Exception:
    pass

sel_label = st.selectbox(
    "Pesquisar",
    options=labels,
    index=min(default_idx, len(labels)-1),
    placeholder="Digite 'Petrobras', 'Bradesco', etc."
)
sel_row = df_b3[df_b3["label"] == sel_label].iloc[0]
ticker = sel_row["ticker"]

# ---------- Cotação automática sempre (sem toggle) ----------
px = yahoo_price_b3(ticker)
if np.isnan(px):
    st.warning("Não consegui obter a cotação agora via Yahoo Finance. Você pode tentar outro ticker.")
spot_str = f"{px:.2f}".replace(".", ",") if not np.isnan(px) else ""
spot_num = float(px) if not np.isnan(px) else np.nan

col_price, col_dummy = st.columns([1.0, 1.0])
with col_price:
    st.text_input("Preço à vista (S)", value=spot_str, disabled=True, key="spot_display")

# ---------- Área de colagem da option chain ----------
st.markdown("### 3) Colar a option chain do **{}** (opcoes.net)".format(ticker))
raw_text = st.text_area(
    "Cole aqui a tabela (Ctrl/Cmd+C no site → Ctrl/Cmd+V aqui)",
    height=220,
    placeholder="Cole a tabela com as colunas: Ticker, Vencimento, Tipo, Strike, Último, Vol. Impl. (%), Delta..."
)

df_opts = parse_pasted_table(raw_text)

# Exibe resumo leve (sem poluir o design)
if not df_opts.empty:
    expir_sorted = sorted(df_opts["expiration"].dropna().unique())
    st.success(f"Tabela reconhecida: **{len(df_opts)}** linhas • Vencimentos detectados: {', '.join(str(x) for x in expir_sorted[:6])}{'...' if len(expir_sorted)>6 else ''}")
else:
    st.info("Cole a tabela do opcoes.net para prosseguir.")

# ---------- Selecionar vencimento ----------
chosen_exp = None
if not df_opts.empty:
    exp_list = sorted(df_opts["expiration"].dropna().unique())
    exp_label = st.selectbox("📅 Escolha um vencimento",
                             options=exp_list,
                             index=0 if len(exp_list) else None)

    if exp_label:
        chosen_exp = exp_label

# ---------- Montar pares e Top 3 ----------
tab_sug, tab_comp, tab_help = st.tabs(["🏆 Sugestões", "📈 Comparar estratégias", "📘 Explicações"])

with tab_sug:
    if df_opts.empty or np.isnan(spot_num) or chosen_exp is None:
        st.info("Aguardando **tabela colada**, **cotação** e **vencimento** para gerar as sugestões.")
    else:
        df_exp = df_opts[df_opts["expiration"] == chosen_exp].copy()
        sigma = hv20  # proxy de IV anual
        pairs = make_pairs(df_exp, spot=spot_num, r=r, sigma=sigma, dmin=dmin, dmax=dmax)

        if pairs.empty:
            st.warning("Não há strangles possíveis com os filtros atuais. Ajuste |Δ| ou cole uma cadeia mais completa.")
        else:
            # rank local
            pairs = pairs.sort_values("score", ascending=False).reset_index(drop=True)
            topN = pairs.head(3)

            # Render amigável
            st.markdown("#### 🏆 Top 3 (melhor **prêmio/risco**)")

            def fmt_line(i, rw):
                # crédito por lote (100) em R$
                cred_lote = rw["credit"] * 100.0
                be_low = rw["be_low"]; be_high = rw["be_high"]
                poe_p = rw["poe_put"] * 100 if not pd.isna(rw["poe_put"]) else np.nan
                poe_c = rw["poe_call"] * 100 if not pd.isna(rw["poe_call"]) else np.nan
                band_p = rw["band_put"]; band_c = rw["band_call"]
                kp = rw["Kp"]; kc = rw["Kc"]
                symp = str(rw.get("sym_put", "PUT")) or "PUT"
                symc = str(rw.get("sym_call", "CALL")) or "CALL"

                return (
                    f"**#{i}** → Vender **PUT {symp} (K={kp:.2f})** + **CALL {symc} (K={kc:.2f})**  \n"
                    f"**Crédito por lote:** R$ {cred_lote:,.2f}  \n"
                    f"**Break-evens:** [{be_low:.2f}, {be_high:.2f}]  \n"
                    f"**PoE PUT:** {poe_p:.1f}% • **PoE CALL:** {poe_c:.1f}%  \n"
                    f"**Bandas:** PUT **{band_p}** • CALL **{band_c}**  \n"
                    f"**Dica:** ⏳ se faltarem ≤ 7 dias, e **S** encostar em **K_call** ⇒ recomprar a CALL; "
                    f"🎯 capturar ~75% do crédito e encerrar."
                )

            for idx, rw in enumerate(topN.itertuples(index=False), start=1):
                st.markdown(fmt_line(idx, rw))

            # Tabela compacta
            st.markdown("##### 📋 Sugestões ranqueadas (compacto)")
            show = topN[["expiration","Kp","Kc","credit","be_low","be_high","poe_put","poe_call","band_put","band_call","sym_put","sym_call"]].copy()
            show = show.rename(columns={
                "expiration":"Venc.",
                "Kp":"K_put",
                "Kc":"K_call",
                "credit":"Crédito (R$/ação)",
                "be_low":"Break-even ↓",
                "be_high":"Break-even ↑",
                "poe_put":"PoE PUT",
                "poe_call":"PoE CALL",
                "band_put":"Banda PUT",
                "band_call":"Banda CALL",
                "sym_put":"Ticker PUT",
                "sym_call":"Ticker CALL",
            })
            show["PoE PUT"]  = (show["PoE PUT"]*100).map(lambda x: f"{x:.1f}%")
            show["PoE CALL"] = (show["PoE CALL"]*100).map(lambda x: f"{x:.1f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

with tab_comp:
    st.markdown("### 📈 Comparar estratégias (Strangle × Iron Condor × Jade Lizard)")
    if df_opts.empty or np.isnan(spot_num) or chosen_exp is None:
        st.info("Cole a cadeia, selecione o vencimento e aguarde a cotação para comparar.")
    else:
        df_exp = df_opts[df_opts["expiration"] == chosen_exp].copy()
        sigma = hv20
        pairs = make_pairs(df_exp, spot=spot_num, r=r, sigma=sigma, dmin=dmin, dmax=dmax)
        if pairs.empty:
            st.warning("Sem strangles elegíveis para comparar. Ajuste os filtros.")
        else:
            base = pairs.iloc[0]  # usa a melhor como base
            Kp, Kc = float(base["Kp"]), float(base["Kc"])
            credit = float(base["credit"])
            # asas (heurística): 3% para cada lado
            wing_pct = st.slider("Largura das asas (% do preço à vista)", 1, 20, 6, 1)
            Kp_w = max(0.01, Kp - (wing_pct/100.0)*spot_num)
            Kc_w = Kc + (wing_pct/100.0)*spot_num
            # crédito IC: supor compra de PUT Kp_w e CALL Kc_w custando 30% do crédito base (heurística)
            ic_cost = 0.30 * credit
            ic_credit = max(0.0, credit - ic_cost)
            loss_max_ic = (Kc_w - Kc) - ic_credit  # aprox

            # Jade Lizard: PUT vendida (Kp), CALL vendida (Kc), CALL comprada (Kc_w)
            jl_credit = credit - 0.15*credit  # compra da call de proteção, heurístico
            no_upside_risk = jl_credit >= (Kc_w - Kc)

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Strangle — Crédito", f"R$ {credit*100:,.2f}")
                st.write(f"**Zona neutra (Kp–Kc)**: {Kp:.2f} — {Kc:.2f}")
                prob_inside = 1 - (float(base['poe_put']) + float(base['poe_call']))
                st.write(f"**PoE ficar dentro**: {max(0.0, prob_inside)*100:.0f}%")
            with colB:
                st.metric("Iron Condor — Crédito", f"R$ {ic_credit*100:,.2f}")
                st.write(f"**Asas (P,C)**: {Kp_w:.2f}, {Kc_w:.2f}")
                st.write(f"**Perda máx. aprox.**: R$ {max(0.0,loss_max_ic)*100:,.2f}")
            with colC:
                st.metric("Jade Lizard — Crédito", f"R$ {jl_credit*100:,.2f}")
                st.write(f"**Asa (CALL)**: {Kc_w:.2f}")
                st.write(f"**Sem risco de alta?** {'Sim' if no_upside_risk else 'Não'}")

            st.caption(
                "Observação: créditos/custos de proteção nas comparações usam **heurísticas** quando não existem prêmios das asas na cadeia colada."
            )

with tab_help:
    st.markdown("### 📘 Explicações rápidas")
    st.markdown(
        "- **Strangle vendido**: vender **PUT (Kp)** + **CALL (Kc)**. Lucro máximo = **crédito** se o preço **S** ficar entre Kp e Kc no vencimento.  \n"
        "- **Iron Condor**: o strangle acima **+ compra** das asas (PUT abaixo de Kp e CALL acima de Kc) para **limitar perdas**.  \n"
        "- **Jade Lizard**: **PUT vendida** + **CALL vendida** + **CALL comprada** acima; se o crédito ≥ (Kc_w − Kc), **sem risco de alta**.  \n"
        "- **P/L no vencimento** (por ação):  \n"
        "  • Strangle: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + crédito  \n"
        "  • Iron Condor: Strangle + max(0, Kp_w − S) + max(0, S − Kc_w) − custo_das_asas  \n"
        "  • Jade Lizard: −max(0, Kp − S) − max(0, S − Kc) + max(0, S − Kc_w) + crédito_líquido"
    )

# ---------- Rodapé leve ----------
st.divider()
st.caption("v9 • Educação em opções • Este app não é recomendação de investimento.")

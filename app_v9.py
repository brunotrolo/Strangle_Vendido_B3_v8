# app_v9.py
# --------------------------------------------
# Strangle Vendido Coberto — v9
# - Busca dinâmica de tickers na B3 (dadosdemercado.com.br/acoes)
# - Cotação automática via yfinance (sempre)
# - Input para colar option chain do opcoes.net
# - TOP3 em tabela + blocos explicativos c/ lotes e prêmio estimado (métrica + fórmula)
# - Parser robusto (dedup de colunas e primeira ocorrência)
# --------------------------------------------

import re
import io
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, date
from typing import Tuple, Optional

# yfinance para cotação
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------------- Utils numéricos ----------------

SQRT_2 = math.sqrt(2.0)
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _parse_percent_to_vol(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%","").replace(" ", "")
    # tratamento pt-BR vs en-US
    if s.count(",") == 1 and s.count(".") > 1:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    v = _safe_float(s)
    return v/100.0 if pd.notna(v) else np.nan

def _parse_money_ptbr(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    s = s.replace(".", "").replace(",", ".")
    return _safe_float(s)

def yearfrac(d1: date, d2: date) -> float:
    return max((d2 - d1).days, 0) / 365.0

def format_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------- Dados de Tickers (B3) ----------------

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_b3_tickers() -> pd.DataFrame:
    url = "https://www.dadosdemercado.com.br/acoes"
    headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp)", "Accept-Language": "pt-BR,pt;q=0.9"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    best = None
    for tb in tables:
        cols_lower = [str(c).strip().lower() for c in tb.columns]
        if any("ticker" in c for c in cols_lower) and any("nome" in c or "empresa" in c for c in cols_lower):
            best = tb.copy()
            break
    if best is None:
        best = tables[0].copy()

    colmap = {}
    for c in best.columns:
        cl = str(c).strip().lower()
        if "ticker" in cl or "código" in cl:
            colmap[c] = "ticker"
        elif "nome" in cl or "empresa" in cl:
            colmap[c] = "nome"
    best = best.rename(columns=colmap)
    if "ticker" not in best.columns:
        for c in best.columns:
            if best[c].astype(str).str.match(r"^[A-Z]{4}\d{0,2}$").any():
                best = best.rename(columns={c:"ticker"})
                break
    if "nome" not in best.columns:
        best["nome"] = ""
    best["ticker"] = best["ticker"].astype(str).str.strip().str.upper()
    best["nome"]   = best["nome"].astype(str).str.strip()
    best = best[best["ticker"].str.match(r"^[A-Z]{4}\d{0,2}$")]
    best = best.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return best[["ticker","nome"]]

def fallback_tickers() -> pd.DataFrame:
    data = [
        ("PETR4","Petrobras PN"),
        ("VALE3","Vale ON"),
        ("ITUB4","Itaú Unibanco PN"),
        ("BBDC4","Bradesco PN"),
        ("BBAS3","Banco do Brasil ON"),
        ("CSAN3","Cosan ON"),
        ("ABEV3","Ambev ON"),
        ("WEGE3","WEG ON")
    ]
    return pd.DataFrame(data, columns=["ticker","nome"])

def get_ticker_list_for_select() -> Tuple[pd.DataFrame, str]:
    warn = ""
    try:
        df = fetch_b3_tickers()
        if df.empty:
            raise ValueError("Lista vazia")
        return df, warn
    except Exception as e:
        warn = f"⚠️ Não foi possível atualizar a lista do site (usando lista básica local). Motivo: {e}"
        return fallback_tickers(), warn

# ---------------- Cotação (yfinance) ----------------

@st.cache_data(ttl=5*60, show_spinner=False)
def get_spot_from_yf(ticker_b3: str) -> Optional[float]:
    if yf is None:
        return None
    code = ticker_b3.upper().strip()
    yf_code = f"{code}.SA"
    try:
        t = yf.Ticker(yf_code)
        px = None
        info = getattr(t, "fast_info", None)
        if info:
            px = info.get("last_price")
        if px is None:
            hist = t.history(period="1d")
            if not hist.empty:
                px = float(hist["Close"].iloc[-1])
        if px is None and hasattr(t, "info"):
            px = t.info.get("regularMarketPrice")
        return float(px) if px is not None else None
    except Exception:
        return None

# ---------------- Parsing da option chain colada ----------------

HEADER_MAP_PT = {
    "ticker":"symbol","ativo":"symbol",
    "tipo":"type","call":"type","put":"type",
    "strike":"strike",
    "último":"last","ultimo":"last",
    "vol. impl. (%)":"impliedVol","vol impl (%)":"impliedVol","volatilidade implícita":"impliedVol",
    "delta":"delta",
    "vencimento":"expiration","data vencimento":"expiration",
}

def _normalize_header(h: str) -> str:
    h0 = str(h).strip().lower()
    for a,b in [("ç","c"),("ã","a"),("â","a"),("á","a"),("é","e"),("ê","e"),("í","i"),("ó","o"),("ô","o"),("ú","u"),("ü","u"),("õ","o")]:
        h0 = h0.replace(a,b)
    h0 = re.sub(r"\s+", " ", h0).strip()
    for k,v in HEADER_MAP_PT.items():
        if k in h0:
            return v
    return h0

def _dedup_cols(cols):
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def _pick_first(df: pd.DataFrame, base: str) -> Optional[str]:
    if base in df.columns:
        return base
    pref = [c for c in df.columns if c == base or c.startswith(base + ".")]
    return pref[0] if pref else None

def parse_pasted_table(txt: str) -> pd.DataFrame:
    raw = str(txt).strip()
    if not raw:
        return pd.DataFrame()

    if "\t" in raw:
        df = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, engine="python")
    else:
        lines = [re.split(r"\s{2,}", ln.strip()) for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame()
        lens = [len(x) for x in lines]
        width = max(lens)
        norm = [row + [""]*(width-len(row)) for row in lines]
        df = pd.DataFrame(norm[1:], columns=norm[0])

    df.columns = [_normalize_header(c) for c in df.columns]
    df.columns = _dedup_cols(df.columns)

    targets = ["symbol","type","strike","last","impliedVol","delta","expiration"]
    col_sel = {}
    for t in targets:
        c = _pick_first(df, t)
        if c:
            col_sel[t] = c

    if not col_sel:
        return pd.DataFrame()

    df = df[list(col_sel.values())].copy()
    df = df.rename(columns={v:k for k,v in col_sel.items()})

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        df["type"] = df["type"].replace({"CALL":"C","PUT":"P","C":"C","P":"P","E":"P","A":"C"})
    if "strike" in df.columns:
        df["strike"] = df["strike"].map(_parse_money_ptbr)
    if "last" in df.columns:
        df["last"] = df["last"].map(_parse_money_ptbr)
    if "impliedVol" in df.columns:
        df["impliedVol"] = df["impliedVol"].map(_parse_percent_to_vol)
    if "delta" in df.columns:
        df["delta"] = df["delta"].apply(lambda x: _safe_float(str(x).replace(",", ".") if pd.notna(x) else x))
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], dayfirst=True, errors="coerce").dt.date

    df = df.dropna(subset=["type","strike","expiration"], how="any")
    df = df[df["type"].isin(["C","P"])].reset_index(drop=True)
    return df

# ---------------- Probabilidades ----------------

def d1_d2(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma is None or sigma <= 0 or T <= 0:
        return np.nan, np.nan
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return d1, d2
    except Exception:
        return np.nan, np.nan

def prob_ITM_call(S,K,r,sigma,T):
    _, d2 = d1_d2(S,K,r,sigma,T)
    return norm_cdf(d2) if not (np.isnan(d2)) else np.nan

def prob_ITM_put(S,K,r,sigma,T):
    p_call = prob_ITM_call(S,K,r,sigma,T)
    return 1.0 - p_call if pd.notna(p_call) else np.nan

# ---------------- App ----------------

st.set_page_config(page_title="Strangle Vendido Coberto — v9", page_icon="💼", layout="wide")
st.markdown("## 💼 Strangle Vendido Coberto — v9")
st.caption("Cole a option chain do opcoes.net, escolha o vencimento e veja as sugestões didáticas de strangle coberto.")

with st.sidebar:
    st.markdown("### ⚙️ Parâmetros")
    r = st.number_input("Taxa livre de risco anual (r)", min_value=0.00, max_value=1.00, value=0.11, step=0.01,
                        help="Usada no Black–Scholes. Ex.: 0,11 = 11% a.a.")
    delta_min = st.number_input("|Δ| mínimo", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                help="Filtro por moneyness via Delta. ~0,05–0,35 é comum.")
    delta_max = st.number_input("|Δ| máximo", min_value=0.0, max_value=1.0, value=0.35, step=0.01,
                                help="Maior |Δ| aceito para as opções vendidas.")
    contract_size = st.number_input("Tamanho do contrato (ações/contrato)", min_value=1, max_value=10000, value=100, step=1)

    st.markdown("---")
    st.markdown("### 🚪 Instruções de SAÍDA")
    dias_alerta = st.number_input("Dias até vencimento (alerta)", min_value=0, max_value=60, value=7, step=1)
    prox_pct = st.number_input("Proximidade ao strike (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    meta_captura = st.number_input("Meta de captura do prêmio (%)", min_value=10, max_value=100, value=75, step=5)

# -------- Tickers --------
df_tks, warn_msg = get_ticker_list_for_select()
if warn_msg:
    st.info(warn_msg)

st.markdown("### 🔎 Escolha um ticker da B3 (pesquise por nome ou código)")
options_labels = (df_tks["ticker"] + " — " + df_tks["nome"].fillna("")).tolist()
default_idx = options_labels.index("PETR4 — Petrobras PN") if "PETR4 — Petrobras PN" in options_labels else 0
sel_label = st.selectbox("Ticker (B3) — pesquise por nome ou código", options_labels,
                         index=default_idx if 0 <= default_idx < len(options_labels) else 0)
sel = sel_label.split(" — ")[0].strip()

# -------- Preço à vista (sempre yfinance) --------
spot = get_spot_from_yf(sel)
if spot is None:
    st.warning("⚠️ Não consegui obter a cotação via yfinance. Informe manualmente.")
    spot = st.number_input("Preço à vista (S)", min_value=0.0, value=10.0, step=0.01)
else:
    st.metric("Preço à vista (S)", f"{spot:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

# -------- Option Chain --------
st.markdown(f"### 3) Colar a option chain de **{sel}** (opcoes.net)")
st.caption("No opcoes.net: copie a tabela completa (Ctrl/Cmd+C) e cole aqui (Ctrl/Cmd+V).")
txt = st.text_area("Cole aqui a tabela (a primeira linha deve conter os cabeçalhos)", height=240)

df_raw = parse_pasted_table(txt) if txt else pd.DataFrame()
if not df_raw.empty:
    st.success(f"Tabela colada reconhecida com {len(df_raw)} linhas.")
    exps = sorted([d for d in df_raw["expiration"].dropna().unique().tolist()])
    st.markdown("### 📅 Vencimento")
    exp_sel = st.selectbox("Escolha um vencimento", exps) if exps else None

    if exp_sel:
        df_exp = df_raw[df_raw["expiration"] == exp_sel].copy()
        calls = df_exp[df_exp["type"]=="C"].copy()
        puts  = df_exp[df_exp["type"]=="P"].copy()

        # OTM
        calls["OTM"] = calls["strike"] > float(spot)
        puts["OTM"]  = puts["strike"]  < float(spot)

        # filtros por |delta|
        def _absd_ok(d):
            if pd.isna(d): 
                return True
            return (abs(d) >= delta_min) and (abs(d) <= delta_max)

        if "delta" in calls.columns:
            calls = calls[calls["delta"].apply(_absd_ok)]
        if "delta" in puts.columns:
            puts  = puts[ puts["delta"].apply(_absd_ok)]

        calls = calls[calls["OTM"]].copy()
        puts  = puts[ puts["OTM"]].copy()

        # mid = last (fallback)
        calls["mid"] = calls.get("last", np.nan)
        puts["mid"]  = puts.get("last", np.nan)

        default_sigma = 0.20
        T = yearfrac(date.today(), exp_sel)

        def _sigma_series(df_leg):
            if "impliedVol" in df_leg.columns and df_leg["impliedVol"].notna().any():
                return df_leg["impliedVol"].fillna(default_sigma)
            return pd.Series(default_sigma, index=df_leg.index)

        sigma_call = _sigma_series(calls)
        sigma_put  = _sigma_series(puts)

        calls["prob_ITM"] = [prob_ITM_call(spot, K, r, s, T) for K, s in zip(calls["strike"], sigma_call)]
        puts["prob_ITM"]  = [prob_ITM_put (spot, K, r, s, T) for K, s in zip(puts["strike"],  sigma_put )]

        calls_rank = calls.sort_values(["prob_ITM","mid"], ascending=[True, False]).head(30)
        puts_rank  = puts.sort_values(["prob_ITM","mid"],  ascending=[True, False]).head(30)

        pairs = []
        for _, p in puts_rank.iterrows():
            for _, c in calls_rank.iterrows():
                pmid = (p.get("mid", 0.0) or 0.0)
                cmid = (c.get("mid", 0.0) or 0.0)
                if pd.isna(pmid) or pd.isna(cmid):
                    continue
                credito = float(pmid) + float(cmid)
                k_put, k_call = float(p["strike"]), float(c["strike"])
                be_low  = k_put - credito
                be_high = k_call + credito
                poe_p = float(p.get("prob_ITM", np.nan))
                poe_c = float(c.get("prob_ITM", np.nan))
                poe_sum = (poe_p if pd.notna(poe_p) else 0.15) + (poe_c if pd.notna(poe_c) else 0.15)
                score = (credito + 1e-6) / (poe_sum + 1e-6)
                pairs.append({
                    "PUT": p.get("symbol",""),
                    "Kp": k_put,
                    "CALL": c.get("symbol",""),
                    "Kc": k_call,
                    "credito": float(credito),
                    "be_low": be_low, 
                    "be_high": be_high,
                    "poe_put": poe_p, 
                    "poe_call": poe_c,
                    "score": score,
                })

        recs = pd.DataFrame(pairs)
        if recs.empty:
            st.warning("Não há CALL e PUT OTM suficientes com os filtros atuais. Ajuste |Δ| ou cole uma cadeia mais completa.")
        else:
            top3 = recs.sort_values("score", ascending=False).head(3).reset_index(drop=True)

            # ---------- TABELA TOP 3 ----------
            st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
            tbl = top3.copy()
            tbl_disp = pd.DataFrame({
                "Rank": [1,2,3][:len(tbl)],
                "PUT": tbl["PUT"],
                "Kp": tbl["Kp"].round(2),
                "CALL": tbl["CALL"],
                "Kc": tbl["Kc"].round(2),
                "Crédito/ação (R$)": tbl["credito"].round(2),
                "Break-evens": tbl.apply(lambda r: f"[{r['be_low']:.2f}, {r['be_high']:.2f}]", axis=1),
                "PoE PUT (%)": (100*tbl["poe_put"]).round(1),
                "PoE CALL (%)": (100*tbl["poe_call"]).round(1),
            })
            st.dataframe(tbl_disp, use_container_width=True)

            # ---------- BLOCO POR RECOMENDAÇÃO + LOTES ----------
            st.markdown("### 📋 Sugestões detalhadas (com lotes e prêmio total)")

            if "lot_map" not in st.session_state:
                st.session_state["lot_map"] = {}

            for i, rw in top3.iterrows():
                rank = i + 1
                key_lotes = f"lotes_rank_{rank}"
                if key_lotes not in st.session_state["lot_map"]:
                    st.session_state["lot_map"][key_lotes] = 1

                with st.container(border=True):
                    st.markdown(
                        f"**#{rank}** → Vender **PUT {rw['PUT']} (K={rw['Kp']:.2f})** + "
                        f"**CALL {rw['CALL']} (K={rw['Kc']:.2f})**"
                    )
                    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.0])
                    col1.metric("Crédito/ação (R$)", f"{rw['credito']:.2f}")
                    col2.metric("Break-evens", f"[{rw['be_low']:.2f}, {rw['be_high']:.2f}]")
                    col3.metric("PoE PUT / CALL (%)", f"{100*rw['poe_put']:.1f} / {100*rw['poe_call']:.1f}")
                    lotes = col4.number_input("Lotes", min_value=0, max_value=10000,
                                              value=st.session_state['lot_map'][key_lotes], step=1, key=key_lotes)
                    st.session_state["lot_map"][key_lotes] = lotes

                    premio_total = rw["credito"] * contract_size * lotes
                    c1, c2 = st.columns([1, 2])
                    c1.metric("🎯 Prêmio estimado (total)", format_brl(premio_total))
                    c2.markdown(
                        f"**Cálculo:** `crédito/ação × contrato × lotes` = "
                        f"`{rw['credito']:.2f} × {contract_size} × {lotes}` → **{format_brl(premio_total)}**"
                    )

                    with st.expander("📘 O que significa cada item?"):
                        st.markdown(
                            "- **Crédito/ação:** soma dos prêmios recebidos ao **vender** a PUT e a CALL (por ação).  \n"
                            "- **Break-evens:** intervalo em que o resultado no vencimento ainda é ≥ 0 "
                            f"([{rw['be_low']:.2f}, {rw['be_high']:.2f}]).  \n"
                            "- **PoE (Prob. expirar ITM):** estimativa por Black–Scholes (σ da cadeia, quando disponível).  \n"
                            f"- **Lotes:** número de strangles (PUT+CALL) vendidos. **Contrato** = {contract_size} ações.  \n"
                            f"- **Regras práticas de saída:** ⏳ faltam ≤ **{dias_alerta}** dias • S encostando em **K_call** ⇒ recomprar a CALL • 🎯 capturar ~**{meta_captura}%** do crédito."
                        )

            # ---------- RESUMO DE PRÊMIOS ----------
            st.markdown("### 💰 Resumo dos prêmios (com os lotes escolhidos)")
            resumo = []
            for i, rw in top3.iterrows():
                rank = i+1
                lotes = st.session_state["lot_map"].get(f"lotes_rank_{rank}", 0)
                premio_total = rw["credito"] * contract_size * lotes
                resumo.append({
                    "Rank": rank,
                    "PUT": rw["PUT"], "Kp": round(rw["Kp"],2),
                    "CALL": rw["CALL"], "Kc": round(rw["Kc"],2),
                    "Lotes": lotes,
                    "Crédito/ação (R$)": round(rw["credito"],2),
                    "Prêmio total (R$)": round(premio_total, 2),
                })
            st.dataframe(pd.DataFrame(resumo), use_container_width=True)

else:
    st.info("Cole a tabela do **opcoes.net** para prosseguir.")

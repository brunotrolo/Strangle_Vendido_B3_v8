# app_v9.py
# Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)
# Requisitos: streamlit, pandas, numpy, python-dateutil, yfinance, requests, lxml
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from datetime import datetime, date
from dateutil.parser import parse as dtparse
import yfinance as yf
import requests

st.set_page_config(page_title="Strangle Vendido Coberto — v9", layout="wide")

# ========================= Utils =========================

def _strip_b3_ticker(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().upper()
    # Normaliza tickers (PETR4, BBDC4 etc.)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def _br_to_float(x):
    """Converte string BR para float (1.234,56 -> 1234.56)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def _parse_brl(s):
    """Converte valores em BRL (ex: 'R$ 1.234,56') para float."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = s.replace("R$", "").strip()
    return _br_to_float(s)

def _guess_date(s):
    """Tenta interpretar datas no padrão brasileiro ou ISO."""
    if pd.isna(s):
        return pd.NaT
    if isinstance(s, (datetime, date)):
        return pd.to_datetime(s)
    s = str(s).strip()
    # tenta padrões DD/MM/AAAA ou AAAA-MM-DD
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except:
        return pd.NaT

def _clean_col(s: str):
    """Limpa cabeçalhos vindos do opcoes.net"""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    reps = {
        "vencimento": "expiration",
        "tipo": "type",
        "strike": "strike",
        "preço exerc.": "strike",
        "último": "last",
        "últ": "last",
        "ultimo": "last",
        "fechamento": "last",
        "compra": "bid",
        "venda": "ask",
        "quantidade": "volume",
        "vol.": "volume",
        "volume": "volume",
        "ajuste": "settlement",
        "delta": "delta",
        "gamma": "gamma",
        "vega": "vega",
        "theta": "theta",
        "iv": "iv",
        "implicita": "iv",
        "i.v.": "iv",
        "horário": "time",
        "hora": "time",
        "data/hora": "time",
        "opção": "option",
        "opcao": "option",
        "ativo": "underlying",
        "papel": "underlying",
        "negócios": "trades",
        "qtd": "volume",
    }
    for k, v in reps.items():
        s = s.replace(k, v)
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def _parse_float_str(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return np.nan
    # Trata "" e "—"
    s = s.replace("—", "")
    s = s.replace("–", "-")
    # Trata "1.234,56" ou "1,234.56"
    if "," in s and "." in s:
        # Heurística: se tem '.' como separador de milhares e ',' decimal (BR)
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def _normalize_option_chain_table(raw_text):
    """
    Recebe texto colado do opcoes.net e tenta montar DataFrame com colunas padronizadas.
    """
    # Remove múltiplos espaços e normaliza separadores
    raw = raw_text.replace("\t", ";")
    raw = re.sub(r"[ ]{2,}", " ", raw)
    raw = raw.strip()

    # Converte para linhas
    lines = raw.splitlines()
    # tenta achar linha de cabeçalho
    header_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"(?i)(vencimento|preço exerc\.|strike|tipo|último|compra|venda|volume|ajuste|delta|iv)", ln):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    header = lines[header_idx]
    body = lines[header_idx+1:]
    cols = [c.strip() for c in re.split(r";|\s{2,}| \| ", header) if c.strip() != ""]

    # Corrige duplicatas simples
    # Garante ao menos TYPE/STRIKE/EXPIRATION/BID/ASK/LAST se possível
    df_rows = []
    for ln in body:
        parts = [c.strip() for c in re.split(r";|\s{2,}| \| ", ln)]
        # Alinha pelo número de colunas do header
        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols)]
        df_rows.append(parts)

    df = pd.DataFrame(df_rows, columns=cols)

    # Renomeia
    df.columns = [_clean_col(c) for c in df.columns]

    # Mapeamentos mínimos
    # 'type' -> CALL/PUT ; 'strike' float ; 'expiration' datetime ; 'bid','ask','last' float
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        # uniformiza
        df["type"] = df["type"].replace({"C": "CALL", "V": "PUT"})

    if "strike" in df.columns:
        df["strike"] = df["strike"].apply(_parse_float_str)

    # preços
    for c in ["bid", "ask", "last", "settlement"]:
        if c in df.columns:
            df[c] = df[c].apply(_parse_float_str)

    # greeks, iv
    for c in ["delta", "gamma", "vega", "theta", "iv"]:
        if c in df.columns:
            df[c] = df[c].apply(_parse_float_str)

    # volume, trades
    for c in ["volume", "trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # expiration/time
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].apply(_guess_date)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)

    # Se "last" não existir, tenta média de bid/ask
    if "last" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["last"] = (df["bid"] + df["ask"]) / 2.0

    df = df[~df["strike"].isna()]
    df = df[~df["type"].isna()]
    df = df[~df["expiration"].isna()]

    df["abs_delta"] = df["delta"].abs() if "delta" in df.columns else np.nan

    df = df.sort_values(["expiration","type","strike"], ascending=[True, True, True]).reset_index(drop=True)
    return df

def _pair_strangles(df_exp, spot, mindelta, maxdelta):
    calls = df_exp[(df_exp["type"].str.contains("CALL")) & (df_exp["strike"] > spot)].copy()
    puts  = df_exp[(df_exp["type"].str.contains("PUT"))  & (df_exp["strike"] < spot)].copy()

    if not np.isnan(mindelta):
        calls = calls[calls["abs_delta"] >= mindelta]
        puts  = puts[puts["abs_delta"]  >= mindelta]
    if not np.isnan(maxdelta):
        calls = calls[calls["abs_delta"] <= maxdelta]
        puts  = puts[puts["abs_delta"]  <= maxdelta]

    out = []
    if calls.empty or puts.empty:
        return out

    for _, rc in calls.iterrows():
        for _, rp in puts.iterrows():
            out.append({
                "Kc": rc["strike"], "Kp": rp["strike"],
                "call_row": rc.to_dict(),
                "put_row": rp.to_dict()
            })
    return out

def payoff_strangle(S, Kp, Kc, credit):
    """
    P/L no vencimento por ação:
    Π(S) = -max(0, Kp - S) - max(0, S - Kc) + credit
    """
    S = np.asarray(S, dtype=float)
    return -np.maximum(0.0, Kp - S) - np.maximum(0.0, S - Kc) + credit

def payoff_iron_condor(S, Kp, Kc, Kp_w, Kc_w, credit, wings_cost):
    """
    Iron Condor = Strangle (vendido) + compra de PUT abaixo de Kp e compra de CALL acima de Kc
    """
    S = np.asarray(S, dtype=float)
    base = payoff_strangle(S, Kp, Kc, credit)
    add_put  = np.maximum(0.0, Kp_w - S)
    add_call = np.maximum(0.0, S - Kc_w)
    return base + add_put + add_call - wings_cost

def jade_lizard(S, Kp, Kc, Kc_w, credit):
    """
    Jade Lizard simplificado: put vendida + call vendida + "compra parcial" de call longa
    Mantra: crédito ≥ (Kc_w − Kc) -> sem risco na alta.
    """
    S = np.asarray(S, dtype=float)
    return -np.maximum(0.0, Kp - S) - np.maximum(0.0, S - Kc) + np.maximum(0.0, S - Kc_w) + credit

def yahoo_price_b3(ticker: str):
    """
    Busca preço de fechamento recente do papel B3 via Yahoo Finance (ticker.SA).
    """
    tk = _strip_b3_ticker(ticker)
    if not tk:
        return None
    try:
        data = yf.Ticker(f"{tk}.SA").history(period="5d")
        if data is None or data.empty:
            return None
        last = float(data["Close"].iloc[-1])
        return last
    except Exception:
        return None

# ========================= Layout =========================

st.title("Strangle Vendido Coberto — B3 (v9)")

st.markdown(
    """
    Esta ferramenta permite **colar a option chain do opcoes.net** e simular **Strangle vendido**,
    bem como variações como **Iron Condor** e **Jade Lizard**.  
    Use os passos abaixo e os parâmetros na **barra lateral**.
    """
)

st.markdown("### 1) Selecione o Ticker (B3)")
colA, colB = st.columns([1,1.2])

# Tenta buscar lista de tickers (opcional)
with colA:
    st.caption("Sugestões de tickers populares (B3)")
    try:
        # fallback simples
        tickers_pop = ["PETR4", "VALE3", "ITUB4", "BBDC4", "CSAN3", "BBAS3"]
        sel_label = st.selectbox(
            "Escolha um ticker popular (opcional)",
            options=[""] + tickers_pop,
            index=0,
            key="ticker_pop"
        )
        sel = sel_label if sel_label else ""
    except Exception:
        st.warning("Não consegui carregar a lista da B3. Digite o ticker manualmente abaixo.")
        sel = st.text_input("Ticker (ex.: PETR4, BBDC4)", value="PETR4", key="ticker_fallback").upper().strip()

with colB:
    auto_price = st.toggle("Usar cotação automática (Yahoo Finance)", value=True, key="use_yf")
    if auto_price:
        px = yahoo_price_b3(sel)
        if px is None:
            st.warning("Não consegui obter o preço no Yahoo. Informe manualmente no campo abaixo.")
    else:
        px = None
    spot_input = st.text_input("Preço à vista (S)", value=(f"{px:.2f}".replace(".", ",") if px else ""), key="spot")
    spot = _br_to_float(spot_input) if spot_input else (px if px else np.nan)

# ========================= Sidebar =========================
st.sidebar.header("⚙️ Parâmetros")

st.sidebar.markdown("##### Filtros de strikes pelo |Δ| (se disponível na tabela)")
hv20_pct = st.sidebar.number_input("HV20 (σ anual – %)", min_value=0.0, max_value=200.0, value=17.12, step=0.10, key="hv20")
r_anual_pct = st.sidebar.number_input("r (anual – %)", min_value=0.0, max_value=100.0, value=11.00, step=0.25, key="r_anual")
mindelta = st.sidebar.number_input("|Δ| mínimo", min_value=0.00, max_value=1.00, step=0.01, value=0.05, key="mindelta",
                                   help="Filtra opções com |Δ| >= mínimo")
maxdelta = st.sidebar.number_input("|Δ| máximo", min_value=0.00, max_value=1.00, step=0.01, value=0.35, key="maxdelta",
                                   help="Filtra opções com |Δ| <= máximo")

st.sidebar.markdown("##### Bandas (heurística de IV realizada, %)")
b_baixo = st.sidebar.number_input("Corte Baixo", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="b_baixo")
b_medio = st.sidebar.number_input("Corte Médio", min_value=0.0, max_value=100.0, value=25.0, step=0.1, key="b_medio")
bands_cfg = {"Baixo": (0, b_baixo), "Médio": (b_baixo, b_medio), "Alto": (b_medio, 55)}

st.sidebar.markdown("##### Instruções de SAÍDA — Regras práticas")
dte_alert = st.sidebar.number_input("Dias até vencimento (alerta)", min_value=0, max_value=60, value=7, key="dte_alert")
prox_pct = st.sidebar.number_input("Proximidade ao strike (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="prox_pct")
take_profit = st.sidebar.number_input("Meta de captura do prêmio (%)", min_value=10, max_value=95, value=75, step=5, key="tp_pct")

st.sidebar.markdown("##### Cobertura")
contract_size = st.sidebar.number_input("Tamanho do contrato", min_value=1, max_value=1000, value=100, key="contract_size")
qty_shares = st.sidebar.number_input(f"Ações em carteira ({sel})", min_value=0, max_value=1_000_000, value=0, key="qty_shares")
cash_avail = st.sidebar.text_input(f"Caixa disponível (R$) ({sel})", value="10.000,00", key="cash_avail")
cash_avail = _br_to_float(cash_avail)

# ========================= Colar option chain =========================
st.markdown("### 3) Colar a option chain do **opcoes.net** (CTRL/CMD+V)")
raw_text = st.text_area(
    "Cole aqui a TABELA COMPLETA (CALLs e PUTs) do site **opcoes.net** (com Δ/IV se possível)",
    height=260,
    key="raw_table"
)

if raw_text.strip():
    df = _normalize_option_chain_table(raw_text)
    if df.empty:
        st.error("Não consegui entender a tabela colada. Verifique se você copiou **todas** as colunas/linhas do opcoes.net.")
    else:
        # Mostra preview
        st.markdown("#### Prévia da Tabela Normalizada")
        st.dataframe(df.head(20), use_container_width=True)

        if np.isnan(spot):
            st.warning("Informe o preço à vista (S) ou habilite a cotação automática.")
        else:
            # Se houver múltiplos vencimentos, deixa escolher
            exps = sorted(df["expiration"].dropna().unique())
            if len(exps) == 0:
                st.error("Não encontrei coluna de vencimento válida.")
            else:
                chosen_exp = st.selectbox("Escolha um vencimento:", options=exps, key="exp_select")
                df_exp = df[df["expiration"] == chosen_exp].copy()

                # Parea CALLs e PUTs conforme filtros |Δ|
                pairs = _pair_strangles(df_exp, spot, mindelta if mindelta > 0 else np.nan, maxdelta if maxdelta > 0 else np.nan)
                if not pairs:
                    st.warning("Nenhuma combinação CALL/PUT encontrada com os filtros e preço atual.")
                else:
                    st.markdown(f"**S (spot)** = {spot:.2f} | **Vencimento**: {pd.to_datetime(chosen_exp).date()}")
                    # Mostra pares resumidos
                    show = pd.DataFrame([{"Kp": p["Kp"], "Kc": p["Kc"], 
                                          "Δ_put": p["put_row"].get("delta", np.nan),
                                          "Δ_call": p["call_row"].get("delta", np.nan),
                                          "IV_put%": p["put_row"].get("iv", np.nan),
                                          "IV_call%": p["call_row"].get("iv", np.nan)} for p in pairs])
                    st.dataframe(show, use_container_width=True, height=220)

                    st.markdown("### 4) Escolher strikes e créditos")
                    col1, col2 = st.columns(2)
                    with col1:
                        Kp = _br_to_float(st.text_input("Strike da PUT vendida (Kp)", value=f"{pairs[0]['Kp']:.2f}".replace(".", ",")))
                        Kc = _br_to_float(st.text_input("Strike da CALL vendida (Kc)", value=f"{pairs[0]['Kc']:.2f}".replace(".", ",")))
                        credito = _br_to_float(st.text_input("Crédito líquido por ação (R$)", value="0,80"))
                    with col2:
                        usar_ic = st.toggle("Simular Iron Condor", value=False, key="use_ic")
                        if usar_ic:
                            Kp_w = _br_to_float(st.text_input("Asa PUT comprada (Kp_w)", value=f"{(pairs[0]['Kp']-1):.2f}".replace(".", ",")))
                            Kc_w = _br_to_float(st.text_input("Asa CALL comprada (Kc_w)", value=f"{(pairs[0]['Kc']+1):.2f}".replace(".", ",")))
                            custo_asas = _br_to_float(st.text_input("Custo das asas (R$)", value="0,30"))
                        else:
                            Kp_w = np.nan
                            Kc_w = np.nan
                            custo_asas = 0.0

                    # Geração de grade de preços para plot
                    Smin = min(Kp, Kc) - max(1.0, 0.05*spot)
                    Smax = max(Kp, Kc) + max(1.0, 0.05*spot)
                    grid = np.linspace(Smin, Smax, 401)

                    pl_strangle = payoff_strangle(grid, Kp, Kc, credito)
                    if usar_ic and not (np.isnan(Kp_w) or np.isnan(Kc_w)):
                        pl_ic = payoff_iron_condor(grid, Kp, Kc, Kp_w, Kc_w, credito, custo_asas)
                    else:
                        pl_ic = None

                    # Jade Lizard (opcional)
                    usar_jl = st.toggle("Simular Jade Lizard", value=False, key="use_jl")
                    if usar_jl:
                        # por simplicidade, Kc_w para checar 'sem risco na alta'
                        Kc_w_jl = _br_to_float(st.text_input("CALL longa (Kc_w) para Jade Lizard", value=f"{(Kc+1):.2f}".replace(".", ",")))
                        pl_jl = jade_lizard(grid, Kp, Kc, Kc_w_jl, credito)
                        sem_risco_alta = (credito >= (Kc_w_jl - Kc))
                    else:
                        Kc_w_jl = np.nan
                        pl_jl = None
                        sem_risco_alta = False

                    # Resultados resumidos
                    st.markdown("### 5) Resultados & Payoff")

                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=grid, y=pl_strangle, mode="lines", name="Strangle"))
                    if pl_ic is not None:
                        fig.add_trace(go.Scatter(x=grid, y=pl_ic, mode="lines", name="Iron Condor"))
                    if pl_jl is not None:
                        fig.add_trace(go.Scatter(x=grid, y=pl_jl, mode="lines", name="Jade Lizard"))

                    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
                    fig.add_vline(x=spot, line_dash="dot", opacity=0.6)
                    fig.update_layout(
                        title="Payoff por ação no vencimento",
                        xaxis_title="Preço do subjacente (S)",
                        yaxis_title="P/L por ação (R$)",
                        height=420,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Métricas simples
                    be_low  = Kp - credito
                    be_high = Kc + credito
                    st.markdown("#### Break-evens e Resumo")
                    colm1, colm2, colm3 = st.columns(3)
                    with colm1:
                        st.metric("Break-even baixo", f"{be_low:.2f}")
                    with colm2:
                        st.metric("Break-even alto", f"{be_high:.2f}")
                    with colm3:
                        st.metric("Crédito (R$/ação)", f"{credito:.2f}")

                    # Notas rápidas
                    with st.expander("Regras práticas (saída/ajustes)"):
                        st.write(f"- **DTE alerta**: {dte_alert} dias; **Proximidade ao strike**: {prox_pct:.1f}% ; **Take profit**: {take_profit}% do prêmio.")
                        st.write("- Ajuste antecipado se o preço tocar **próximo do strike** ou se IV contrair abruptamente.")
                        st.write("- Evitar operar quando HV20 < banda baixa e IV implícita também baixa (pior relação prêmio/risco).")

                    # Cobertura & alocação
                    st.markdown("#### Cobertura e alocação (estimativos)")
                    if qty_shares > 0:
                        lots_covered = qty_shares // contract_size
                        st.text(f"Lotes cobertos (por ações em carteira): {int(lots_covered)}")
                    if not np.isnan(cash_avail):
                        st.text(f"Caixa disponível estimado: R$ {cash_avail:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

                    if usar_ic and not (np.isnan(Kp_w) or np.isnan(Kc_w)):
                        colm4, colm5 = st.columns(2)
                        with colm4:
                            st.text(f"Asa (PUT): {Kp_w:.2f} — Custo asas: R$ {custo_asas:.2f}")
                        with colm5:
                            st.text(f"Asa (CALL): {Kc_w:.2f}")
                    if usar_jl:
                        st.text(f"Sem risco de alta? {'Sim' if sem_risco_alta else 'Não'}")

                st.markdown("#### 📘 Explicações e fórmulas")
                st.markdown(
                    f"- **Strangle**: vender PUT (Kp={Kp:.2f}) e CALL (Kc={Kc:.2f}). "
                    f"Lucro = crédito se **S** ficar entre os strikes.\n"
                    f"- **Iron Condor**: Strangle + compra das 'asas' (Kp_w={Kp_w if 'Kp_w' in locals() and not np.isnan(Kp_w) else '—'}, "
                    f"Kc_w={Kc_w if 'Kc_w' in locals() and not np.isnan(Kc_w) else '—'}) → limita a perda máxima.\n"
                    f"- **Jade Lizard**: PUT vendida + CALL vendida + CALL longa (Kc_w). "
                    f"Se **crédito ≥ (Kc_w − Kc)**, não há risco na alta.\n\n"
                    "P/L por ação (vencimento):\n"
                    "- Strangle: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + crédito.\n"
                    "- Iron Condor = Strangle + max(0, Kp_w − S) + max(0, S − Kc_w) − custo_das_asas.\n"
                    "- Jade Lizard = −max(0, Kp − S) − max(0, S − Kc) + max(0, S − Kc_w) + crédito_líquido."
                )
else:
    st.info("Cole a **tabela completa** do opcoes.net acima (incluindo CALLs e PUTs). Dica: selecione a tabela no site, **CTRL/CMD+C** e depois **CTRL/CMD+V** aqui.")

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
def _br_to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace("\xa0", " ")
    neg = s.startswith("-")
    s = s.replace("+", "").replace("-", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
        return -v if neg else v
    except:
        return np.nan

def _parse_date_br(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    try:
        d = dtparse(s, dayfirst=True).date()
        return d.isoformat()
    except:
        return None

def _guess_sep_from_text(text):
    return "\t" if "\t" in text else r"\s{2,}"

def _normalize_header(cols):
    mapping = {
        "ticker": "symbol",
        "símbolo": "symbol",
        "vencimento": "expiration",
        "dias úteis": "bdays",
        "tipo": "type",
        "strike": "strike",
        "último": "last",
        "ult.": "last",
        "bid": "bid",
        "ask": "ask",
        "vol. impl. (%)": "impliedVol",
        "vol. impl.": "impliedVol",
        "delta": "delta",
        "gamma": "gamma",
        "vega": "vega",
        "theta ($)": "theta",
        "theta (%)": "theta_pct",
        "a/i/otm": "moneyness",
        "dist. (%) do strike": "dist_strike",
        "f.m.": "fm",
        "mod.": "mod",
        "nome": "name",
        "empresa": "name",
        "razão social": "name",
    }
    out = []
    for c in cols:
        key = str(c).strip().lower()
        key = re.sub(r"\s+", " ", key)
        out.append(mapping.get(key, c))
    return out

def _clean_dataframe(df_raw):
    df = df_raw.copy()
    df.columns = _normalize_header(df.columns)

    for col in ["symbol","type","strike","last","impliedVol","delta","expiration"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["strike","last","impliedVol","delta","bid","ask"]:
        if col in df.columns:
            df[col] = df[col].map(_br_to_float)

    df["type"] = df["type"].astype(str).str.upper().str.replace("Ç","C").str.strip()

    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].astype(str).map(_parse_date_br)

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()

    if "last" in df.columns and df["last"].isna().all():
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
            prem_call = rc["last"] if pd.notna(rc["last"]) else 0.0
            prem_put  = rp["last"] if pd.notna(rp["last"]) else 0.0
            credito   = (prem_call or 0.0) + (prem_put or 0.0)
            if credito <= 0 or pd.isna(credito):
                continue
            Kc = rc["strike"]; Kp = rp["strike"]
            be_low  = Kp - credito
            be_high = Kc + credito
            item = {
                "PUT": rp["symbol"], "CALL": rc["symbol"],
                "K_put": Kp, "K_call": Kc,
                "Prêmio PUT": prem_put, "Prêmio CALL": prem_call,
                "Crédito (R$)": f"R$ {credito:,.2f}".replace(",", "X").replace(".", ",").replace("X","."),
                "Crédito num": credito,
                "BE_inferior": be_low, "BE_superior": be_high,
                "Δ_put": rp["abs_delta"], "Δ_call": rc["abs_delta"],
                "PoE_put": f"{(abs(rp['delta']*100) if pd.notna(rp['delta']) else np.nan):.1f}%",
                "PoE_call": f"{(abs(rc['delta']*100) if pd.notna(rc['delta']) else np.nan):.1f}%",
                "expiration": rp["expiration"],
            }
            out.append(item)
    return out

def _format_money(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

# ========================= B3 (ticker + empresa) e Yahoo =========================
@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_b3_universe():
    """
    Lê listas da Wikipédia PT e retorna DataFrame com colunas:
    ['ticker', 'name', 'label'] (ticker sem .SA), sem duplicatas.
    """
    urls = [
        "https://pt.wikipedia.org/wiki/Lista_de_companhias_listadas_na_B3",
        "https://pt.wikipedia.org/wiki/Empresas_listadas_na_B3",
    ]
    records = []
    seen = set()
    for url in urls:
        try:
            html = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text
            tables = pd.read_html(html)
        except Exception:
            continue
        for tb in tables:
            tbc = tb.copy()
            tbc.columns = _normalize_header([str(c) for c in tbc.columns])
            cols = list(tbc.columns)

            tick_cols = [c for c in cols if re.search(r"(symbol|ticker|c[oó]digo|negocia)", c, flags=re.I)]
            name_cols = [c for c in cols if re.search(r"(name|empresa|raz[aã]o)", c, flags=re.I)]
            if not tick_cols:
                continue
            tcol = tick_cols[0]
            ncol = name_cols[0] if name_cols else None

            ser_tk = tbc[tcol].astype(str).str.upper().str.strip()
            ser_tk = ser_tk.str.replace(r"[^A-Z0-9]", "", regex=True)
            ser_tk = ser_tk[ser_tk.str.match(r"^[A-Z]{4}[0-9]{1,2}$")]

            ser_nm = (tbc[ncol].astype(str).str.strip() if ncol and ncol in tbc.columns
                      else pd.Series([""] * len(ser_tk), index=ser_tk.index))

            for idx in ser_tk.index:
                tk = ser_tk.loc[idx]
                nm = ser_nm.loc[idx] if idx in ser_nm.index else ""
                if tk not in seen:
                    seen.add(tk)
                    records.append({"ticker": tk, "name": nm})

    if not records:
        # fallback mínimo
        records = [
            {"ticker":"PETR4","name":"Petrobras"},
            {"ticker":"VALE3","name":"Vale"},
            {"ticker":"ITUB4","name":"Itaú Unibanco"},
            {"ticker":"BBDC4","name":"Bradesco"},
            {"ticker":"ABEV3","name":"Ambev"},
            {"ticker":"BBAS3","name":"Banco do Brasil"},
            {"ticker":"WEGE3","name":"WEG"},
        ]
    dfu = pd.DataFrame(records).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    dfu["name"] = dfu["name"].fillna("").astype(str)
    dfu["label"] = np.where(
        dfu["name"].str.len()>0,
        dfu["ticker"] + " — " + dfu["name"],
        dfu["ticker"]
    )
    dfu = dfu.sort_values("ticker").reset_index(drop=True)
    return dfu

def yahoo_price_b3(ticker_no_suffix: str) -> float | None:
    tk = yf.Ticker(f"{ticker_no_suffix}.SA")
    last = None
    try:
        fi = getattr(tk, "fast_info", None)
        if fi:
            last = fi.get("last_price") or fi.get("last_price_raw")
    except Exception:
        last = None
    if last is None:
        try:
            dfh = tk.history(period="5d", interval="1d")
            if not dfh.empty:
                last = float(dfh["Close"].iloc[-1])
        except Exception:
            last = None
    if last is None:
        try:
            dfi = tk.history(period="1d", interval="1m")
            if not dfi.empty:
                last = float(dfi["Close"].dropna().iloc[-1])
        except Exception:
            last = None
    return float(last) if last is not None else None

# ========================= Header =========================
st.title("💼 Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)")
st.write("Cole a option chain do **opcoes.net**, escolha o vencimento e veja as **sugestões didáticas** de strangle coberto e a **comparação de estratégias**.")

df_b3 = fetch_b3_universe()

colA, colB = st.columns([1.6,1.0])
with colA:
    st.markdown("#### 🔎 Escolha um ticker da B3 (pesquise por **nome** ou **código**)")
    options = df_b3["label"].tolist() if df_b3 is not None and not df_b3.empty else []
    # índice seguro
    default_idx = 0
    if options:
        try:
            if "PETR4" in df_b3["ticker"].values:
                default_idx = int(np.where(df_b3["ticker"].values == "PETR4")[0][0])
            else:
                default_idx = 0
        except Exception:
            default_idx = 0
        default_idx = min(max(default_idx, 0), len(options)-1)
        sel_label = st.selectbox(
            "Pesquisar",
            options=options,
            index=default_idx,
            placeholder="Digite 'Bradesco', 'Petrobras', etc.",
            key="ticker_select"
        )
        sel_row = df_b3[df_b3["label"] == sel_label].iloc[0]
        sel = sel_row["ticker"]
    else:
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
hv20_pct = st.sidebar.number_input("HV20 (σ anual – %)", min_value=0.0, max_value=200.0, value=17.12, step=0.10, key="hv20")
r_anual_pct = st.sidebar.number_input("r (anual – %)", min_value=0.0, max_value=100.0, value=11.00, step=0.25, key="r_anual")

st.sidebar.markdown("##### Filtros de moneyness")
mindelta = st.sidebar.number_input("|Δ| mínimo", min_value=0.00, max_value=1.00, step=0.01, value=0.05, key="mindelta",
    help="Filtro de moneyness por |Δ| (aprox. prob. de ficar ITM). Ex.: 0,05 = 5%.")
maxdelta = st.sidebar.number_input("|Δ| máximo", min_value=0.00, max_value=1.00, step=0.01, value=0.35, key="maxdelta",
    help="Filtro superior de |Δ|. Vendedores costumam usar ~0,05–0,35.")

st.sidebar.markdown("##### Bandas de risco por perna")
b_baixo = st.sidebar.slider("Faixa Baixo (0–X%)", min_value=0, max_value=55, value=15, key="band_baixo",
    help="Prob. ITM ≈ |Δ| × 100. Até aqui: **Baixo**.")
b_medio = st.sidebar.slider("Faixa Médio (X–Y%)", min_value=b_baixo, max_value=55, value=35, key="band_medio",
    help="Até aqui: **Médio**. Acima de Y: **Alto**.")
bands_cfg = {"Baixo": (0, b_baixo), "Médio": (b_baixo, b_medio), "Alto": (b_medio, 55)}

st.sidebar.markdown("##### Instruções de SAÍDA — Regras práticas")
dte_alert = st.sidebar.number_input("Dias até vencimento (alerta)", min_value=0, max_value=60, value=7, key="dte_alert")
prox_pct = st.sidebar.number_input("Proximidade ao strike (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="prox_pct")
take_profit = st.sidebar.number_input("Meta de captura do prêmio (%)", min_value=10, max_value=95, value=75, step=5, key="tp_pct")

st.sidebar.markdown("##### Cobertura")
contract_size = st.sidebar.number_input("Tamanho do contrato", min_value=1, max_value=1000, value=100, key="contract_size")
qty_shares = st.sidebar.number_input(lambda: f"Ações em carteira ({sel})", min_value=0, max_value=1_000_000, value=0, key="qty_shares")
cash_avail = st.sidebar.text_input(lambda: f"Caixa disponível (R$) ({sel})", value="10.000,00", key="cash_avail")
cash_avail = _br_to_float(cash_avail)

# ========================= Colar option chain =========================
st.markdown("### 3) Colar a option chain do **opcoes.net** (CTRL/CMD+V)")
raw = st.text_area("Cole aqui a tabela (copie do site e cole aqui)", height=240, key="pastebox")

df = None
if raw.strip():
    sep = _guess_sep_from_text(raw)
    try:
        if sep == "\t":
            df_raw = pd.read_csv(StringIO(raw), sep="\t")
        else:
            df_raw = pd.read_csv(StringIO(re.sub(r"[ ]{2,}", "\t", raw)), sep="\t")
    except Exception:
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        rows = [re.split(r"\t|[ ]{2,}", ln.strip()) for ln in lines]
        if not rows:
            df_raw = pd.DataFrame()
        else:
            header = rows[0]
            body = rows[1:] if len(rows) > 1 else []
            df_raw = pd.DataFrame(body, columns=header)
    if not df_raw.empty:
        df = _clean_dataframe(df_raw)

if df is not None and not df.empty:
    exps = sorted(df["expiration"].dropna().unique().tolist())
    st.markdown("### 📅 Vencimento")
    chosen_exp = st.selectbox("Escolha um vencimento:", options=exps, key="exp_select")

    df_exp = df[df["expiration"] == chosen_exp].copy()

    with st.expander("Ver prévia da cadeia (limpa)", expanded=False):
        st.dataframe(df_exp[["symbol","type","strike","last","impliedVol","delta","expiration"]], height=260, use_container_width=True)

    if pd.isna(spot):
        st.error("Preço à vista (S) não definido. Ative a cotação automática ou informe manualmente.")
    else:
        combos = _pair_strangles(df_exp, spot, mindelta, maxdelta)
        if not combos:
            st.warning("Não há CALL e PUT OTM suficientes com os filtros atuais. Ajuste |Δ| ou escolha outro vencimento.")
        else:
            dfc = pd.DataFrame(combos)

            def band_label(x):
                if pd.isna(x): return "—"
                v = x*100
                if v <= b_baixo: return "Baixo"
                if v <= b_medio: return "Médio"
                return "Alto"

            dfc["Banda_put"]  = dfc["Δ_put"].map(band_label)
            dfc["Banda_call"] = dfc["Δ_call"].map(band_label)

            max_lotes_call = qty_shares // contract_size if contract_size>0 else 0
            dfc["Aloc. PUT (R$) por lote"] = dfc["K_put"] * contract_size
            dfc["Lotes PUT cash-secured"]   = np.floor(cash_avail / dfc["Aloc. PUT (R$) por lote"]).astype(int)
            dfc["Lotes CALL cobertos"]      = max_lotes_call
            dfc["Lotes máx. cobertos"]      = dfc[["Lotes PUT cash-secured","Lotes CALL cobertos"]].min(axis=1)

            dist_rel = ((dfc["K_call"] - spot).abs() + (spot - dfc["K_put"]).abs())/max(spot,1e-6)
            dfc["score"] = dfc["Crédito num"] / (dist_rel.replace(0, np.nan))
            dfc = dfc.sort_values(["Lotes máx. cobertos","score","Crédito num"], ascending=[False, False, False])

            try:
                d_exp = date.fromisoformat(chosen_exp)
                dte = (d_exp - date.today()).days
            except:
                dte = None

            def saida_row(rw):
                kput, kcall = rw["K_put"], rw["K_call"]
                alerta_time = (dte is not None and dte <= dte_alert)
                prox_kput  = abs((spot - kput)/kput)*100 <= prox_pct
                prox_kcall = abs((spot - kcall)/kcall)*100 <= prox_pct
                dicas = []
                if alerta_time:
                    dicas.append(f"⏳ faltam ≤ {dte_alert} dias")
                if prox_kcall:
                    dicas.append("S encostando no **K_call** ⇒ recomprar a CALL")
                if prox_kput:
                    dicas.append("S encostando no **K_put** ⇒ recomprar a PUT")
                dicas.append(f"🎯 capturar ~{take_profit}% do crédito e encerrar")
                return " | ".join(dicas)

            dfc["Obs. saída"] = dfc.apply(saida_row, axis=1)

            # ---- TOP 3 (apresentação mais legível) ----
            top3 = dfc[dfc["Lotes máx. cobertos"]>0].head(3).copy()
            if top3.empty:
                top3 = dfc.head(3).copy()

            st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
            for i, rw in top3.reset_index(drop=True).iterrows():
                credito_lote = rw["Crédito num"] * contract_size
                md = (
                    f"**#{i+1}**\n"
                    f"- **PUT**: {rw['PUT']}  |  **Kₚ**: `{rw['K_put']:.2f}`  |  **PoE**: {rw['PoE_put']}  |  **Banda**: **{rw['Banda_put']}**\n"
                    f"- **CALL**: {rw['CALL']}  |  **K꜀**: `{rw['K_call']:.2f}` |  **PoE**: {rw['PoE_call']} |  **Banda**: **{rw['Banda_call']}**\n"
                    f"- **Crédito por lote**: **{_format_money(credito_lote)}**  "
                    f"(**por ação**: {rw['Crédito (R$)']})\n"
                    f"- **Break-evens**: **[{rw['BE_inferior']:.2f}, {rw['BE_superior']:.2f}]**\n"
                    f"- **Cobertura estimada (máx.)**: PUT cash-secured **{int(rw['Lotes PUT cash-secured'])}** lotes · "
                    f"CALL cobertos **{int(rw['Lotes CALL cobertos'])}** lotes · "
                    f"**{int(rw['Lotes máx. cobertos'])}** lotes combinando\n"
                    f"- **Dica**: {rw['Obs. saída']}"
                )
                st.markdown(md)

            with st.expander("📋 Tabela completa (esta sessão)"):
                show = dfc.copy()
                show["Crédito por lote"] = show["Crédito num"] * contract_size
                cols = [
                    "PUT","K_put","Δ_put","Banda_put","CALL","K_call","Δ_call","Banda_call",
                    "Crédito (R$)","Crédito por lote","BE_inferior","BE_superior",
                    "Lotes PUT cash-secured","Lotes CALL cobertos","Lotes máx. cobertos","Obs. saída"
                ]
                st.dataframe(show[cols], use_container_width=True, height=380)

            # ===== Comparar estratégias =====
            st.markdown("### 📈 Comparar estratégias (Strangle × Iron Condor × Jade Lizard)")
            base = top3.iloc[0] if not top3.empty else dfc.iloc[0]
            Kp, Kc = base["K_put"], base["K_call"]
            credito = base["Crédito num"]
            asas_pct = st.slider("Largura das asas (% do preço à vista)", 2, 15, 8, key="asas")
            asa_abs = (asas_pct/100.0)*spot
            Kp_w = max(0.01, Kp - asa_abs)
            Kc_w = Kc + asa_abs

            perda_max_aprox = max((Kp - Kp_w), (Kc_w - Kc)) - credito
            perda_max_aprox = max(perda_max_aprox, 0)
            sem_risco_alta = credito >= (Kc_w - Kc)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Strangle — Crédito", _format_money(credito))
                st.text(f"Zona neutra (Kp–Kc): {Kp:.2f} — {Kc:.2f}")
            with c2:
                st.metric("Iron Condor — Crédito", _format_money(credito*0.95))
                st.text(f"Asas (P,C): {Kp_w:.2f}, {Kc_w:.2f}")
                st.text(f"Perda máx. aprox.: {_format_money(perda_max_aprox)}")
            with c3:
                st.metric("Jade Lizard — Crédito", _format_money(credito*0.95))
                st.text(f"Asa (CALL): {Kc_w:.2f}")
                st.text(f"Sem risco de alta? {'Sim' if sem_risco_alta else 'Não'}")

            st.markdown("#### 📘 Explicações e fórmulas")
            st.markdown(
                f"- **Strangle**: vender PUT (Kp={Kp:.2f}) + CALL (Kc={Kc:.2f}). Lucro = crédito se **S** ficar entre os strikes.\n"
                f"- **Iron Condor**: Strangle + compra das asas (Kp_w={Kp_w:.2f}, Kc_w={Kc_w:.2f}) → limita a perda máxima.\n"
                f"- **Jade Lizard**: PUT vendida + CALL vendida + CALL comprada (Kc_w). Se **crédito ≥ (Kc_w − Kc)**, não há risco na alta.\n\n"
                "P/L por ação (vencimento):\n"
                "- Strangle: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + crédito.\n"
                "- Iron Condor = Strangle + max(0, Kp_w − S) + max(0, S − Kc_w) − custo_das_asas.\n"
                "- Jade Lizard = −max(0, Kp − S) − max(0, S − Kc) + max(0, S − Kc_w) + crédito_líquido."
            )
else:
    st.info("Cole a **tabela completa** do opcoes.net acima para começar. Dica: clique na tabela no site, **CTRL/CMD+C** e depois **CTRL/CMD+V** aqui.")

# app_v9_paste_only.py
# v9 — Somente "copiar e colar" do opcoes.net (sem upload)
# Correções importantes:
#  - Normalização de 'type': se vier A/E (ou qualquer coisa ≠ CALL/PUT), zera p/ NaN e aplica heurística (strike vs spot)
#  - Modo diagnóstico avançado: escondido por padrão (toggle)
#  - Mantém comparador (Strangle × Iron Condor × Jade Lizard) e regras didáticas

import io, re, unicodedata, math
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests

st.set_page_config(page_title="Strangle Vendido Coberto — v9", page_icon="💼", layout="wide")
st.title("💼 Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)")
st.caption("Cole a tabela do opcoes.net, escolha o vencimento e veja as **sugestões de strangle coberto** e a **comparação de estratégias**.")

CALL_SERIES = set(list("ABCDEFGHIJKL"))
PUT_SERIES  = set(list("MNOPQRSTUVWX"))
SQRT_2 = np.sqrt(2.0)

# -------------------- Utils --------------------
def _strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _norm_colname(c: str) -> str:
    c0 = str(c).strip()
    c1 = re.sub(r'\s+', ' ', c0)
    c2 = _strip_accents(c1).lower()
    return c2

def norm_cdf(x: float) -> float:
    try:
        return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))
    except Exception:
        return float("nan")

def yearfrac(start: date, end: date) -> float:
    return max(1e-9, (end - start).days / 365.0)

def bs_d1(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_d2(d1, sigma, T):
    if np.isnan(d1) or sigma <= 0 or T <= 0:
        return np.nan
    return d1 - sigma * np.sqrt(T)

def call_delta(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T)
    if np.isnan(d1): return np.nan
    return norm_cdf(d1)

def put_delta(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T)
    if np.isnan(d1): return np.nan
    return norm_cdf(d1) - 1.0

def prob_ITM_call(S, K, r, sigma, T):
    d1 = bs_d1(S, K, r, sigma, T); d2 = bs_d2(d1, sigma, T)
    return norm_cdf(d2) if not np.isnan(d2) else np.nan

def prob_ITM_put(S, K, r, sigma, T):
    p_call = prob_ITM_call(S, K, r, sigma, T)
    return (1.0 - p_call) if not np.isnan(p_call) else np.nan

def _br_to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "-", "—"):
        return np.nan
    s = s.replace("\xa0", " ").replace("\u202f", " ")
    s_clean = re.sub(r"[^0-9,.\-eE]", "", s)
    if "," in s_clean:
        s_clean = s_clean.replace(".", "")
        s_clean = s_clean.replace(",", ".")
        try: return float(s_clean)
        except: return np.nan
    else:
        try: return float(s_clean)
        except:
            s2 = s_clean.replace(".", "")
            try: return float(s2)
            except: return np.nan

def _excel_serial_to_date(n):
    try:
        n = float(n)
        if n <= 0: return np.nan
        base = datetime(1899, 12, 30)
        return (base + timedelta(days=int(n))).date()
    except Exception:
        return np.nan

def _parse_date_any(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (pd.Timestamp, datetime)): return x.date()
    if isinstance(x, date): return x
    if isinstance(x, (int, float)) and not np.isnan(x):
        d = _excel_serial_to_date(x)
        if isinstance(d, date): return d
    s = str(x).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(x, dayfirst=True).date()
    except Exception:
        return np.nan

# -------------------- Tickers B3 --------------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_b3_tickers():
    url = "https://www.dadosdemercado.com.br/acoes"
    try:
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    tickers = []
    for tb in soup.find_all("table"):
        head = tb.find("thead")
        if not head: continue
        ths = [th.get_text(strip=True).lower() for th in head.find_all("th")]
        if not ths: continue
        header_text = " ".join(ths)
        if (("código" in header_text or "codigo" in header_text or "ticker" in header_text)
            and ("ação" in header_text or "empresa" in header_text or "nome" in header_text)):
            body = tb.find("tbody")
            if body:
                for tr in body.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 2: continue
                    tk = tds[0].get_text(strip=True).upper()
                    nm = tds[1].get_text(strip=True)
                    if re.match(r"^[A-Z]{4}\d$", tk):
                        tickers.append((tk, nm))
            break
    if not tickers:
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/acoes/([A-Z]{4}\d)$", a["href"])
            if m:
                tk = m.group(1)
                nm = a.get_text(strip=True) or tk
                tickers.append((tk, nm))
    return sorted({tk: nm for tk, nm in tickers}.items(), key=lambda x: x[0])

tickers_list = fetch_b3_tickers()
if not tickers_list:
    st.warning("Não consegui carregar a lista de tickers. Você pode digitar manualmente.")

col_tk1, col_tk2 = st.columns([2,1])
with col_tk1:
    if tickers_list:
        labels = [f"{tk} — {nm}" for tk,nm in tickers_list]
        pick_label = st.selectbox("🔎 Escolha um ticker da B3", options=labels, index=0)
        TICKER = pick_label.split(" — ")[0]
    else:
        TICKER = st.text_input("Digite o ticker (ex.: PETR4, VALE3, ITUB4)", value="PETR4").strip().upper()
with col_tk2:
    st.metric("Ticker", TICKER or "—")
if not TICKER:
    st.stop()

# -------------------- Sidebar (parâmetros) --------------------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    r = st.number_input("Taxa livre de risco anual (r)", 0.0, 1.0, 0.11, 0.005, format="%.3f",
                        help="Usada no Black–Scholes. No Brasil, aproxime pela SELIC anualizada.")
    delta_min = st.number_input("|Δ| mínimo", 0.0, 1.0, 0.00, 0.01,
                                help="Filtro de moneyness por Delta. Ex.: 0,05–0,35 para OTM.")
    delta_max = st.number_input("|Δ| máximo", 0.0, 1.0, 0.35, 0.01,
                                help="Valor máximo de |Δ| (mais OTM).")
    risk_selection = st.multiselect("Bandas de risco por perna", ["Baixo","Médio","Alto"],
                                    default=["Baixo","Médio","Alto"],
                                    help="PoE por perna: Baixo 0–15%, Médio 15–35%, Alto 35–55%.")
    lookback_days = st.number_input("Janela p/ IV Rank/Percentil (dias)", 60, 500, 252, 1,
                                    help="Compara a IV atual vs histórico (usa HV20 se IV faltar).")
    st.markdown("---")
    show_exit_help = st.checkbox("Mostrar instruções de SAÍDA", value=True,
                                 help="Recomprar perna ameaçada perto do vencimento ou encerrar após capturar 70–80% do prêmio.")
    days_exit_thresh = st.number_input("Dias até vencimento p/ alerta", 1, 30, 10,
                                       help="Com ≤ N dias, alertas mais proativos.")
    prox_pct = st.number_input("Proximidade ao strike (%)", 1, 20, 5,
                               help="Marca strike ‘ameaçado’ quando S está a menos de X% dele.") / 100.0
    capture_target = st.number_input("Meta de captura do prêmio (%)", 10, 95, 70,
                                     help="Encerrar com ganho parcial (ex.: 70% capturado).") / 100.0
    st.markdown("---")
    ADV = st.toggle("🔬 Mostrar diagnóstico avançado", value=False)

# -------------------- Spot/HV20 --------------------
@st.cache_data(show_spinner=False)
def load_spot_and_hv20(yahoo_ticker: str):
    try:
        y = yf.Ticker(yahoo_ticker)
        hist = y.history(period="2y")
        if len(hist) >= 30:
            spot_val = float(hist["Close"].iloc[-1])
            rets = hist["Close"].pct_change()
            hv20 = (rets.rolling(20).std() * np.sqrt(252)).dropna()
            return float(spot_val), float(hv20.iloc[-1])
    except Exception:
        pass
    return np.nan, np.nan

spot, hv20 = load_spot_and_hv20(f"{TICKER}.SA")
c1,c2,c3 = st.columns(3)
with c1: st.metric("Preço à vista (S)", f"{spot:,.2f}" if not np.isnan(spot) else "—")
with c2: st.metric("HV20 (σ anual – proxy)", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
with c3: st.metric("r (anual)", f"{r:.2%}")
if np.isnan(spot) or spot <= 0:
    st.error("Não consegui obter o preço à vista. Verifique o ticker.")
    st.stop()

# -------------------- ENTRADA: COLAR --------------------
st.markdown(f"### 3) Colar a option chain de **{TICKER}** (opcoes.net)")
raw_text = st.text_area(
    "Cole aqui a **tabela** (Ctrl/Cmd+C no site → Ctrl/Cmd+V aqui)",
    height=240,
    help="Pode colar **HTML (<table>)**, **CSV** (; , \\t) ou **texto tabular**."
)

def read_pasted_table(text: str) -> pd.DataFrame:
    raw = text.strip()
    if raw == "":
        raise ValueError("Texto vazio.")
    # 1) HTML
    if "<table" in raw.lower() and "</table>" in raw.lower():
        try:
            dfs = pd.read_html(raw)
            df = max(dfs, key=lambda d: d.shape[1])
            return df.dropna(how="all")
        except Exception:
            pass
    # 2) CSV/TSV
    lines = [ln for ln in raw.splitlines() if ln.strip()][:20]
    cand = {";":0, ",":0, "\t":0, "|":0}
    for ln in lines:
        for sep in cand.keys():
            cand[sep] += ln.count(sep)
    sep = max(cand, key=cand.get)
    try:
        df = pd.read_csv(io.StringIO(raw), sep=sep)
        if df.shape[1] >= 2:
            return df.dropna(how="all")
    except Exception:
        pass
    # 3) Espaços múltiplos
    try:
        df = pd.read_csv(io.StringIO(raw), sep=r"\s{2,}", engine="python")
        if df.shape[1] >= 2:
            return df.dropna(how="all")
    except Exception:
        pass
    # 4) Fallback: largura fixa
    rows = [re.split(r"\s{2,}|\t|\|", ln.strip()) for ln in raw.splitlines() if ln.strip()]
    width = max(len(r) for r in rows)
    rows = [r + [""]*(width-len(r)) for r in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.dropna(how="all")

if not raw_text.strip():
    st.info("👉 Cole a tabela para continuar.")
    st.stop()

try:
    df_raw = read_pasted_table(raw_text)
    st.success(f"Tabela colada reconhecida com {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas.")
    if ADV: st.dataframe(df_raw.head())
except Exception as e:
    st.error(f"Não consegui entender o texto colado: {e}")
    st.stop()

# -------------------- Normalização da chain --------------------
def normalize_opcoesnet_df(df: pd.DataFrame) -> pd.DataFrame:
    rev_map = {_norm_colname(c): c for c in df.columns}
    aliases = {
        "symbol": ["ticker", "codigo", "código", "opcao", "opção", "simbolo", "símbolo"],
        "expiration": ["vencimento", "venc", "expiracao", "expiração", "expiry"],
        "type": ["tipo", "class", "opcao_tipo"],
        "strike": ["strike", "preco de exercicio", "preço de exercicio", "preço de exercício", "exercicio", "k"],
        "last": ["ultimo", "último", "preco", "preço", "fechamento", "close", "ultimo negocio", "último negocio"],
        "bid": ["bid", "compra", "melhor compra", "oferta de compra"],
        "ask": ["ask", "venda", "melhor venda", "oferta de venda"],
        "impliedVol": ["vol impl (%)", "vol impl. (%)", "volatilidade implicita", "volatilidade implícita", "iv", "iv (%)"],
        "delta": ["delta", "Δ"],
    }
    def find_col(alias_list):
        for al in alias_list:
            norm = _norm_colname(al)
            if norm in rev_map: return rev_map[norm]
        for al in alias_list:
            norm_al = re.sub(r'[^a-z0-9%$ ]','', _norm_colname(al))
            for norm_col, orig in rev_map.items():
                if re.sub(r'[^a-z0-9%$ ]','', norm_col) == norm_al:
                    return orig
        return None

    out = pd.DataFrame(index=df.index)
    c_symbol = find_col(aliases["symbol"])
    if c_symbol is not None: out["symbol"] = df[c_symbol]

    c_type = find_col(aliases["type"])
    if c_type is not None:
        tnorm = df[c_type].astype(str).str.upper().str.strip()
        # Mapeia CALL/PUT; qualquer outro valor vira NaN (para acionar heurística)
        mapped = tnorm.replace({"CALL":"C","COMPRA":"C","C":"C","PUT":"P","VENDA":"P","P":"P"})
        mapped = mapped.where(mapped.isin(["C","P"]), np.nan)
        out["type"] = mapped
    else:
        out["type"] = np.nan

    # fallback do tipo pelo código (letra da série), se ainda estiver NaN
    if ("type" not in out.columns) or out["type"].isna().all():
        if "symbol" in out.columns:
            def infer_from_symbol(code: str):
                if not isinstance(code, str): return np.nan
                s = re.sub(r'[^A-Z0-9]', '', code.upper())
                m = re.search(r'([A-Z])\d+$', s)
                if not m: return np.nan
                letra = m.group(1)
                if letra in CALL_SERIES: return 'C'
                if letra in PUT_SERIES:  return 'P'
                return np.nan
            out["type"] = out.get("type", pd.Series(np.nan, index=out.index))
            out["type"] = out["type"].fillna(out["symbol"].map(infer_from_symbol))

    # strike
    c_strike = find_col(aliases["strike"])
    out["strike"] = pd.to_numeric(df[c_strike].map(_br_to_float), errors="coerce") if c_strike else np.nan

    # preços → mid
    c_bid = find_col(aliases["bid"])
    c_ask = find_col(aliases["ask"])
    c_last = find_col(aliases["last"])
    if c_bid is not None: out["bid"] = pd.to_numeric(df[c_bid].map(_br_to_float), errors="coerce")
    if c_ask is not None: out["ask"] = pd.to_numeric(df[c_ask].map(_br_to_float), errors="coerce")
    if c_last is not None: out["last"] = pd.to_numeric(df[c_last].map(_br_to_float), errors="coerce")
    bid = out["bid"] if "bid" in out.columns else pd.Series(np.nan, index=df.index)
    ask = out["ask"] if "ask" in out.columns else pd.Series(np.nan, index=df.index)
    last= out["last"] if "last" in out.columns else pd.Series(np.nan, index=df.index)
    has_quote = bid.notna() | ask.notna()
    out["mid"] = np.where(has_quote, (bid.fillna(0)+ask.fillna(0))/2.0, last)

    # IV / Δ originais (se vierem)
    c_iv = find_col(aliases["impliedVol"])
    if c_iv is not None:
        out["impliedVol"] = pd.to_numeric(df[c_iv].map(lambda v: _br_to_float(v)/100.0), errors="coerce")
    c_delta = find_col(aliases["delta"])
    if c_delta is not None:
        out["delta"] = pd.to_numeric(df[c_delta].map(_br_to_float), errors="coerce")

    # vencimento
    c_exp = find_col(aliases["expiration"])
    exp_series = df[c_exp] if c_exp is not None else (df.iloc[:,1] if df.shape[1]>=2 else None)
    if exp_series is not None:
        parsed = exp_series.map(_parse_date_any)
        if parsed.isna().mean() > 0.9:
            parsed = pd.to_datetime(exp_series.astype(str).str.strip(), dayfirst=True, errors="coerce").dt.date
        out["expiration"] = parsed
    else:
        out["expiration"] = np.nan

    # diagnóstico opcional
    if ADV:
        with st.expander("🛠️ Diagnóstico de leitura"):
            st.write("Colunas padronizadas:", list(out.columns))
            if "type" in out.columns:
                st.write("Contagem por tipo:", out["type"].value_counts(dropna=False))
            if "strike" in out.columns:
                ex = out["strike"].dropna().astype(float)
                st.write("Qtde de strikes válidos:", int(ex.shape[0]))
                st.write("Exemplo strikes:", ex.head(10).tolist())
            if "expiration" in out.columns:
                st.write("Exemplo vencimentos:", out["expiration"].dropna().astype(str).head(5).tolist())
    return out

chain_all = normalize_opcoesnet_df(df_raw)

# -------------------- Escolha de vencimento --------------------
valid_exps = sorted([d if isinstance(d, date) else (d.date() if isinstance(d, datetime) else None)
                     for d in chain_all["expiration"].dropna().unique().tolist()])
valid_exps = [d for d in valid_exps if isinstance(d, date)]
if not valid_exps:
    st.error("Nenhum vencimento válido encontrado. Garanta que ‘Vencimento’ está como data (ex.: 17/10/2025).")
    st.stop()

st.markdown("#### 📅 Vencimento")
exp_choice = st.selectbox("Escolha um vencimento:", options=valid_exps, index=0)

def yearfrac_days(to_date: date):
    return yearfrac(date.today(), to_date), (to_date - date.today()).days

# -------------------- Enriquecimento (Δ/IV) com HEURÍSTICA sempre que preciso --------------------
def enrich_chain(df: pd.DataFrame, spot: float, r: float, hv20: float, exp_date: date):
    T, _ = yearfrac_days(exp_date)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Se 'type' tiver valores que NÃO são C/P, zera p/ NaN (força heurística)
    if "type" in df.columns:
        df.loc[~df["type"].isin(["C","P"]), "type"] = np.nan

    # Heurística se faltar C/P (ou parte deles)
    need_type = df["type"].isna() if "type" in df.columns else pd.Series(True, index=df.index)
    if need_type.any():
        guess = pd.Series(np.nan, index=df.index, dtype="object")
        guess.loc[(df["strike"] > spot)] = "C"
        guess.loc[(df["strike"] < spot)] = "P"
        df["type"] = df.get("type", guess).fillna(guess)

    # IV efetiva
    if "impliedVol" not in df.columns:
        df["impliedVol"] = np.nan
    df["iv_eff"] = df["impliedVol"]
    if df["iv_eff"].isna().all() or (df["iv_eff"]<=0).all():
        df["iv_eff"] = hv20

    # Delta: calcula se faltar (agora com 'type' garantido como C/P)
    if "delta" not in df.columns:
        df["delta"] = np.nan
    need = df["delta"].isna()
    if need.any():
        vals = []
        for _, row in df.loc[need].iterrows():
            K = float(row["strike"]) if not pd.isna(row["strike"]) else np.nan
            sigma = float(row.get("iv_eff", np.nan)) if not np.isnan(row.get("iv_eff", np.nan)) else hv20
            sigma = max(sigma, 1e-6)
            t = str(row.get("type"))
            if t == "C" and not np.isnan(K):
                d = call_delta(spot, K, r, sigma, T)
            elif t == "P" and not np.isnan(K):
                d = put_delta(spot, K, r, sigma, T)
            else:
                d = np.nan
            vals.append(d)
        df.loc[need, "delta"] = vals

    return df, T

# -------------------- Filtros de lado --------------------
def side_filters(df: pd.DataFrame, spot: float, delta_min: float, delta_max: float):
    calls = df[(df["strike"]>spot)].copy()
    puts  = df[(df["strike"]<spot)].copy()
    # respeita 'type' se existir
    if "type" in df.columns:
        calls = calls[calls["type"]=="C"]
        puts  = puts[puts["type"]=="P"]
    # Janela de delta
    for dfx in (calls, puts):
        if "delta" in dfx.columns:
            dfx["abs_delta"] = dfx["delta"].abs()
            dfx = dfx[(dfx["abs_delta"]>=delta_min) & (dfx["abs_delta"]<=delta_max)]
        # preço disponível
        if "mid" in dfx.columns:
            dfx = dfx[dfx["mid"].notna()]
        yield dfx

# -------------------- Montagem --------------------
def try_strangles_for_exp(df_exp: pd.DataFrame, spot: float, r: float, hv20: float, exp_date: date,
                          delta_min: float, delta_max: float, shares_owned: int, cash_available: float, lot_size: int):
    dfE, T = enrich_chain(df_exp.copy(), spot, r, hv20, exp_date)
    calls, puts = list(side_filters(dfE, spot, delta_min, delta_max))

    if ADV:
        with st.expander(f"🔎 Diagnóstico de filtros (OTM) — {exp_date}"):
            st.write(f"Calls OTM: {len(calls)} | Puts OTM: {len(puts)}")
            if not calls.empty: st.write("Exemplo calls:", calls[["strike","mid","delta"]].head())
            if not puts.empty:  st.write("Exemplo puts:",  puts[["strike","mid","delta"]].head())

    if calls.empty or puts.empty:
        return pd.DataFrame()

    max_qty_call = (shares_owned // lot_size) if lot_size > 0 else 0
    def max_qty_put_for_strike(Kp: float) -> int:
        if lot_size <= 0 or Kp <= 0: return 0
        return int(cash_available // (Kp * lot_size))

    def prob_side(row, side):
        K = float(row["strike"])
        sigma = float(row.get("iv_eff", np.nan)) if not np.isnan(row.get("iv_eff", np.nan)) else hv20
        sigma = max(sigma, 1e-6)
        return prob_ITM_call(spot, K, r, sigma, T) if side=="C" else prob_ITM_put(spot, K, r, sigma, T)

    calls["prob_ITM"] = [prob_side(rw, "C") for _, rw in calls.iterrows()]
    puts["prob_ITM"]  = [prob_side(rw, "P") for _, rw in puts.iterrows()]
    for dfx in (calls, puts):
        mask = dfx["prob_ITM"].isna() & dfx.get("delta", pd.Series(np.nan, index=dfx.index)).notna()
        dfx.loc[mask, "prob_ITM"] = dfx.loc[mask, "delta"].abs()

    combos = []
    for _, c in calls.iterrows():
        for _, p in puts.iterrows():
            Kc, Kp = float(c["strike"]), float(p["strike"])
            if not (Kp < spot < Kc): continue
            c_mid = float(c.get("mid", np.nan)); p_mid = float(p.get("mid", np.nan))
            if np.isnan(c_mid) or np.isnan(p_mid): continue
            mid_credit = c_mid + p_mid
            qty_call_cap = max_qty_call
            qty_put_cap  = max_qty_put_for_strike(Kp)
            qty = min(qty_call_cap if qty_call_cap>0 else 0, qty_put_cap if qty_put_cap>0 else 0)
            if qty < 1: continue
            be_low, be_high = Kp - mid_credit, Kc + mid_credit
            probITM_put = float(p.get("prob_ITM", np.nan))
            probITM_call = float(c.get("prob_ITM", np.nan))
            poe_total = (pd.Series([probITM_put]).fillna(0).iloc[0] + pd.Series([probITM_call]).fillna(0).iloc[0])
            poe_total = min(1.0, max(0.0, poe_total))
            poe_inside = max(0.0, 1.0 - poe_total)
            combos.append({
                "ticker": TICKER,
                "call_symbol": c.get("symbol", ""), "put_symbol": p.get("symbol", ""),
                "K_call": Kc, "K_put": Kp,
                "probITM_call": probITM_call, "probITM_put": probITM_put,
                "credit_total_por_contrato": mid_credit,
                "qty": qty,
                "BE_low": be_low, "BE_high": be_high,
                "iv_eff_avg": float(np.nanmean([c.get("iv_eff", np.nan), p.get("iv_eff", np.nan)])),
                "expiration": exp_date, "lot_size": lot_size,
            })
    return pd.DataFrame(combos)

# -------------------- Cobertura --------------------
st.markdown("### 4) Cobertura e tamanho do contrato")
colA, colB, colC = st.columns(3)
with colA:
    shares_owned = st.number_input(f"Ações em carteira ({TICKER})", 0, step=100, value=0,
                                   help="Ações livres para cobrir CALLs (1 lote = ‘Tamanho do contrato’).")
with colB:
    cash_available = st.number_input(f"Caixa disponível (R$) ({TICKER})", 0.0, step=100.0, value=10000.0, format="%.2f",
                                     help="Dinheiro para cobrir a PUT (strike × tamanho do contrato).")
with colC:
    lot_size = st.number_input(f"Tamanho do contrato ({TICKER})", 1, step=1, value=100,
                               help="Na B3, normalmente 100.")

# -------------------- Montagem + fallback de vencimento --------------------
chain_all = chain_all.copy()
valid_exps_sorted = sorted(valid_exps)
try_order = [exp_choice] + [d for d in valid_exps_sorted if d != exp_choice]  # tenta o escolhido primeiro
combo_df = pd.DataFrame()
used_exp = None
for e in try_order:
    tmp = try_strangles_for_exp(
        chain_all[chain_all["expiration"]==e], spot, r, hv20, e,
        delta_min, delta_max, shares_owned, cash_available, lot_size
    )
    if not tmp.empty:
        combo_df = tmp.copy()
        used_exp = e
        break

if combo_df.empty:
    st.warning(
        "Não há strangles possíveis com os filtros/limites atuais **em nenhum vencimento** desta cadeia colada.\n\n"
        "Dicas: aumente |Δ| máximo, verifique se ‘Último’ (ou Bid/Ask) está preenchido, ou cole uma cadeia mais completa."
    )
    st.stop()

if used_exp != exp_choice:
    st.info(f"⚠️ No vencimento **{exp_choice}** não havia CALL e PUT suficientes. "
            f"Usei automaticamente **{used_exp}** (mais próximo com ambos os lados).")

# -------------------- Ranking e exibição --------------------
spot_val = spot
combo_df["retorno_pct"] = combo_df["credit_total_por_contrato"] / spot_val
combo_df["poe_total"] = (combo_df["probITM_call"].fillna(0) + combo_df["probITM_put"].fillna(0)).clip(0,1)
combo_df["poe_inside"] = (1 - combo_df["poe_total"]).clip(0,1)
combo_df["risk_score"] = combo_df["poe_total"] / 2.0
combo_df["score_final"] = combo_df["retorno_pct"] / (combo_df["risk_score"] + 0.01)

st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
top3 = combo_df.sort_values(by=["score_final","credit_total_por_contrato"], ascending=[False,False]).head(3)
display_cols = ["call_symbol","K_call","put_symbol","K_put","credit_total_por_contrato",
                "poe_total","poe_inside","retorno_pct","score_final","qty","BE_low","BE_high"]
st.dataframe(
    top3[display_cols].rename(columns={
        "call_symbol":"CALL","put_symbol":"PUT","credit_total_por_contrato":"Crédito/ação",
        "poe_total":"PoE_total","poe_inside":"PoE_dentro","retorno_pct":"Retorno %","score_final":"Score"
    }).style.format({
        "K_call":"%.2f","K_put":"%.2f","Crédito/ação":"R$ %.2f","PoE_total":"{:.0%}",
        "PoE_dentro":"{:.0%}","Retorno %":"{:.2%}","Score":"{:.2f}",
        "BE_low":"R$ %.2f","BE_high":"R$ %.2f"
    }),
    use_container_width=True
)

st.subheader("📋 Sugestões ranqueadas")
show_cols = ["ticker","call_symbol","K_call","put_symbol","K_put","credit_total_por_contrato",
             "retorno_pct","poe_total","poe_inside","qty","BE_low","BE_high","iv_eff_avg","expiration","score_final"]
st.dataframe(
    combo_df[show_cols].rename(columns={
        "call_symbol":"CALL","put_symbol":"PUT","iv_eff_avg":"IV (média)",
        "expiration":"Vencimento","credit_total_por_contrato":"Crédito/ação",
        "retorno_pct":"Retorno %","poe_total":"PoE_total","poe_inside":"PoE_dentro","score_final":"Score"
    }).style.format({
        "K_call":"%.2f","K_put":"%.2f","Crédito/ação":"R$ %.2f","Retorno %":"{:.2%}",
        "PoE_total":"{:.0%}","PoE_dentro":"{:.0%}","BE_low":"R$ %.2f","BE_high":"R$ %.2f",
        "iv_eff_avg":"{:.0%}","Score":"{:.2f}"
    }),
    use_container_width=True
)

# -------------------- Payoff --------------------
st.markdown("### 📈 Payoff no Vencimento (P/L por ação)")
combo_df["__id__"] = (
    combo_df["K_put"].round(2).astype(str) + "–" + combo_df["K_call"].round(2).astype(str) +
    " | crédito≈" + combo_df["credit_total_por_contrato"].round(2).astype(str)
)
sel = st.selectbox("Estrutura:", options=combo_df["__id__"].tolist())
row = combo_df[combo_df["__id__"]==sel].iloc[0]
Kc, Kp, credit = float(row["K_call"]), float(row["K_put"]), float(row["credit_total_por_contrato"])
S_grid = np.linspace(max(0.01, Kp*0.8), Kc*1.2, 400)
payoff = -np.maximum(0.0, S_grid - Kc) - np.maximum(0.0, Kp - S_grid) + credit
fig = plt.figure()
plt.plot(S_grid, payoff)
plt.axhline(0, linestyle="--"); plt.axvline(Kp, linestyle=":"); plt.axvline(Kc, linestyle=":")
plt.title(f"{TICKER} — Payoff | Kp={Kp:.2f}, Kc={Kc:.2f}, Crédito≈R$ {credit:.2f}/ação")
plt.xlabel("Preço no vencimento (S)"); plt.ylabel("P/L por ação (R$)")
st.pyplot(fig, use_container_width=True)

# -------------------- Comparar estratégias --------------------
def _nearest_strike(df, typ, target, side):
    d = df[df["type"]==typ].copy() if "type" in df.columns else df.copy()
    if side == "above":
        d = d[d["strike"] > target]
        if d.empty: return None
        k = float(d["strike"].min())
    else:
        d = d[d["strike"] < target]
        if d.empty: return None
        k = float(d["strike"].max())
    row = d.loc[d["strike"]==k].iloc[0]
    premium = float(row["mid"]) if "mid" in d.columns else (float(row.get("last", np.nan)) if not pd.isna(row.get("last", np.nan)) else np.nan)
    return k, premium, row.get("symbol", None)

def payoff_arrays_strangle(S, Kp, Kc, credit):
    return -np.maximum(0.0, Kp - S) - np.maximum(0.0, S - Kc) + credit

def payoff_arrays_iron_condor(S, Kp, Kc, Kp_w, Kc_w, credit_net):
    short_strangle = -np.maximum(0.0, Kp - S) - np.maximum(0.0, S - Kc)
    long_wings = +np.maximum(0.0, Kp_w - S) + np.maximum(0.0, S - Kc_w)
    return short_strangle + long_wings + credit_net

def payoff_arrays_jade_lizard(S, Kp, Kc, Kc_w, credit_net):
    short_put = -np.maximum(0.0, Kp - S)
    short_call = -np.maximum(0.0, S - Kc)
    long_call  = +np.maximum(0.0, S - Kc_w)
    return short_put + short_call + long_call + credit_net

with st.expander("📈 Comparar estratégias (Strangle × Iron Condor × Jade Lizard)"):
    tmp = combo_df.copy()
    tmp["__base__"] = (
        tmp["K_put"].round(2).astype(str) + "–" + tmp["K_call"].round(2).astype(str) +
        " | crédito≈" + tmp["credit_total_por_contrato"].round(2).astype(str)
    )
    pick = st.selectbox("Strangle base:", options=tmp["__base__"].tolist())
    rowb = tmp[tmp["__base__"]==pick].iloc[0]
    Kp_b, Kc_b, cred_b = float(rowb["K_put"]), float(rowb["K_call"]), float(rowb["credit_total_por_contrato"])

    wing_pct = st.slider("Largura das asas (% do preço à vista)", 2, 15, 5, 1,
                         help="Distância das asas (PUT comprada e CALL comprada) do strangle base.") / 100.0
    df_used = chain_all[chain_all["expiration"]==rowb["expiration"]].copy()
    df_used, _ = enrich_chain(df_used, spot, r, hv20, rowb["expiration"])

    Kc_target = Kc_b + wing_pct * spot
    Kp_target = Kp_b - wing_pct * spot
    kc_w_tuple = _nearest_strike(df_used, 'C', Kc_target, side='above')
    kp_w_tuple = _nearest_strike(df_used, 'P', Kp_target, side='below')
    if kc_w_tuple is None or kp_w_tuple is None:
        st.warning("Não encontrei strikes para as asas. Aumente a largura.")
    else:
        Kc_w, prem_cw, _ = kc_w_tuple
        Kp_w, prem_pw, _ = kp_w_tuple
        cost_wings = (0.0 if np.isnan(prem_cw) else prem_cw) + (0.0 if np.isnan(prem_pw) else prem_pw)
        credit_condor = cred_b - cost_wings
        credit_jl = cred_b - (0.0 if np.isnan(prem_cw) else prem_cw)

        S_grid2 = np.linspace(max(0.01, Kp_w*0.8), Kc_w*1.2, 500)
        pay_str = payoff_arrays_strangle(S_grid2, Kp_b, Kc_b, cred_b)
        pay_ic  = payoff_arrays_iron_condor(S_grid2, Kp_b, Kc_b, Kp_w, Kc_w, credit_condor)
        pay_jl  = payoff_arrays_jade_lizard(S_grid2, Kp_b, Kc_b, Kc_w, credit_jl)

        colA,colB,colC = st.columns(3)
        with colA:
            st.metric("Strangle — Crédito", f"R$ {cred_b:.2f}")
            st.metric("Zona neutra (Kp–Kc)", f"{Kp_b:.2f} — {Kc_b:.2f}")
            st.metric("PoE ficar dentro", f"{rowb['poe_inside']:.0%}")
        with colB:
            st.metric("Iron Condor — Crédito", f"R$ {credit_condor:.2f}")
            st.metric("Asas (P,C)", f"{Kp_w:.2f}, {Kc_w:.2f}")
            max_loss_ic = max(0.0, (Kp_b - Kp_w) - credit_condor, (Kc_w - Kc_b) - credit_condor)
            st.metric("Perda máx. aprox.", f"R$ {max_loss_ic:.2f}")
        with colC:
            st.metric("Jade Lizard — Crédito", f"R$ {credit_jl:.2f}")
            st.metric("Asa (CALL)", f"{Kc_w:.2f}")
            no_upside = credit_jl >= (Kc_w - Kc_b)
            st.metric("Sem risco de alta?", "Sim" if no_upside else "Não")

        for name, arr in [("Strangle vendido", pay_str), ("Iron Condor", pay_ic), ("Jade Lizard", pay_jl)]:
            fig = plt.figure()
            plt.plot(S_grid2, arr); plt.axhline(0, linestyle="--"); plt.axvline(Kp_b, linestyle=":"); plt.axvline(Kc_b, linestyle=":")
            if name != "Strangle vendido":
                plt.axvline(Kp_w, linestyle=":"); plt.axvline(Kc_w, linestyle=":")
            plt.title(f"{TICKER} — {name}")
            plt.xlabel("Preço do ativo no vencimento (S)"); plt.ylabel("P/L por ação (R$)")
            st.pyplot(fig, use_container_width=True)

        with st.expander("📘 Explicações e fórmulas"):
            st.markdown(f"""
- **Strangle**: vender PUT (Kp={Kp_b:.2f}) + CALL (Kc={Kc_b:.2f}). Lucro = **crédito** se S ficar entre os strikes.  
- **Iron Condor**: Strangle + compra das asas (Kp_w, Kc_w) que **limitam a perda máxima**.  
- **Jade Lizard**: PUT vendida + CALL vendida + CALL comprada (Kc_w); se **crédito ≥ (Kc_w − Kc)**, **sem risco na alta**.

**P/L por ação (vencimento)**  
- Strangle: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + crédito.  
- Iron Condor: Strangle + max(0, Kp_w − S) + max(0, S − Kc_w) − custo_das_asas.  
- Jade Lizard: −max(0, Kp − S) − max(0, S − Kc) + max(0, S − Kc_w) + crédito_líquido.
""")
# -------------------- Fim --------------------

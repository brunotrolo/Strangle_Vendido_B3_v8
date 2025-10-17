# app_v9_b3_file.py
# v9 didático (1 ticker por vez) com:
# - Lista B3 (Dados de Mercado)
# - Upload da chain (Excel/CSV do opcoes.net) com detecção automática de cabeçalho (linha 2 OU linha 1)
# - Mapeamento automático + fallback por POSIÇÃO (coluna B) para "Vencimento" (dd/mm/aaaa)
# - Tooltips (help=) em TODOS os controles da barra lateral
# - Sugestões de strangle + saídas didáticas
# - Aba "📈 Comparar estratégias" (Strangle × Iron Condor × Jade Lizard)

import io
import re
import unicodedata
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

# -------------------------------
# Configuração
# -------------------------------
st.set_page_config(
    page_title="Strangle Vendido Coberto — v9 (B3 + arquivo)",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Strangle Vendido Coberto — v9 (B3 + arquivo)")
st.caption("Escolha um ticker da B3, envie a planilha do opções.net (.xlsx/.csv) e receba sugestões + comparação (Strangle × Iron Condor × Jade Lizard).")

# -------------------------------
# Utilidades
# -------------------------------
CALL_SERIES = set(list("ABCDEFGHIJKL"))
PUT_SERIES  = set(list("MNOPQRSTUVWX"))
SQRT_2 = np.sqrt(2.0)

def _strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + np.math.erf(x / SQRT_2))

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
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    if s in ('', 'nan', '-', '—'): return np.nan
    s = s.replace('.', '').replace(',', '.')
    s = re.sub(r'[^0-9.\-eE]', '', s)
    try: return float(s)
    except: return np.nan

def _excel_serial_to_date(n):
    try:
        n = float(n)
        if n <= 0: return np.nan
        base = datetime(1899, 12, 30)  # pandas/Excel epoch
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

# -------------------------------
# Tickers da B3 (Dados de Mercado)
# -------------------------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_b3_tickers():
    url = "https://www.dadosdemercado.com.br/acoes"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
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
        if ("código" in header_text or "codigo" in header_text or "ticker" in header_text) and ("ação" in header_text or "empresa" in header_text or "nome" in header_text):
            for tr in tb.find("tbody").find_all("tr"):
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
                nm = a.get_text(strip=True)
                tickers.append((tk, nm or tk))
    return sorted({tk: nm for tk, nm in tickers}.items(), key=lambda x: x[0])

tickers_list = fetch_b3_tickers()
if not tickers_list:
    st.warning("Não consegui carregar a lista de tickers do site. Você ainda pode digitar o ticker manualmente.")

# -------------------------------
# Escolha de 1 ticker por vez
# -------------------------------
col_tk1, col_tk2 = st.columns([2,1])
with col_tk1:
    if tickers_list:
        tickers_labels = [f"{tk} — {nm}" for tk, nm in tickers_list]
        pick_label = st.selectbox("🔎 Escolha um ticker da B3", options=tickers_labels, index=0)
        TICKER = pick_label.split(" — ")[0]
    else:
        TICKER = st.text_input("Digite o ticker (ex.: PETR4, VALE3, ITUB4)", value="PETR4").strip().upper()
with col_tk2:
    st.metric("Ticker selecionado", TICKER if TICKER else "—")
if not TICKER: st.stop()

# -------------------------------
# Sidebar — parâmetros com tooltips
# -------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    r = st.number_input(
        "Taxa livre de risco anual (r)",
        min_value=0.0, max_value=1.0, value=0.11, step=0.005, format="%.3f",
        help="Usada no Black–Scholes. No Brasil, aproxime pela SELIC anualizada. Ex.: 0,11 = 11% a.a."
    )
    delta_min = st.number_input(
        "|Δ| mínimo", min_value=0.0, max_value=1.0, value=0.00, step=0.01,
        help="Filtro de ‘moneyness’ por Delta (menor = mais OTM)."
    )
    delta_max = st.number_input(
        "|Δ| máximo", min_value=0.0, max_value=1.0, value=0.35, step=0.01,
        help="Vendedores costumam usar |Δ| ~ 0,05–0,35 (opções OTM)."
    )
    risk_selection = st.multiselect(
        "Bandas de risco por perna", ["Baixo","Médio","Alto"], default=["Baixo","Médio","Alto"],
        help="Classifica pela prob. de exercício (PoE) de cada perna: Baixo 0–15%, Médio 15–35%, Alto 35–55%."
    )
    lookback_days = st.number_input(
        "Janela p/ IV Rank/Percentil (dias)", min_value=60, max_value=500, value=252, step=1,
        help="Compara a IV atual vs histórico (proxy por HV20 se IV faltar). IV alta favorece vender prêmio."
    )
    st.markdown("---")
    show_exit_help = st.checkbox(
        "Mostrar instruções de SAÍDA", value=True,
        help="Recomprar a perna ameaçada perto do vencimento OU encerrar após capturar 70–80% do prêmio."
    )
    days_exit_thresh = st.number_input(
        "Dias até vencimento p/ alerta", min_value=1, max_value=30, value=10,
        help="Com ≤ N dias, as mensagens de saída ficam mais proativas."
    )
    prox_pct = st.number_input(
        "Proximidade ao strike (%)", min_value=1, max_value=20, value=5,
        help="Considera o strike ‘ameaçado’ quando S está a menos de X% dele."
    ) / 100.0
    capture_target = st.number_input(
        "Meta de captura do prêmio (%)", min_value=10, max_value=95, value=70,
        help="Meta para encerrar com ganho parcial (ex.: 70% já capturado → zera o risco)."
    ) / 100.0

# -------------------------------
# Spot e HV20 (proxy IV)
# -------------------------------
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
    st.error("Não foi possível obter o preço à vista (Yahoo). Verifique o ticker.")
    st.stop()

# -------------------------------
# Upload (opcoes.net) — leitura automática
# -------------------------------
st.markdown(f"### 3) Envie a *option chain* do **opcoes.net** (Excel/CSV) para **{TICKER}**")
uploaded = st.file_uploader(
    "Detecção automática do cabeçalho: tenta linha 2 (header=1) e linha 1 (header=0). Vencimento será lido do nome OU da coluna B se necessário.",
    type=["xlsx","xls","csv"]
)
if uploaded is None:
    st.info("👉 Envie o arquivo para continuar.")
    st.stop()

def _auto_read_opcoesnet(file) -> pd.DataFrame:
    """
    Lê .xlsx/.csv do opcoes.net com detecção automática:
    - testa header=1 e header=0
    - busca 'Vencimento' por aliases; se não achar, usa a COLUNA B (iloc[:,1]) como fallback (formato dd/mm/aaaa)
    - normaliza colunas e mapeia aliases
    - calcula 'mid' robusto
    - retorno: DataFrame com index alinhado e colunas: [symbol?, type, strike, bid?, ask?, last?, mid, impliedVol?, delta?, expiration]
    """
    name = file.name.lower()
    file_bytes = file.getvalue()  # lê uma vez

    def try_read(header):
        if name.endswith(".csv"):
            text = file_bytes.decode("utf-8", errors="ignore")
            return pd.read_csv(io.StringIO(text), header=header)
        else:
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", header=header)

    # 1) Ler com header=1, senão header=0
    df = None
    header_used = None
    for h in (1, 0):
        try:
            _df = try_read(h)
            if _df is not None and _df.shape[1] >= 2:
                df = _df.dropna(how="all").copy()
                header_used = h
                break
        except Exception:
            continue

    if df is None:
        raise RuntimeError("Falha ao ler a planilha: tente exportar novamente em .xlsx ou .csv.")

    # Normaliza nomes (original -> normalizado)
    def _clean_cols(cols):
        out = []
        for c in cols:
            c0 = str(c).strip()
            c1 = re.sub(r'\s+', ' ', c0)
            c2 = _strip_accents(c1).lower()
            out.append((c0, c2))
        return out

    norm_pairs = _clean_cols(df.columns)
    rev_map  = {norm: orig for orig, norm in norm_pairs}

    # Aliases tolerantes
    aliases = {
        "symbol": ["ticker", "codigo", "código", "opcao", "opção", "símbolo", "simbolo"],
        "expiration": ["vencimento", "venc", "expiracao", "expiração", "expiry"],
        "bdays": ["dias uteis", "dias úteis", "dias-uteis", "dias_uteis", "dias comerciais"],
        "type": ["tipo", "class", "opcao_tipo"],
        "strike": ["strike", "preco de exercicio", "preço de exercicio", "preço de exercício", "exercicio", "k"],
        "last": ["ultimo", "último", "preco", "preço", "fechamento", "close", "último negocio", "ultimo negocio"],
        "bid": ["bid", "compra", "melhor compra", "oferta de compra"],
        "ask": ["ask", "venda", "melhor venda", "oferta de venda"],
        "impliedVol": ["vol impl (%)", "vol impl. (%)", "volatilidade implicita", "volatilidade implícita", "iv", "iv (%)"],
        "delta": ["delta", "Δ"],
        "dist_pct": ["dist (%) do strike", "dist % do strike", "distancia (%) do strike"],
    }

    def find_col(alias_list):
        # match direto
        for al in alias_list:
            if al in rev_map:
                return rev_map[al]
        # match relaxado
        for al in alias_list:
            for norm, orig in rev_map.items():
                if re.sub(r'[^a-z0-9%$ ]','', norm) == re.sub(r'[^a-z0-9%$ ]','', al):
                    return orig
        return None

    # DataFrame de saída com MESMO índice
    out = pd.DataFrame(index=df.index)

    # symbol (opcional)
    col_symbol = find_col(aliases["symbol"])
    if col_symbol is not None:
        out["symbol"] = df[col_symbol]

    # type
    col_type = find_col(aliases["type"])
    if col_type is not None:
        out["type"] = df[col_type].astype(str).str.upper().str.strip().replace({
            "CALL":"C","COMPRA":"C","C":"C",
            "PUT":"P","VENDA":"P","P":"P"
        })
    else:
        out["type"] = np.nan
        if "symbol" in out.columns:
            def infer_type(code: str):
                if not isinstance(code, str): return np.nan
                s = re.sub(r'[^A-Z0-9]', '', str(code).upper().strip())
                m = re.search(r'([A-Z])\d+$', s)
                if not m: return np.nan
                letra = m.group(1)
                if letra in CALL_SERIES: return 'C'
                if letra in PUT_SERIES:  return 'P'
                return np.nan
            out["type"] = out["symbol"].map(infer_type)

    # strike
    col_strike = find_col(aliases["strike"])
    out["strike"] = pd.to_numeric(df[col_strike].map(_br_to_float), errors="coerce") if col_strike else np.nan

    # preços
    col_bid = find_col(aliases["bid"])
    if col_bid is not None:
        out["bid"] = pd.to_numeric(df[col_bid].map(_br_to_float), errors="coerce")
    col_ask = find_col(aliases["ask"])
    if col_ask is not None:
        out["ask"] = pd.to_numeric(df[col_ask].map(_br_to_float), errors="coerce")
    col_last = find_col(aliases["last"])
    if col_last is not None:
        out["last"] = pd.to_numeric(df[col_last].map(_br_to_float), errors="coerce")

    # IV / Delta
    col_iv = find_col(aliases["impliedVol"])
    if col_iv is not None:
        out["impliedVol"] = pd.to_numeric(df[col_iv].map(lambda v: _br_to_float(v)/100.0), errors="coerce")
    col_delta = find_col(aliases["delta"])
    if col_delta is not None:
        out["delta"] = pd.to_numeric(df[col_delta].map(_br_to_float), errors="coerce")

    # MID robusto
    bid_series  = out["bid"]  if "bid"  in out.columns else pd.Series(np.nan, index=df.index)
    ask_series  = out["ask"]  if "ask"  in out.columns else pd.Series(np.nan, index=df.index)
    last_series = out["last"] if "last" in out.columns else pd.Series(np.nan, index=df.index)
    has_quote = bid_series.notna() | ask_series.notna()
    mid_from_quotes = (bid_series.fillna(0) + ask_series.fillna(0)) / 2.0
    out["mid"] = np.where(has_quote, mid_from_quotes, last_series)

    # expiration (por nome OU fallback por POSIÇÃO — coluna B)
    col_exp = find_col(aliases["expiration"])
    col_bdays = find_col(aliases["bdays"])
    exp_series = None

    if col_exp is not None:
        exp_series = df[col_exp]
    elif col_bdays is not None:
        def _est(d):
            try:
                n = int(_br_to_float(d))
                return (pd.Timestamp(date.today()) + BDay(n)).date()
            except Exception:
                return np.nan
        out["expiration"] = df[col_bdays].map(_est)
    else:
        # 👉 Fallback: usar a COLUNA B (segunda coluna) conforme você informou
        try:
            if df.shape[1] >= 2:
                exp_series = df.iloc[:, 1]  # coluna B
        except Exception:
            exp_series = None

    if exp_series is not None:
        parsed = exp_series.map(_parse_date_any)
        # checagem de qualidade: pelo menos 10% datas válidas
        valid_ratio = parsed.notna().mean() if len(parsed) else 0.0
        if valid_ratio >= 0.1:
            out["expiration"] = parsed
        else:
            # última tentativa: tentar converter com dayfirst à força
            try:
                forced = pd.to_datetime(exp_series.astype(str).str.strip(), dayfirst=True, errors="coerce").dt.date
                out["expiration"] = forced
            except Exception:
                out["expiration"] = np.nan

    if "expiration" not in out.columns:
        out["expiration"] = np.nan

    # IV placeholder (fallback real via HV20 ocorre depois)
    if "impliedVol" not in out.columns:
        out["impliedVol"] = np.nan

    # garantias mínimas
    for c in ["type","strike","expiration"]:
        if c not in out.columns: out[c] = np.nan

    # Debug opcional
    with st.expander("🛠️ Diagnóstico de leitura (opcional)"):
        st.write(f"Header usado: {header_used} (1 = linha 2, 0 = linha 1)")
        st.write("Algumas colunas mapeadas:", list(out.columns))
        st.write("Amostra (5 linhas):")
        st.dataframe(out.head())

    return out

try:
    chain_all = _auto_read_opcoesnet(uploaded)
except Exception as e:
    st.error(f"Falha ao ler o arquivo: {e}")
    st.stop()

# -------------------------------
# Seleção de vencimento
# -------------------------------
valid_exps = sorted([d for d in chain_all["expiration"].dropna().unique().tolist() if isinstance(d, (date, datetime))])
if not valid_exps:
    st.error("Nenhum vencimento válido encontrado. Verifique o arquivo (coluna 'Vencimento' ou a coluna B).")
    st.stop()

exp_choice = st.selectbox("📅 Vencimento", options=valid_exps, index=0)
T = yearfrac(date.today(), exp_choice if isinstance(exp_choice,date) else exp_choice.date())
days_to_exp = (exp_choice - date.today()).days if isinstance(exp_choice,date) else (exp_choice.date() - date.today()).days
df = chain_all[chain_all["expiration"] == exp_choice].copy()

# Preço médio (mid) e IV efetiva
if "impliedVol" not in df.columns:
    df["impliedVol"] = np.nan
df["iv_eff"] = df["impliedVol"]
if df["iv_eff"].isna().all():
    df["iv_eff"] = hv20

# Separa OTM
calls = df[(df["type"]=="C") & (df["strike"]>spot)].copy()
puts  = df[(df["type"]=="P") & (df["strike"]<spot)].copy()

# Completa Deltas se faltar
for side_df, side in [(calls,"C"), (puts,"P")]:
    if "delta" not in side_df.columns:
        side_df["delta"] = np.nan
    need = side_df["delta"].isna()
    if need.any():
        vals = []
        for _, row in side_df.loc[need].iterrows():
            K = float(row["strike"])
            sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
            sigma = max(sigma, 1e-6)
            d = call_delta(spot, K, r, sigma, T) if side=="C" else put_delta(spot, K, r, sigma, T)
            vals.append(d)
        side_df.loc[need, "delta"] = vals

# Filtro |Δ|
def dfilter(dfi: pd.DataFrame) -> pd.DataFrame:
    if "delta" not in dfi.columns: return dfi
    dfo = dfi.copy()
    dfo["abs_delta"] = dfo["delta"].abs()
    dfo = dfo[(dfo["abs_delta"]>=delta_min) & (dfo["abs_delta"]<=delta_max)]
    return dfo

calls = dfilter(calls); puts = dfilter(puts)

# Prob ITM + bandas
def probs(df_side: pd.DataFrame, side: str) -> pd.DataFrame:
    df_side = df_side.copy()
    pr = []
    for _, row in df_side.iterrows():
        K = float(row["strike"]); sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
        sigma = max(sigma, 1e-6)
        p_itm = prob_ITM_call(spot,K,r,sigma,T) if side=="C" else prob_ITM_put(spot,K,r,sigma,T)
        if np.isnan(p_itm) and not np.isnan(row.get("delta", np.nan)):
            p_itm = abs(float(row["delta"]))
        pr.append(p_itm)
    df_side["prob_ITM"] = pr
    return df_side

calls = probs(calls, "C")
puts  = probs(puts, "P")

bands = {"Baixo":(0.00,0.15), "Médio":(0.15,0.35), "Alto":(0.35,0.55)}
def label_band(p):
    if np.isnan(p): return "Fora"
    for k,(a,b) in bands.items():
        if a <= p <= b: return k
    return "Fora"

calls["band"] = calls["prob_ITM"].apply(label_band)
puts["band"]  = puts["prob_ITM"].apply(label_band)
calls = calls[calls["band"].isin(risk_selection)]
puts  = puts[puts["band"].isin(risk_selection)]

# Cobertura por ticker
st.markdown("### 4) Cobertura e tamanho do contrato")
colA, colB, colC = st.columns(3)
with colA:
    shares_owned = st.number_input(f"Ações em carteira ({TICKER})", min_value=0, step=100, value=0,
                                   help="Quantidade de ações livres na carteira para cobrir as CALLs (1 lote = ‘Tamanho do contrato’).")
with colB:
    cash_available = st.number_input(f"Caixa disponível (R$) ({TICKER})", min_value=0.0, step=100.0, value=10000.0, format="%.2f",
                                     help="Dinheiro reservado para cobrir a PUT (compra ao strike × tamanho do contrato).")
with colC:
    lot_size = st.number_input(f"Tamanho do contrato ({TICKER})", min_value=1, step=1, value=100,
                               help="Na B3, ações geralmente 100; use o valor do seu contrato.")

max_qty_call = (shares_owned // lot_size) if lot_size > 0 else 0
def max_qty_put_for_strike(Kp: float) -> int:
    if lot_size <= 0 or Kp <= 0: return 0
    return int(cash_available // (Kp * lot_size))

# Combinação em strangles cobertos
combos = []
for _, c in calls.iterrows():
    for _, p in puts.iterrows():
        Kc, Kp = float(c["strike"]), float(p["strike"])
        if not (Kp < spot < Kc): continue
        mid_credit = float(c["mid"]) + float(p["mid"])
        qty_call_cap = max_qty_call
        qty_put_cap  = max_qty_put_for_strike(Kp)
        qty = min(qty_call_cap if qty_call_cap>0 else 0, qty_put_cap if qty_put_cap>0 else 0)
        if qty < 1: continue
        be_low, be_high = Kp - mid_credit, Kc + mid_credit
        probITM_put = float(p.get("prob_ITM")); probITM_call = float(c.get("prob_ITM"))
        poe_total = min(1.0, max(0.0, probITM_put + probITM_call))
        poe_inside = max(0.0, 1.0 - poe_total)
        combos.append({
            "ticker": TICKER,
            "call_symbol": c.get("symbol", ""), "put_symbol": p.get("symbol", ""),
            "K_call": Kc, "K_put": Kp,
            "band_call": c.get("band"), "band_put": p.get("band"),
            "probITM_call": probITM_call, "probITM_put": probITM_put,
            "poe_total": poe_total, "poe_inside": poe_inside,
            "delta_call": float(c.get("delta", np.nan)), "delta_put": float(p.get("delta", np.nan)),
            "credit_total_por_contrato": mid_credit,
            "qty": qty,
            "BE_low": be_low, "BE_high": be_high,
            "iv_eff_avg": float(np.nanmean([c.get("iv_eff"), p.get("iv_eff")])),
            "expiration": exp_choice, "lot_size": lot_size,
        })

if not combos:
    st.warning("Não há strangles possíveis respeitando a cobertura (ações e caixa) com os filtros atuais.")
    st.stop()

combo_df = pd.DataFrame(combos)
combo_df["retorno_pct"] = combo_df["credit_total_por_contrato"] / spot
combo_df["risk_score"] = (combo_df["probITM_call"] + combo_df["probITM_put"]) / 2.0
combo_df["score_final"] = combo_df["retorno_pct"] / (combo_df["risk_score"] + 0.01)

# Instruções de saída (didáticas)
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
            msg.append("⚠️ Ambos strikes sob pressão perto do vencimento.")
        elif near_call:
            msg.append("⚠️ CALL ameaçada: preço próximo de K_call.")
        elif near_put:
            msg.append("⚠️ PUT ameaçada: preço próximo de K_put.")
        msg.append(f"🕐 Faltam {days_to_exp} d; regra didática: zerar o risco recomprando a perna ameaçada.")
        msg.append("➡️ Ação: compre de volta a opção vendida (CALL ou PUT).")
    else:
        msg.append("✅ Conforto: preço distante dos strikes ou ainda há tempo.")
        msg.append("➡️ Ação: mantenha e monitore. Considere encerrar se capturar boa parte do prêmio.")
    msg.append(f"💰 Meta: encerrar ao capturar {int(capture_target*100)}% do prêmio.")
    msg.append(f"🔧 Zeragem total (~): R$ {target_debit_both:.2f}/ação. Perna: ~ R$ {target_debit_per_leg:.2f}/ação.")
    return "  \n".join(msg), ("⚠️" if (time_critical and (near_call or near_put)) else "✅")

if show_exit_help:
    exit_texts, alerts = [], []
    for _, rrow in combo_df.iterrows():
        text, alert = build_exit_guidance(rrow)
        exit_texts.append(text); alerts.append(alert)
    combo_df["Instrucao_saida"] = exit_texts
    combo_df["Alerta_saida"] = alerts

# Top 3 e tabela
st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
top3 = combo_df.sort_values(by=["score_final","credit_total_por_contrato"], ascending=[False,False]).head(3)
display_cols = ["call_symbol","K_call","put_symbol","K_put","credit_total_por_contrato","poe_total","retorno_pct","score_final","qty","BE_low","BE_high"]
if show_exit_help: display_cols += ["Alerta_saida"]
st.dataframe(top3[display_cols].rename(columns={
    "call_symbol":"CALL","put_symbol":"PUT","credit_total_por_contrato":"Crédito/ação",
    "poe_total":"PoE_total","retorno_pct":"Retorno %","score_final":"Score"
}).style.format({"K_call":"%.2f","K_put":"%.2f","Crédito/ação":"R$ %.2f","PoE_total":"{:.0%}","Retorno %":"{:.2%}","Score":"{:.2f}","BE_low":"R$ %.2f","BE_high":"R$ %.2f"}), use_container_width=True)

st.subheader("📋 Sugestões ranqueadas")
show_cols = ["ticker","call_symbol","K_call","probITM_call","delta_call",
             "put_symbol","K_put","probITM_put","delta_put",
             "credit_total_por_contrato","retorno_pct","poe_total","poe_inside",
             "qty","BE_low","BE_high","iv_eff_avg","expiration","score_final"]
if show_exit_help:
    show_cols += ["Alerta_saida","Instrucao_saida"]
fmt = {"K_call":"%.2f","K_put":"%.2f","probITM_call":"{:.0%}","probITM_put":"{:.0%}",
       "delta_call":"{:.2f}","delta_put":"{:.2f}","credit_total_por_contrato":"R$ {:.2f}",
       "retorno_pct":"{:.2%}","poe_total":"{:.0%}","poe_inside":"{:.0%}",
       "BE_low":"R$ {:.2f}","BE_high":"R$ {:.2f}","iv_eff_avg":"{:.0%}","score_final":"{:.2f}"}
st.dataframe(
    combo_df[show_cols].rename(columns={
        "call_symbol":"CALL","put_symbol":"PUT","iv_eff_avg":"IV (média)",
        "expiration":"Vencimento","credit_total_por_contrato":"Crédito/ação",
        "retorno_pct":"Retorno %","poe_total":"PoE_total","poe_inside":"PoE_dentro","score_final":"Score",
        "Instrucao_saida":"📘 Instrução de saída", "Alerta_saida":"Alerta"
    }).style.format(fmt),
    use_container_width=True
)

# Payoff (uma estrutura)
st.markdown("### 📈 Payoff no Vencimento (P/L por ação)")
combo_df["__id__"] = (
    combo_df["call_symbol"].astype(str) + " & " + combo_df["put_symbol"].astype(str) +
    " | Kc=" + combo_df["K_call"].round(2).astype(str) +
    " Kp=" + combo_df["K_put"].round(2).astype(str) +
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

# -------------------------------
# v9 — Comparar Estratégias
# -------------------------------
def _nearest_strike(df, typ, target, side):
    d = df[df["type"]==typ].copy()
    if d.empty: return None
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
    st.markdown("Selecione um **strangle** base; o app monta automaticamente as asas do Condor/Jade Lizard.")
    base_id = (
        combo_df["K_put"].round(2).astype(str) + "–" + combo_df["K_call"].round(2).astype(str) +
        " | crédito≈" + combo_df["credit_total_por_contrato"].round(2).astype(str)
    )
    tmp = combo_df.copy()
    tmp["__base__"] = base_id
    pick = st.selectbox("Strangle base:", options=tmp["__base__"].tolist())
    rowb = tmp[tmp["__base__"]==pick].iloc[0]
    Kp_b, Kc_b, cred_b = float(rowb["K_put"]), float(rowb["K_call"]), float(rowb["credit_total_por_contrato"])

    wing_pct = st.slider("Largura das asas (% do preço à vista)", min_value=2, max_value=15, value=5, step=1,
                         help="Define o quão distantes estarão as asas (PUT comprada e CALL comprada) do strangle base.") / 100.0
    Kc_target = Kc_b + wing_pct * spot
    Kp_target = Kp_b - wing_pct * spot
    kc_w_tuple = _nearest_strike(df, 'C', Kc_target, side='above')
    kp_w_tuple = _nearest_strike(df, 'P', Kp_target, side='below')
    if kc_w_tuple is None or kp_w_tuple is None:
        st.warning("Não foi possível localizar strikes para as asas. Aumente a largura.")
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
            poe_inside = float(rowb.get("poe_inside", np.nan))
            if not np.isnan(poe_inside): st.metric("PoE ficar dentro", f"{poe_inside:.0%}")
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

        with st.expander("📘 Explicações, fórmulas e guia do gráfico"):
            st.markdown(f"""
**Estruturas**  
- **Strangle vendido coberto** — Vende PUT (Kp={Kp_b:.2f}) + CALL (Kc={Kc_b:.2f}). Ganha o **crédito** se S ∈ [{Kp_b:.2f}, {Kc_b:.2f}].  
- **Iron Condor coberto** — Strangle + compra PUT (Kp_w={Kp_w:.2f}) e CALL (Kc_w={Kc_w:.2f}) de proteção (perda máxima limitada).  
- **Jade Lizard** — PUT vendida (Kp), CALL vendida (Kc) e CALL comprada (Kc_w). Se **crédito ≥ (Kc_w − Kc)**, não há **risco de alta**.

**Fórmulas do P/L (por ação, no vencimento)**  
- Strangle: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + **crédito**.  
- Iron Condor: Π(S) = Strangle + max(0, Kp_w − S) + max(0, S − Kc_w) − **custo_das_asas**.  
- Jade Lizard: Π(S) = −max(0, Kp − S) − max(0, S − Kc) + max(0, S − Kc_w) + **crédito_líquido**.

**Probabilidade (didática)**  
PoE_total ≈ PoE_put + PoE_call (truncado em 100%).  
PoE_dentro = 1 − PoE_total. (Estimado por BS/Δ quando IV faltar.)

**Guia do gráfico**  
- Linha **0** (horizontal) = **ponto de equilíbrio**.  
- Linhas pontilhadas = **strikes** (Kp, Kc) e, quando houver, **asas** (Kp_w, Kc_w).  
- Curva = seu **P/L por ação** no vencimento para cada preço **S**.
""")

# -------------------------------
# Fim
# -------------------------------

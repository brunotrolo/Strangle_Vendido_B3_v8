# app_v9_opcoesnet_file.py
# Fluxo: 1 ticker por vez + upload do Excel do opcoes.net
# Correções empíricas a partir da planilha enviada:
#  - Cabeçalho na linha 1 (header=0), com fallback em header=1
#  - Strike com vírgula decimal e possíveis NBSP/esp. invisíveis
#  - Sem Bid/Ask: usar "Último" como preço (mid)
#  - Mapas de colunas PT-BR tolerantes a acentos e variações
#  - Montagem de strangle vendido coberto + comparador (Strangle × Iron Condor × Jade Lizard)

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

# ---------------------- Config -----------------------------------
st.set_page_config(page_title="Strangle Vendido Coberto — v9", page_icon="💼", layout="wide")
st.title("💼 Strangle Vendido Coberto — v9 (B3 + planilha opcoes.net)")
st.caption("Selecione um ticker, envie a planilha (.xlsx/.csv) do opcoes.net e veja sugestões + comparação (Strangle × Iron Condor × Jade Lizard).")

# ---------------------- Utilidades --------------------------------
CALL_SERIES = set(list("ABCDEFGHIJKL"))
PUT_SERIES  = set(list("MNOPQRSTUVWX"))
SQRT_2 = np.sqrt(2.0)

def _strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _norm_colname(c: str) -> str:
    c0 = str(c).strip()
    c1 = re.sub(r'\s+', ' ', c0)
    c2 = _strip_accents(c1).lower()
    return c2

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
    """Converte '1.234,56' ou '1234.56' para float.
       Se já for numérico, retorna direto. Remove NBSP e símbolos (R$, %)."""
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
        s_clean = s_clean.replace(".", "")     # remove milhares
        s_clean = s_clean.replace(",", ".")    # vírgula -> ponto
        try: return float(s_clean)
        except: return np.nan
    else:
        # não remova '.' — é possivelmente separador decimal
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

# ---------------------- Tickers B3 (para UX) ----------------------
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
    # Procura uma tabela com colunas "código"/"ticker" e "ação"/"empresa"
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
        # fallback conservador
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/acoes/([A-Z]{4}\d)$", a["href"])
            if m:
                tk = m.group(1)
                nm = a.get_text(strip=True) or tk
                tickers.append((tk, nm))
    return sorted({tk: nm for tk, nm in tickers}.items(), key=lambda x: x[0])

tickers_list = fetch_b3_tickers()
if not tickers_list:
    st.warning("Não consegui carregar a lista de tickers do site. Você ainda pode digitar o ticker manualmente.")

# ---------------------- Escolha do ticker -------------------------
col_tk1, col_tk2 = st.columns([2,1])
with col_tk1:
    if tickers_list:
        labels = [f"{tk} — {nm}" for tk, nm in tickers_list]
        pick_label = st.selectbox("🔎 Escolha um ticker da B3", options=labels, index=0)
        TICKER = pick_label.split(" — ")[0]
    else:
        TICKER = st.text_input("Digite o ticker (ex.: PETR4, VALE3, ITUB4)", value="PETR4").strip().upper()
with col_tk2:
    st.metric("Ticker selecionado", TICKER if TICKER else "—")
if not TICKER: st.stop()

# ---------------------- Parâmetros (sidebar c/ tooltips) ----------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    r = st.number_input("Taxa livre de risco anual (r)", min_value=0.0, max_value=1.0, value=0.11, step=0.005, format="%.3f",
                        help="Usada no Black–Scholes. No Brasil, aproxime pela SELIC anualizada. Ex.: 0,11 = 11% a.a.")
    delta_min = st.number_input("|Δ| mínimo", min_value=0.0, max_value=1.0, value=0.00, step=0.01,
                                help="Filtro de ‘moneyness’ por Delta. Vendedores costumam usar |Δ| ~ 0,05–0,35.")
    delta_max = st.number_input("|Δ| máximo", min_value=0.0, max_value=1.0, value=0.35, step=0.01,
                                help="Valor máximo do |Δ| para filtrar opções mais OTM.")
    risk_selection = st.multiselect("Bandas de risco por perna", ["Baixo","Médio","Alto"], default=["Baixo","Médio","Alto"],
                                    help="Classificação didática pela PoE: Baixo 0–15%, Médio 15–35%, Alto 35–55%.")
    lookback_days = st.number_input("Janela p/ IV Rank/Percentil (dias)", min_value=60, max_value=500, value=252, step=1,
                                    help="Compara a IV atual vs histórico (usa HV20 se IV faltar).")
    st.markdown("---")
    show_exit_help = st.checkbox("Mostrar instruções de SAÍDA", value=True,
                                 help="Recomprar a perna ameaçada perto do vencimento ou encerrar após capturar 70–80% do prêmio.")
    days_exit_thresh = st.number_input("Dias até vencimento p/ alerta", min_value=1, max_value=30, value=10,
                                       help="Com ≤ N dias, mensagens de saída ficam mais proativas.")
    prox_pct = st.number_input("Proximidade ao strike (%)", min_value=1, max_value=20, value=5,
                               help="Considera strike ‘ameaçado’ quando S está a menos de X% dele.") / 100.0
    capture_target = st.number_input("Meta de captura do prêmio (%)", min_value=10, max_value=95, value=70,
                                     help="Encerrar com ganho parcial (ex.: 70% capturado → zera o risco).") / 100.0

# ---------------------- Spot & HV20 --------------------------------
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

# ---------------------- Leitura do arquivo -------------------------
st.markdown(f"### 3) Envie a *option chain* do **opcoes.net** (Excel/CSV) para **{TICKER}**")
uploaded = st.file_uploader(
    "Detecção automática: tenta header=0 (linha 1) e fallback header=1 (linha 2). 'Vencimento' é dd/mm/aaaa.",
    type=["xlsx","xls","csv"]
)
if uploaded is None:
    st.info("👉 Envie o arquivo para continuar.")
    st.stop()

def _auto_read_opcoesnet(file) -> pd.DataFrame:
    name = file.name.lower()
    file_bytes = file.getvalue()

    def try_read(header):
        if name.endswith(".csv"):
            text = file_bytes.decode("utf-8", errors="ignore")
            return pd.read_csv(io.StringIO(text), header=header)
        else:
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", header=header)

    df, header_used = None, None
    # Primeiro tenta header=0 (linha 1), depois header=1 (linha 2)
    for h in (0, 1):
        try:
            _df = try_read(h)
            if _df is not None and _df.shape[1] >= 2:
                df = _df.dropna(how="all").copy(); header_used = h; break
        except Exception:
            continue
    if df is None:
        raise RuntimeError("Falha ao ler a planilha (.xlsx/.csv).")

    # Normaliza nomes -> mapa norm->original
    norm_pairs = [(_norm_colname(c), c) for c in df.columns]
    rev_map  = {norm: orig for norm, orig in norm_pairs}

    # aliases PT-BR
    aliases = {
        "symbol": ["ticker", "codigo", "código", "opcao", "opção", "simbolo", "símbolo"],
        "expiration": ["vencimento", "venc", "expiracao", "expiração", "expiry"],
        "bdays": ["dias uteis", "dias úteis", "dias-uteis", "dias_uteis", "dias comerciais"],
        "type": ["tipo", "class", "opcao_tipo"],
        "strike": ["strike", "preco de exercicio", "preço de exercicio", "preço de exercício", "exercicio", "k"],
        "last": ["ultimo", "último", "preco", "preço", "fechamento", "close", "ultimo negocio", "último negocio"],
        "bid": ["bid", "compra", "melhor compra", "oferta de compra"],
        "ask": ["ask", "venda", "melhor venda", "oferta de venda"],
        "impliedVol": ["vol impl (%)", "vol impl. (%)", "volatilidade implicita", "volatilidade implícita", "iv", "iv (%)"],
        "delta": ["delta", "Δ"],
    }

    def find_col(alias_list):
        # 1) match exato do normalizado
        for al in alias_list:
            norm = _norm_colname(al)
            if norm in rev_map: return rev_map[norm]
        # 2) aproximação leve (remove símbolos)
        for al in alias_list:
            norm_al = re.sub(r'[^a-z0-9%$ ]','', _norm_colname(al))
            for norm_col, orig in rev_map.items():
                if re.sub(r'[^a-z0-9%$ ]','', norm_col) == norm_al:
                    return orig
        return None

    out = pd.DataFrame(index=df.index)

    # symbol (opcional)
    c_symbol = find_col(aliases["symbol"])
    if c_symbol is not None:
        out["symbol"] = df[c_symbol]

    # type
    c_type = find_col(aliases["type"])
    if c_type is not None:
        tnorm = df[c_type].astype(str).str.upper().str.strip()
        out["type"] = tnorm.replace({"CALL":"C","COMPRA":"C","C":"C","PUT":"P","VENDA":"P","P":"P"})
    else:
        out["type"] = np.nan

    # fallback: deduz do código
    if out.get("type", pd.Series(dtype=object)).isna().all() and "symbol" in out.columns:
        def infer_from_symbol(code: str):
            if not isinstance(code, str): return np.nan
            s = re.sub(r'[^A-Z0-9]', '', code.upper())
            m = re.search(r'([A-Z])\d+$', s)
            if not m: return np.nan
            letra = m.group(1)
            if letra in CALL_SERIES: return 'C'
            if letra in PUT_SERIES:  return 'P'
            return np.nan
        out["type"] = out["symbol"].map(infer_from_symbol)

    # strike — BR robusto + fallback to_numeric
    c_strike = find_col(aliases["strike"])
    if c_strike:
        raw = df[c_strike]
        strike_parsed = pd.to_numeric(raw.map(_br_to_float), errors="coerce")
        if strike_parsed.notna().sum() == 0:
            strike_parsed = pd.to_numeric(raw, errors="coerce")
        out["strike"] = strike_parsed
    else:
        out["strike"] = np.nan

    # preços — usa Último se não houver Bid/Ask
    c_bid = find_col(aliases["bid"])
    c_ask = find_col(aliases["ask"])
    c_last = find_col(aliases["last"])
    if c_bid is not None:
        out["bid"] = pd.to_numeric(df[c_bid].map(_br_to_float), errors="coerce")
    if c_ask is not None:
        out["ask"] = pd.to_numeric(df[c_ask].map(_br_to_float), errors="coerce")
    if c_last is not None:
        out["last"] = pd.to_numeric(df[c_last].map(_br_to_float), errors="coerce")

    # mid
    bid_series  = out["bid"]  if "bid"  in out.columns else pd.Series(np.nan, index=df.index)
    ask_series  = out["ask"]  if "ask"  in out.columns else pd.Series(np.nan, index=df.index)
    last_series = out["last"] if "last" in out.columns else pd.Series(np.nan, index=df.index)
    has_quote = bid_series.notna() | ask_series.notna()
    mid_from_quotes = (bid_series.fillna(0) + ask_series.fillna(0)) / 2.0
    out["mid"] = np.where(has_quote, mid_from_quotes, last_series)

    # IV / Delta
    c_iv = find_col(aliases["impliedVol"])
    if c_iv is not None:
        out["impliedVol"] = pd.to_numeric(df[c_iv].map(lambda v: _br_to_float(v)/100.0), errors="coerce")
    c_delta = find_col(aliases["delta"])
    if c_delta is not None:
        out["delta"] = pd.to_numeric(df[c_delta].map(_br_to_float), errors="coerce")

    # expiration — por nome (coluna Vencimento)
    c_exp = find_col(aliases["expiration"])
    exp_series = None
    if c_exp is not None:
        exp_series = df[c_exp]
    else:
        # fallback: coluna B
        if df.shape[1] >= 2:
            exp_series = df.iloc[:, 1]

    if exp_series is not None:
        parsed = exp_series.map(_parse_date_any)
        # força dayfirst
        if parsed.isna().mean() > 0.9:
            parsed = pd.to_datetime(exp_series.astype(str).str.strip(), dayfirst=True, errors="coerce").dt.date
        out["expiration"] = parsed
    else:
        out["expiration"] = np.nan

    # Completa faltantes obrigatórios
    for col in ["type","strike","mid","expiration"]:
        if col not in out.columns:
            out[col] = np.nan

    # Diagnóstico
    with st.expander("🛠️ Diagnóstico de leitura"):
        st.write(f"Header usado: {header_used} (0 = linha 1, 1 = linha 2)")
        st.write("Colunas mapeadas:", list(out.columns))
        if "type" in out.columns:
            st.write("Contagem por type (bruto):", out["type"].value_counts(dropna=False))
        if "strike" in out.columns:
            ex_strikes = out["strike"].dropna().astype(float)
            st.write("Qtde de strikes válidos:", int(ex_strikes.shape[0]))
            st.write("Exemplo strikes:", ex_strikes.head(10).tolist())
        if "expiration" in out.columns:
            st.write("Exemplo vencimentos:", out["expiration"].dropna().astype(str).head(5).tolist())
        st.dataframe(out.head())

    return out

# Lê arquivo
try:
    chain_all = _auto_read_opcoesnet(uploaded)
except Exception as e:
    st.error(f"Falha ao ler o arquivo: {e}")
    st.stop()

# ---------------------- Seleção do vencimento ---------------------
valid_exps = sorted([d if isinstance(d, date) else (d.date() if isinstance(d, datetime) else None)
                     for d in chain_all["expiration"].dropna().unique().tolist()])
valid_exps = [d for d in valid_exps if isinstance(d, date)]
if not valid_exps:
    st.error("Nenhum vencimento válido encontrado. Garanta que a coluna 'Vencimento' está como dd/mm/aaaa.")
    st.stop()

exp_choice = st.selectbox("📅 Vencimento", options=valid_exps, index=0)
T = yearfrac(date.today(), exp_choice)
days_to_exp = (exp_choice - date.today()).days
df = chain_all[chain_all["expiration"] == exp_choice].copy()

# ---------------------- Normalizações e IV/Δ ----------------------
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
if "type" not in df.columns or df["type"].isna().all():
    # inferência pela posição do strike vs spot
    guess = pd.Series(None, index=df.index, dtype="object")
    guess.loc[(df["strike"] > spot)] = "C"
    guess.loc[(df["strike"] < spot)] = "P"
    df["type"] = guess

# IV efetiva (proxy HV20 se faltar)
if "impliedVol" not in df.columns:
    df["impliedVol"] = np.nan
df["iv_eff"] = df["impliedVol"]
if df["iv_eff"].isna().all() or (df["iv_eff"]<=0).all():
    df["iv_eff"] = hv20

# Completa deltas faltantes por BS
if "delta" not in df.columns:
    df["delta"] = np.nan
need = df["delta"].isna()
if need.any():
    vals = []
    for _, row in df.loc[need].iterrows():
        K = float(row["strike"]) if not pd.isna(row["strike"]) else np.nan
        sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
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

# ---------------------- Seleção OTM + filtros ---------------------
calls = df[(df["type"]=="C") & (df["strike"]>spot)].copy()
puts  = df[(df["type"]=="P") & (df["strike"]<spot)].copy()

if calls.empty:
    calls = df[df["strike"]>spot].copy(); calls["type"]="C"
if puts.empty:
    puts = df[df["strike"]<spot].copy(); puts["type"]="P"

def dfilter(dfi: pd.DataFrame) -> pd.DataFrame:
    dfo = dfi.copy()
    if "delta" not in dfo.columns:
        return dfo
    dfo["abs_delta"] = dfo["delta"].abs()
    dfo = dfo[(dfo["abs_delta"]>=delta_min) & (dfo["abs_delta"]<=delta_max)]
    return dfo

calls = dfilter(calls); puts = dfilter(puts)

with st.expander("🔎 Diagnóstico de filtros (OTM)"):
    st.write(f"Calls OTM após filtros: {len(calls)} | Puts OTM após filtros: {len(puts)}")
    if not calls.empty: st.write("Exemplo calls:", calls[["strike","mid","delta"]].head())
    if not puts.empty:  st.write("Exemplo puts:",  puts[["strike","mid","delta"]].head())

# ---------------------- Cobertura --------------------------------
st.markdown("### 4) Cobertura e tamanho do contrato")
colA, colB, colC = st.columns(3)
with colA:
    shares_owned = st.number_input(f"Ações em carteira ({TICKER})", min_value=0, step=100, value=0,
        help="Ações livres para cobrir CALLs (1 lote = ‘Tamanho do contrato’).")
with colB:
    cash_available = st.number_input(f"Caixa disponível (R$) ({TICKER})", min_value=0.0, step=100.0, value=10000.0, format="%.2f",
        help="Dinheiro reservado para cobrir a PUT (strike × tamanho do contrato).")
with colC:
    lot_size = st.number_input(f"Tamanho do contrato ({TICKER})", min_value=1, step=1, value=100,
        help="Na B3, ações geralmente 100; ajuste se o seu contrato for diferente.")

max_qty_call = (shares_owned // lot_size) if lot_size > 0 else 0
def max_qty_put_for_strike(Kp: float) -> int:
    if lot_size <= 0 or Kp <= 0: return 0
    return int(cash_available // (Kp * lot_size))

# ---------------------- Montagem dos strangles --------------------
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
        probITM_put = float(p.get("prob_ITM", np.nan)); probITM_call = float(c.get("prob_ITM", np.nan))

        # se ainda não calculamos PoE, calcule agora
        sigma_p = float(p.get("iv_eff", np.nan)) if not np.isnan(p.get("iv_eff", np.nan)) else hv20
        sigma_c = float(c.get("iv_eff", np.nan)) if not np.isnan(c.get("iv_eff", np.nan)) else hv20
        if np.isnan(probITM_put):
            probITM_put = prob_ITM_put(spot, Kp, r, max(sigma_p,1e-6), T)
        if np.isnan(probITM_call):
            probITM_call = prob_ITM_call(spot, Kc, r, max(sigma_c,1e-6), T)

        if np.isnan(probITM_put) and not np.isnan(p.get("delta", np.nan)):
            probITM_put = abs(float(p["delta"]))
        if np.isnan(probITM_call) and not np.isnan(c.get("delta", np.nan)):
            probITM_call = abs(float(c["delta"]))

        poe_total = min(1.0, max(0.0, (probITM_put or 0.0) + (probITM_call or 0.0)))
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
            "expiration": exp_choice, "lot_size": lot_size,
        })

if not combos:
    st.warning("Não há strangles possíveis com os filtros/limites atuais. Verifique no diagnóstico se 'strike' e 'Último' foram reconhecidos.")
    st.stop()

combo_df = pd.DataFrame(combos)
combo_df["retorno_pct"] = combo_df["credit_total_por_contrato"] / spot
combo_df["poe_total"] = (combo_df["probITM_call"].fillna(0) + combo_df["probITM_put"].fillna(0)).clip(0,1)
combo_df["poe_inside"] = (1 - combo_df["poe_total"]).clip(0,1)
combo_df["risk_score"] = combo_df["poe_total"] / 2.0
combo_df["score_final"] = combo_df["retorno_pct"] / (combo_df["risk_score"] + 0.01)

# ---------------------- Instruções de saída -----------------------
with st.expander("📘 Regras didáticas de saída"):
    st.markdown("""
- **Quando sair**:  
  • Se o preço encostar em **K_put** ou **K_call** **perto do vencimento** (poucos dias).  
  • Ou quando você já tiver **capturado ~70–80% do prêmio**.

- **Como sair**:  
  • **Recomprar** a perna ameaçada (a PUT ou a CALL).  
  • Ou **encerrar o strangle inteiro** recomprando ambas as pernas.
""")

# ---------------------- Top 3 -------------------------------------
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

# ---------------------- Tabela completa ---------------------------
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

# ---------------------- Payoff de uma estrutura -------------------
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

# ---------------------- Comparar estratégias ----------------------
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
    tmp = combo_df.copy()
    tmp["__base__"] = (
        tmp["K_put"].round(2).astype(str) + "–" + tmp["K_call"].round(2).astype(str) +
        " | crédito≈" + tmp["credit_total_por_contrato"].round(2).astype(str)
    )
    pick = st.selectbox("Strangle base:", options=tmp["__base__"].tolist())
    rowb = tmp[tmp["__base__"]==pick].iloc[0]
    Kp_b, Kc_b, cred_b = float(rowb["K_put"]), float(rowb["K_call"]), float(rowb["credit_total_por_contrato"])

    wing_pct = st.slider("Largura das asas (% do preço à vista)", min_value=2, max_value=15, value=5, step=1,
                         help="Distância das asas (PUT comprada e CALL comprada) do strangle base.") / 100.0
    Kc_target = Kc_b + wing_pct * spot
    Kp_target = Kp_b - wing_pct * spot
    kc_w_tuple = _nearest_strike(df, 'C', Kc_target, side='above')
    kp_w_tuple = _nearest_strike(df, 'P', Kp_target, side='below')
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

# ---------------------- Fim ---------------------------------------

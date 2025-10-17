# app_v9_fileinput.py
# Strangle Vendido Coberto — v9 (fonte: arquivo CSV/XLSX)
# Mantém: sugestões de strangle, instruções de saída, comparação (Strangle × Iron Condor × Jade Lizard)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io, re
from datetime import date, datetime
import yfinance as yf

CALL_SERIES = set(list("ABCDEFGHIJKL"))
PUT_SERIES  = set(list("MNOPQRSTUVWX"))
SQRT_2 = np.sqrt(2.0)

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

def normalize_chain_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        'symbol': ['symbol','símbolo','ticker','codigo','código','asset','opção','opcao','código da opção','codigo da opcao'],
        'expiration': ['expiration','venc','vencimento','expiração','expiracao','expiry','venc.','vencimento (d/m/a)'],
        'type': ['type','tipo','opção','opcao','option_type','class'],
        'strike': ['strike','preço de exercício','preco de exercicio','exercicio','k','strike_price','preço de exercício (r$)'],
        'bid': ['bid','compra','melhor compra','oferta de compra'],
        'ask': ['ask','venda','melhor venda','oferta de venda'],
        'last': ['last','último','ultimo','preço','preco','close','último negócio','ultimo negocio','último neg.','ult.'],
        'impliedVol': ['iv','ivol','impliedvol','implied_vol','vol implícita','vol implicita','vol. impl. (%)','vol impl (%)','iv (%)'],
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
            s = re.sub(r'[^A-Z0-9]', '', str(code).upper().strip())
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

def read_chain_from_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(file)
        best_len, best_df = -1, None
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
                if df.shape[0] > best_len and df.shape[1] >= 4:
                    best_len, best_df = df.shape[0], df
            except Exception:
                continue
        if best_df is None:
            raise RuntimeError("Não foi possível ler nenhuma planilha válida do Excel.")
        return normalize_chain_columns(best_df)
    else:
        data = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(data))
        return normalize_chain_columns(df)

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

def compute_iv_rank_percentile(series_df: pd.DataFrame, lookback: int):
    if series_df is None or series_df.empty: return (np.nan, np.nan, np.nan)
    s = series_df.sort_values("date").tail(lookback)["iv"].dropna()
    if len(s) < 5: return (np.nan, np.nan, np.nan)
    iv_now = float(s.iloc[-1]); iv_min, iv_max = float(s.min()), float(s.max())
    iv_rank = (iv_now - iv_min) / max(1e-9, (iv_max - iv_min))
    iv_percentile = float((s <= iv_now).mean())
    return (iv_now, iv_rank, iv_percentile)

# ---------------------------
# App
# ---------------------------

st.set_page_config(page_title="Strangle Vendido Coberto – v9 (arquivo local)", layout="wide")
st.title("💼 Strangle Vendido Coberto — Sugeridor (v9)")
st.caption("Fonte de dados: arquivo CSV/XLSX importado • Sugestões de strangle • Saídas didáticas • Comparador de estratégias (v9)")

with st.sidebar:
    st.header("Parâmetros gerais")
    r = st.number_input("Taxa livre de risco anual (r)", min_value=0.0, max_value=1.0, value=0.11, step=0.005, format="%.3f")
    st.markdown("---")
    uploaded = st.file_uploader("Importe a chain (CSV ou Excel)", type=["csv","xlsx","xls"])
    st.caption("Use a planilha/CSV com colunas como: Símbolo, Vencimento, Tipo, Strike, Bid/Ask/Último, Delta, IV...")
    st.markdown("---")
    delta_min = st.number_input("|Δ| mínimo", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    delta_max = st.number_input("|Δ| máximo", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    risk_selection = st.multiselect("Bandas de risco por perna", ["Baixo","Médio","Alto"], default=["Baixo","Médio","Alto"])
    st.caption("Baixo=0–15% · Médio=15–35% · Alto=35–55%")
    lookback_days = st.number_input("Janela p/ IV Rank/Percentil (dias)", min_value=60, max_value=500, value=252, step=1)
    st.markdown("---")
    show_exit_help = st.checkbox("Mostrar instruções de SAÍDA", value=True)
    days_exit_thresh = st.number_input("Dias até vencimento p/ alerta", min_value=1, max_value=30, value=10)
    prox_pct = st.number_input("Proximidade ao strike (%)", min_value=1, max_value=20, value=5) / 100.0
    capture_target = st.number_input("Meta de captura do prêmio (%)", min_value=10, max_value=95, value=70) / 100.0

if uploaded is None:
    st.info("👉 Importe um arquivo CSV/XLSX com a chain de opções para começar.")
    st.stop()

try:
    chain_all = read_chain_from_uploaded(uploaded)
except Exception as e:
    st.error(f"Falha ao ler o arquivo: {e}")
    st.stop()

# tenta inferir os tickers-base a partir do prefixo dos símbolos das opções
tickers_in_file = sorted(list(set([re.sub(r'\d+$','', str(s)[:5]).upper() if isinstance(s,str) else '' for s in chain_all.get('symbol', [])])))
tickers_in_file = [t for t in tickers_in_file if re.match(r'^[A-Z]{4,5}$', t)]
if not tickers_in_file:
    st.error("Não foi possível inferir o ticker base a partir da coluna 'symbol'. Verifique se a planilha tem a coluna correta.")
    st.stop()

tabs = st.tabs(tickers_in_file)

@st.cache_data(show_spinner=False)
def load_spot(tk_base: str):
    yahoo_ticker = f"{tk_base}.SA"
    return load_spot_and_iv_proxy(yahoo_ticker)

def pipeline_for_ticker(tk_base: str):
    def belongs(row_symbol: str):
        s = str(row_symbol).upper().strip()
        base = re.sub(r'\d+$','', re.sub(r'[^A-Z0-9]','', s))
        return base == tk_base
    df = chain_all[chain_all["symbol"].map(belongs)].copy()
    if df.empty:
        st.warning("Nenhuma linha para este ticker no arquivo.")
        return None

    spot, hv20, iv_series = load_spot(tk_base)
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Preço à vista (S)", f"{spot:,.2f}" if not np.isnan(spot) else "—")
    with c2: st.metric("HV20 (σ anual – proxy)", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
    with c3: st.metric("r (anual)", f"{r:.2%}")
    if np.isnan(spot) or spot<=0:
        st.error("Não foi possível obter o preço à vista (Yahoo).")
        return None

    exps = sorted([d for d in df["expiration"].dropna().unique().tolist() if isinstance(d, (date, datetime))], key=lambda x: x)
    if not exps:
        st.error("Nenhum vencimento válido encontrado no arquivo.")
        return None
    exp_choice = st.selectbox("Vencimento", options=exps, key=f"exp_{tk_base}")
    T = yearfrac(date.today(), exp_choice if isinstance(exp_choice,date) else exp_choice.date())
    days_to_exp = (exp_choice - date.today()).days if isinstance(exp_choice,date) else (exp_choice.date() - date.today()).days
    df = df[df["expiration"] == exp_choice].copy()
    if df.empty:
        st.warning("Sem opções para o vencimento escolhido.")
        return None

    if "bid" in df.columns and "ask" in df.columns and (df["bid"].notna().any() or df["ask"].notna().any()):
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    else:
        df["mid"] = df.get("last", np.nan).fillna(0)
    if "impliedVol" not in df.columns:
        df["impliedVol"] = np.nan
    df["iv_eff"] = df["impliedVol"]
    if df["iv_eff"].isna().all():
        df["iv_eff"] = hv20

    calls = df[(df["type"]=="C") & (df["strike"]>spot)].copy()
    puts  = df[(df["type"]=="P") & (df["strike"]<spot)].copy()

    for side_df, side in [(calls,"C"), (puts,"P")]:
        if "delta" not in side_df.columns:
            side_df["delta"] = np.nan
        need = side_df["delta"].isna()
        if need.any():
            values = []
            for _, row in side_df.loc[need].iterrows():
                K = float(row["strike"])
                sigma = float(row["iv_eff"]) if not pd.isna(row["iv_eff"]) and row["iv_eff"]>0 else hv20
                sigma = max(sigma, 1e-6)
                d = call_delta(spot, K, r, sigma, T) if side == "C" else put_delta(spot, K, r, sigma, T)
                values.append(d)
            side_df.loc[need, "delta"] = values

    def dfilter(dfi: pd.DataFrame) -> pd.DataFrame:
        if "delta" not in dfi.columns: return dfi
        dfo = dfi.copy(); dfo["abs_delta"] = dfo["delta"].abs()
        if delta_min>0: dfo = dfo[dfo["abs_delta"]>=delta_min]
        if delta_max>0: dfo = dfo[dfo["abs_delta"]<=delta_max]
        return dfo
    calls, puts = dfilter(calls), dfilter(puts)

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
    calls, puts = probs(calls,"C"), probs(puts,"P")

    bands = {"Baixo":(0.00,0.15),"Médio":(0.15,0.35),"Alto":(0.35,0.55)}
    def label(p):
        if np.isnan(p): return "Fora"
        for k,(a,b) in bands.items():
            if a <= p <= b: return k
        return "Fora"
    calls["band"] = calls["prob_ITM"].apply(label)
    puts["band"]  = puts["prob_ITM"].apply(label)
    calls = calls[calls["band"].isin(risk_selection)]
    puts  = puts[puts["band"].isin(risk_selection)]

    if calls.empty or puts.empty:
        st.warning("Sem CALLs/PUTs OTM dentro dos filtros/risco.")
        return None

    colA, colB, colC = st.columns(3)
    with colA:
        shares_owned = st.number_input(f"Ações em carteira ({tk_base})", min_value=0, step=100, value=0, key=f"shares_{tk_base}")
    with colB:
        cash_available = st.number_input(f"Caixa disponível (R$) ({tk_base})", min_value=0.0, step=100.0, value=10000.0, format="%.2f", key=f"cash_{tk_base}")
    with colC:
        lot_size = st.number_input(f"Tamanho do contrato ({tk_base})", min_value=1, step=1, value=100, key=f"lot_{tk_base}")

    max_qty_call = (shares_owned // lot_size) if lot_size > 0 else 0
    def max_qty_put_for_strike(Kp: float) -> int:
        if lot_size <= 0 or Kp <= 0: return 0
        return int(cash_available // (Kp * lot_size))

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
                "ticker": tk_base,
                "call_symbol": c.get("symbol"), "put_symbol": p.get("symbol"),
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
        st.warning("Não há strangles possíveis respeitando cobertura (ações/caixa).")
        return None
    combo_df = pd.DataFrame(combos)
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

    exit_texts, alerts = [], []
    for _, rrow in combo_df.iterrows():
        text, alert = build_exit_guidance(rrow)
        exit_texts.append(text); alerts.append(alert)
    combo_df["Instrucao_saida"] = exit_texts
    combo_df["Alerta_saida"] = alerts

    st.session_state.setdefault("v9_ctx", {})
    st.session_state["v9_ctx"][tk_base] = {
        "chain": df.copy(),
        "spot": float(spot),
        "r": float(r),
        "expiration": exp_choice,
        "T": T,
        "lot_size": int(lot_size),
    }

    st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
    top3 = combo_df.sort_values(by=["score_final","credit_total_por_contrato"], ascending=[False,False]).head(3)
    display_cols = ["call_symbol","K_call","put_symbol","K_put","credit_total_por_contrato","poe_total","retorno_pct","score_final","qty","BE_low","BE_high","Alerta_saida"]
    st.dataframe(top3[display_cols].rename(columns={
        "call_symbol":"CALL","put_symbol":"PUT","credit_total_por_contrato":"Crédito/ação",
        "poe_total":"PoE_total","retorno_pct":"Retorno %","score_final":"Score"
    }).style.format({"K_call":"%.2f","K_put":"%.2f","Crédito/ação":"R$ %.2f","PoE_total":"{:.0%}","Retorno %":"{:.2%}","Score":"{:.2f}","BE_low":"R$ %.2f","BE_high":"R$ %.2f"}), width='stretch')

    st.subheader("📌 Sugestões por banda (ranqueadas)")
    def pick_by_band(df, band, n=3):
        sub = df[(df.get("band_call","")==band) & (df.get("band_put","")==band)].copy()
        if sub.empty: return sub
        return sub.sort_values(by=["score_final","credit_total_por_contrato"], ascending=[False,False]).head(n)
    final = []
    for band in ["Baixo","Médio","Alto"]:
        pick = pick_by_band(combo_df, band, n=3)
        if not pick.empty:
            pick.insert(0, "Risco", band)
            final.append(pick)
    if final:
        result = pd.concat(final, ignore_index=True)
    else:
        result = combo_df.copy()

    show_cols = ["ticker","Risco","call_symbol","K_call","probITM_call","delta_call",
                 "put_symbol","K_put","probITM_put","delta_put",
                 "credit_total_por_contrato","retorno_pct","poe_total","poe_inside",
                 "qty","BE_low","BE_high","iv_eff_avg","expiration","score_final","Alerta_saida","Instrucao_saida"]

    fmt = {"K_call":"%.2f","K_put":"%.2f",
           "probITM_call":"{:.0%}","probITM_put":"{:.0%}",
           "delta_call":"{:.2f}","delta_put":"{:.2f}",
           "credit_total_por_contrato":"R$ {:.2f}",
           "retorno_pct":"{:.2%}","poe_total":"{:.0%}","poe_inside":"{:.0%}",
           "BE_low":"R$ {:.2f}","BE_high":"R$ {:.2f}",
           "iv_eff_avg":"{:.0%}","score_final":"{:.2f}"}

    st.dataframe(
        result[show_cols].rename(columns={
            "call_symbol":"CALL","put_symbol":"PUT","iv_eff_avg":"IV (média)",
            "expiration":"Vencimento","credit_total_por_contrato":"Crédito/ação",
            "retorno_pct":"Retorno %","poe_total":"PoE_total","poe_inside":"PoE_dentro","score_final":"Score",
            "Instrucao_saida":"📘 Instrução de saída", "Alerta_saida":"Alerta"
        }).style.format(fmt),
        width='stretch'
    )

    # ---- Comparador de estratégias (v9)
    with st.expander("📈 Comparar estratégias (v9)"):
        render_compare_tab(tk_base, result)

    # ---- Payoff simples
    st.markdown("### 📈 Payoff no Vencimento (P/L por ação)")
    result["id"] = (result.get("Risco","") + " | " + result["call_symbol"].astype(str) + " & " +
                    result["put_symbol"].astype(str) + " | Kc=" + result["K_call"].round(2).astype(str) +
                    " Kp=" + result["K_put"].round(2).astype(str) +
                    " | crédito≈" + result["credit_total_por_contrato"].round(2).astype(str))
    sel = st.selectbox("Estrutura:", options=result["id"].tolist(), key=f"sel_{tk_base}")
    row = result[result["id"]==sel].iloc[0]
    Kc, Kp, credit = float(row["K_call"]), float(row["K_put"]), float(row["credit_total_por_contrato"])
    S_grid = np.linspace(max(0.01, Kp*0.8), Kc*1.2, 400)
    payoff = -np.maximum(0.0, S_grid - Kc) - np.maximum(0.0, Kp - S_grid) + credit
    fig = plt.figure()
    plt.plot(S_grid, payoff); plt.axhline(0, linestyle="--"); plt.axvline(Kp, linestyle=":"); plt.axvline(Kc, linestyle=":")
    plt.title(f"{tk_base} — Payoff | Kp={Kp:.2f}, Kc={Kc:.2f}, Crédito≈R$ {credit:.2f}/ação")
    plt.xlabel("Preço no vencimento (S)"); plt.ylabel("P/L por ação (R$)")
    st.pyplot(fig, width='stretch')

    return result

# -------- v9: comparação ---------
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

def payoff_arrays_strangle(S_grid, Kp, Kc, credit):
    return -np.maximum(0.0, Kp - S_grid) - np.maximum(0.0, S_grid - Kc) + credit

def payoff_arrays_iron_condor(S_grid, Kp, Kc, Kp_w, Kc_w, credit_net):
    short_strangle = -np.maximum(0.0, Kp - S_grid) - np.maximum(0.0, S_grid - Kc)
    long_wings = +np.maximum(0.0, Kp_w - S_grid) + np.maximum(0.0, S_grid - Kc_w)
    return short_strangle + long_wings + credit_net

def payoff_arrays_jade_lizard(S_grid, Kp, Kc, Kc_w, credit_net):
    short_put = -np.maximum(0.0, Kp - S_grid)
    short_call = -np.maximum(0.0, S_grid - Kc)
    long_call  = +np.maximum(0.0, S_grid - Kc_w)
    return short_put + short_call + long_call + credit_net

def render_compare_tab(tk, combos_df):
    ctx = st.session_state.get("v9_ctx", {}).get(tk, None)
    if ctx is None or combos_df is None or combos_df.empty:
        st.info("Gere sugestões primeiro para habilitar a comparação.")
        return

    chain = ctx["chain"]; spot = ctx["spot"]

    st.markdown("#### Selecione o **strangle** base")
    opt_id = (combos_df.get("Risco","") + " | Kp=" + combos_df["K_put"].round(2).astype(str) +
              " · Kc=" + combos_df["K_call"].round(2).astype(str) +
              " · crédito≈" + combos_df["credit_total_por_contrato"].round(2).astype(str))
    combos_df = combos_df.copy(); combos_df["__id__"] = opt_id
    pick = st.selectbox("Estrutura:", options=opt_id.tolist(), key=f"v9_pick_{tk}")
    row = combos_df.loc[combos_df["__id__"]==pick].iloc[0]

    Kp = float(row["K_put"]);  Kc = float(row["K_call"]); credit = float(row["credit_total_por_contrato"])

    wing_pct = st.slider("Largura das asas (Iron Condor / Jade Lizard)", min_value=2, max_value=15, value=5, step=1) / 100.0
    Kc_target = Kc + wing_pct * spot; Kp_target = Kp - wing_pct * spot
    kc_w_tuple = _nearest_strike(chain, 'C', Kc_target, side='above')
    kp_w_tuple = _nearest_strike(chain, 'P', Kp_target, side='below')
    if kc_w_tuple is None or kp_w_tuple is None:
        st.warning("Não foi possível localizar strikes para as asas. Aumente a largura.")
        return
    Kc_w, prem_cw, _ = kc_w_tuple; Kp_w, prem_pw, _ = kp_w_tuple

    cost_wings = 0.0
    if not np.isnan(prem_cw): cost_wings += prem_cw
    if not np.isnan(prem_pw): cost_wings += prem_pw
    credit_condor = credit - cost_wings
    cost_jl = prem_cw if not np.isnan(prem_cw) else 0.0
    credit_jl = credit - cost_jl

    poe_inside = float(row.get("poe_inside", np.nan))

    st.markdown("### Resumo comparativo (por ação)")
    colA,colB,colC = st.columns(3)
    with colA:
        st.metric("Strangle — Crédito", f"R$ {credit:.2f}")
        st.metric("Zona neutra (Kp–Kc)", f"{Kp:.2f} — {Kc:.2f}")
        if not np.isnan(poe_inside): st.metric("PoE ficar dentro", f"{poe_inside:.0%}")
    with colB:
        st.metric("Iron Condor — Crédito", f"R$ {credit_condor:.2f}")
        st.metric("Asas (P,C)", f"{Kp_w:.2f}, {Kc_w:.2f}")
        max_loss_ic = max(0.0, (Kp - Kp_w) - credit_condor, (Kc_w - Kc) - credit_condor)
        st.metric("Perda máx. aprox.", f"R$ {max_loss_ic:.2f}")
    with colC:
        st.metric("Jade Lizard — Crédito", f"R$ {credit_jl:.2f}")
        st.metric("Asa (CALL)", f"{Kc_w:.2f}")
        no_upside = credit_jl >= (Kc_w - Kc)
        st.metric("Sem risco de alta?", "Sim" if no_upside else "Não")

    S_grid = np.linspace(max(0.01, Kp_w*0.8), Kc_w*1.2, 500)
    pay_str = payoff_arrays_strangle(S_grid, Kp, Kc, credit)
    pay_ic  = payoff_arrays_iron_condor(S_grid, Kp, Kc, Kp_w, Kc_w, credit_condor)
    pay_jl  = payoff_arrays_jade_lizard(S_grid, Kp, Kc, Kc_w, credit_jl)

    st.markdown("### Payoff comparativo (por ação, no vencimento)")
    for name, arr in [("Strangle vendido", pay_str), ("Iron Condor", pay_ic), ("Jade Lizard", pay_jl)]:
        fig = plt.figure()
        plt.plot(S_grid, arr); plt.axhline(0, linestyle="--"); plt.axvline(Kp, linestyle=":"); plt.axvline(Kc, linestyle=":")
        if name != "Strangle vendido":
            plt.axvline(Kp_w, linestyle=":"); plt.axvline(Kc_w, linestyle=":")
        plt.title(f"{tk} — {name}")
        plt.xlabel("Preço do ativo no vencimento (S)"); plt.ylabel("P/L por ação (R$)")
        st.pyplot(fig, width='stretch')

    with st.expander("📘 Explicações didáticas"):
        st.markdown(f"""
**Strangle vendido coberto** — Vende PUT (Kp={Kp:.2f}) + CALL (Kc={Kc:.2f}).  
Ganha o crédito se S ∈ [{Kp:.2f}, {Kc:.2f}]. Risco em extremos.

**Iron Condor coberto** — Compra PUT (Kp_w={Kp_w:.2f}) e CALL (Kc_w={Kc_w:.2f}) de proteção.  
Limita perdas; crédito menor; mesma zona neutra.

**Jade Lizard** — Vende PUT (Kp) + CALL (Kc) e compra CALL (Kc_w).  
Se crédito ≥ (Kc_w − Kc), não há risco de alta (ganho limitado ao crédito).
""")

for tk, tab in zip(tickers_in_file, tabs):
    with tab:
        st.subheader(f"Ativo: {tk}")
        pipeline_for_ticker(tk)

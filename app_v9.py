# app_v9.py
# Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)
# Requisitos: streamlit, pandas, numpy, python-dateutil
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from datetime import datetime, date
from dateutil.parser import parse as dtparse

st.set_page_config(page_title="Strangle Vendido Coberto — v9", layout="wide")

# ============== UTIL ==============
def _br_to_float(x):
    """
    Converte strings PT-BR (1.234.567,89) -> float(1234567.89).
    Devolve np.nan para vazios/traços.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() in {"nan", "none"}:
        return np.nan
    # remove espaços não quebráveis
    s = s.replace("\xa0", " ")
    # mantém sinal +/-
    neg = s.startswith("-")
    s = s.replace("+", "").replace("-", "")
    # remove pontos de milhar e troca vírgula por ponto
    s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
        return -v if neg else v
    except:
        return np.nan

def _parse_date_br(s):
    """
    Recebe 'dd/mm/aaaa' ou 'dd/mm/aa' e retorna YYYY-MM-DD (str).
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    try:
        d = dtparse(s, dayfirst=True).date()
        return d.isoformat()
    except:
        return None

def _guess_sep_from_text(text):
    # opcoes.net costuma copiar com TAB. Se não, fallback “>=2 espaços”.
    if "\t" in text:
        return "\t"
    return r"\s{2,}"

def _normalize_header(cols):
    """
    Normaliza headers típicos do opcoes.net em PT-BR -> chaves padrão.
    """
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
    }
    out = []
    for c in cols:
        key = str(c).strip().lower()
        key = re.sub(r"\s+", " ", key)
        out.append(mapping.get(key, c))
    return out

def _clean_dataframe(df_raw):
    # renomeia colunas
    df = df_raw.copy()
    df.columns = _normalize_header(df.columns)

    # padroniza colunas essenciais, criando se não vierem
    for col in ["symbol","type","strike","last","impliedVol","delta","expiration"]:
        if col not in df.columns:
            df[col] = np.nan

    # normaliza números PT-BR nas colunas principais
    for col in ["strike","last","impliedVol","delta","bid","ask"]:
        if col in df.columns:
            df[col] = df[col].map(_br_to_float)

    # tipo -> "CALL"/"PUT" (aceita qualquer caixa)
    df["type"] = df["type"].astype(str).str.upper().str.replace("Ç","C").str.strip()
    # 'CALL' / 'PUT' já vem do site; manter qualquer outra variação
    # expiration
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].astype(str).map(_parse_date_br)

    # símbolo
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()

    # Se não houver 'last', tenta mid=(bid+ask)/2
    if "last" in df.columns:
        if df["last"].isna().all():
            if "bid" in df.columns and "ask" in df.columns:
                df["last"] = (df["bid"] + df["ask"]) / 2.0

    # remove linhas sem strike ou sem tipo/venc
    df = df[~df["strike"].isna()]
    df = df[~df["type"].isna()]
    df = df[~df["expiration"].isna()]

    # garante delta em módulo para filtro, mas mantém sinal para PoE-itm aproximado
    df["abs_delta"] = df["delta"].abs() if "delta" in df.columns else np.nan

    # ordena
    df = df.sort_values(["expiration","type","strike"], ascending=[True, True, True])
    df = df.reset_index(drop=True)
    return df

def _pair_strangles(df_exp, spot, mindelta, maxdelta):
    """
    Recebe DF de UM vencimento, separa CALL OTM (K > S) e PUT OTM (K < S),
    aplica filtro de |Δ| e cruza todas as combinações.
    Retorna lista de dicts com métricas.
    """
    # filtros OTM
    calls = df_exp[(df_exp["type"].str.contains("CALL")) & (df_exp["strike"] > spot)].copy()
    puts  = df_exp[(df_exp["type"].str.contains("PUT"))  & (df_exp["strike"] < spot)].copy()

    # filtro por |Δ|
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

def _label_band(|delta_abs|, bands):
    """
    bands: dict com faixas em % ITM aproximado pelo |Δ| (0–1).
    Retorna rótulo "Baixo/Médio/Alto" conforme limites.
    """
    if pd.isna(|delta_abs|):
        return "—"
    p = |delta_abs| * 100.0
    low_hi  = bands["Baixo"][1]    # fim da faixa baixa
    mid_hi  = bands["Médio"][1]    # fim da faixa média
    if p <= low_hi:
        return "Baixo"
    elif p <= mid_hi:
        return "Médio"
    else:
        return "Alto"

def _format_money(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

# ============== HEADER (estilo anterior) ==============
st.title("💼 Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)")
st.write("Cole a option chain do **opcoes.net**, escolha o vencimento e veja as **sugestões didáticas** de strangle coberto e a **comparação de estratégias**.")

colA, colB, colC = st.columns([1.1,1,1])
with colA:
    st.markdown("#### 🔎 Escolha um ticker da B3")
    ticker = st.text_input("Ticker (livre)", value="PETR4")
with colB:
    spot = st.text_input("Preço à vista (S)", value="17,68")
    spot = _br_to_float(spot)
with colC:
    hv20 = st.text_input("HV20 (σ anual – proxy)", value="17,12%")
    r_anual = st.text_input("r (anual)", value="11,00%")

# ============== SIDEBAR (com tooltips operantes) ==============
st.sidebar.header("⚙️ Parâmetros (explicativos)")
mindelta = st.sidebar.number_input(
    "|Δ| mínimo",
    min_value=0.00, max_value=1.00, step=0.01, value=0.05,
    help="Filtro de moneyness por |Δ| (aprox. PoE ITM). Ex.: 0,05 = 5%."
)
maxdelta = st.sidebar.number_input(
    "|Δ| máximo",
    min_value=0.00, max_value=1.00, step=0.01, value=0.35,
    help="Filtro superior de |Δ|. Vendedores usam ~0,05–0,35."
)

st.sidebar.markdown("##### Bandas de risco por perna")
b_baixo = st.sidebar.slider(
    "Faixa Baixo (0–X%)",
    min_value=0, max_value=55, value=15,
    help="Prob. ITM ≈ |Δ| × 100. Até aqui rotulamos a perna como **Baixo**."
)
b_medio = st.sidebar.slider(
    "Faixa Médio (X–Y%)",
    min_value=b_baixo, max_value=55, value=35,
    help="Até este limite, rotulamos a perna como **Médio**. Acima disso é **Alto**."
)
bands_cfg = {
    "Baixo": (0, b_baixo),
    "Médio": (b_baixo, b_medio),
    "Alto":  (b_medio, 55)
}

st.sidebar.markdown("##### Instruções de SAÍDA — Regras práticas")
dte_alert = st.sidebar.number_input(
    "Dias até vencimento (alerta)",
    min_value=0, max_value=60, value=7,
    help="Quando faltarem ≤ N dias, as mensagens de saída ficam mais proativas."
)
prox_pct = st.sidebar.number_input(
    "Proximidade ao strike (%)",
    min_value=0.0, max_value=20.0, value=1.0, step=0.1,
    help="Considera o strike 'ameaçado' quando S está a menos de X% dele."
)
take_profit = st.sidebar.number_input(
    "Meta de captura do prêmio (%)",
    min_value=10, max_value=95, value=75, step=5,
    help="Ex.: 70–80% do crédito já capturado ⇒ encerra (zera o risco)."
)

contract_size = st.sidebar.number_input(
    "Tamanho do contrato",
    min_value=1, max_value=1000, value=100,
    help="Normalmente 100 ações por contrato."
)
qty_shares = st.sidebar.number_input(
    f"Ações em carteira ({ticker})",
    min_value=0, max_value=1_000_000, value=0,
    help="Para cobrir a CALL vendida (covered call)."
)
cash_avail = st.sidebar.text_input(
    f"Caixa disponível (R$) ({ticker})",
    value="10.000,00",
    help="Para cobrir a PUT vendida (cash-secured)."
)
cash_avail = _br_to_float(cash_avail)

# ============== COLAR OPTION CHAIN ==============
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
        # fallback simples: uma coluna só -> tentar por linhas e split
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        rows = [re.split(r"\t|[ ]{2,}", ln.strip()) for ln in lines]
        df_raw = pd.DataFrame(rows[1:], columns=rows[0])

    # limpar
    df = _clean_dataframe(df_raw)

if df is not None and not df.empty:
    # lista de vencimentos
    exps = sorted(df["expiration"].dropna().unique().tolist())
    st.markdown("### 📅 Vencimento")
    chosen_exp = st.selectbox("Escolha um vencimento:", options=exps)

    # filtra por vencimento escolhido
    df_exp = df[df["expiration"] == chosen_exp].copy()

    # mostrar resumo leve (sem poluição)
    with st.expander("Ver prévia da cadeia (limpa)", expanded=False):
        st.dataframe(df_exp[["symbol","type","strike","last","impliedVol","delta","expiration"]], height=260, use_container_width=True)

    # pairing
    combos = _pair_strangles(df_exp, spot, mindelta, maxdelta)

    if not combos:
        st.warning("Não há CALL e PUT OTM suficientes com os filtros atuais. Ajuste |Δ| ou escolha outro vencimento.")
    else:
        dfc = pd.DataFrame(combos)

        # bandas por perna (usando |Δ|)
        def band_label(x):
            return "—" if pd.isna(x) else ("Baixo" if x*100 <= b_baixo else ("Médio" if x*100 <= b_medio else "Alto"))

        dfc["Banda_put"]  = dfc["Δ_put"].map(band_label)
        dfc["Banda_call"] = dfc["Δ_call"].map(band_label)

        # cobertura: CALL coberta por ações, PUT por caixa
        # lotes máximos
        max_lotes_call = qty_shares // contract_size if contract_size>0 else 0
        # risco put: alocação nominal ≈ K_put * contrato
        dfc["Aloc. PUT (R$) por lote"] = dfc["K_put"] * contract_size
        dfc["Lotes PUT cash-secured"]   = np.floor(cash_avail / dfc["Aloc. PUT (R$) por lote"]).astype(int)
        dfc["Lotes CALL cobertos"]      = max_lotes_call
        dfc["Lotes máx. cobertos"]      = dfc[["Lotes PUT cash-secured","Lotes CALL cobertos"]].min(axis=1)

        # rank simples: maior crédito / (distância relativa das asas até S)
        dist_rel = ((dfc["K_call"] - spot).abs() + (spot - dfc["K_put"]).abs())/max(spot,1e-6)
        dfc["score"] = dfc["Crédito num"] / (dist_rel.replace(0, np.nan))
        dfc = dfc.sort_values(["Lotes máx. cobertos","score","Crédito num"], ascending=[False, False, False])

        # dica de saída (usa proximidade e DTE)
        # dias até venc = diferença real
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
            if prox_kput:
                dicas.append("S encostando no **K_put** ⇒ recomprar a PUT")
            if prox_kcall:
                dicas.append("S encostando no **K_call** ⇒ recomprar a CALL")
            dicas.append(f"🎯 capturar ~{take_profit}% do crédito e encerrar")
            return " | ".join(dicas)

        dfc["Obs. saída"] = dfc.apply(saida_row, axis=1)
        dfc["Crédito total"] = dfc["Crédito num"]  # por ação; por lote multiplicar por contract_size

        # TOP 3
        top3 = dfc[dfc["Lotes máx. cobertos"]>0].head(3).copy()
        if top3.empty:
            top3 = dfc.head(3).copy()  # se não houver cobertura, mostra mesmo assim

        st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
        # Imprime no estilo “anterior”, com linhas simples e fortes, sem cards
        for i, rw in top3.reset_index(drop=True).iterrows():
            credito_lote = rw["Crédito num"] * contract_size
            st.markdown(
                f"**#{i+1} →** Vender **PUT {rw['PUT']} (K={rw['K_put']:.2f})** + "
                f"**CALL {rw['CALL']} (K={rw['K_call']:.2f})** "
                f"| **Crédito por lote:** **{_format_money(credito_lote)}** "
                f"| **Break-evens:** **[{rw['BE_inferior']:.2f}, {rw['BE_superior']:.2f}]** "
                f"| **PoE PUT:** {rw['PoE_put']} • **PoE CALL:** {rw['PoE_call']} "
                f"| **Bandas:** PUT **{rw['Banda_put']}** • CALL **{rw['Banda_call']}**\n"
                f"**Dica:** {rw['Obs. saída']}"
            )

        with st.expander("📋 Tabela completa (esta sessão)"):
            show = dfc.copy()
            show["Crédito por lote"] = show["Crédito num"] * contract_size
            cols = [
                "PUT","K_put","Δ_put","Banda_put","CALL","K_call","Δ_call","Banda_call",
                "Crédito (R$)","Crédito por lote","BE_inferior","BE_superior",
                "Lotes PUT cash-secured","Lotes CALL cobertos","Lotes máx. cobertos","Obs. saída"
            ]
            st.dataframe(show[cols], use_container_width=True, height=360)

        # ========== Comparar estratégias (resumo didático) ==========
        st.markdown("### 📈 Comparar estratégias (Strangle × Iron Condor × Jade Lizard)")
        # base = melhor do top3
        base = top3.iloc[0] if not top3.empty else dfc.iloc[0]
        Kp, Kc = base["K_put"], base["K_call"]
        credito = base["Crédito num"]
        asas_pct = st.slider("Largura das asas (% do preço à vista)", 2, 15, 8)
        asa_abs = (asas_pct/100.0)*spot
        Kp_w = max(0.01, Kp - asa_abs) # compra da PUT
        Kc_w = Kc + asa_abs            # compra da CALL

        # iron condor (aprox): perda máx por lado = (distância asa) - crédito
        perda_max_aprox = max((Kp - Kp_w), (Kc_w - Kc)) - credito
        perda_max_aprox = max(perda_max_aprox, 0)

        # jade lizard: sem risco de alta se crédito >= (Kc_w - Kc)
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


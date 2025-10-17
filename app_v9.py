# app_v9.py
# ============================================
# Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)
# - Pegar opção colada (tabela opcoes.net)
# - Escolher vencimento
# - Sugerir Strangles (Top 3) com detalhes didáticos
# - Aba "Comparar estratégias" (Strangle × Iron Condor × Jade Lizard)
# - Tooltips explicativos nos parâmetros (ajuda ao passar o mouse)
# ============================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------
# Utilidades numéricas
# ------------------------
SQRT_2 = math.sqrt(2.0)

def norm_cdf(x: float) -> float:
    if x is None or isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return float('nan')
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def d1_d2(S, K, r, sigma, T):
    try:
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return float('nan'), float('nan')
        num = math.log(S / K) + (r + 0.5 * sigma * sigma) * T
        den = sigma * math.sqrt(T)
        d1 = num / den
        d2 = d1 - den
        return d1, d2
    except Exception:
        return float('nan'), float('nan')

def prob_ITM_call(S, K, r, sigma, T):
    # P(S_T > K) ≈ N(d2) no world risk-neutral
    _, d2 = d1_d2(S, K, r, sigma, T)
    return norm_cdf(d2) if not math.isnan(d2) else float('nan')

def prob_ITM_put(S, K, r, sigma, T):
    # P(S_T < K) = 1 - P(S_T > K) ≈ 1 - N(d2)
    _, d2 = d1_d2(S, K, r, sigma, T)
    return 1.0 - norm_cdf(d2) if not math.isnan(d2) else float('nan')

# ------------------------
# Conversões e parsing
# ------------------------
def brazil_to_float(x):
    """
    Converte string no formato brasileiro para float.
    Exemplos:
      "1.234,56" -> 1234.56
      "0,42"     -> 0.42
      "—", "", None -> NaN
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "—" or s == "-":
        return np.nan
    # Remove espaços, substitui milhares ".", decimal ","
    s = s.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_date_ptbr(x):
    """Converte 'dd/mm/aaaa' para 'YYYY-MM-DD'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # Já em ISO?
    try:
        # tenta aaaa-mm-dd
        dt = pd.to_datetime(s, errors="raise", dayfirst=False)
        return dt.date().isoformat()
    except Exception:
        pass
    # tenta dd/mm/aaaa
    try:
        dt = pd.to_datetime(s, format="%d/%m/%Y", errors="raise", dayfirst=True)
        return dt.date().isoformat()
    except Exception:
        # últimas tentativas genéricas
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            return dt.date().isoformat() if pd.notna(dt) else np.nan
        except Exception:
            return np.nan

def padronizar_colunas(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia cabeçalhos do opcoes.net (PT-BR) para colunas padronizadas:
      symbol, type, strike, last, mid, impliedVol, delta, expiration
    """
    rename_map = {}
    cols = {c.strip(): c for c in df_raw.columns}

    def has(name):
        return name in cols

    # Mapeamentos típicos do opcoes.net:
    if has("Ticker"):            rename_map[cols["Ticker"]] = "symbol"
    if has("Tipo"):              rename_map[cols["Tipo"]] = "type"
    if has("Strike"):            rename_map[cols["Strike"]] = "strike"
    if has("Último"):            rename_map[cols["Último"]] = "last"
    if has("Vol. Impl. (%)"):    rename_map[cols["Vol. Impl. (%)"]] = "impliedVol"
    if has("Delta"):             rename_map[cols["Delta"]] = "delta"
    if has("Vencimento"):        rename_map[cols["Vencimento"]] = "expiration"

    df = df_raw.rename(columns=rename_map).copy()

    # Normaliza 'type' -> 'C'/'P'
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip()
        df["type"] = df["type"].replace({"CALL": "C", "PUT": "P"})

    # Numeric conversions
    if "strike" in df.columns:
        df["strike"] = df["strike"].apply(brazil_to_float)
    if "last" in df.columns:
        df["last"] = df["last"].apply(brazil_to_float)
    # mid não existe no site — manter como NaN e usar last como fallback
    df["mid"] = np.nan
    # impliedVol: percentual -> decimal
    if "impliedVol" in df.columns:
        df["impliedVol"] = df["impliedVol"].apply(brazil_to_float) / 100.0
    # delta (pode vir com virgula)
    if "delta" in df.columns:
        df["delta"] = df["delta"].apply(brazil_to_float)

    # expiration -> ISO
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].apply(parse_date_ptbr)

    # Tira linhas totalmente vazias
    df = df.dropna(how="all")
    return df

def parse_pasted_table(text: str) -> pd.DataFrame:
    """
    Lê a tabela colada (normalmente tabs) em DataFrame e padroniza colunas.
    """
    # Muitas vezes o opcoes.net usa \t; usaremos regex para tabs e múltiplos espaços
    try:
        data = pd.read_csv(io.StringIO(text), sep=r"\t+|\s{2,}", engine="python")
    except Exception:
        # fallback: qualquer separador de espaços
        data = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")

    # Remove colunas "em branco" criadas por separador
    data = data.loc[:, ~data.columns.astype(str).str.contains("^Unnamed")]
    return padronizar_colunas(data)

# ------------------------
# Cobertura e score
# ------------------------
def band_from_prob(prob_ex):
    if pd.isna(prob_ex):
        return "—"
    x = 100.0 * float(prob_ex)
    if x <= 15: return "Baixa"
    if x <= 35: return "Média"
    return "Alta"

def safe_mid(row):
    val = row.get('mid', np.nan)
    if pd.isna(val) or not isinstance(val, (int, float)):
        val = row.get('last', np.nan)
    try:
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        return 0.0

def fmt_money(x): 
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")
    except Exception:
        return "R$ 0,00"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{float(x)*100:.1f}%"

def covered_lots_limits(Kp, contract_size, shares_owned, cash_available):
    max_lots_call = max(0, int(shares_owned // contract_size))  # CALL coberta por ações
    max_lots_put  = max(0, int((cash_available // 1) // (Kp * contract_size)))  # PUT coberta por caixa no strike
    return max_lots_call, max_lots_put

def select_top3(puts, calls, spot, r, sigma, T, contract_size, shares_owned, cash_available, topn=3):
    if puts.empty or calls.empty:
        return pd.DataFrame()

    p_df = puts.copy()
    c_df = calls.copy()
    p_df["prem_put"]  = p_df.apply(safe_mid, axis=1)
    c_df["prem_call"] = c_df.apply(safe_mid, axis=1)

    combos = []
    for _, prow in p_df.iterrows():
        for _, crow in c_df.iterrows():
            try:
                Kp = float(prow["strike"]); Kc = float(crow["strike"])
            except Exception:
                continue
            prem_p = float(prow["prem_put"]); prem_c = float(crow["prem_call"])
            credito = prem_p + prem_c

            p_ex_call = prob_ITM_call(spot, Kc, r, sigma, T)
            p_ex_put  = prob_ITM_put(spot, Kp, r, sigma, T)

            banda_c = band_from_prob(p_ex_call)
            banda_p = band_from_prob(p_ex_put)

            be_low  = Kp - credito
            be_high = Kc + credito

            max_call, max_put = covered_lots_limits(Kp, contract_size, shares_owned, cash_available)
            max_lots = min(max_call, max_put)

            score = credito * (1.0 - 0.5 * ((p_ex_call if not pd.isna(p_ex_call) else 0) + (p_ex_put if not pd.isna(p_ex_put) else 0)))

            combos.append({
                "Vencimento": prow.get("expiration", crow.get("expiration","")),
                "PUT": prow.get("symbol",""),
                "K_put": Kp,
                "Prêmio PUT": prem_p,
                "Δ_put": abs(float(prow.get("delta", np.nan))) if "delta" in prow else np.nan,
                "PoE_put": p_ex_put,
                "Banda_put": banda_p,

                "CALL": crow.get("symbol",""),
                "K_call": Kc,
                "Prêmio CALL": prem_c,
                "Δ_call": abs(float(crow.get("delta", np.nan))) if "delta" in crow else np.nan,
                "PoE_call": p_ex_call,
                "Banda_call": banda_c,

                "Crédito total": credito,
                "BE_inferior": be_low,
                "BE_superior": be_high,

                "Lotes máx. cobertos": max_lots,
                "Score": score
            })

    dfc = pd.DataFrame(combos)
    if dfc.empty:
        return dfc

    dfc = dfc[(dfc["Crédito total"] > 0.0) & (dfc["Lotes máx. cobertos"] > 0)]
    if dfc.empty:
        return dfc

    dfc = dfc.sort_values(["Score", "Crédito total"], ascending=[False, False]).head(topn)

    dfc["Crédito (R$)"] = dfc["Crédito total"] * contract_size
    dfc["Obs. saída"] = dfc.apply(
        lambda r: (f"Sair se S≈{r['K_put']:.2f} (PUT) ou S≈{r['K_call']:.2f} (CALL), "
                   f"ou encerrar após capturar ~70–80% do crédito (≈{fmt_money(0.75*r['Crédito (R$)'])})."),
        axis=1
    )

    # formatações
    for col in ["K_put","K_call","Prêmio PUT","Prêmio CALL","Crédito total","BE_inferior","BE_superior"]:
        dfc[col] = dfc[col].astype(float).round(2)
    for col in ["Δ_put","Δ_call"]:
        if col in dfc: dfc[col] = dfc[col].astype(float).round(2)
    for col in ["PoE_put","PoE_call"]:
        dfc[col] = dfc[col].astype(float)

    dfc["PoE_put"]  = dfc["PoE_put"].map(fmt_pct)
    dfc["PoE_call"] = dfc["PoE_call"].map(fmt_pct)
    dfc["Crédito (R$)"] = dfc["Crédito (R$)"].map(fmt_money)

    return dfc[[
        "Vencimento",
        "PUT","K_put","Prêmio PUT","Δ_put","PoE_put","Banda_put",
        "CALL","K_call","Prêmio CALL","Δ_call","PoE_call","Banda_call",
        "Crédito total","Crédito (R$)","BE_inferior","BE_superior",
        "Lotes máx. cobertos","Obs. saída"
    ]]

# ------------------------
# Payoff helpers
# ------------------------
def payoff_strangle(S, Kp, Kc, credito):
    # Π = -max(0,Kp-S) - max(0,S-Kc) + credito
    return -np.maximum(0, Kp - S) - np.maximum(0, S - Kc) + credito

def payoff_iron_condor(S, Kp, Kc, Kp_w, Kc_w, credito_condor):
    # Strangle + asas (long put em Kp_w, long call em Kc_w)
    # custo_das_asas embutido no 'credito_condor' (crédito líquido)
    return (-np.maximum(0, Kp - S) - np.maximum(0, S - Kc)
            + np.maximum(0, Kp_w - S) + np.maximum(0, S - Kc_w)
            + credito_condor)

def payoff_jade_lizard(S, Kp, Kc, Kc_w, credito_jl):
    # PUT vendida + CALL vendida + CALL comprada (Kc_w)
    # se crédito >= (Kc_w - Kc) => sem risco de alta
    return (-np.maximum(0, Kp - S) - np.maximum(0, S - Kc)
            + np.maximum(0, S - Kc_w) + credito_jl)

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Strangle Vendido Coberto — v9", page_icon="💼", layout="wide")

st.title("💼 Strangle Vendido Coberto — v9 (colar tabela do opcoes.net)")
st.caption("Cole a option chain do opcoes.net, escolha o vencimento e veja as sugestões didáticas de strangle coberto e a comparação de estratégias.")

# 1) Ticker básico (apenas “estético”, já que a cadeia virá colada)
tickers_b3 = ["PETR4", "VALE3", "BBDC4", "ITUB4", "ABEV3", "CSNA3", "MGLU3", "WEGE3"]
col_tk1, col_tk2, col_spot = st.columns([1,1,1.2])
with col_tk1:
    ticker = st.selectbox("🔎 Escolha um ticker da B3", tickers_b3, index=2)
with col_tk2:
    st.text_input("Ticker (livre)", value=ticker, key="ticker_free")
with col_spot:
    spot = st.number_input("Preço à vista (S)", min_value=0.01, value=17.68, step=0.01, format="%.2f", help="Preço atual do ativo. Você pode editar manualmente se quiser.")

# Sidebar — Parâmetros com help
st.sidebar.header("⚙️ Parâmetros")

r_anual = st.sidebar.number_input(
    "Taxa livre de risco anual (r)",
    min_value=0.0, max_value=1.0, value=0.11, step=0.01, format="%.2f",
    help="Usada no Black–Scholes. No Brasil, aproxime pela SELIC anualizada. Ex.: 0,11 = 11% a.a."
)

iv_proxy = st.sidebar.number_input(
    "HV20 / IV anual (proxy, σ)",
    min_value=0.0, max_value=3.0, value=0.20, step=0.01, format="%.2f",
    help="Volatilidade anual de referência. Se a cadeia colada tiver IV por opção, ela será usada por perna; caso contrário, usa-se este valor."
)

delta_min = st.sidebar.number_input(
    "|Δ| mínimo", min_value=0.00, max_value=1.00, value=0.05, step=0.01, format="%.2f",
    help="Filtro de ‘moneyness’ por |Delta|. Vendedores costumam usar ~0,05–0,35 (opções OTM)."
)
delta_max = st.sidebar.number_input(
    "|Δ| máximo", min_value=0.00, max_value=1.00, value=0.35, step=0.01, format="%.2f",
    help="Limite superior de |Delta| para considerar na sugestão."
)

risk_bands_help = (
    "Classificação didática pela probabilidade de exercício (PoE) de cada perna:\n"
    "• Baixa: 0–15%\n• Média: 15–35%\n• Alta: 35–55%\n"
    "A sugestão cruza CALL e PUT na mesma banda."
)
st.sidebar.text_input("Bandas de risco por perna", value="Baixa / Média / Alta", help=risk_bands_help, disabled=True)

# Cobertura
st.sidebar.header("🛡️ Cobertura")
shares_owned = st.sidebar.number_input("Ações em carteira (CALL coberta)", min_value=0, value=1000, step=100,
                                       help="Quantidade de ações disponíveis para cobrir a CALL.")
cash_available = st.sidebar.number_input("Caixa disponível (PUT coberta) — R$", min_value=0.0, value=10000.0, step=500.0, format="%.2f",
                                         help="Valor em caixa para cobrir a PUT no strike (compra de 100 ações por contrato).")
contract_size = st.sidebar.number_input("Tamanho do contrato", min_value=1, value=100, step=1,
                                        help="No Brasil, normalmente 100 ações por contrato.")

st.sidebar.header("🚪 Instruções de SAÍDA")
exit_help = (
    "Recomprar a perna ameaçada quando o preço encostar no strike perto do vencimento, "
    "ou encerrar após capturar ~70–80% do prêmio."
)
st.sidebar.text_input("Regras práticas", value="Encosta no strike ou 70–80% do prêmio capturado.", help=exit_help, disabled=True)

# 2) Colar a option chain
st.markdown("### 3) Colar a option chain do **opcoes.net** (Ctrl/Cmd+C no site → Ctrl/Cmd+V aqui)")
pasted = st.text_area("Cole aqui a tabela", height=260, placeholder="Cole aqui a tabela completa (CALLs e PUTs) do opcoes.net")

if pasted.strip():
    try:
        df_all = parse_pasted_table(pasted)
    except Exception as e:
        st.error(f"Falha ao interpretar a tabela colada: {e}")
        st.stop()

    # Verifica colunas essenciais
    essential = ["symbol","type","strike","last","expiration"]
    missing = [c for c in essential if c not in df_all.columns]
    if missing:
        st.error(f"Colunas essenciais ausentes: {missing}. Verifique se colou a tabela completa.")
        st.stop()

    # Se não houver impliedVol por opção, usaremos iv_proxy
    df_all["impliedVol"] = df_all["impliedVol"] if "impliedVol" in df_all.columns else np.nan
    # delta pode faltar
    if "delta" not in df_all.columns:
        df_all["delta"] = np.nan

    # Vencimentos disponíveis
    vencs = sorted([v for v in df_all["expiration"].dropna().unique()])
    st.markdown("### 📅 Vencimento")
    chosen = st.selectbox("Escolha um vencimento", vencs, index=0 if vencs else None)

    # Filtra por vencimento
    df = df_all[df_all["expiration"] == chosen].copy()

    # Define sigma por perna: se faltou IV individual, usa proxy (iv_proxy)
    df["sigma"] = df["impliedVol"].fillna(iv_proxy)

    # Tempo até o vencimento (anos)
    try:
        T = (pd.to_datetime(chosen) - pd.Timestamp.today()).days / 365.0
        T = max(T, 1/365)  # evita zero
    except Exception:
        T = 30/365

    # Filtros OTM e Δ
    df["abs_delta"] = df["delta"].abs()
    # Se delta não existe, não filtrar por delta obrigatoriamente (considera todos) — mas limitará depois pelo OTM.
    if df["abs_delta"].notna().any():
        df = df[(df["abs_delta"].isna()) | ((df["abs_delta"] >= delta_min) & (df["abs_delta"] <= delta_max))]

    calls = df[(df["type"] == "C") & (df["strike"] > spot)].copy()
    puts  = df[(df["type"] == "P") & (df["strike"] < spot)].copy()

    # Top 3
    st.markdown("### 🏆 Top 3 (melhor prêmio/risco)")
    top3 = select_top3(
        puts=puts, calls=calls, spot=spot, r=r_anual,
        sigma=float(iv_proxy),  # para PoE na rankeação; já que σ por opção pode faltar
        T=T, contract_size=contract_size, shares_owned=shares_owned, cash_available=cash_available
    )

    if top3.empty:
        st.warning("Não há strangles cobertos viáveis com os filtros/limites atuais. "
                   "Ajuste |Δ|, verifique prêmios (‘Último’) e cobertura (ações/caixa).")
    else:
        st.dataframe(top3, use_container_width=True, hide_index=True)
        for i, row in top3.iterrows():
            st.markdown(
                f"**#{i+1}** → Vender **PUT {row['PUT']} (K={row['K_put']:.2f})** + "
                f"**CALL {row['CALL']} (K={row['K_call']:.2f})** | "
                f"Crédito por lote: **{row['Crédito (R$)']}** | "
                f"Break-evens: **[{row['BE_inferior']:.2f}, {row['BE_superior']:.2f}]** | "
                f"PoE PUT: **{row['PoE_put']}**, PoE CALL: **{row['PoE_call']}** | "
                f"Lotes cobertos: **{int(row['Lotes máx. cobertos'])}**  \n"
                f"_Dica_: {row['Obs. saída']}"
            )

        # 4) Visualizações e comparação
        st.markdown("---")
        st.markdown("### 📈 Payoff no Vencimento (P/L por ação) + Comparar estratégias")

        tabs = st.tabs(["📈 Payoff Strangle", "🔀 Comparar estratégias", "📘 Explicações & Fórmulas"])

        # Base: usa a #1 do Top 3
        base = top3.iloc[0]
        Kp_base = float(base["K_put"])
        Kc_base = float(base["K_call"])
        credito_base = float(base["Crédito total"])

        S_min = max(0.01, spot * 0.7)
        S_max = spot * 1.3
        S_grid = np.linspace(S_min, S_max, 301)

        with tabs[0]:
            pl = payoff_strangle(S_grid, Kp_base, Kc_base, credito_base)
            fig = plt.figure()
            plt.axhline(0, linewidth=1)
            plt.axvline(Kp_base, linestyle="--", linewidth=1)
            plt.axvline(Kc_base, linestyle="--", linewidth=1)
            plt.plot(S_grid, pl, linewidth=2)
            plt.title(f"Strangle: Kp={Kp_base:.2f}, Kc={Kc_base:.2f} | crédito≈{credito_base:.2f}/ação")
            plt.xlabel("Preço no vencimento (S_T)")
            plt.ylabel("P/L por ação")
            st.pyplot(fig, clear_figure=True)

        with tabs[1]:
            st.subheader("Strangle × Iron Condor × Jade Lizard")
            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown(f"**Strangle base**: {Kp_base:.2f}–{Kc_base:.2f} | crédito≈{credito_base:.2f}")
            with colB:
                # define asas % do preço atual (simples)
                wing_pct = st.slider("Largura das asas (% do preço à vista)", 2, 20, 10, 1)
            with colC:
                st.write("")

            # Iron Condor: compra PUT abaixo e CALL acima
            Kp_w = max(0.01, Kp_base - (wing_pct/100.0)*spot)
            Kc_w = Kc_base + (wing_pct/100.0)*spot
            # custo das asas não conhecido; assuma pequeno e gere um crédito líquido próximo do strangle:
            credito_condor = max(0.0, credito_base - 0.02*spot)  # ajuste simples

            # Jade Lizard: PUT vendida (Kp), CALL vendida (Kc), CALL comprada (Kc_w)
            credito_jl = credito_base * 0.95  # suposição simples
            jl_no_risco_alta = credito_jl >= (Kc_w - Kc_base)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strangle — Crédito", fmt_money(credito_base * contract_size))
                st.write(f"Zona neutra (Kp–Kc): **{Kp_base:.2f} — {Kc_base:.2f}**")
            with col2:
                st.metric("Iron Condor — Crédito", fmt_money(credito_condor * contract_size))
                st.write(f"Asas (P,C): **{Kp_w:.2f}**, **{Kc_w:.2f}**")
            with col3:
                st.metric("Jade Lizard — Crédito", fmt_money(credito_jl * contract_size))
                st.write(f"Asa (CALL): **{Kc_w:.2f}** — Sem risco de alta? **{'Sim' if jl_no_risco_alta else 'Não'}**")

            # Gráficos
            pl_s = payoff_strangle(S_grid, Kp_base, Kc_base, credito_base)
            pl_c = payoff_iron_condor(S_grid, Kp_base, Kc_base, Kp_w, Kc_w, credito_condor)
            pl_j = payoff_jade_lizard(S_grid, Kp_base, Kc_base, Kc_w, credito_jl)

            fig2 = plt.figure()
            plt.axhline(0, linewidth=1)
            plt.plot(S_grid, pl_s, linewidth=2, label="Strangle")
            plt.plot(S_grid, pl_c, linewidth=2, label="Iron Condor")
            plt.plot(S_grid, pl_j, linewidth=2, label="Jade Lizard")
            plt.legend()
            plt.title("Comparação de Payoffs (por ação)")
            plt.xlabel("Preço no vencimento (S_T)")
            plt.ylabel("P/L por ação")
            st.pyplot(fig2, clear_figure=True)

        with tabs[2]:
            st.markdown("#### 📘 Explicações rápidas")
            st.markdown(
                f"""
- **Strangle**: vender PUT (Kp={Kp_base:.2f}) + CALL (Kc={Kc_base:.2f}).  
  **Lucro máx.** ≈ crédito. **Break-evens**: {Kp_base-credito_base:.2f} e {Kc_base+credito_base:.2f}.  
  **Perigo**: encostar Kp (queda) ou Kc (alta).

- **Iron Condor**: Strangle + compra de uma PUT mais baixa (Kp_w={Kp_w:.2f}) e uma CALL mais alta (Kc_w={Kc_w:.2f}).  
  **Objetivo**: limitar perda máxima.

- **Jade Lizard**: PUT vendida + CALL vendida + CALL comprada em Kc_w={Kc_w:.2f}.  
  Se **crédito ≥ (Kc_w − Kc)** ⇒ **sem risco de alta**.  
  Aqui: crédito≈{credito_jl:.2f} {'≥' if jl_no_risco_alta else '<'} {(Kc_w-Kc_base):.2f} ⇒ **{'Sim' if jl_no_risco_alta else 'Não'}**.

- **Saída** (geral): recomprar a perna ameaçada quando S encostar no strike perto do vencimento,  
  ou encerrar após capturar ~70–80% do prêmio.
                """
            )

else:
    st.info("Cole a tabela do **opcoes.net** acima (todas as linhas das CALLs e PUTs). Em seguida, escolha o vencimento.")

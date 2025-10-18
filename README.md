# 💼 Strangle Vendido Coberto — v9

> **App educacional e prático** para montar *strangles cobertos* na B3 com leitura de option chain, cálculo de prêmio/risco, probabilidades e recomendações didáticas.

<p align="center">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-%F0%9F%8C%88-red?style=flat">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python">
  <img alt="Market" src="https://img.shields.io/badge/Market-B3-black?style=flat">
  <img alt="License" src="https://img.shields.io/badge/Use-Educacional-green?style=flat">
</p>

---

## 🧭 Visão Geral
O **Strangle Vendido Coberto — v9** é um aplicativo em **Python + Streamlit** para **investidores da B3 (Brasil)** que buscam **renda com controle de risco** vendendo **PUT + CALL OTM** (strangle) de forma **coberta**.  
Você cola a *option chain* (opcoes.net.br), o app identifica vencimentos, filtra OTM, calcula **prêmio**, **break‑evens**, **PoE (prob. de exercício)** e entrega um **Top 3** claro e didático, com **regras de saída**.

> ⚠️ **Aviso**: projeto **didático**, não é recomendação de investimento.

---

## ✨ Destaques
- 📊 **Leitura automática** da planilha de opções (colar do opcoes.net.br)
- 🧾 **Busca de tickers dinâmica** (ticker + nome) via dadosdemercado.com.br
- 💹 **Preço à vista automático** via `yfinance`
- 🔍 **Detecção de vencimentos** e separação por tipo (CALL/PUT)
- 💰 **Cálculo de prêmio** (PUT + CALL, por ação e total)
- 🧮 **Break‑evens** + **PoE** (modelo Black–Scholes)
- 🧱 **Filtro OTM** e ranqueamento por **prêmio/risco**
- 🏆 **Top 3** com explicações simples e regras práticas de saída
- 💼 **Simulação de lotes** → mostra **prêmio total em R$**

---

## 🔗 Sumário
- [🧭 Visão Geral](#-visão-geral)
- [✨ Destaques](#-destaques)
- [🧮 Como o app calcula](#-como-o-app-calcula)
- [📘 Glossário Rápido](#-glossário-rápido)
- [🧩 Estruturas](#-estruturas)
  - [Strangle Vendido Coberto](#strangle-vendido-coberto)
  - [Iron Condor](#iron-condor)
  - [Jade Lizard](#jade-lizard)
- [🧰 Fluxo do App](#-fluxo-do-app)
- [🏆 Top 3 — Como interpretar](#-top-3--como-interpretar)
- [💡 Dicas de uso](#-dicas-de-uso)
- [⚙️ Parâmetros (Sidebar)](#️-parâmetros-sidebar)
- [🚀 Como rodar](#-como-rodar)
- [🗺️ Roadmap](#️-roadmap)
- [📚 Créditos](#-créditos)

---

## 🧮 Como o app calcula

### 💰 Prêmio (por ação)
```
Crédito/ação = Prêmio PUT + Prêmio CALL
```

### 📏 Break‑evens
```
Inferior  = Strike_PUT  – Crédito/ação
Superior  = Strike_CALL + Crédito/ação
```

### 🎲 Probabilidade de Exercício (PoE)
Modelo **Black–Scholes** (σ da cadeia quando disponível; HV20 como proxy):
```
CALL: PoE ≈ N(d2)      |  PUT: PoE ≈ N(−d2)
d2 = [ ln(S/K) + (r − 0.5*σ²)*T ] / (σ*√T)
```

### 💼 Prêmio total (R$)
```
Prêmio total = (Crédito/ação) × (Contrato=100) × (Lotes)
```

---

## 📘 Glossário Rápido

| Sigla | Descrição |
|---|---|
| **S (Spot)** | Preço atual da ação (via yfinance) |
| **Strike** | Preço de exercício da opção |
| **IV** | Volatilidade implícita (quando disponível na cadeia) |
| **HV20** | Volatilidade histórica anualizada (proxy) |
| **PoE** | Prob. de exercício no vencimento (expirar ITM) |
| **OTM/ITM** | Fora/Dentro do dinheiro |
| **Contrato** | 100 ações na B3 |

---

## 🧩 Estruturas

### Strangle Vendido Coberto
> Vende **1 PUT OTM** + **1 CALL OTM**.  
> Melhor quando o mercado tende a **lateralizar** e IV está **elevada**.

**Pontos‑chave**
- Lucro máx. = **crédito recebido**
- Risco: ser exercido em uma das pontas
- Ajustes comuns: **rolagem** e **recompra** de uma perna

---

### Iron Condor
> Strangle + asas (compra de PUT/CALL mais fora) → **perda máxima limitada**.

**Pontos‑chave**
- Lucro menor, risco **controlado**
- Indicado p/ **volatilidade incerta**

---

### Jade Lizard
> PUT vendida + CALL vendida + **CALL comprada** mais acima  
> Se **crédito ≥ (K_call_comp − K_call_ven)** ⇒ **sem risco na alta**.

**Pontos‑chave**
- Mantém risco de **queda**
- Útil se você **aceita** comprar as ações

---

## 🧰 Fluxo do App

1) **Escolha o ticker** pelo nome OU código (lista dinâmica da B3)  
2) **Preço à vista** do ativo via **yfinance** (automático)  
3) **Cole** a planilha da *option chain* (opcoes.net.br)  
4) **Selecione o vencimento** (detectado da planilha)  
5) O app filtra **OTM**, calcula **prêmio/PoE/BE** e mostra o **Top 3**  
6) Informe **lotes** para ver **prêmio total**

> 💡 Se a cotação parecer defasada, recarregue a página (cache curto).

---

## 🏆 Top 3 — Como interpretar

| Coluna | Explicação |
|---|---|
| **PUT / CALL** | Códigos das opções sugeridas |
| **Strike (R$)** | Strike da PUT e CALL |
| **Prêmio PUT / CALL (R$)** | Valor unitário de cada perna |
| **Crédito/ação (R$)** | **Prêmio PUT + CALL**, por ação |
| **Break‑evens (mín–máx)** | Faixa de preço onde o P/L ≥ 0 no vencimento |
| **PoE PUT / CALL (%)** | Probabilidade estimada de exercício (queda/subida) |

> **Regras práticas de saída**  
> ⏳ faltam ≤ 7 dias → atenção redobrada  
> 📈 se **S** encostar no **Strike da CALL**, **recomprar a CALL**  
> 🎯 ao capturar **~75% do crédito**, **encerrar**

---

## 💡 Dicas de uso

| Situação | Preferir |
|---|---|
| Mercado lateral + IV alta | **Strangle** |
| Alta incerteza de volatilidade | **Iron Condor** |
| Viés de alta moderada | **Jade Lizard** |
| IV muito baixa | **Evitar vender** |

---

## ⚙️ Parâmetros (Sidebar)

- **HV20 (σ anual – proxy)**: usada quando a IV não está disponível
- **r (anual)**: taxa livre de risco (aprox. SELIC)
- **Ações em carteira** e **Caixa**: definem cobertura
- **Tamanho do contrato**: 100 (B3)
- **Dias até vencimento (alerta)** e **Meta de captura do prêmio**

---

## 🚀 Como rodar

### Local (Windows/macOS/Linux)
```bash
pip install -U streamlit yfinance pandas numpy requests beautifulsoup4 lxml
streamlit run app_v9.py
```

### Streamlit Cloud
1. Conecte seu repositório (com `app_v9.py` na raiz)
2. Defina o arquivo principal: `app_v9.py`
3. `requirements.txt` sugerido:
   ```txt
   streamlit
   yfinance
   pandas
   numpy
   requests
   beautifulsoup4
   lxml
   ```
4. Deploy 🚀

---

## 🗺️ Roadmap
- 🔔 **Alertas automáticos** (Telegram/e-mail)
- 🧰 **Histórico** de operações (log + export)
- 🎲 **Monte Carlo** para PoE dinâmico
- 🤖 **ML** (XGBoost/LSTM) para sinais de reversão
- 🔌 Integrações (Salesforce Flow / APIs)

---

## 📚 Créditos
Projeto didático com inspiração em práticas de educação financeira no Brasil e fundamentos de **finanças quantitativas** (Black–Scholes, gregos, IV/HV).  
**Stack**: Python 3.11 · Streamlit · yfinance · pandas/numpy · bs4/lxml.

> **Autor(es)**: ChatGPT + Colaborador(a)  
> **Licença**: Uso educacional — sem garantia, sem recomendação de investimento.

---

### 🧷 Anexos úteis
- Site para colar a cadeia: https://opcoes.net.br  
- Lista de ações da B3: https://www.dadosdemercado.com.br/acoes
- API de preços: `yfinance`

> Dúvidas, ideias ou PRs são muito bem‑vindos! 😉

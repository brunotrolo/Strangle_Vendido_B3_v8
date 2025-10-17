
# 💼 Strangle Vendido Coberto — Versão 8 (v8)

## 📘 Visão Geral
O **Strangle Vendido Coberto v8** é um aplicativo educacional e operacional desenvolvido em **Python + Streamlit**, projetado para **investidores que vendem opções cobertas (CALL e PUT)** e desejam **maximizar o prêmio recebido com baixo risco de exercício**.

A versão 8 é a mais didática e completa até agora, incluindo:

- ⚙️ **Sugestões inteligentes** de *strangles vendidos cobertos* (CALL + PUT)
- 📊 **Ranking prêmio/risco** com *PoE_total*, *Score* e *Retorno %*
- 🧠 **Explicações automáticas de saída**, ensinando *quando e como zerar o risco*
- 🔔 **Alertas visuais** (⚠️) quando o ativo se aproxima dos strikes
- 📘 **Instruções operacionais de saída** (quanto recomprar, quando encerrar)
- 🏆 **Top 3 recomendações** por risco
- 🔍 **Catálogo automático de tickers da B3**
- 📈 **Payoff visual interativo**
- 🧮 **Cálculo de probabilidade de exercício (PoE) com modelo Black‑Scholes**
- 🔗 **Integração automática com o site [opcoes.net.br](https://opcoes.net.br)** para baixar a chain

---

## 🧩 Conceitos Didáticos (Resumo)

| Conceito | Explicação |
|-----------|------------|
| **Strangle vendido coberto** | Venda simultânea de 1 CALL OTM (coberta por ações em carteira) e 1 PUT OTM (coberta por caixa garantido). |
| **Objetivo** | Capturar o maior prêmio possível com baixa probabilidade de exercício. |
| **PoE_total** | Probabilidade combinada de qualquer perna ser exercida (`probITM_call + probITM_put`). |
| **PoE_dentro** | Probabilidade do preço ficar entre os strikes no vencimento (`1 − PoE_total`). |
| **Retorno potencial (%)** | Prêmio recebido ÷ preço do ativo. |
| **Score (prêmio/risco)** | Métrica inteligente = `Retorno % ÷ (risk_score + 0.01)`. |
| **IV Rank / IV Percentil** | Mede se a volatilidade atual está alta ou baixa comparada ao histórico. |

---

## 🔔 Estratégia de Saída e Rolagem (NOVO na v8)

Uma das maiores evoluções da v8 é o módulo **didático de saída**, que ajuda o investidor a **zerar o risco** e **ficar com parte do prêmio** quando a operação se aproxima do strike.

### 🧭 Quando sair da operação?
- Quando o **preço do ativo** está **a menos de ±5% do strike** (configurável), **e** faltam **≤ 10 dias para o vencimento**.  
- Ou quando você já **capturou 70–80% do prêmio máximo**.  
- Nesse ponto, a relação risco/retorno se torna desfavorável — o prêmio restante é pequeno, mas o risco de exercício é alto.

### 💡 Como encerrar (zerar o risco)
- **Recomprar a opção vendida ameaçada** (CALL se o preço subir, PUT se cair).  
- Assim, você mantém a maior parte do prêmio e **evita ser exercido**.  
- Se quiser manter a estratégia, pode **rolar a perna ameaçada** para o próximo vencimento, com strike mais OTM.

### 📘 Explicação automática no app
Para cada strangle sugerido, o app mostra:
> ⚠️ CALL ameaçada: preço próximo de K_call.  
> 🕐 Faltam 6 dias; regra didática: **recomprar a CALL** para zerar o risco.  
> 💰 Encerrar quando capturar 70% do prêmio.  
> 🔧 Ordem sugerida: recomprar por ~R$ 0,20/ação (20% do crédito).

Essas instruções são automáticas e variam conforme o ativo, o strike e o tempo até o vencimento.

---

## 🧮 Campos da Tabela Principal

| Coluna | Descrição |
|---------|-----------|
| **Ticker** | Código do ativo (ex: PETR4) |
| **CALL / PUT** | Códigos das opções sugeridas |
| **K_call / K_put** | Preços de exercício das pernas |
| **probITM_call / put** | Probabilidade de cada perna ser exercida |
| **Δ (delta)** | Sensibilidade aproximada da opção ao preço do ativo |
| **Crédito/ação** | Soma dos prêmios recebidos pelas duas pernas |
| **Retorno %** | Retorno estimado sobre o valor do ativo |
| **PoE_total** | Chance combinada de ser exercido |
| **PoE_dentro** | Chance de o preço ficar entre os strikes |
| **Score** | Ranking de atratividade (prêmio/risco) |
| **📘 Instrução de saída** | Texto didático explicando quando e como sair |
| **⚠️ Alerta** | Mostra se há risco imediato de exercício |

---

## ⚙️ Como Usar

1. Execute localmente:
   ```bash
   pip install -r requirements.txt
   streamlit run app_v8.py
   ```

2. No painel lateral:
   - Escolha **tickers** (ex: PETR4, VALE3, ITUB4)
   - Defina suas **ações em carteira** e **caixa disponível**
   - Ajuste a **proximidade ao strike (%)**, **dias críticos** e **meta de captura do prêmio**
   - Marque “Mostrar instruções operacionais de saída”

3. O app exibirá:
   - **Resumo do ticker**
   - **Top 3 recomendações**
   - **Tabela completa de strangles sugeridos**
   - **Payoff visual**
   - **Explicações automáticas de saída**

---

## 🏦 Exemplo Prático — PETR4

Suponha que PETR4 está em **R$ 38,00**, e você venda:

- CALL 40 (strike acima) → prêmio R$ 0,45  
- PUT 36 (strike abaixo) → prêmio R$ 0,35  
- Crédito total: R$ 0,80/ação (R$ 80 por contrato)

**Cenário:**  
- Se PETR4 permanecer entre 36 e 40 até o vencimento → você ganha todo o prêmio (R$ 80).  
- Se subir até ~R$ 39,80 faltando 5 dias → recomprar a CALL (zerar o risco).  
- Lucro líquido ≈ 70–80% do prêmio, risco eliminado.

---

## 📈 Interpretação do Payoff

- A linha azul representa o **lucro/prejuízo por ação** no vencimento.  
- As linhas pontilhadas mostram os **strikes (K_put e K_call)**.  
- A região central é o intervalo “seguro” — onde o preço pode oscilar sem exercício.

---

## 🧮 Fórmulas Utilizadas

- **d₁ =** ln(S/K) + (r + σ²/2)T / (σ√T)  
- **d₂ =** d₁ − σ√T  
- **probITM_call = N(d₂)**  
- **probITM_put = 1 − N(d₂)**  
- **PoE_total = probITM_call + probITM_put**  
- **Retorno % = (prêmio total ÷ preço do ativo)**  
- **Score = Retorno % ÷ (risk_score + 0.01)**

---

## 🧭 Boas Práticas e Dicas

- Prefira **strangles OTM**, com **PoE_total < 40%**.  
- Evite strikes muito próximos do preço atual quando o vencimento está próximo.  
- Utilize a **função de exportar CSV** para acompanhar seus trades.  
- Capture **70–80% do prêmio** e **encerre antes do vencimento**.  
- Se quiser manter a posição, **role** para o próximo vencimento.

---

## 🌐 Fontes de Dados
- [opcoes.net.br](https://opcoes.net.br) – chain de opções da B3  
- [Yahoo Finance](https://finance.yahoo.com) – preço à vista e séries históricas  
- [dadosdemercado.com.br](https://www.dadosdemercado.com.br/acoes) – lista de tickers da B3

---

## 📦 Exportação e Publicação

Você pode publicar o app gratuitamente em:

- 🌐 **Hugging Face Spaces** (recomendado) – compatível com Streamlit nativamente  
- 🧱 **Render.com** – via Docker  
- 💾 **Deta Space** – se quiser evoluir para armazenar trades e alertas automáticos

---

## 📚 Créditos e Licença
Desenvolvido para fins **didáticos e educativos** por investidores que estudam operações com derivativos cobertos na B3.  
Uso livre e gratuito sob a licença **MIT**.


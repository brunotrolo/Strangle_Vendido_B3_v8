# 💼 Strangle Vendido Coberto — Versão v9
### 📈 Comparação de Estratégias: Strangle × Iron Condor × Jade Lizard

---

## 🧭 Visão Geral

Este aplicativo ajuda investidores a **vender opções cobertas (CALL e PUT)** de forma inteligente e didática.  
O foco é **maximizar a captura de prêmios** com **baixo risco de ser exercido**, simulando e comparando estratégias clássicas de renda com opções na B3 🇧🇷.

A versão **v9** traz uma grande novidade:
> A aba **📈 Comparar Estratégias**, que permite visualizar e comparar os resultados de **Strangle Vendido**, **Iron Condor Coberto** e **Jade Lizard** — com gráficos, métricas e explicações automáticas.

---

## ⚙️ Funcionalidades Principais

| Recurso | Descrição |
|----------|------------|
| 🧾 **Leitura automática** | Coleta os dados de opções diretamente do site [opcoes.net.br](https://opcoes.net.br) |
| 💰 **Cálculo de prêmios** | Mostra o valor dos prêmios, strikes e rentabilidade esperada |
| 📊 **Probabilidades (PoE)** | Estima a probabilidade de exercício via modelo Black–Scholes |
| ⚖️ **Classificação de risco** | Classifica cada operação em Baixo, Médio e Alto risco |
| 📘 **Instruções automáticas de saída** | Explica quando e como sair da operação (recompra parcial, rolagem etc.) |
| 📈 **Comparador de estratégias (v9)** | Exibe payoffs e métricas das três estruturas mais usadas por vendedores de opções |

---

## 🚀 Como Usar

1. **Escolha um ou mais tickers** (ex.: PETR4, VALE3, ITUB4).
2. Informe:
   - Quantidade de ações em carteira 💼  
   - Caixa disponível 💵  
   - Tamanho do contrato 📦 (geralmente 100)
3. O app buscará a **opções chain** automaticamente.
4. Selecione o **vencimento** desejado.
5. Veja as sugestões de **Strangles vendidos cobertos** com:
   - Prêmio total, retorno percentual, probabilidade de exercício e classificação de risco.
6. Clique em **📈 Comparar Estratégias (v9)** para abrir o novo módulo.

---

## 📈 Nova Aba — “Comparar Estratégias (v9)”

Essa aba foi criada para te ajudar a **entender as diferenças práticas** entre 3 estruturas muito usadas na venda coberta de opções.

### 🔸 1. Strangle Vendido Coberto

- **Vende 1 PUT OTM + 1 CALL OTM**, ambas cobertas (caixa e ações).  
- Ideal para **mercado lateral** e **IV alta**.  
- **Lucro máximo**: prêmio recebido.  
- **Risco**: ser exercido em um dos lados.  
- **Melhor quando**: o ativo tende a oscilar dentro de uma faixa.

🧮 **Payoff:**  
`Lucro = +Prêmio – |Movimento fora dos strikes|`

---

### 🔸 2. Iron Condor Coberto 🦅

- Mesmo strangle vendido, **mas compra duas asas protetoras** (PUT e CALL mais distantes).  
- Reduz o risco máximo e define **perda limitada**.  
- Ideal para **mercado instável** ou quando o investidor quer dormir tranquilo 😴.  
- **Lucro máximo**: prêmio líquido (menor que o strangle).  
- **Perda máxima**: limitada à diferença entre strikes menos o crédito recebido.

🧮 **Payoff:**  
`Lucro = +Prêmio líquido – |Perda nas asas|`

---

### 🔸 3. Jade Lizard 🦎

- Combina **venda de PUT + venda de CALL + compra de CALL mais OTM**.  
- Mantém boa parte do prêmio da PUT e **elimina o risco de alta**, se o crédito ≥ diferença entre as CALLs.  
- **Lucro máximo**: prêmio líquido.  
- **Risco**: somente no lado da PUT.

🧮 **Payoff:**  
`Lucro = +Prêmio líquido – |Risco de queda|`

---

## 📊 Gráficos e Interpretação Visual

Na aba de comparação, o app gera **3 gráficos de payoff**:

| Cor / Linha | Representa |
|--------------|-------------|
| 🟦 Linha contínua | Lucro/Prejuízo por ação no vencimento |
| ⚫ Linha tracejada vertical | Strikes de CALL e PUT |
| 🟢 Zona central | Faixa de lucro máximo (entre os strikes) |
| 🔴 Região abaixo/acima | Zonas de risco (exercício PUT ou CALL) |

Você pode ajustar a **largura das asas (%)** para ver como muda o risco e o crédito das estruturas.

---

## 💡 Dicas de Interpretação

| Situação de Mercado | Estratégia Ideal | Motivo |
|----------------------|------------------|---------|
| 📉 Volatilidade alta e preço lateral | **Strangle Vendido** | Gera mais prêmio |
| ⚖️ Volatilidade média e incerteza | **Iron Condor** | Limita risco e mantém renda |
| 📈 Tendência leve de alta | **Jade Lizard** | Protege na alta e ainda gera prêmio |
| 💥 IV muito baixa | Nenhuma venda | Prêmios ruins — espere melhores condições |

---

## 🧮 Métricas Calculadas

| Métrica | Descrição |
|----------|------------|
| **Retorno (%)** | Lucro potencial sobre o preço à vista |
| **PoE_total** | Probabilidade de exercício (qualquer perna) |
| **PoE_dentro** | Probabilidade de ficar entre os strikes |
| **IV Rank / Percentil** | Nível relativo da volatilidade implícita |
| **Score (prêmio/risco)** | Combina prêmio recebido e risco de exercício |

---

## 📘 Instruções de Saída e Rolagem

- ⚠️ **Perto do vencimento (≤ 10 dias)** e **preço encostando no strike (±5%)** → **Recomprar a perna ameaçada**.  
- 💰 **Capturou 70–80% do prêmio?** → **Zere** a operação e evite tail risk.  
- 🔄 **Quer manter a posição?** → **Role** para o próximo vencimento com strikes OTM.  

🧩 O app mostra mensagens automáticas como:  
> “⚠️ CALL ameaçada: preço próximo de K_call. Faltam 5 dias. Sugestão: recomprar a CALL e garantir 80% do prêmio.”

---

## 🧠 Glossário Didático

| Termo | Significado |
|--------|--------------|
| **Delta (Δ)** | Sensibilidade do preço da opção ao ativo subjacente |
| **IV (Implied Volatility)** | Expectativa de volatilidade do mercado |
| **IV Rank / Percentil** | Mede se a IV atual está alta ou baixa historicamente |
| **PoE (Probabilidade de Exercício)** | Chance da opção terminar “dentro do dinheiro” |
| **BE (Break-even)** | Pontos de equilíbrio — onde o lucro zera |
| **Rolagem** | Substituir opção atual por outra mais distante no tempo |
| **Strangle** | Venda simultânea de uma CALL e uma PUT OTM |
| **Iron Condor** | Strangle com asas protetoras compradas |
| **Jade Lizard** | PUT vendida + CALL vendida + CALL comprada mais OTM |

---

## 🧰 Deploy Gratuito

### 🟣 **Streamlit Cloud**
1. Faça login em [streamlit.io](https://streamlit.io/cloud).  
2. Clique em “New app” → conecte seu GitHub → selecione o repositório.  
3. Arquivo principal: `app_v9.py`  
4. `requirements.txt` deve incluir:
   ```
   streamlit
   requests
   yfinance
   pandas
   numpy
   matplotlib
   bs4
   lxml
   pyppeteer
   requests-html
   nest_asyncio
   ```

### 🟢 **Deta Space (alternativa com banco local)**
Ideal para salvar histórico das operações e futuras notificações.  
- Faça login em [deta.space](https://deta.space).  
- Publique o projeto via terminal com:
  ```bash
  deta new --python
  deta deploy
  ```
- Armazene suas operações vendidas e resultados.

---

## 🧩 Versão Atual

| Versão | Novidades |
|---------|------------|
| **v8** | Instruções automáticas de saída e alertas de exercício |
| **v9** | Nova aba de comparação 📈 Strangle × Iron Condor × Jade Lizard |

---

## ✨ Exemplo Prático — PETR4

> PETR4 a R$ 38,00  
> Strangle: CALL 40 / PUT 36  
> Iron Condor: CALL 41 / PUT 35 (asas)  
> Jade Lizard: PUT 36 / CALL 40 + CALL 41

| Estratégia | Crédito | Risco Máx | Prob. Ficar Dentro |
|-------------|----------|------------|---------------------|
| Strangle | R$ 1,20 | Ilimitado | 72% |
| Iron Condor | R$ 0,90 | R$ 0,90 | 70% |
| Jade Lizard | R$ 1,10 | R$ 1,10 (queda) | 71% |

**Conclusão:** Jade Lizard oferece **proteção de alta** e **bom prêmio**, sendo ideal se o investidor já tem posição comprada.

---

## 🧩 Conclusão

> A versão **v9** transforma o app em uma **ferramenta de aprendizado interativo** sobre venda coberta de opções na B3.

Ela não só mostra o **melhor Strangle**, mas **ensina o porquê**, comparando outras estruturas que **reduzem risco e mantêm rentabilidade**.  
Use o app como **laboratório de aprendizado e tomada de decisão disciplinada**.

📘 Desenvolvido para investidores que desejam unir **estratégia, segurança e didática.**

---

🛠️ **Autor:** ChatGPT (colaboração com Bruno Teixeira)  
📅 **Versão:** 9.0  
📍 **Mercado-alvo:** B3 – Brasil, opções cobertas (PETR4, VALE3, ITUB4, CSAN3, etc.)

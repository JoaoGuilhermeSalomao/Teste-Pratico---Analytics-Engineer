# Case de Análise de Dados — Hábitos e Desempenho Estudantil

Este projeto teve como objetivo explorar um conjunto de dados sobre **hábitos de estudo, estilo de vida e saúde mental de estudantes**, a fim de identificar **quais fatores mais influenciam o desempenho acadêmico**.
Todo o processo foi realizado em **Python**, utilizando as bibliotecas **pandas**, **matplotlib**, **seaborn** e **statsmodels**.

---

## 1. Exploração Inicial

Comecei o projeto realizando uma análise exploratória da base de dados, observando:

* Os **tipos de variáveis** e valores ausentes (apenas 9% faltantes em `parental_education_level`);
* A **distribuição** das principais colunas (notas, horas de estudo, sono, uso de redes sociais, saúde mental, etc.);
* E a inexistência de **linhas duplicadas** (0 registros).

Criei histogramas para visualizar as distribuições e compreender o comportamento geral das variáveis contínuas, o que me ajudou a identificar padrões e possíveis outliers.

---

## 2. Engenharia e Tratamento de Dados

Durante a etapa de **engenharia de dados**, realizei diversas transformações com o objetivo de **corrigir inconsistências, tratar ausências e criar variáveis derivadas** que representassem melhor os comportamentos analisados.

| Ação                                                                                   | Descrição                                                                                                 | Justificativa                                                                                                                       |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Substituição dos valores ausentes em `parental_education_level` por “Not Informed”** | Cerca de 9% dos registros estavam vazios nessa coluna.                                                    | Como essa variável é **categórica e qualitativa**, decidi criar uma categoria neutra (“Not Informed”) para não distorcer a amostra. |
| **Criação da variável ordinal `parents_education_ord`**                                | Converti os níveis de escolaridade para valores de 1 a 6 (Primary → 1 ... PhD → 6).                       | Essa transformação permite o uso da variável em **correlações e regressões**, respeitando sua ordem hierárquica.                    |
| **Criação da variável binária `works_flag`**                                           | Converti respostas textuais (“Yes”, “Sim”, “True”) para 0 e 1.                                            | Isso facilita a aplicação de testes estatísticos e comparações entre grupos de estudantes que trabalham ou não.                     |
| **Criação de faixas (`buckets`) para horas de estudo e tempo em redes sociais**        | Agrupei as colunas `study_hours_per_day` e `social_media_hours` em intervalos (ex.: 0–1h, 1–2h, 2–4h...). | Essa abordagem melhora a **interpretação visual** e reduz o impacto de outliers.                                                    |
| **Conversão de `diet_quality` em valores numéricos (`diet_quality_num`)**              | Mapeei as categorias: Poor → 1, Fair → 2, Good → 3.                                                       | Assim pude realizar cálculos quantitativos mantendo a **ordem implícita de qualidade** da dieta.                                    |
| **Criação do `lifestyle_index`**                                                       | Desenvolvi um índice composto (0–100) com pesos 0.4 para sono, 0.3 para dieta e 0.3 para exercício.       | Esse índice resume a **qualidade de vida** dos estudantes, combinando múltiplos fatores.                                            |
| **Normalização Min–Max (0–100)**                                                       | Escalei as variáveis base para a mesma faixa de valores.                                                  | Isso evita que variáveis com amplitudes maiores dominem o índice e facilita comparações.                                            |
| **Preenchimento de ausentes numéricos com a mediana**                                  | Apliquei essa técnica em variáveis contínuas.                                                             | A mediana é menos sensível a outliers e preserva a distribuição central dos dados.                                                  |
| **Checagem final com `df.info()` e `df.head()`**                                       | Revisei a integridade e os tipos das colunas após as transformações.                                      | Isso garantiu que todas as variáveis estivessem prontas para a análise estatística.                                                 |

Esses tratamentos garantiram que o dataset ficasse **completo, consistente e estatisticamente interpretável**, refletindo de forma fiel os comportamentos que eu desejava investigar.

---

## 3. Técnicas Estatísticas Utilizadas

Empreguei diferentes técnicas para analisar os dados:

* **Regressão Linear Simples (OLS):** para medir relações entre variáveis contínuas;
* **Regressão Linear Múltipla com Interação:** para avaliar efeitos combinados;
* **Correlação de Pearson:** para medir força e direção de relações lineares;
* **Teste t de Médias Independentes:** para comparar grupos distintos;
* **Pivot Table + Heatmap:** para analisar interações categóricas;
* **Boxplots e Violin Plots:** para comparar distribuições e medianas.

Cada análise foi acompanhada de visualizações para facilitar a interpretação e validar visualmente os resultados.

---

## 4. Visualização de Dados

| Hipótese                                                               | Tipo de Gráfico           | Justificativa                                                                         |
| ---------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| **H1 — Quem mais estuda tira as melhores notas**                       | Scatterplot com regressão | Permite observar a relação linear positiva entre estudo e nota.                       |
| **H2 — Quem tem boa saúde mental tira boas notas**                     | Boxplot                   | Mostra a variação das notas conforme o nível de saúde mental.                         |
| **H3 — Quem passa muito tempo nas redes sociais tira notas piores**    | Scatterplot + Boxplot     | Demonstra a relação negativa entre tempo em redes e desempenho.                       |
| **H4 — Quanto mais estudar e melhor a saúde mental, maiores as notas** | Heatmap                   | Representa o efeito combinado de duas variáveis categorizadas.                        |
| **H5 — Quem tem um bom lifestyle tira notas melhores**                 | Scatterplot + Boxplot     | Compara as medianas e variações entre grupos de lifestyle.                            |
| **H6 — Pessoas que trabalham tendem a ter notas mais baixas**          | Violin Plot               | Compara distribuições e densidades entre grupos.                                      |
| **H7 — Pais com maior escolaridade → notas maiores**                   | Boxplot                   | Facilita a comparação de desempenho entre diferentes níveis de escolaridade dos pais. |
| **H8 — Um bom lifestyle influencia em uma melhor saúde mental**        | Scatterplot               | Mostra a tendência entre lifestyle e saúde mental.                                    |

---

## 5. Resultados e Interpretação das Hipóteses

| Hipótese                                             | Resultado Estatístico                | Interpretação                                                                                     |
| ---------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **H1 — Quem mais estuda tira as melhores notas**     | R² = 0.681, Coef. = 9.49, p < 0.001  | Relação forte e positiva — cada hora adicional de estudo aumenta significativamente a nota média. |
| **H2 — Boa saúde mental → boas notas**               | R² = 0.103, Coef. = 1.90, p < 0.001  | Relação positiva moderada — estudantes com melhor bem-estar tendem a ter desempenho mais alto.    |
| **H3 — Mais redes sociais → notas piores**           | R² = 0.028, Coef. = -2.40, p < 0.001 | Relação negativa significativa — uso excessivo de redes sociais está ligado a notas menores.      |
| **H4 — Estudo + saúde mental → notas mais altas**    | R² = 0.788, p < 0.001                | Efeito combinado positivo entre as variáveis.                                                     |
| **H5 — Bom lifestyle → notas melhores**              | R² = 0.026, p < 0.001                | Relação fraca, mas ainda positiva.                                                                |
| **H6 — Trabalhar → notas mais baixas**               | p = 0.39                             | Diferença não significativa.                                                                      |
| **H7 — Pais com maior escolaridade → notas maiores** | R² = 0.000, p = 0.803                | Nenhuma relação significativa.                                                                    |
| **H8 — Bom lifestyle → melhor saúde mental**         | R² = 0.000, p = 0.634                | Relação estatisticamente nula.                                                                    |

---

## 6. Conclusões Gerais

1. **O tempo de estudo é o principal fator associado ao desempenho.**
   É o único com impacto realmente expressivo e estatisticamente robusto.

2. **Saúde mental e redes sociais também influenciam as notas.**
   Estudantes com boa saúde mental tendem a ter notas mais altas, enquanto o uso excessivo de redes está ligado a piores resultados.

3. **Aspectos externos**, como o trabalho e a escolaridade dos pais, **têm impacto mínimo**, sugerindo que o desempenho depende mais dos próprios hábitos.

4. **O lifestyle saudável** contribui levemente, mas o efeito não é significativo.

---

## 7. Dicionário de Termos e Métricas

| Termo                            | Significado                                    | Interpretação                                                         |
| -------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| **R² (R-squared)**               | Grau de explicação do modelo                   | Mede quanto da variação na nota é explicada pela variável analisada.  |
| **Coeficiente (β)**              | Força e direção da relação                     | Valor positivo → relação direta; negativo → relação inversa.          |
| **p-valor**                      | Probabilidade de o resultado ocorrer por acaso | Se **p < 0.05**, a relação é estatisticamente significativa.          |
| **OLS (Ordinary Least Squares)** | Método de regressão linear tradicional         | Ajusta uma linha que minimiza o erro entre valores previstos e reais. |
| **Boxplot**                      | Gráfico de caixa                               | Mostra mediana, quartis e outliers.                                   |
| **Heatmap**                      | Mapa de calor                                  | Representa médias ou correlações em matriz.                           |
| **Violin Plot**                  | Gráfico de violino                             | Exibe a densidade e simetria das distribuições.                       |
| **Pivot Table**                  | Tabela dinâmica                                | Resume dados categóricos de forma matricial.                          |


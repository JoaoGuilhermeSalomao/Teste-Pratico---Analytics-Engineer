Perfeito — aqui está o README completo e atualizado, sem emojis e já com a seção do Streamlit integrada de forma profissional, para você copiar e colar diretamente no repositório:

⸻

Case de Análise de Dados — Hábitos e Desempenho Estudantil

Este projeto teve como objetivo explorar um conjunto de dados sobre hábitos de estudo, estilo de vida e saúde mental de estudantes, a fim de identificar quais fatores mais influenciam o desempenho acadêmico.
Todo o processo foi realizado em Python, utilizando as bibliotecas pandas, matplotlib, seaborn e statsmodels.

⸻

1. Exploração Inicial

O projeto iniciou com uma análise exploratória da base de dados, observando:
	•	Os tipos de variáveis e valores ausentes (apenas 9% faltantes em parental_education_level);
	•	A distribuição das principais colunas (notas, horas de estudo, sono, uso de redes sociais, saúde mental, etc.);
	•	A inexistência de linhas duplicadas (0 registros).

Foram criados histogramas para visualizar as distribuições e compreender o comportamento geral das variáveis contínuas, permitindo identificar padrões e possíveis outliers.

⸻

2. Engenharia e Tratamento de Dados

Durante a etapa de engenharia de dados, foram realizadas diversas transformações para corrigir inconsistências, tratar ausências e criar variáveis derivadas que representassem melhor os comportamentos analisados.

Ação	Descrição	Justificativa
Substituição dos valores ausentes em parental_education_level por “Not Informed”	Cerca de 9% dos registros estavam vazios nessa coluna.	Como a variável é categórica, foi criada uma categoria neutra (“Not Informed”) para não distorcer a amostra.
Criação da variável ordinal parents_education_ord	Conversão dos níveis de escolaridade para valores de 1 a 6 (Primary → 1 … PhD → 6).	Permite o uso em correlações e regressões, respeitando sua hierarquia.
Criação da variável binária works_flag	Conversão de respostas textuais (“Yes”, “Sim”, “True”) para 0 e 1.	Facilita testes estatísticos e comparações entre grupos.
Criação de faixas (buckets) para horas de estudo e tempo em redes sociais	Agrupamento de study_hours_per_day e social_media_hours em intervalos (ex.: 0–1h, 1–2h, 2–4h…).	Melhora a interpretação visual e reduz o impacto de outliers.
Conversão de diet_quality em valores numéricos	Mapeamento: Poor → 1, Fair → 2, Good → 3.	Permite cálculos quantitativos mantendo a ordem implícita.
Criação do lifestyle_index	Índice composto (0–100) com pesos 0.4 (sono), 0.3 (dieta) e 0.3 (exercício).	Resume a qualidade de vida em um único indicador.
Normalização Min–Max (0–100)	Escala das variáveis base para a mesma faixa.	Evita distorções entre variáveis de magnitudes diferentes.
Preenchimento de ausentes numéricos com a mediana	Aplicado em variáveis contínuas.	A mediana é menos sensível a outliers.
Checagem final (df.info() e df.head())	Verificação de integridade e tipos.	Garante que o dataset está pronto para modelagem.

Esses tratamentos garantiram que o dataset ficasse completo, consistente e estatisticamente interpretável.

⸻

3. Técnicas Estatísticas Utilizadas

Foram empregadas diferentes técnicas para análise:
	•	Regressão Linear Simples (OLS): para medir relações entre variáveis contínuas.
	•	Regressão Linear Múltipla com Interação: para avaliar efeitos combinados.
	•	Correlação de Pearson: para medir força e direção de relações lineares.
	•	Teste t de Médias Independentes: para comparar grupos distintos.
	•	Pivot Table + Heatmap: para analisar interações categóricas.
	•	Boxplots e Violin Plots: para comparar distribuições e medianas.

Cada análise foi acompanhada de visualizações que facilitaram a interpretação e validação dos resultados.

⸻

4. Visualização de Dados

Hipótese	Tipo de Gráfico	Justificativa
H1 — Quem mais estuda tira as melhores notas	Scatterplot com regressão	Mostra relação linear positiva entre estudo e nota.
H2 — Boa saúde mental → boas notas	Boxplot	Mostra variação das notas por nível de saúde mental.
H3 — Mais redes sociais → notas piores	Scatterplot + Boxplot	Mostra relação negativa entre tempo em redes e desempenho.
H4 — Estudo + saúde mental → notas mais altas	Heatmap	Representa o efeito combinado das variáveis.
H5 — Bom lifestyle → notas melhores	Scatterplot + Boxplot	Compara medianas e variações entre grupos de lifestyle.
H6 — Trabalhar → notas mais baixas	Violin Plot	Compara densidades de distribuição entre grupos.
H7 — Escolaridade dos pais → notas maiores	Boxplot	Compara desempenho conforme o nível de escolaridade dos pais.
H8 — Lifestyle → saúde mental	Scatterplot	Mostra a tendência entre hábitos e bem-estar.


⸻

5. Resultados e Interpretação

Hipótese	Resultado Estatístico	Interpretação
H1	R² = 0.681, Coef. = 9.49, p < 0.001	Relação forte e positiva: cada hora adicional de estudo aumenta a nota média.
H2	R² = 0.103, Coef. = 1.90, p < 0.001	Relação positiva moderada: bem-estar influencia desempenho.
H3	R² = 0.028, Coef. = -2.40, p < 0.001	Relação negativa: uso excessivo de redes está ligado a notas menores.
H4	R² = 0.788, p < 0.001	Efeito combinado positivo entre estudo e saúde mental.
H5	R² = 0.026, p < 0.001	Relação leve e positiva.
H6	p = 0.39	Diferença não significativa.
H7	R² = 0.000, p = 0.803	Nenhuma relação significativa.
H8	R² = 0.000, p = 0.634	Sem relação estatística significativa.


⸻

6. Conclusões Gerais
	1.	O tempo de estudo é o principal fator relacionado ao desempenho acadêmico.
	2.	Saúde mental e uso de redes sociais têm impacto moderado.
	3.	Trabalho e escolaridade dos pais influenciam pouco.
	4.	Um estilo de vida saudável contribui levemente, mas sem significância estatística.

⸻

7. Dicionário de Termos e Métricas

Termo	Significado	Interpretação
R² (R-squared)	Grau de explicação do modelo	Mede quanto da variação na nota é explicada pela variável.
Coeficiente (β)	Força e direção da relação	Positivo → direta; negativo → inversa.
p-valor	Probabilidade de o resultado ocorrer por acaso	p < 0.05 → relação significativa.
OLS	Regressão Linear Tradicional	Minimiza o erro entre valores previstos e reais.
Boxplot	Gráfico de caixa	Mostra mediana, quartis e outliers.
Heatmap	Mapa de calor	Representa médias ou correlações em matriz.
Violin Plot	Gráfico de violino	Exibe densidade e simetria das distribuições.
Pivot Table	Tabela dinâmica	Resume dados categóricos em matriz.


⸻

8. Dashboard Interativa — Streamlit

Além da análise estatística tradicional, foi desenvolvida uma interface interativa com Streamlit, que permite tratamento, exploração e modelagem de dados de forma visual e intuitiva.

Funcionalidades Principais

1. Tratamento de Dados Automatizado
	•	Detecção automática de valores nulos, outliers e variáveis categóricas.
	•	Interface para o usuário escolher o tratamento: preencher, remover, normalizar, codificar ou ignorar.
	•	Opção de aplicar tratamentos gerais em todas as colunas de um tipo (numéricas, categóricas, etc.).

2. Análise Exploratória
	•	Geração automática de histogramas, boxplots e mapa de calor com correlações (com valores anotados).
	•	Possibilidade de escolher qual variável será o eixo Y e qual modelo de gráfico usar.
	•	Exibição das principais estatísticas descritivas da base.

3. Modelagem e Predição
	•	Escolha da coluna alvo (target) e das variáveis preditoras.
	•	Seleção do modelo de Machine Learning (Regressão Linear, Random Forest, etc.).
	•	Exibição automática das métricas de desempenho (R², RMSE, MAE) e importância das variáveis.

4. Visualização Avançada
	•	Possibilidade de criar múltiplos gráficos simultaneamente.
	•	Dropdown interativo para selecionar hipóteses específicas e visualizar seus gráficos correspondentes.
	•	Dashboard organizada por seções (Exploração, Modelagem e Resultados).

5. Exportação
	•	Geração automática dos resultados (métricas, gráficos e textos) em JSON dentro de notebook_artifacts/.
	•	Opção de baixar relatórios em PDF diretamente pela interface.

⸻

9. Referências e Execução do Projeto
	•	Notebook completo:   [`analise_habitos.ipynb`](./analise_habitos.ipynb)
	•	Dashboard: app.py (executar com streamlit run app.py)
	•	Deploy: [Streamlit Cloud](https://teste-pratico---analytics-engineer-funppavd9ozydtjbeuwic6.streamlit.app/)
	•	Artefatos gerados: notebook_artifacts/summary.json, metrics.json, hypotheses.json e gráficos .png

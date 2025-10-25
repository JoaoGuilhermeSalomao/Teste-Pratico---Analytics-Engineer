# app.py
import io
import os
import json
import glob
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as _mse, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# =========================
# Config
# =========================
st.set_page_config(page_title="Tratamento & Análise de Dados", page_icon="", layout="wide")
sns.set_theme(style="whitegrid", palette="Blues", font_scale=1.05)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# =========================
# Helpers genéricos
# =========================
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def safe_title(s):
    try:
        return s.replace("_", " ").title()
    except Exception:
        return str(s)

def rmse_compat(y_true, y_pred):
    try:
        return _mse(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(_mse(y_true, y_pred)))

# =========================
# Perfil de dados (nulos e outliers)
# =========================
def profile_nulls(df: pd.DataFrame):
    n = df.isna().sum()
    pct = (n / len(df) * 100).round(2) if len(df) else 0
    res = pd.DataFrame({"coluna": n.index, "nulos": n.values, "%": pct if isinstance(pct, pd.Series) else 0})
    return res.sort_values("nulos", ascending=False).reset_index(drop=True)

def outlier_mask_iqr(s: pd.Series):
    s_clean = s.dropna()
    if s_clean.empty:
        return pd.Series(False, index=s.index)
    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < low) | (s > high)
    return mask.reindex(s.index, fill_value=False)

def profile_outliers(df_num: pd.DataFrame):
    rows = []
    for col in df_num.columns:
        m = outlier_mask_iqr(df_num[col])
        rate = float(m.mean())
        rows.append({
            "coluna": col,
            "taxa_outliers": rate,
            "p01": df_num[col].quantile(0.01),
            "p05": df_num[col].quantile(0.05),
            "p95": df_num[col].quantile(0.95),
            "p99": df_num[col].quantile(0.99)
        })
    return pd.DataFrame(rows).sort_values("taxa_outliers", ascending=False).reset_index(drop=True)

# =========================
# Tratamento (respeita enabled/ignore)
# =========================
def apply_missing(df: pd.DataFrame, cfg_miss: dict, cat_fill_constants: dict, enabled: bool, ignored_cols: list):
    if not enabled:
        return df.copy()

    df_out = df.copy()
    num_cols = [c for c in get_numeric_cols(df_out) if c not in ignored_cols]
    cat_cols = [c for c in get_categorical_cols(df_out) if c not in ignored_cols]

    for col in num_cols:
        strat = cfg_miss["per_numeric"].get(col, cfg_miss["global_numeric"])
        if strat == "ignore":
            continue
        if strat == "drop":
            df_out = df_out.dropna(subset=[col])
        elif strat == "median":
            df_out[col] = df_out[col].fillna(df_out[col].median())
        elif strat == "mean":
            df_out[col] = df_out[col].fillna(df_out[col].mean())
        elif strat == "zero":
            df_out[col] = df_out[col].fillna(0)

    for col in cat_cols:
        strat = cfg_miss["per_categorical"].get(col, cfg_miss["global_categorical"])
        if strat == "ignore":
            continue
        if strat == "drop":
            df_out = df_out.dropna(subset=[col])
        elif strat == "mode":
            if df_out[col].isna().any():
                mode_val = df_out[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df_out[col] = df_out[col].fillna(fill_val)
        elif strat == "constant":
            fill_val = cat_fill_constants.get(col, "Unknown")
            df_out[col] = df_out[col].fillna(fill_val)

    return df_out

def apply_outliers(df: pd.DataFrame, cfg_out: dict, enabled: bool, ignored_cols: list):
    if not enabled:
        return df.copy()

    df_out = df.copy()
    num_cols = [c for c in get_numeric_cols(df_out) if c not in ignored_cols]

    for col in num_cols:
        strat = cfg_out["per_numeric"].get(col, cfg_out["global_strategy"])
        if strat in ["none", "ignore"]:
            continue

        mask = outlier_mask_iqr(df_out[col])

        if strat == "drop":
            df_out = df_out.loc[~mask].copy()
        elif strat in ["clip_1_99", "clip_5_95", "custom"]:
            if strat == "clip_1_99":
                low_p, high_p = 1.0, 99.0
            elif strat == "clip_5_95":
                low_p, high_p = 5.0, 95.0
            else:
                low_p, high_p = cfg_out["custom_perc"].get(col, (cfg_out["custom_low"], cfg_out["custom_high"]))
            low = df_out[col].quantile(low_p/100.0)
            high = df_out[col].quantile(high_p/100.0)
            df_out[col] = df_out[col].clip(lower=low, upper=high)

    return df_out

def apply_encoding(df: pd.DataFrame, cfg_enc: dict, enabled: bool, ignored_cols: list):
    if not enabled:
        return df.copy()

    df_out = df.copy()
    cat_cols = [c for c in get_categorical_cols(df_out) if c not in ignored_cols]

    for col in list(cat_cols):
        strat = cfg_enc["per_categorical"].get(col, cfg_enc.get("global_categorical", "keep"))
        if strat == "ignore":
            continue
        if strat == "drop":
            df_out = df_out.drop(columns=[col])
        elif strat == "keep":
            continue
        elif strat == "onehot":
            dummies = pd.get_dummies(df_out[col], prefix=col, drop_first=True, dtype=float)
            df_out = pd.concat([df_out.drop(columns=[col]), dummies], axis=1)
        elif strat == "ordinal":
            order = cfg_enc["ordinal_orders"].get(col, None)
            if order is None or len(order) == 0:
                order = sorted([str(v) for v in df_out[col].dropna().unique().tolist()])
            mapping = {v: i for i, v in enumerate(order)}
            df_out[col] = df_out[col].map(mapping).astype(float)
    return df_out

# =========================
# Modelos
# =========================
def train_regressor(df, target, features, model_name, test_size=0.2, random_state=42, use_robust=False):
    X = df[features].copy()
    y = df[target].copy()
    keep = X.join(y).dropna()
    X, y = keep[features], keep[target]

    if model_name == "Linear Regression":
        scaler = RobustScaler(with_centering=True, with_scaling=True) if use_robust else StandardScaler(with_mean=True, with_std=True)
        model = Pipeline([("scaler", scaler), ("reg", LinearRegression())])
    elif model_name == "Ridge (L2)":
        scaler = RobustScaler(with_centering=True, with_scaling=True) if use_robust else StandardScaler(with_mean=True, with_std=True)
        model = Pipeline([("scaler", scaler), ("reg", Ridge(alpha=1.0))])
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("Modelo não suportado.")

    if len(X) < 3:
        raise ValueError("Poucos dados após remoção de ausentes para treinar. Ajuste o tratamento ou selecione outras colunas.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, y_pred),
        "RMSE": rmse_compat(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

    feature_importance = None
    if model_name in ["Linear Regression", "Ridge (L2)"]:
        reg = model[-1]
        coefs = getattr(reg, "coef_", None)
        if coefs is not None and len(features) == len(coefs):
            feature_importance = pd.DataFrame({"feature": features, "importance": coefs}).sort_values("importance", ascending=False)
    elif model_name == "Random Forest":
        feature_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    return model, metrics, feature_importance, (X_test, y_test, y_pred)

# =========================
# Gráficos
# =========================
def make_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(df[x], df[y], alpha=0.6)
    if df[[x, y]].dropna().shape[0] > 2:
        sns.regplot(x=x, y=y, data=df, scatter=False, ax=ax, line_kws={"color": "red"})
    ax.set_title(f"Relação entre {safe_title(x)} e {safe_title(y)}")
    ax.set_xlabel(safe_title(x)); ax.set_ylabel(safe_title(y))
    fig.tight_layout()
    return fig

def make_boxplot(df, cat, num):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.boxplot(x=cat, y=num, data=df, ax=ax)
    ax.set_title(f"Distribuição de {safe_title(num)} por {safe_title(cat)}")
    ax.set_xlabel(safe_title(cat)); ax.set_ylabel(safe_title(num))
    fig.tight_layout()
    return fig

def make_hist(df, col, bins=30):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(df[col].dropna(), bins=bins, edgecolor="black")
    ax.set_title(f"Histograma — {safe_title(col)}")
    ax.set_xlabel(safe_title(col)); ax.set_ylabel("Contagem")
    fig.tight_layout()
    return fig

def make_corr_heatmap(df):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sem colunas numéricas para correlação", ha="center", va="center")
        ax.axis("off")
        return fig
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))  # era (9,7)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlGnBu",
        linewidths=0.3, square=True, ax=ax
    )
    ax.set_title("Mapa de Calor de Correlação (Numéricas)")
    fig.tight_layout()
    return fig

# =========================
# Construtor de múltiplos gráficos (IDs estáveis)
# =========================
def _init_chart_state():
    if "charts" not in st.session_state:
        st.session_state["charts"] = []

def _default_chart_config(all_num, all_cat):
    default_type = "Dispersão (x vs y)"
    default_x = "study_hours_per_day" if "study_hours_per_day" in all_num else (all_num[0] if all_num else None)
    default_y = "exam_score" if "exam_score" in all_num else (all_num[0] if all_num else None)
    default_cat = all_cat[0] if all_cat else None
    return {
        "id": str(uuid4()),
        "type": default_type,
        "x": default_x,
        "y": default_y,
        "cat": default_cat,
        "num": default_y,
        "bins": 30
    }

def render_chart_form(cfg_idx, cfg, all_num, all_cat, df_for_charts):
    chart_id = cfg.get("id", f"chart_{cfg_idx}")
    st.markdown(f"**Gráfico #{cfg_idx+1}**")
    colA, colB, colC = st.columns([1, 1, 1])

    chart_type = colA.selectbox(
        "Tipo",
        ["Dispersão (x vs y)", "Boxplot (cat vs num)", "Histograma", "Heatmap de Correlação"],
        index=["Dispersão (x vs y)", "Boxplot (cat vs num)", "Histograma", "Heatmap de Correlação"].index(cfg.get("type", "Dispersão (x vs y)")),
        key=f"chart_type_{chart_id}"
    )
    cfg["type"] = chart_type

    fig = None

    if chart_type == "Dispersão (x vs y)":
        x_opt = all_num if all_num else [None]
        y_opt = all_num if all_num else [None]
        cfg["x"] = colB.selectbox("X (num)", options=x_opt, index=(x_opt.index(cfg.get("x")) if cfg.get("x") in x_opt else 0), key=f"chart_x_{chart_id}")
        cfg["y"] = colC.selectbox("Y (num)", options=y_opt, index=(y_opt.index(cfg.get("y")) if cfg.get("y") in y_opt else 0), key=f"chart_y_{chart_id}")
        if all_num:
            fig = make_scatter(df_for_charts, cfg["x"], cfg["y"])

    elif chart_type == "Boxplot (cat vs num)":
        cat_opt = all_cat if all_cat else [None]
        num_opt = all_num if all_num else [None]
        cfg["cat"] = colB.selectbox("Categoria (x)", options=cat_opt, index=(cat_opt.index(cfg.get("cat")) if cfg.get("cat") in cat_opt else 0), key=f"chart_cat_{chart_id}")
        cfg["num"] = colC.selectbox("Numérica (y)", options=num_opt, index=(num_opt.index(cfg.get("num")) if cfg.get("num") in num_opt else 0), key=f"chart_num_{chart_id}")
        if all_cat and all_num:
            fig = make_boxplot(df_for_charts, cfg["cat"], cfg["num"])

    elif chart_type == "Histograma":
        num_opt = all_num if all_num else [None]
        cfg["num"] = colB.selectbox("Coluna (num)", options=num_opt, index=(num_opt.index(cfg.get("num")) if cfg.get("num") in num_opt else 0), key=f"chart_histcol_{chart_id}")
        cfg["bins"] = colC.slider("Bins", 10, 100, cfg.get("bins", 30), 5, key=f"chart_bins_{chart_id}")
        if all_num:
            fig = make_hist(df_for_charts, cfg["num"], bins=cfg["bins"])

    elif chart_type == "Heatmap de Correlação":
        st.caption("Mostra a correlação entre colunas numéricas do dataset (Pearson).")
        fig = make_corr_heatmap(df_for_charts)

    c1, c2 = st.columns([1, 1])
    remove = c1.button("🗑️ Remover", key=f"remove_{chart_id}")
    duplicate = c2.button("➕ Duplicar", key=f"dup_{chart_id}")

    if remove:
        st.session_state["charts"].pop(cfg_idx)
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()
    if duplicate:
        new_cfg = cfg.copy(); new_cfg["id"] = str(uuid4())
        st.session_state["charts"].insert(cfg_idx + 1, new_cfg)
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    st.markdown("---")
    return fig

# =========================
# PDF util
# =========================
def build_pdf(summary_text: str, figures: list):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig_txt = plt.figure(figsize=(8.27, 11.69))
        fig_txt.clf()
        fig_txt.text(0.08, 0.94, "Relatório — Tratamento & Análise", fontsize=18, weight="bold")
        y = 0.9
        for line in summary_text.split("\n"):
            fig_txt.text(0.08, y, line, fontsize=11)
            y -= 0.035
        pdf.savefig(fig_txt); plt.close(fig_txt)

        for f in figures:
            pdf.savefig(f)
            plt.close(f)
    buffer.seek(0)
    return buffer

# =========================
# Notebook Mode — carregamento de resultados prontos
# =========================
def try_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def collect_notebook_results(ipynb_path: str):
    """
    Tenta carregar resultados do repositório:
    1) Prioriza pasta notebook_artifacts/ (metrics.json, summary.json, hypotheses.json, *.png/*.jpg)
    2) Se não existir, tenta ler o .ipynb com nbformat e extrair:
       - Títulos de hipóteses (markdown com 'Hipótese')
       - Parágrafos subsequentes
       - (Imagens embutidas não são extraídas; preferir artifacts)
    """
    base_dir = os.path.dirname(ipynb_path) if ipynb_path else "."
    artifacts_dir = os.path.join(base_dir, "notebook_artifacts")

    result = {
        "hypotheses": [],   # list of {title, text}
        "metrics": None,    # dict
        "summary": None,    # str
        "fig_paths": []     # list of image paths
    }

    # 1) Artefatos
    if os.path.isdir(artifacts_dir):
        # JSONs
        result["metrics"] = try_load_json(os.path.join(artifacts_dir, "metrics.json"))
        hjson = try_load_json(os.path.join(artifacts_dir, "hypotheses.json"))
        if isinstance(hjson, list):
            result["hypotheses"] = hjson
        sjson = try_load_json(os.path.join(artifacts_dir, "summary.json"))
        if isinstance(sjson, str):
            result["summary"] = sjson

        # imagens
        imgs = sorted(glob.glob(os.path.join(artifacts_dir, "*.png"))) + \
               sorted(glob.glob(os.path.join(artifacts_dir, "*.jpg"))) + \
               sorted(glob.glob(os.path.join(artifacts_dir, "*.jpeg")))
        result["fig_paths"] = imgs

        return result

    # 2) Parse do .ipynb (fallback)
    if ipynb_path and os.path.exists(ipynb_path):
        try:
            import nbformat
            nb = nbformat.read(ipynb_path, as_version=4)
            # Coleta "Hipótese" dos markdowns
            for cell in nb["cells"]:
                if cell.get("cell_type") == "markdown":
                    src = cell.get("source", "")
                    lines = [l.strip() for l in src.splitlines()]
                    if any("Hipótese" in l or "Hipotese" in l for l in lines):
                        title = next((l for l in lines if l.startswith("#")), lines[0] if lines else "Hipótese")
                        body = "\n".join([l for l in lines if not l.startswith("#")]).strip()
                        result["hypotheses"].append({"title": title, "text": body})
                elif cell.get("cell_type") == "code":
                    # tentamos achar um dicionário de métricas impresso em JSON
                    for out in cell.get("outputs", []):
                        txt = None
                        if out.get("output_type") in ["stream", "display_data", "execute_result"]:
                            data = out.get("data", {})
                            if "text/plain" in data:
                                txt = data["text/plain"]
                            elif out.get("text"):
                                txt = out.get("text")
                        if txt and "{" in txt and "}" in txt and "RMSE" in txt:
                            try:
                                # tentativa ingênua de capturar JSON
                                start = txt.find("{"); end = txt.rfind("}")
                                jtxt = txt[start:end+1]
                                mdict = json.loads(jtxt)
                                result["metrics"] = mdict
                            except Exception:
                                pass
            return result
        except ModuleNotFoundError:
            st.warning("Para ler o notebook diretamente, instale `nbformat`: `pip install nbformat`.")
        except Exception as e:
            st.error(f"Falha ao ler o notebook: {e}")

    return result

def build_pdf_from_notebook(summary_text: str, hypothesis_texts: list, img_paths: list):
    """Monta PDF com resumo, blocos de hipóteses e imagens dos artefatos."""
    figs = []

    # página 1: resumo
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # resumo
        fig_txt = plt.figure(figsize=(8.27, 11.69))
        fig_txt.clf()
        fig_txt.text(0.08, 0.94, "Relatório — Notebook (Resultados Prontos)", fontsize=18, weight="bold")
        y = 0.9
        for line in summary_text.split("\n"):
            fig_txt.text(0.08, y, line, fontsize=11)
            y -= 0.035
        pdf.savefig(fig_txt); plt.close(fig_txt)

        # hipóteses (texto)
        for hyp in hypothesis_texts:
            fig_h = plt.figure(figsize=(8.27, 11.69))
            fig_h.clf()
            y = 0.94
            ttl = hyp.get("title", "Hipótese")
            body = hyp.get("text", "")
            fig_h.text(0.08, y, ttl, fontsize=14, weight="bold"); y -= 0.04
            for line in body.split("\n"):
                fig_h.text(0.08, y, line, fontsize=11); y -= 0.03
                if y < 0.1:
                    pdf.savefig(fig_h); plt.close(fig_h)
                    fig_h = plt.figure(figsize=(8.27, 11.69)); fig_h.clf(); y = 0.94
            pdf.savefig(fig_h); plt.close(fig_h)

        # imagens
        for path in img_paths:
            try:
                img = plt.imread(path)
                fig_i, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.imshow(img); ax.axis("off")
                ax.set_title(os.path.basename(path))
                pdf.savefig(fig_i); plt.close(fig_i)
            except Exception:
                continue

    buffer.seek(0)
    return buffer

# =========================
# Carregamento do CSV
# =========================
st.sidebar.header("⚙️ Configurações")
default_csv = "data/habitos_e_desempenho_estudantil.csv"
uploaded = st.sidebar.file_uploader("Carregar outro CSV (opcional)", type=["csv"])

if uploaded is not None:
    df = load_csv(uploaded)
    st.sidebar.success("✅ CSV carregado manualmente!")
elif os.path.exists(default_csv):
    df = load_csv(default_csv)
    st.sidebar.info("📂 CSV padrão carregado automaticamente.")
else:
    st.error("❌ Nenhum arquivo CSV encontrado. Faça upload de um arquivo.")
    st.stop()

# =========================
# Seleção de MODO
# =========================
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "🧭 Modo de Dashboard:",
    ["Notebook (resultados prontos)", "Interativo (Streamlit)"],
    index=0
)

st.title("🧰 Tratamento & Análise de Dados")

# =========================
# MODO NOTEBOOK — Resultados Prontos (com abas por hipótese)
# =========================
if mode == "Notebook (resultados prontos)":
    st.subheader("📓 Resultados do Notebook")

    # Caminho do notebook e artefatos
    default_ipynb = "./analise_habitos.ipynb"
    ipynb_up = st.sidebar.file_uploader("Carregar .ipynb (opcional)", type=["ipynb"])
    if ipynb_up is not None:
        tmp_path = "uploaded_notebook.ipynb"
        with open(tmp_path, "wb") as f:
            f.write(ipynb_up.read())
        ipynb_path = tmp_path
        st.sidebar.success("Notebook carregado!")
    else:
        ipynb_path = default_ipynb

    # Carrega resultados
    results = collect_notebook_results(ipynb_path)

    # — Resumo e Métricas gerais (fora das abas)
    with st.container():
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("**Resumo do estudo**")
            summary_txt = results.get("summary") or "Resumo não encontrado. Exporte `notebook_artifacts/summary.json`."
            st.write(summary_txt)
        with c2:
            st.markdown("**Métricas**")
            metrics = results.get("metrics")
            if isinstance(metrics, dict) and metrics:
                # aceita dict simples ou dict de dicts
                if all(isinstance(v, dict) for v in metrics.values()):
                    # dict de modelos
                    mdf = []
                    for model_name, md in metrics.items():
                        for k, v in md.items():
                            mdf.append({"Modelo": model_name, "Métrica": k, "Valor": v})
                    mdf = pd.DataFrame(mdf)
                    st.dataframe(mdf, use_container_width=True, hide_index=True)
                else:
                    mdf = pd.DataFrame(list(metrics.items()), columns=["Métrica", "Valor"])
                    st.dataframe(mdf, use_container_width=True, hide_index=True)
            else:
                st.info("Métricas não encontradas. Exporte `notebook_artifacts/metrics.json` para preenchimento automático.")

    st.markdown("---")

    
    # — Hipóteses via DROPDOWN (seleção única)
    hyps = results.get("hypotheses", [])
    if hyps:
        # lista de títulos
        hyp_titles = [h.get("title", f"H{i+1}") for i, h in enumerate(hyps)]

        st.markdown("### 🧪 Hipóteses")
        sel_title = st.selectbox("Selecione uma hipótese:", options=hyp_titles, index=0)
        sel_idx = hyp_titles.index(sel_title)
        hyp = hyps[sel_idx]

        # Texto explicativo
        st.markdown(f"### 🧠 {hyp.get('title','Hipótese')}")
        st.write(hyp.get("text", ""))

        # Métricas da hipótese (tabela compacta)
        met = hyp.get("metrics", {})
        if isinstance(met, dict) and met:
            hdf = pd.DataFrame(list(met.items()), columns=["Métrica", "Valor"])
            st.dataframe(hdf, use_container_width=True, hide_index=True)
        else:
            st.info("Sem métricas específicas desta hipótese.")

        # Gráficos da hipótese — menores (duas colunas)
        fig_list = hyp.get("figures", [])
        if fig_list:
            cols = st.columns(2)
            for i, fname in enumerate(fig_list):
                path = os.path.join("notebook_artifacts", fname)
                if os.path.exists(path):
                    with cols[i % 2]:
                        st.image(path, caption=os.path.basename(path), use_container_width=True)
                else:
                    st.warning(f"Figura não encontrada: {fname}")
        else:
            st.info("Sem figuras registradas para esta hipótese.")

        # (Opcional) Navegação rápida anterior/próxima
        c_prev, c_next = st.columns([1,1])
        with c_prev:
            if st.button("⬅️ Anterior", use_container_width=True) and sel_idx > 0:
                st.experimental_set_query_params(hyp=sel_idx-1)
                st.rerun()
        with c_next:
            if st.button("Próxima ➡️", use_container_width=True) and sel_idx < len(hyps)-1:
                st.experimental_set_query_params(hyp=sel_idx+1)
                st.rerun()

    else:
        st.info("Nenhuma hipótese localizada. Exporte `notebook_artifacts/hypotheses.json` para preencher automaticamente.")


    st.markdown("---")

    # — Visão Geral (gráficos gerais do notebook)
    st.subheader("🌐 Visão Geral")
    st.caption("Gráficos gerais exportados (ex.: correlação, distribuições).")
    fig_paths = results.get("fig_paths", [])
    # filtramos os que não são de hipótese (opcional): aqui mostramos todos mesmo
    if fig_paths:
        # mostra em grid 2 colunas para ficarem menores
        cols = st.columns(2)
        for i, p in enumerate(fig_paths):
            with cols[i % 2]:
                st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        st.info("Nenhuma imagem geral encontrada. Coloque figuras em `notebook_artifacts/*.png`.")

    # — Download PDF consolidado
    st.markdown("---")
    st.subheader("📄 Exportar PDF (Notebook)")
    # Exporta somente a hipótese selecionada
    pdf_bytes = build_pdf_from_notebook(
        summary_text=results.get("summary") or "Relatório (Notebook)",
        hypothesis_texts=[hyp],  # apenas a selecionada
        img_paths=[os.path.join("notebook_artifacts", f) for f in hyp.get("figures", [])]
    )
    st.download_button(
        label="⬇️ Baixar PDF (Notebook)",
        data=pdf_bytes,
        file_name="relatorio_notebook.pdf",
        mime="application/pdf"
    )


# =========================
# MODO INTERATIVO — Streamlit
# =========================
else:
    # TABS
    tab_data, tab_model, tab_viz, tab_export = st.tabs(
        ["🔎 Dados & Tratamento", "🤖 Modelagem", "📈 Visualizações", "📄 Exportar"]
    )

    with tab_data:
        st.subheader("📋 Análise Inicial")
        colA, colB = st.columns([1,1])
        with colA:
            st.markdown("**Tipos de Dados**")
            types_df = pd.DataFrame({"coluna": df.columns, "dtype": [str(t) for t in df.dtypes]})
            st.dataframe(types_df, use_container_width=True)
            st.markdown("**Nulos por Coluna**")
            st.dataframe(profile_nulls(df), use_container_width=True)
        with colB:
            num_cols_tmp = get_numeric_cols(df)
            st.markdown("**Outliers por IQR (numéricas)**")
            if num_cols_tmp:
                st.dataframe(profile_outliers(df[num_cols_tmp]), use_container_width=True)
            else:
                st.info("Sem colunas numéricas.")

        st.markdown("---")
        st.subheader("🧹 Tratamento Interativo")

        if "treat_cfg" not in st.session_state:
            st.session_state["treat_cfg"] = {
                "ignored_columns": [],
                "missing": {
                    "enabled": True,
                    "global_numeric": "median",
                    "global_categorical": "mode",
                    "per_numeric": {},
                    "per_categorical": {}
                },
                "cat_fill_constants": {},
                "outliers": {
                    "enabled": True,
                    "method": "iqr",
                    "global_strategy": "none",
                    "custom_low": 1.0,
                    "custom_high": 99.0,
                    "per_numeric": {},
                    "custom_perc": {}
                },
                "encoding": {
                    "enabled": True,
                    "global_categorical": "keep",
                    "per_categorical": {},
                    "ordinal_orders": {}
                }
            }

        cfg = st.session_state["treat_cfg"]

        st.markdown("**📝 Ignorar totalmente colunas (remover do pipeline):**")
        cfg["ignored_columns"] = st.multiselect(
            "Selecione colunas para excluir do tratamento, modelagem e gráficos:",
            options=list(df.columns),
            default=cfg.get("ignored_columns", [])
        )

        # === NULOS ===
        with st.expander("✅ Tratamento de Nulos", expanded=True):
            cfg["missing"]["enabled"] = st.checkbox("Habilitar tratamento de nulos", value=cfg["missing"]["enabled"])
            if not cfg["missing"]["enabled"]:
                st.info("Tópico desabilitado — nenhuma imputação será aplicada.")
            col1, col2 = st.columns([1,1])
            cfg["missing"]["global_numeric"] = col1.selectbox(
                "Estratégia global (numéricas):",
                ["median", "mean", "zero", "drop", "ignore"],
                index=["median","mean","zero","drop","ignore"].index(cfg["missing"]["global_numeric"]) if cfg["missing"]["enabled"] else 0,
                disabled=not cfg["missing"]["enabled"]
            )
            cfg["missing"]["global_categorical"] = col2.selectbox(
                "Estratégia global (categóricas):",
                ["mode", "constant", "drop", "ignore"],
                index=["mode","constant","drop","ignore"].index(cfg["missing"]["global_categorical"]) if cfg["missing"]["enabled"] else 0,
                disabled=not cfg["missing"]["enabled"]
            )

            num_cols = [c for c in get_numeric_cols(df) if c not in cfg["ignored_columns"]]
            cat_cols = [c for c in get_categorical_cols(df) if c not in cfg["ignored_columns"]]

            cA, cB = st.columns([1,1])
            if cA.button("Aplicar estratégia global às NUMÉRICAS"):
                for c in num_cols:
                    cfg["missing"]["per_numeric"][c] = cfg["missing"]["global_numeric"]
                st.success("Estratégia global aplicada a todas as numéricas.")
            if cB.button("Aplicar estratégia global às CATEGÓRICAS"):
                for c in cat_cols:
                    cfg["missing"]["per_categorical"][c] = cfg["missing"]["global_categorical"]
                st.success("Estratégia global aplicada a todas as categóricas.")

            st.markdown("**Overrides por coluna (opcional):**")
            if num_cols:
                st.caption("Numéricas:")
                for c in num_cols:
                    cfg["missing"]["per_numeric"].setdefault(c, cfg["missing"]["global_numeric"])
                    cfg["missing"]["per_numeric"][c] = st.selectbox(
                        f"{c}",
                        ["median","mean","zero","drop","ignore"],
                        index=["median","mean","zero","drop","ignore"].index(cfg["missing"]["per_numeric"][c]),
                        key=f"miss_num_{c}",
                        disabled=not cfg["missing"]["enabled"]
                    )
            if cat_cols:
                st.caption("Categóricas:")
                for c in cat_cols:
                    cfg["missing"]["per_categorical"].setdefault(c, cfg["missing"]["global_categorical"])
                    cols3 = st.columns([2,1])
                    cfg["missing"]["per_categorical"][c] = cols3[0].selectbox(
                        f"{c}",
                        ["mode","constant","drop","ignore"],
                        index=["mode","constant","drop","ignore"].index(cfg["missing"]["per_categorical"][c]),
                        key=f"miss_cat_{c}",
                        disabled=not cfg["missing"]["enabled"]
                    )
                    if cfg["missing"]["per_categorical"][c] == "constant" and cfg["missing"]["enabled"]:
                        cfg["cat_fill_constants"].setdefault(c, "Unknown")
                        cfg["cat_fill_constants"][c] = cols3[1].text_input(
                            "Constante", value=cfg["cat_fill_constants"][c], key=f"miss_cat_const_{c}"
                        )

        # === OUTLIERS ===
        with st.expander("📏 Tratamento de Outliers (IQR)", expanded=True):
            cfg["outliers"]["enabled"] = st.checkbox("Habilitar tratamento de outliers (IQR)", value=cfg["outliers"]["enabled"])
            if not cfg["outliers"]["enabled"]:
                st.info("Tópico desabilitado — nenhum tratamento de outliers será aplicado.")

            col1, col2, col3 = st.columns([1,1,1])
            cfg["outliers"]["global_strategy"] = col1.selectbox(
                "Estratégia global:",
                ["none","clip_1_99","clip_5_95","drop","custom","ignore"],
                index=["none","clip_1_99","clip_5_95","drop","custom","ignore"].index(cfg["outliers"]["global_strategy"]) if cfg["outliers"]["enabled"] else 0,
                disabled=not cfg["outliers"]["enabled"]
            )
            cfg["outliers"]["custom_low"] = float(col2.number_input("Low % (global custom)", min_value=0.0, max_value=50.0, value=float(cfg["outliers"]["custom_low"]), step=0.5, disabled=not cfg["outliers"]["enabled"]))
            cfg["outliers"]["custom_high"] = float(col3.number_input("High % (global custom)", min_value=50.0, max_value=100.0, value=float(cfg["outliers"]["custom_high"]), step=0.5, disabled=not cfg["outliers"]["enabled"]))

            num_cols = [c for c in get_numeric_cols(df) if c not in cfg["ignored_columns"]]

            if st.button("Aplicar estratégia global a TODAS as numéricas"):
                for c in num_cols:
                    cfg["outliers"]["per_numeric"][c] = cfg["outliers"]["global_strategy"]
                    if cfg["outliers"]["global_strategy"] == "custom":
                        cfg["outliers"]["custom_perc"][c] = (cfg["outliers"]["custom_low"], cfg["outliers"]["custom_high"])
                st.success("Estratégia global aplicada a todas as numéricas.")

            st.markdown("**Overrides por coluna (opcional):**")
            if num_cols:
                for c in num_cols:
                    cols4 = st.columns([2,1,1])
                    cfg["outliers"]["per_numeric"].setdefault(c, cfg["outliers"]["global_strategy"])
                    cfg["outliers"]["per_numeric"][c] = cols4[0].selectbox(
                        f"{c}",
                        ["none","clip_1_99","clip_5_95","drop","custom","ignore"],
                        index=["none","clip_1_99","clip_5_95","drop","custom","ignore"].index(cfg["outliers"]["per_numeric"][c]),
                        key=f"out_strat_{c}",
                        disabled=not cfg["outliers"]["enabled"]
                    )
                    if cfg["outliers"]["per_numeric"][c] == "custom" and cfg["outliers"]["enabled"]:
                        low_key = f"out_low_{c}"
                        high_key = f"out_high_{c}"
                        low_v = float(cols4[1].number_input("Low %", min_value=0.0, max_value=50.0, value=float(cfg["outliers"]["custom_perc"].get(c, (cfg['outliers']['custom_low'], cfg['outliers']['custom_high']))[0]), step=0.5, key=low_key))
                        high_v = float(cols4[2].number_input("High %", min_value=50.0, max_value=100.0, value=float(cfg["outliers"]["custom_perc"].get(c, (cfg['outliers']['custom_low'], cfg['outliers']['custom_high']))[1]), step=0.5, key=high_key))
                        cfg["outliers"]["custom_perc"][c] = (low_v, high_v)

        # === CATEGÓRICAS ===
        with st.expander("🔤 Codificação de Categóricas", expanded=True):
            cfg["encoding"]["enabled"] = st.checkbox("Habilitar codificação de categóricas", value=cfg["encoding"]["enabled"])
            if not cfg["encoding"]["enabled"]:
                st.info("Tópico desabilitado — nenhuma codificação será aplicada.")

            cat_cols = [c for c in get_categorical_cols(df) if c not in cfg["ignored_columns"]]
            if not cat_cols:
                st.info("Sem colunas categóricas detectadas (ou todas ignoradas).")
            else:
                cfg["encoding"]["global_categorical"] = st.selectbox(
                    "Estratégia global para categóricas:",
                    ["keep","onehot","ordinal","drop","ignore"],
                    index=["keep","onehot","ordinal","drop","ignore"].index(cfg["encoding"]["global_categorical"]),
                    disabled=not cfg["encoding"]["enabled"]
                )
                if st.button("Aplicar estratégia global a TODAS as categóricas"):
                    for c in cat_cols:
                        cfg["encoding"]["per_categorical"][c] = cfg["encoding"]["global_categorical"]
                    st.success("Estratégia global aplicada a todas as categóricas.")

                for c in cat_cols:
                    cols5 = st.columns([2,1])
                    cfg["encoding"]["per_categorical"].setdefault(c, cfg["encoding"]["global_categorical"])
                    cfg["encoding"]["per_categorical"][c] = cols5[0].selectbox(
                        f"{c}",
                        ["keep","onehot","ordinal","drop","ignore"],
                        index=["keep","onehot","ordinal","drop","ignore"].index(cfg["encoding"]["per_categorical"][c]),
                        key=f"enc_{c}",
                        disabled=not cfg["encoding"]["enabled"]
                    )
                    if cfg["encoding"]["per_categorical"][c] == "ordinal" and cfg["encoding"]["enabled"]:
                        default_order = ",".join(sorted([str(v) for v in df[c].dropna().unique().tolist()]))
                        existing = ",".join(cfg["encoding"]["ordinal_orders"].get(c, [])) if cfg["encoding"]["ordinal_orders"].get(c, []) else default_order
                        txt = cols5[1].text_input("Ordem (csv)", value=existing, key=f"enc_ord_{c}")
                        cfg["encoding"]["ordinal_orders"][c] = [v.strip() for v in txt.split(",") if len(v.strip())>0]

        st.markdown("---")
        if st.button("🚀 Aplicar Tratamento e Gerar Base Preparada"):
            try:
                df0 = df.drop(columns=cfg["ignored_columns"], errors="ignore")
                df1 = apply_missing(df0, cfg["missing"], cfg["cat_fill_constants"], enabled=cfg["missing"]["enabled"], ignored_cols=cfg["ignored_columns"])
                df2 = apply_outliers(df1, cfg["outliers"], enabled=cfg["outliers"]["enabled"], ignored_cols=cfg["ignored_columns"])
                df_prepared = apply_encoding(df2, cfg["encoding"], enabled=cfg["encoding"]["enabled"], ignored_cols=cfg["ignored_columns"])

                st.session_state["df_prepared"] = df_prepared
                st.success(f"Base preparada com sucesso! Linhas: {len(df_prepared)}, Colunas: {len(df_prepared.columns)}")
                st.dataframe(df_prepared.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao aplicar tratamento: {e}")

        if "df_prepared" in st.session_state:
            st.markdown("**Prévia da base preparada:**")
            st.dataframe(st.session_state["df_prepared"].head(10), use_container_width=True)

    with tab_model:
        st.subheader("🤖 Modelagem")
        if "df_prepared" not in st.session_state:
            st.info("Aplique o tratamento na aba 'Dados & Tratamento' para gerar a base preparada.")
        else:
            df_model = st.session_state["df_prepared"]
            all_num = get_numeric_cols(df_model)

            col1, col2 = st.columns([1,1])
            with col1:
                target = st.selectbox("Variável alvo (y):", options=all_num, index=(all_num.index("exam_score") if "exam_score" in all_num else 0))
                features = st.multiselect("Variáveis preditoras (X):", options=[c for c in all_num if c != target], default=[])
                model_name = st.selectbox("Modelo:", ["Linear Regression", "Ridge (L2)", "Random Forest"], index=0)
                test_size = st.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
                seed = st.number_input("Random state", min_value=0, value=42, step=1)

                use_robust_scaler_flag = False

                if not features:
                    st.warning("Selecione ao menos uma variável preditora (X).")
                else:
                    try:
                        model, metrics, feat_imp, test_pack = train_regressor(
                            df_model, target, features, model_name,
                            test_size=test_size, random_state=seed, use_robust=use_robust_scaler_flag
                        )
                        X_test, y_test, y_pred = test_pack
                        mtrx = pd.DataFrame({"Métrica": ["R²", "RMSE", "MAE"], "Valor": [metrics["R²"], metrics["RMSE"], metrics["MAE"]]})
                        mtrx["Valor"] = mtrx["Valor"].map(lambda v: f"{v:.4f}")
                        st.markdown("**Métricas (teste):**")
                        st.dataframe(mtrx, hide_index=True, use_container_width=True)

                        if feat_imp is not None and not feat_imp.empty:
                            st.markdown("**Importância / Coeficientes:**")
                            st.dataframe(feat_imp.reset_index(drop=True), use_container_width=True)
                        st.session_state["last_model_metrics"] = metrics
                        st.session_state["last_feat_imp"] = feat_imp
                        st.session_state["last_model_name"] = model_name
                        st.success("Treino concluído!")
                    except Exception as e:
                        st.error(f"Erro no treino: {e}")

    with tab_viz:
        st.subheader("📈 Visualizações (múltiplas)")
        if "df_prepared" not in st.session_state:
            st.info("Aplique o tratamento na aba 'Dados & Tratamento' para gerar a base preparada.")
        else:
            df_v = st.session_state["df_prepared"]
            all_num = get_numeric_cols(df_v)
            all_cat = get_categorical_cols(df_v)

            st.sidebar.subheader("🧱 Construtor de Gráficos")
            add_chart = st.sidebar.button("➕ Adicionar gráfico")

            _init_chart_state()
            if add_chart:
                st.session_state["charts"].append(_default_chart_config(all_num, all_cat))

            if not st.session_state["charts"]:
                st.info("Clique em **➕ Adicionar gráfico** na barra lateral para começar.")
                chart_figs = []
            else:
                chart_figs = []
                for idx, cfg in enumerate(st.session_state["charts"]):
                    fig = render_chart_form(idx, cfg, all_num, all_cat, df_v)
                    if fig is not None:
                        chart_figs.append(fig)
            st.session_state["chart_figs"] = chart_figs

    with tab_export:
        st.subheader("💾 Exportação")
        if "df_prepared" in st.session_state:
            st.markdown("**Baixar CSV tratado**")
            st.download_button(
                label="⬇️ CSV tratado",
                data=st.session_state["df_prepared"].to_csv(index=False).encode("utf-8"),
                file_name="dados_tratados.csv",
                mime="text/csv"
            )
        else:
            st.info("Gere a base preparada na aba 'Dados & Tratamento' para habilitar o download.")

        st.markdown("---")
        st.subheader("📄 Exportar PDF")
        chart_figs = st.session_state.get("chart_figs", [])
        metrics = st.session_state.get("last_model_metrics", None)
        feat_imp = st.session_state.get("last_feat_imp", None)
        model_name = st.session_state.get("last_model_name", None)

        summary_lines = [
            f"Data do relatório: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "",
            "Seção de Tratamento:",
            "- Ignorados: " + (", ".join(st.session_state["treat_cfg"]["ignored_columns"]) if "treat_cfg" in st.session_state else "(nenhum)"),
            "- Nulos: global + overrides por coluna (num/cat); opção ignore e liga/desliga",
            "- Outliers: IQR (none/clip/drop/custom/ignore) + overrides por coluna",
            "- Codificação: keep/one-hot/ordinal/drop/ignore + ordem para ordinal",
            "",
            "Seção de Modelagem:",
            f"- Modelo: {model_name if model_name else '(não treinado)'}",
        ]
        if metrics:
            summary_lines += [
                "Métricas (teste):",
                f"  · R²: {metrics['R²']:.4f}",
                f"  · RMSE: {metrics['RMSE']:.4f}",
                f"  · MAE: {metrics['MAE']:.4f}",
            ]
        if feat_imp is not None and isinstance(feat_imp, pd.DataFrame) and not feat_imp.empty:
            summary_lines.append("Importâncias/Coeficientes:")
            for _, row in feat_imp.iterrows():
                summary_lines.append(f"  · {row['feature']}: {row['importance']:.4f}")

        summary = "\n".join(summary_lines)
        pdf_bytes = build_pdf(summary, chart_figs if chart_figs else [])
        st.download_button(
            label="⬇️ Baixar PDF",
            data=pdf_bytes,
            file_name="relatorio_tratamento_analise.pdf",
            mime="application/pdf"
        )

        st.caption("O PDF inclui uma página de resumo e todas as figuras geradas na aba de Visualizações.")

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, beta
import pandas as pd

# --- Заголовок приложения ---
st.set_page_config(page_title="Доверительные интервалы для пропорций", layout="wide")
st.title("📊 Доверительные интервалы для пропорций")
st.markdown("Интерактивная веб-панель для сравнения методов расчёта CI: Wald, Wilson, Agresti-Coull, Clopper-Pearson и др.")

# --- Функции для доверительных интервалов ---
def wald_confidence_interval(x, n, confidence=0.95):
    if n == 0 or x < 0 or x > n:
        return (np.nan, np.nan)
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    p_hat = x / n
    if p_hat == 0 or p_hat == 1:
        se = 0
    else:
        se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * se
    lower = max(0, p_hat - margin)
    upper = min(1, p_hat + margin)
    return (lower, upper)

def wilson_confidence_interval(x, n, confidence=0.95):
    if n == 0 or x < 0 or x > n:
        return (np.nan, np.nan)
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    p_hat = x / n
    z2 = z ** 2
    denom = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0, lower), min(1, upper))

def agresti_coull_interval(x, n, confidence=0.95):
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    z2 = z ** 2
    n_tilde = n + z2
    p_tilde = (x + z2 / 2) / n_tilde
    se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    margin = z * se
    lower = max(0, p_tilde - margin)
    upper = min(1, p_tilde + margin)
    return (lower, upper)

def clopper_pearson_interval(x, n, confidence=0.95):
    alpha = 1 - confidence
    if x == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, x, n - x + 1)
    if x == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, x + 1, n - x)
    return (lower, upper)

def plus_4_interval(x, n, confidence=0.95):
    if abs(confidence - 0.95) > 0.01:
        return (np.nan, np.nan)
    x_adj = x + 2
    n_adj = n + 4
    p_adj = x_adj / n_adj
    z = 1.96
    se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
    margin = z * se
    lower = max(0, p_adj - margin)
    upper = min(1, p_adj + margin)
    return (lower, upper)


# --- Боковая панель: ввод параметров ---
st.sidebar.header("🔧 Параметры")
n = st.sidebar.slider("Размер выборки $n$", min_value=5, max_value=200, value=30, step=1)
confidence = st.sidebar.slider("Уровень доверия", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
show_animation = st.sidebar.checkbox("Показать анимацию изменения CI при росте $n$")

# --- Основная часть: графики и таблицы ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Доверительные интервалы при $n = {n}$ и доверии = {confidence:.0%}")

    x_values = np.arange(0, n + 1)
    p_values = x_values / n

    methods = {
        "Wald": wald_confidence_interval,
        "Wilson": wilson_confidence_interval,
        "Agresti-Coull": agresti_coull_interval,
        "Clopper-Pearson": clopper_pearson_interval,
    }
    if abs(confidence - 0.95) < 0.01:
        methods["Plus-4"] = plus_4_interval

    colors = {
        "Wald": "#d62728",
        "Wilson": "#1f77b4",
        "Agresti-Coull": "#2ca02c",
        "Clopper-Pearson": "#ff7f0e",
        "Plus-4": "#9467bd"
    }

    fig = go.Figure()

    for name, func in methods.items():
        lowers = [func(x, n, confidence)[0] for x in x_values]
        uppers = [func(x, n, confidence)[1] for x in x_values]

        fig.add_trace(go.Scatter(
            x=lowers, y=p_values, mode='lines', line=dict(color=colors[name]), name=f"{name} (низ)", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=uppers, y=p_values, mode='lines', line=dict(color=colors[name], dash='dot'), name=f"{name} (верх)", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=uppers + lowers[::-1], y=p_values.tolist() + p_values[::-1].tolist(),
            fill='toself', fillcolor=colors[name], opacity=0.1, line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='none'
        ))

    fig.update_layout(
        title=f"Сравнение методов (n={n}, доверие={confidence:.0%})",
        xaxis_title="Доверительный интервал",
        yaxis_title="Доля $\\hat{{p}} = x/n$",
        hovermode="closest",
        height=600,
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1),
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02])
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Рекомендация по методу")

    def recommend_method(n, p_hat=None, confidence=0.95, context="research"):
        if n < 10:
            return "✅ **Wilson** или **Clopper-Pearson**"
        if n < 30 or (p_hat is not None and (p_hat < 0.1 or p_hat > 0.9)):
            return "✅ **Wilson**"
        if abs(confidence - 0.95) < 0.01 and context == "teaching":
            return "✅ **Plus-4** или **Wilson**"
        if context == "conservative":
            return "✅ **Clopper-Pearson**"
        return "✅ **Wilson** (рекомендуется)"

    context = st.radio("Цель анализа", ["research", "teaching", "conservative"])
    example_p = st.selectbox("Пример доли", [0.1, 0.3, 0.5, 0.8, 0.95], format_func=lambda x: f"p̂ = {x}")

    recommendation = recommend_method(n=n, p_hat=example_p, confidence=confidence, context=context)
    st.markdown(recommendation)

    st.markdown("---")
    st.markdown("### ℹ️ Подсказка")
    st.markdown("""
    - **Wilson** — лучший выбор в большинстве случаев
    - **Wald** не используйте при $n < 50$ или $p̂$ около 0/1
    - **Clopper-Pearson** — консервативен (шире интервалы)
    """)

# --- Анимация роста n ---
if show_animation:
    st.subheader("🎬 Анимация: как меняются интервалы при росте $n$")
    st.markdown("Показано изменение интервалов при увеличении $n$ от 5 до 50 (при $\\hat{p} = 0.5$)")

    frames = []
    n_range = list(range(5, 51, 5))
    fig_anim = go.Figure()

    for n_step in n_range:
        x_vals = np.array([int(n_step * 0.5)])
        p_vals = x_vals / n_step

        traces = []
        for name, func in methods.items():
            if name == "Plus-4" and abs(confidence - 0.95) > 0.01:
                continue
            for x in x_vals:
                low, high = func(x, n_step, confidence)
                traces.append(go.Scatter(
                    x=[low, high], y=[p_vals[0], p_vals[0]], mode='lines+markers',
                    line=dict(color=colors.get(name, "black")),
                    name=name, showlegend=(n_step == 5)
                ))
        frames.append(go.Frame(data=traces, layout=go.Layout(title=f"n = {n_step}")))

    # Первый кадр
    x_vals = np.array([int(5 * 0.5)])
    p_vals = x_vals / 5
    init_traces = []
    for name, func in methods.items():
        if name == "Plus-4" and abs(confidence - 0.95) > 0.01:
            continue
        for x in x_vals:
            low, high = func(x, 5, confidence)
            init_traces.append(go.Scatter(
                x=[low, high], y=[p_vals[0], p_vals[0]], mode='lines+markers',
                line=dict(color=colors.get(name, "black")), name=name
            ))

    fig_anim = go.Figure(
        data=init_traces,
        layout=go.Layout(
            title="Анимация: рост n от 5 до 50 (p̂ ≈ 0.5)",
            xaxis=dict(range=[0, 1], title="Доверительный интервал"),
            yaxis=dict(range=[0.4, 0.6], title="Доля"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.1, y=0,
                buttons=[dict(label="▶️ Воспроизвести",
                              method="animate",
                              args=[None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}])]
            )]
        ),
        frames=frames
    )
    st.plotly_chart(fig_anim, use_container_width=True)

# --- Таблица сравнения ---
st.sidebar.subheader("🔢 Пример сравнения (x=15, n=30)")
if st.sidebar.button("Показать таблицу"):
    x = 15
    data = []
    for name, func in methods.items():
        try:
            low, high = func(x, n, confidence)
            width = high - low
            data.append({"Метод": name, "Нижняя": f"{low:.3f}", "Верхняя": f"{high:.3f}", "Ширина": f"{width:.3f}"})
        except:
            pass
    df = pd.DataFrame(data)
    st.sidebar.dataframe(df, use_container_width=True)
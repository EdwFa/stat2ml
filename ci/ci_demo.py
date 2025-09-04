import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
import warnings
warnings.filterwarnings("ignore")

# --- Все функции для доверительных интервалов ---
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
    if abs(confidence - 0.95) > 0.01:  # приблизительно
        return (np.nan, np.nan)  # не рекомендуется
    x_adj = x + 2
    n_adj = n + 4
    p_adj = x_adj / n_adj
    z = 1.96  # z_{0.975}
    se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
    margin = z * se
    lower = max(0, p_adj - margin)
    upper = min(1, p_adj + margin)
    return (lower, upper)


# --- Функция сравнения и построения графика ---
def plot_ci_comparison(n=50, confidence=0.95):
    """Строит график доверительных интервалов для всех x от 0 до n."""
    x_values = np.arange(0, n + 1)
    p_values = x_values / n

    methods = {
        "Wald": wald_confidence_interval,
        "Wilson": wilson_confidence_interval,
        "Agresti-Coull": agresti_coull_interval,
        "Clopper-Pearson": clopper_pearson_interval,
    }

    # Исключение Plus-4, если не 95%
    if abs(confidence - 0.95) < 0.01:
        methods["Plus-4"] = plus_4_interval

    plt.figure(figsize=(12, 8))

    colors = {
        "Wald": "red",
        "Wilson": "blue",
        "Agresti-Coull": "green",
        "Clopper-Pearson": "orange",
        "Plus-4": "purple"
    }

    for name, func in methods.items():
        lowers = []
        uppers = []
        for x in x_values:
            try:
                low, high = func(x, n, confidence)
                lowers.append(low if not np.isnan(low) else None)
                uppers.append(high if not np.isnan(high) else None)
            except:
                lowers.append(None)
                uppers.append(None)
        plt.plot(lowers, p_values, color=colors[name], linestyle='-', alpha=0.7)
        plt.plot(uppers, p_values, color=colors[name], linestyle='-', alpha=0.7, label=name)
        # Заливка между нижней и верхней границей
        plt.fill_betweenx(p_values, lowers, uppers, color=colors[name], alpha=0.1)

    plt.axvline(0, color='k', linewidth=0.5, linestyle='--')
    plt.axvline(1, color='k', linewidth=0.5, linestyle='--')
    plt.xlabel("Доверительный интервал")
    plt.ylabel("Выборочная доля $\\hat{p} = x/n$")
    plt.title(f"Сравнение доверительных интервалов для пропорции (n = {n}, {confidence:.0%} доверие)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.show()


# --- Оценка "лучшего" метода ---
def evaluate_method_coverage(method_func, n, p_true=0.5, confidence=0.95, n_sim=10000):
    """Оценивает вероятность покрытия (coverage) метода при истинном p_true.
    p_true — это гипотетическое истинное значение доли в популяции, которое мы не знаем
    на практике, но можем использовать в симуляциях, чтобы проверить, насколько надёжны те
    или иные статистические методы"""
    covered = 0
    for _ in range(n_sim):
        x = np.random.binomial(n, p_true)
        try:
            low, high = method_func(x, n, confidence)
            if low <= p_true <= high:
                covered += 1
        except:
            continue
    coverage = covered / n_sim
    return coverage

def compare_methods_by_quality(n=30, p_true=0.5, confidence=0.95):
    """Сравнивает методы по покрытию и средней ширине интервала."""
    methods = {
        "Wald": wald_confidence_interval,
        "Wilson": wilson_confidence_interval,
        "Agresti-Coull": agresti_coull_interval,
        "Clopper-Pearson": clopper_pearson_interval,
    }

    if abs(confidence - 0.95) < 0.01:
        methods["Plus-4"] = plus_4_interval

    results = []

    print(f"Оценка качества методов (n={n}, p_true={p_true}, {confidence:.0%} доверие)")
    print(f"{'Метод':<16} {'Покрытие':<10} {'Ср. ширина':<12} {'Качество'}")
    print("-" * 50)

    for name, func in methods.items():
        coverages = []
        widths = []

        for _ in range(5000):  # Моделируем все возможные x
            x = np.random.binomial(n, p_true)
            try:
                low, high = func(x, n, confidence)
                if np.isnan(low) or np.isnan(high):
                    continue
                coverages.append(p_true >= low and p_true <= high)
                widths.append(high - low)
            except:
                continue

        avg_coverage = np.mean(coverages) if coverages else np.nan
        avg_width = np.mean(widths) if widths else np.nan

        # Качество: близость покрытия к confidence и минимальная ширина
        coverage_error = abs(avg_coverage - confidence)
        score = -coverage_error * 1000 + (-avg_width * 100) if not np.isnan(avg_width) else -np.inf

        if avg_coverage >= confidence - 0.02 and avg_coverage <= confidence + 0.02:
            quality = "✅ Хорош"
        elif avg_coverage > confidence:
            quality = "🟨 Консервативен"
        else:
            quality = "❌ Ниже треб."

        results.append((score, name, avg_coverage, avg_width, quality))

    # Сортируем по качеству (оценке)
    results.sort(reverse=True)

    for _, name, cov, width, qual in results:
        print(f"{name:<16} {cov:<10.3f} {width:<12.3f} {qual}")

    print("\n🏆 Лучший метод по балансу покрытия и точности:", results[0][1])
    return results[0][1]


# === ЗАПУСК ПРИМЕРОВ ===
if __name__ == "__main__":
    # График сравнения
    plot_ci_comparison(n=100, confidence=0.9)

    # Сравнение по качеству
    best_method = compare_methods_by_quality(n=100, p_true=0.95, confidence=0.9)
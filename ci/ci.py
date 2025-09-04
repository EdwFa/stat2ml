import math
from scipy.stats import norm

def wilson_confidence_interval(x, n, confidence=0.95):
    """
    Вычисляет доверительный интервал для доли (пропорции) по методу Уилсона.

    Параметры:
    ----------
    x : int
        Количество "успехов" в выборке.
    n : int
        Общий размер выборки.
    confidence : float, optional (по умолчанию 0.95)
        Уровень доверия (например, 0.95 для 95%).

    Возвращает:
    -----------
    tuple
        (нижняя_граница, верхняя_граница) — доверительный интервал.

    Пример:
    -------
    >>> wilson_confidence_interval(x=45, n=100, confidence=0.95)
    (0.355, 0.549)
    """
    if n == 0:
        raise ValueError("Размер выборки n должен быть больше 0.")
    if x < 0 or x > n:
        raise ValueError("Число успехов x должно быть в диапазоне [0, n].")
    if not (0 < confidence < 1):
        raise ValueError("Уровень доверия должен быть между 0 и 1.")

    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)  # z-критическое значение

    p_hat = x / n
    z2 = z ** 2

    # Формула Уилсона
    denominator = 1 + z2 / n
    center_term = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n ** 2)))

    lower = (center_term - margin) / denominator
    upper = (center_term + margin) / denominator

    return (lower, upper)


# Пример использования
if __name__ == "__main__":
    x = 45  # число успехов
    n = 100  # размер выборки
    ci = wilson_confidence_interval(x, n, confidence=0.95)
    print(f"Доверительный интервал (метод Уилсона): [{ci[0]:.3f}, {ci[1]:.3f}]")
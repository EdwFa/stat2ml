from ci.ci import wilson_confidence_interval
from ci.ci_demo import clopper_pearson_interval
from ci.ci_demo import agresti_coull_interval

if __name__ == '__main__':
    x = 98  # число успехов
    n = 100  # размер выборки
    confidence = 0.90 # уровень значимости

    ci = wilson_confidence_interval(x, n, confidence)
    print(f"Доверительный интервал (метод Уилсона): [{ci[0]:.3f}, {ci[1]:.3f}]")
    ci = clopper_pearson_interval(x, n, confidence)
    print(f"Доверительный интервал (метод Клопера-Пирсона): [{ci[0]:.3f}, {ci[1]:.3f}]")
    ci = agresti_coull_interval(x, n, confidence)
    print(f"Доверительный интервал (метод Агрести-Коуэл): [{ci[0]:.3f}, {ci[1]:.3f}]")




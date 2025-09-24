import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import pmdarima
    print(f"✓ pmdarima {pmdarima.__version__} установлен")
    
    # Проверка основных функций
    from pmdarima import auto_arima
    print("✓ Функция auto_arima доступна")
    
except ImportError as e:
    print(f"✗ Ошибка: {e}")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__} установлен")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__} установлен")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib as mpl
    print(f"✓ matplotlib {mpl.__version__} установлен")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib.pyplot доступен")
except ImportError as e:
    print(f"✗ matplotlib.pyplot: {e}")

print("\n=== Все зависимости работают! ===")
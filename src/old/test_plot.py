import matplotlib.pyplot as plt
import numpy as np

print("Testing VS Code plotting...")

# Простой тест
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Тестовый график в VS Code')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

print("Если график не показался, он сохранится в файл...")
plt.savefig('test_plot.png')
import numpy as np
import wavelets as lw
import matplotlib.pyplot as plt

n = 1000

x = np.linspace(0, 3.14, n, dtype=np.float64)
data = np.sin(3 * x) + np.exp(x / 5) + np.cos(1 / (0.01 + x))
wavelets = np.zeros_like(x)
result = np.zeros_like(x)

dataCpy = data.copy()
lw.wavelet_transform1D(dataCpy.ctypes.data, n, wavelets.ctypes.data)
lw.invwavelet_transform1D(wavelets.ctypes.data, n, result.ctypes.data)

f1 = plt.figure(1)
f1.clear()
plt.subplot(2, 1, 1)
plt.plot(data, label="data")
plt.plot(result, label="result")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(result - data, label="result - data")
plt.legend()

plt.show()

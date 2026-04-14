import numpy as np
import matplotlib.pyplot as plt

# CASO 1
fs = 500
t = np.linspace(0, 1, fs, endpoint=False)

# Señales
x1 = np.sin(2*np.pi*250*t)     # 250 Hz
x2 = np.sin(2*np.pi*1000*t)    # 1000 Hz

x = x1 + x2

# FFT
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# 2. CASO 2
fs2 = 3000
t2 = np.linspace(0, 1, fs2, endpoint=False)

x1_2 = np.sin(2*np.pi*250*t2)
x2_2 = np.sin(2*np.pi*1000*t2)

x_ok = x1_2 + x2_2

X_fft2 = np.fft.fft(x_ok)
frecuencias2 = np.fft.fftfreq(len(x_ok), 1/fs2)

n2 = len(x_ok) // 2
f_positivas2 = frecuencias2[:n2]
magnitud2 = np.abs(X_fft2[:n2]) * 2 / len(x_ok)

plt.figure(figsize=(8,4))
plt.plot(t, x)
plt.title("Señal en el tiempo")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,4))
plt.stem(f_positivas, magnitud, basefmt=" ")
plt.title("FFT con aliasing")
plt.xlim(0,1500)
plt.grid(True)
plt.show()

# Tiempo sin aliasing
plt.figure(figsize=(8,4))
plt.plot(t2, x_ok)
plt.xlim(0, 0.01)
plt.title("Señal en el tiempo")
plt.grid(True)
plt.show()

# FFT sin aliasing
plt.figure(figsize=(8,4))
plt.stem(f_positivas2, magnitud2, basefmt=" ")
plt.title("FFT sin aliasing")
plt.xlim(0,1500)
plt.grid(True)
plt.show()
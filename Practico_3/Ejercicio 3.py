import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. PARÁMETROS
# -------------------------------
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# -------------------------------
# 2. GENERAR SEÑALES
# -------------------------------
x1 = np.sin(2*np.pi*50*t)    # 50 Hz
x2 = np.sin(2*np.pi*200*t)   # 200 Hz

# Señal compuesta
x = x1 + x2

# -------------------------------
# 3. FFT
# -------------------------------
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# -------------------------------
# 4. FILTRADO (eliminar 50 Hz)
# -------------------------------
X_fft_filtrada = X_fft.copy()

mask = np.where(np.abs(np.abs(frecuencias) - 50) < 1)
X_fft_filtrada[mask] = 0

senal_filtrada = np.fft.ifft(X_fft_filtrada).real
magnitud_filtrada = np.abs(X_fft_filtrada[:n_mitad]) * 2 / N

# -------------------------------
# 5. GRÁFICAS SEPARADAS
# -------------------------------

# Señal 1
plt.figure(figsize=(8,4))
plt.plot(t, x1)
plt.title("Señal 1 (50 Hz)")
plt.grid(True)
plt.show()

# Señal 2
plt.figure(figsize=(8,4))
plt.plot(t, x2)
plt.title("Señal 2 (200 Hz)")
plt.grid(True)
plt.show()

# Señal compuesta
plt.figure(figsize=(8,4))
plt.plot(t, x, label="Original")
plt.plot(t, senal_filtrada, label="Filtrada (sin 50 Hz)", color='red')
plt.legend()
plt.title("Señal compuesta")
plt.grid(True)
plt.show()

# FFT original
plt.figure(figsize=(8,4))
plt.plot(f_positivas, magnitud)
plt.title("Espectro de Frecuencia (Original)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,300)
plt.grid(True)
plt.show()

# FFT filtrada
plt.figure(figsize=(8,4))
plt.plot(f_positivas, magnitud_filtrada, color='green')
plt.title("Espectro de Frecuencia (Filtrada)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,300)
plt.grid(True)
plt.show()


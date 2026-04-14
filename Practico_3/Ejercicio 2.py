import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal con ruido
x = np.sin(2*np.pi*60*t) + 0.5*np.random.randn(len(t))

# FFT
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# --- FILTRADO EN FRECUENCIA ---
X_fft_filtrada = X_fft.copy()

# Dejamos solo la frecuencia de 60 Hz (quitamos el resto)
mask = np.abs(np.abs(frecuencias) - 60) > 5
X_fft_filtrada[mask] = 0

# Reconstrucción de la señal
senal_filtrada = np.fft.ifft(X_fft_filtrada).real

# Magnitud filtrada
magnitud_filtrada = np.abs(X_fft_filtrada[:n_mitad]) * 2 / N

# --- GRÁFICAS ---
plt.figure(figsize=(10,10))

# Señal en el tiempo
plt.subplot(3,1,1)
plt.plot(t, x, label="Original con ruido", alpha=0.5)
plt.plot(t, senal_filtrada, label="Filtrada", color='red')
plt.title("Señal en el tiempo")
plt.legend()
plt.grid(True)

# Espectro original
plt.subplot(3,1,2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencia (con ruido)")
plt.xlim(0,150)
plt.grid(True)

# Espectro filtrado
plt.subplot(3,1,3)
plt.plot(f_positivas, magnitud_filtrada, color='green')
plt.title("Espectro después del filtrado")
plt.xlim(0,150)
plt.grid(True)

plt.tight_layout()
plt.show()

# --- RESULTADO EN CONSOLA ---
print("Frecuencia principal esperada: 60 Hz")
print("En la FFT se observa un pico dominante en 60 Hz")
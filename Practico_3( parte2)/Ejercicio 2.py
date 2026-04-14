import numpy as np
import matplotlib.pyplot as plt
import librosa

y, fs = librosa.load('Under Pressure.mp3')

ruido = 0.2 * np.random.randn(len(y))
y_ruido = y + ruido

N = len(y)

Y_fft = np.fft.fft(y)
Y_fft_ruido = np.fft.fft(y_ruido)

frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]

magnitud = np.abs(Y_fft[:n_mitad]) * 2 / N
magnitud_ruido = np.abs(Y_fft_ruido[:n_mitad]) * 2 / N

t = np.linspace(0, len(y)/fs, len(y))

plt.figure(figsize=(10,8))
plt.subplot(4,1,1)
plt.plot(t[:30000], y[:30000], alpha=0.5, label="original")
plt.legend()
plt.title("Señal en el tiempo")
plt.grid(True)
plt.subplot(4,1,2)
plt.plot(t[:30000], y_ruido[:30000], alpha=0.6, label="Con ruido")
plt.legend()
plt.title("Señal en el tiempo")
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(f_positivas, magnitud)
plt.xlim(0,5000)
plt.title("FFT señal original")
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(f_positivas, magnitud_ruido)
plt.xlim(0,5000)
plt.title("FFT señal con ruido")
plt.grid(True)

plt.tight_layout()
plt.show()
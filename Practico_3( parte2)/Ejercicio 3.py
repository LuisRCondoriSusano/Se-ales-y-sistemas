import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

y, fs = librosa.load('audio_con_ruido.wav')

N = len(y)
Y_fft = np.fft.fft(y)
frecuencias = np.fft.fftfreq(N, 1/fs)

frecuencia_corte = 100

Y_fft_filtrada = Y_fft.copy()
mask = np.abs(frecuencias) > frecuencia_corte
Y_fft_filtrada[mask] = 0

y_filtrado = np.fft.ifft(Y_fft_filtrada).real

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]

magnitud = np.abs(Y_fft[:n_mitad]) * 2 / N
magnitud_filtrada = np.abs(Y_fft_filtrada[:n_mitad]) * 2 / N

indice = np.argmax(magnitud)
frecuencia_dominante = f_positivas[indice]

t = np.linspace(0, len(y)/fs, len(y))

plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(t[:2000], y[:2000], label="Original")
plt.plot(t[:2000], y_filtrado[:2000], alpha=0.7, label="Filtrada")
plt.legend()
plt.title("Señal en el tiempo")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(f_positivas, magnitud)
plt.xlim(0,5000)
plt.title("FFT Original")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(f_positivas, magnitud_filtrada)
plt.xlim(0,5000)
plt.title("FFT Filtrada (Pasa-bajo)")
plt.grid(True)

plt.tight_layout()
plt.show()

sf.write('audio_filtrado.wav', y_filtrado, fs)

print("Frecuencia dominante:", frecuencia_dominante, "Hz")
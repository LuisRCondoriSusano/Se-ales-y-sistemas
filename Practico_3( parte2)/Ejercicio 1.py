import numpy as np
import matplotlib.pyplot as plt
import librosa

y, fs = librosa.load('Under Pressure.mp3')

N = len(y)
Y_fft = np.fft.fft(y)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(Y_fft[:n_mitad]) * 2 / N

indice = np.argmax(magnitud)
frecuencia_dominante = f_positivas[indice]

t = np.linspace(0, len(y)/fs, len(y))

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, y)
plt.title("Señal de audio (tiempo)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.xlim(0, 5000)
plt.title("Espectro de frecuencia")
plt.grid(True)

plt.tight_layout()
plt.show()

print("Frecuencia dominante:", frecuencia_dominante, "Hz")
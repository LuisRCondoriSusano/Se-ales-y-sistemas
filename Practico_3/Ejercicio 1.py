import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
x = np.sin(2 * np.pi * 40 * t)

N = len(x)
X_fft = np.fft.fft(x)
freqs = np.fft.fftfreq(N, 1/fs)
mitad = N // 2

magnitud = np.abs(X_fft[:mitad]) * 2 / N

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t[:150], x[:150], color='blue')
plt.title("Señal en el tiempo: sin(2π·40·t)")
plt.xlabel("Tiempo [s]")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqs[:mitad], magnitud, color='red', alpha=0.8)  # ← plot en vez de stem
plt.title("Espectro FFT — pico en 40 Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 100)
plt.grid(True)

plt.tight_layout()
plt.show()
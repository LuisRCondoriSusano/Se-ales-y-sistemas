import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

f_desconocida = np.random.randint(50, 200)
x = np.sin(2*np.pi*f_desconocida*t)

N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

indice = np.argmax(magnitud)
frecuencia_detectada = f_positivas[indice]

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, x)
plt.title("Señal en el tiempo")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.xlim(0,250)
plt.title("Espectro de frecuencia")
plt.grid(True)

plt.tight_layout()
plt.show()

print("Frecuencia real:", f_desconocida)
print("Frecuencia detectada:", frecuencia_detectada)
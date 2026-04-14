import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('perro.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitud = 20 * np.log(np.abs(fshift) + 1)

rows, cols = img.shape
crow, ccol = rows//2, cols//2  

mask = np.zeros((rows, cols), np.uint8)
r = 30
cv2.circle(mask, (ccol, crow), r, 1, -1)

fshift_filtered = fshift * mask

magnitud_filtrada = 20 * np.log(np.abs(fshift_filtered) + 1)

f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(magnitud, cmap='gray')
plt.title('Espectro de Magnitud')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(mask, cmap='gray')
plt.title('Máscara Pasa-bajo')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(img_back, cmap='gray')
plt.title('Imagen Filtrada')
plt.axis('off')

plt.tight_layout()
plt.show()
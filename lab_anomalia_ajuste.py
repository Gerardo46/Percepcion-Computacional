"""Script que emula las celdas de un cuaderno de trabajo.

El código aplica técnicas básicas para eliminar ruido sal y pimienta y para
mejorar el contraste de imágenes en escala de grises. Los resultados se
almacenan en carpetas locales bajo ``outputs/``.
"""

import os
import cv2
import numpy as np

def add_salt_pepper(img, amount=0.05):
    out = img.copy()
    num_salt = int(np.ceil(amount * img.size * 0.5))
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    out[coords[0], coords[1]] = 255
    num_pepper = int(np.ceil(amount * img.size * 0.5))
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    out[coords[0], coords[1]] = 0
    return out

def custom_median_filter(img, ksize=3):
    pad = ksize // 2
    padded = np.pad(img, pad, mode='edge')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            out[i, j] = np.median(window)
    return out

def psnr(target, ref):
    mse = np.mean((target.astype(np.float64) - ref.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# 1.1 Tests unitarios básicos
test_img = np.zeros((10, 10), dtype=np.uint8)
assert psnr(test_img, test_img) == float('inf')
ref = np.full((10, 10), 50, dtype=np.uint8)
tgt = np.full((10, 10), 52, dtype=np.uint8)
print(f"PSNR test (dif 2 niveles): {psnr(tgt, ref):.2f} dB")

# 2. Eliminación de anomalías (ruido sal y pimienta)
# 2.1 Carga y validación
ROOT_DIR = os.path.dirname(__file__)
i1 = cv2.imread(os.path.join(ROOT_DIR, 'Cars14.png'), cv2.IMREAD_GRAYSCALE)
i2 = cv2.imread(os.path.join(ROOT_DIR, 'Cars37.png'), cv2.IMREAD_GRAYSCALE)
txt = lambda img, name: (_ for _ in ()).throw(ValueError(f"No se pudo cargar {name}")) if img is None else None
txt(i1, 'Cars14.png')
txt(i2, 'Cars37.png')
noisy1 = add_salt_pepper(i1, amount=0.02)
noisy2 = add_salt_pepper(i2, amount=0.02)

# 2.2 Aplicación de filtros
den1_custom = custom_median_filter(noisy1, ksize=3)
den2_custom = custom_median_filter(noisy2, ksize=3)
den1_cv = cv2.medianBlur(noisy1, 3)
den2_cv = cv2.medianBlur(noisy2, 3)

# 2.3 Medición de desempeño
print("PSNR Imagen 1 custom :", psnr(den1_custom, i1))
print("PSNR Imagen 1 OpenCV :", psnr(den1_cv, i1))
print("PSNR Imagen 2 custom :", psnr(den2_custom, i2))
print("PSNR Imagen 2 OpenCV :", psnr(den2_cv, i2))

# 2.4 Guardado de resultados
output_dir = os.path.join(ROOT_DIR, 'outputs', 'anomalias')
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, 'i1_noisy.png'), noisy1)
cv2.imwrite(os.path.join(output_dir, 'i1_custom.png'), den1_custom)
cv2.imwrite(os.path.join(output_dir, 'i1_opencv.png'), den1_cv)
cv2.imwrite(os.path.join(output_dir, 'i2_noisy.png'), noisy2)
cv2.imwrite(os.path.join(output_dir, 'i2_custom.png'), den2_custom)
cv2.imwrite(os.path.join(output_dir, 'i2_opencv.png'), den2_cv)
print(f"Resultados guardados en {output_dir}")

# 3. Ajuste de intensidad (mejora de contraste)
# 3.1 Carga y validación
i3 = cv2.imread(os.path.join(ROOT_DIR, 'coche02.jpeg'), cv2.IMREAD_GRAYSCALE)
i4 = cv2.imread(os.path.join(ROOT_DIR, 'Cars65.png'), cv2.IMREAD_GRAYSCALE)
txt(i3, 'coche02.jpeg')
txt(i4, 'Cars65.png')

# 3.2 Funciones de ajuste y medida
def contrast_stretch(img):
    mn, mx = img.min(), img.max()
    if mx == mn:
        return img.copy()
    return (((img - mn) * 255.0 / (mx - mn))).astype(np.uint8)

def contrast_measure(img):
    return float(img.std())

# 3.3 Aplicación de técnicas
stretch3 = contrast_stretch(i3)
stretch4 = contrast_stretch(i4)
eq3 = cv2.equalizeHist(i3)
eq4 = cv2.equalizeHist(i4)

# 3.4 Medición de contraste
print(f"Contraste Original 3: {contrast_measure(i3):.2f}")
print(f"Contraste Stretch 3:  {contrast_measure(stretch3):.2f}")
print(f"Contraste Equalize 3: {contrast_measure(eq3):.2f}")
print(f"Contraste Original 4: {contrast_measure(i4):.2f}")
print(f"Contraste Stretch 4:  {contrast_measure(stretch4):.2f}")
print(f"Contraste Equalize 4: {contrast_measure(eq4):.2f}")

# 3.5 Guardado de resultados
output_dir2 = os.path.join(ROOT_DIR, 'outputs', 'contraste')
os.makedirs(output_dir2, exist_ok=True)
cv2.imwrite(os.path.join(output_dir2, 'i3_original.png'), i3)
cv2.imwrite(os.path.join(output_dir2, 'i3_stretch.png'), stretch3)
cv2.imwrite(os.path.join(output_dir2, 'i3_eq.png'), eq3)
cv2.imwrite(os.path.join(output_dir2, 'i4_original.png'), i4)
cv2.imwrite(os.path.join(output_dir2, 'i4_stretch.png'), stretch4)
cv2.imwrite(os.path.join(output_dir2, 'i4_eq.png'), eq4)
print(f"Resultados guardados en {output_dir2}")
# 4. Conclusiones
# - Eliminación de anomalías: comparar el PSNR entre filtros y seleccionar el de mayor valor.
# - Ajuste de intensidad: la desviación estándar indica el mejor contraste; se elige la técnica que lo maximice.

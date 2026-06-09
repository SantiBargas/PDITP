import cv2
import numpy as np

# 1. Cargar la imagen
img = cv2.imread('group_01/00003.png')
original = img.copy()

# 2. Preprocesamiento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizado para atenuar la textura de la rejilla del fondo
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# 3. Umbralizado (Binarización)
# Como el fondo es claro y los pescados oscuros, usamos THRESH_BINARY_INV
# El método de Otsu calcula automáticamente el mejor umbral
_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. Operaciones Morfológicas para limpiar la máscara
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (19, 19))  # Tamaño del kernel ajustable según el tamaño de los pescados
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (35, 35)))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (35, 35)))

# 5. Encontrar Contornos
# RETR_EXTERNAL solo detecta los contornos exteriores (ignora huecos internos)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear imágenes para dibujar los resultados
img_bboxes = original.copy()


for i, cnt in enumerate(contours):
    # Filtrar por área para evitar pequeños ruidos que hayan quedado
    if cv2.contourArea(cnt) < 4000:
        continue
        
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img_bboxes, [box], 0, (0, 0, 255), 2)


# 6. Guardar o mostrar resultados
#cv2.imwrite('mascara_binaria.jpg', mask)
#cv2.imwrite('pescados_bounding_box.jpg', img_bboxes)

# Si estás en un entorno local con interfaz gráfica:
cv2.namedWindow('Máscara', cv2.WINDOW_NORMAL)
cv2.namedWindow('Bounding Boxes', cv2.WINDOW_NORMAL)


cv2.resizeWindow('Máscara', 800, 600)  # Tamaño inicial cómodo
cv2.resizeWindow('Bounding Boxes', 800, 600)

cv2.imshow('Máscara', mask)
cv2.imshow('Bounding Boxes', img_bboxes)



cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
from scipy.signal import find_peaks
import nibabel as nib

"""
Método:

rescaling(image): Toma una imagen como entrada y realiza el reescalado de la imagen utilizando el valor mínimo y máximo 
de los píxeles de la imagen. Devuelve la imagen reescalada.

Variables:

image: Variable de entrada que representa la imagen a reescalar.
min_value: Variable que almacena el valor mínimo de los píxeles de la imagen.
max_value: Variable que almacena el valor máximo de los píxeles de la imagen.
image_data_rescaled: Variable que almacena la imagen reescalada. Es el resultado de aplicar la fórmula
(image - min_value) / (max_value - min_value) para cada pixel de la imagen.

El propósito principal de esta función es normalizar los valores de los píxeles en una imagen, asegurando que 
se encuentren en un rango específico. Esto puede ser útil para el procesamiento y análisis de imágenes, ya que 
muchos algoritmos y técnicas requieren que los valores estén en un rango predefinido para un funcionamiento óptimo.
Al reescalar la imagen, se ajustan los valores de los píxeles para que se distribuyan en el rango de 0 a 1, 
lo que facilita el análisis y la comparación de diferentes imágenes.

"""

def rescaling(image):
  min_value = image.min()
  max_value = image.max()

  image_data_rescaled = (image - min_value) / (max_value - min_value)
  return image_data_rescaled


"""
Método:

zscore(image): Este método toma una imagen como entrada y realiza la normalización z-score en la imagen. 
Calcula la media y la desviación estándar de los píxeles de la imagen que sean mayores que 10. Luego, aplica 
la fórmula (image - mean) / standard_deviation para cada píxel de la imagen y devuelve la imagen normalizada 
z-score.

Variables:

image: Variable de entrada que representa la imagen a normalizar z-score.
mean: Variable que almacena la media de los píxeles de la imagen mayores que 10.
standard_deviation: Variable que almacena la desviación estándar de los píxeles de la imagen mayores que 10.
image_zscore: Variable que almacena la imagen normalizada z-score. Es el resultado de aplicar la fórmula 
(image - mean) / standard_deviation para cada pixel de la imagen.

El propósito principal de esta función es normalizar los valores de los píxeles en una imagen utilizando la 
técnica de z-score. La normalización z-score es útil para estandarizar los valores de los píxeles y compararlos 
en relación con la distribución de los datos en la imagen. Al calcular la media y la desviación estándar de 
los píxeles que son mayores que 10, se excluyen los valores atípicos o ruido de bajo nivel que pueden afectar 
negativamente la normalización. Luego, se aplica la fórmula del z-score a cada píxel de la imagen para obtener 
una imagen normalizada z-score. Esto puede ser útil para resaltar características relevantes en la imagen y 
reducir el impacto de los valores atípicos en el análisis y procesamiento posterior de la imagen.
"""
def zscore(image):
  mean = image[image > 10].mean()
  standard_deviation = image[image > 10].std()
  image_zscore = (image - mean)/(standard_deviation) 
  return image_zscore

"""
Método:

white_stripe(X, tipo): Este método toma una imagen médica X y un tipo de imagen tipo como entrada. Realiza un
procesamiento especial en la imagen según el tipo de imagen especificado. Primero, calcula el histograma de 
los valores de píxeles de la imagen y encuentra los picos del histograma que tienen una altura superior a 100.
Luego, obtiene los valores correspondientes a los picos en los bordes del histograma. A continuación, realiza 
un procesamiento especial en la imagen según el tipo de imagen especificado. Si el tipo de imagen es 
'FLAIR.nii.gz' o 'T1.nii.gz', divide la imagen X por el segundo valor de los picos. Si el tipo de imagen es 
'IR.nii.gz', divide la imagen X por el primer valor de los picos. Finalmente, devuelve la imagen procesada.

Variables:

X: Variable de entrada que representa la imagen médica a procesar.
tipo: Variable que especifica el tipo de imagen médica a procesar.
hist: Variable que almacena el histograma de los valores de píxeles de la imagen.
big_edges: Variable que almacena los bordes del histograma.
picos: Variable que almacena los índices de los picos del histograma.
val_picos: Variable que almacena los valores correspondientes a los picos en los bordes del histograma.
image_data_rescaled: Variable que almacena la imagen procesada después del procesamiento especial según el 
tipo de imagen.

El propósito de esta función es realizar un procesamiento especial en una imagen médica para resaltar o 
normalizar ciertas características según el tipo de imagen. El procesamiento se basa en el análisis del 
histograma de los valores de píxeles de la imagen y la identificación de los picos relevantes en el histograma. 
Luego, se realiza una operación específica en la imagen según el tipo de imagen especificado para resaltar
 o normalizar ciertas características de interés. La función devuelve la imagen procesada resultante.

"""

def white_stripe(X, tipo):
  hist, big_edges = np.histogram(X.flatten(), bins= 'auto')
  picos, _ = find_peaks(hist, height= 100)
  val_picos = big_edges[picos]

  if tipo == 'FLAIR.nii.gz' or tipo == 'T1.nii.gz':
    image_data_rescaled = X/ val_picos[1]

  if tipo == 'IR.nii.gz':
    image_data_rescaled = X/val_picos[0]

  return image_data_rescaled

"""
Este método realiza el ajuste de histograma entre una imagen de datos y una imagen de referencia.

Método:

histogram_matching(image_data, ks, ref): Este método toma una imagen de datos image_data, un número 
de pasos ks y una imagen de referencia ref como entrada. Realiza el ajuste de histograma entre la imagen
de datos y la imagen de referencia. Primero, carga la imagen de referencia correspondiente según el tipo 
de imagen especificado por ref. Luego, define los rangos de los percentiles y los valores correspondientes 
a esos percentiles tanto para la imagen de datos como para la imagen de referencia. A continuación, interpola 
los valores de la imagen de datos según los percentiles de la imagen de datos y los percentiles de la imagen 
de referencia para realizar el ajuste de histograma. Finalmente, devuelve la imagen de datos con el histograma 
ajustado.

Variables:

image_data: Variable de entrada que representa la imagen de datos a la que se le aplicará el ajuste de histograma.
ks: Variable que representa el número de pasos para definir los percentiles en el ajuste de histograma.
ref: Variable que especifica el tipo de imagen de referencia a utilizar para el ajuste de histograma.
data_target: Variable que almacena los datos de la imagen de referencia correspondiente según el tipo de imagen especificado por ref.
ini: Variable que representa el valor inicial para definir los percentiles.
fin: Variable que representa el valor final para definir los percentiles.
step: Variable que representa el paso entre los valores de los percentiles.
percentiles_data: Variable que almacena los valores de los percentiles para la imagen de datos.
percentiles_target: Variable que almacena los valores de los percentiles para la imagen de referencia.
p1: Variable que almacena los percentiles calculados para la imagen de datos.
p2: Variable que almacena los percentiles calculados para la imagen de referencia.

El propósito de esta función es realizar el ajuste de histograma entre una imagen de datos y una imagen de
referencia. El ajuste de histograma se basa en la correspondencia de los percentiles entre las dos imágenes.
Se calculan los percentiles para la imagen de datos y la imagen de referencia, y luego se realiza la 
interpolación de los valores de la imagen de datos según los percentiles correspondientes en la imagen de 
referencia. Esto permite que la distribución de valores en la imagen de datos se ajuste a la distribución de
valores en la imagen de referencia. La función devuelve la imagen de datos con el histograma ajustado.
"""

def histogram_matching(image_data, ks, ref):

  if ref == 'T1.nii.gz':
    data_target = nib.load('ref_images/T1_copia.nii.gz').get_fdata()
  if ref == 'IR.nii.gz':
    data_target = nib.load('ref_images/IR_copia.nii.gz').get_fdata()
  if ref == 'FLAIR.nii.gz':
    data_target = nib.load('ref_images/FLAIR_copia.nii.gz').get_fdata()   

  ini=0
  fin = 100
  step = (fin - ini)/(ks-1)

  percentiles_data = np.arange(ini, fin+step, step)
  percentiles_target = np.arange(ini, fin+step, step)

  p1 = np.percentile(image_data, percentiles_data)
  p2 = np.percentile(data_target, percentiles_target)

  
  return np.interp(image_data, p1, p2)

"""
Descripción: Esta función realiza un filtrado promedio en una imagen tridimensional. El filtrado promedio se 
realiza tomando el promedio de los valores de los píxeles vecinos en cada posición de la imagen.

Variables:

image: La imagen tridimensional a la que se le aplicará el filtrado promedio.
depth, height, width: Las dimensiones de la imagen tridimensional.
filtered_image: Una matriz de ceros del mismo tamaño que la imagen de entrada, que se utilizará para almacenar la imagen filtrada.
z, y, x: Variables de bucle que representan las coordenadas de los píxeles en la imagen.

Descripción del proceso:

La función itera sobre cada píxel de la imagen, excluyendo los bordes, ya que se requieren vecinos en todas
las direcciones. Para cada píxel, se obtienen los valores de los píxeles vecinos en todas las direcciones: 
arriba, abajo, izquierda, derecha, adelante, atrás y el propio píxel.

Luego, se calcula el promedio de los valores de los píxeles vecinos utilizando la función np.mean.
Finalmente, el valor promedio calculado se asigna al píxel correspondiente en la imagen filtrada.
En resumen, esta función aplica un filtrado promedio tridimensional a una imagen, suavizando los valores de 
los píxeles al calcular el promedio de los valores de los píxeles vecinos. Esto puede ayudar a reducir el 
ruido y las variaciones locales en la imagen.

"""
def mean_filter_3d(image):
    depth, height, width = image.shape
    filtered_image = np.zeros_like(image)
    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = [
                    image[z-1, y, x],
                    image[z+1, y, x],
                    image[z, y-1, x],
                    image[z, y+1, x],
                    image[z, y, x-1],
                    image[z, y, x+1],
                    image[z, y, x]
                ]
                filtered_value = np.mean(neighbors)
                filtered_image[z, y, x] = filtered_value

    return filtered_image

"""
Descripción: Esta función realiza un filtrado mediano en una imagen tridimensional. El filtrado mediano se 
realiza tomando el valor mediano de los píxeles vecinos en cada posición de la imagen.

Parámetros:

image: La imagen tridimensional a la que se le aplicará el filtrado mediano.
Variables:

depth, height, width: Las dimensiones de la imagen tridimensional.
filtered_image: Una matriz de ceros del mismo tamaño que la imagen de entrada, que se utilizará para 
almacenar la imagen filtrada.
z, y, x: Variables de bucle que representan las coordenadas de los píxeles en la imagen.
Descripción del proceso:

La función itera sobre cada píxel de la imagen, excluyendo los bordes, ya que se requieren vecinos en todas 
las direcciones. Para cada píxel, se obtienen los valores de los píxeles vecinos en todas las direcciones: 
arriba, abajo, izquierda, derecha, adelante, atrás y el propio píxel.

Luego, se calcula el valor mediano de los valores de los píxeles vecinos utilizando la función np.median.
Finalmente, el valor mediano calculado se asigna al píxel correspondiente en la imagen filtrada.
En resumen, esta función aplica un filtrado mediano tridimensional a una imagen, suavizando los valores de
los píxeles al tomar el valor mediano de los píxeles vecinos. El filtrado mediano es útil para eliminar ruido 
impulsivo y preservar mejor los detalles en comparación con el filtrado promedio.
"""
def median_filter_3d(image):
    depth, height, width = image.shape
    filtered_image = np.zeros_like(image)
    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = [
                    image[z-1, y, x],
                    image[z+1, y, x],
                    image[z, y-1, x],
                    image[z, y+1, x],
                    image[z, y, x-1],
                    image[z, y, x+1],
                    image[z, y, x]
                ]
                filtered_value = np.median(neighbors)
                filtered_image[z, y, x] = filtered_value

    return filtered_image

"""
Esta función calcula la derivada de una imagen tridimensional utilizando la diferencia de intensidad entre 
los píxeles vecinos en las tres direcciones (x, y, z). La derivada se calcula utilizando el método de 
diferencia de intensidad al cuadrado.

Parámetros:

image: La imagen tridimensional a la que se le calculará la derivada.
Variables:

df: Una matriz de ceros del mismo tamaño que la imagen de entrada, que se utilizará para almacenar los valores
de la derivada.
x, y, z: Variables de bucle que representan las coordenadas de los píxeles en la imagen.
Descripción del proceso:

La función itera sobre cada píxel de la imagen, excluyendo los bordes, ya que se requieren vecinos en todas 
las direcciones. Para cada píxel, se calculan las diferencias de intensidad al cuadrado entre los píxeles 
vecinos en las direcciones x, y, z.
Luego, se calcula la magnitud de la derivada utilizando la fórmula de la raíz cuadrada de la suma de las 
diferencias de intensidad al cuadrado en las tres direcciones.
Finalmente, el valor de la derivada calculada se asigna al píxel correspondiente en la matriz df.
En resumen, esta función calcula la derivada tridimensional de una imagen utilizando la diferencia de intensidad
entre los píxeles vecinos en las tres direcciones. Proporciona una medida de la variación de intensidad en 
cada punto de la imagen y puede ser útil en aplicaciones de análisis de características o detección de bordes 
en imágenes médicas.
"""
def derivada(image):
  df = np.zeros_like(image)
  for x in range(1, image.shape[0] - 2):
    for y in range(1, image.shape[1] - 2):
      for z in range(1, image.shape[2] - 2):
        dfdx = np.power( image[x+1, y, z]- image[x-1,y,z] , 2)
        dfdy = np.power( image[x, y+1, z]- image[x,y-1,z] , 2)
        dfdz = np.power( image[x, y, z+1]- image[x,y,z-1] , 2)
        df[x,y,z] =np.sqrt(dfdx + dfdy + dfdz)
  return df

"""
Esta función aplica un filtro de media a una imagen tridimensional, considerando únicamente los píxeles que 
cumplen una condición de derivada por debajo de un umbral dado. Los píxeles que cumplen la condición son 
reemplazados por el valor medio de su vecindario, mientras que los píxeles que no cumplen la condición se 
mantienen sin cambios.

Parámetros:

image_data: La imagen tridimensional a la que se le aplicará el filtro de media con bordes.
tol: El umbral de tolerancia para la derivada. Los píxeles cuya derivada absoluta esté por debajo de este 
umbral se considerarán para el filtrado.

Variables:

df: La derivada de la imagen tridimensional, calculada utilizando la función derivada.
depth, height, width: Las dimensiones de la imagen tridimensional.
filtered_image: Una matriz de ceros del mismo tamaño que la imagen de entrada, que se utilizará para 
almacenar los valores filtrados.

Descripción del proceso:

La función calcula la derivada de la imagen de entrada utilizando la función derivada y almacena los resultados
en la matriz df. A continuación, se itera sobre cada píxel de la imagen, excluyendo los bordes, ya que se 
requieren vecinos en todas las direcciones.

Para cada píxel, se verifica si la derivada absoluta está por debajo del umbral especificado (tol).
Si la derivada cumple la condición, se extrae el vecindario tridimensional del píxel y se calcula el valor 
medio de los píxeles en el vecindario.

El valor medio calculado se asigna al píxel correspondiente en la matriz filtered_image.
Si la derivada no cumple la condición, se mantiene el valor original del píxel en filtered_image.

Al final del proceso, se devuelve la imagen filtered_image con el filtro de media aplicado a los píxeles que
cumplen la condición de derivada.

"""
def meanwithBorder(image_data, tol=10):
  df = derivada(image_data)
  depth, height, width = image_data.shape
  filtered_image = np.zeros_like(image_data)
  for x in range(1, depth - 1):
      for y in range(1, height - 1):
          for z in range(1, width - 1):
              if np.abs(df[x,y,z]) < tol:
                neighborhood = image_data[x-1:x+2 , y- 1: y+2 , z-1: z+2]
                filtered_value = np.mean(neighborhood)
                filtered_image[x, y, z] = filtered_value

              else:
                filtered_image[x, y, z] = image_data[x, y, z]

  return filtered_image
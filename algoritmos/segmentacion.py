import numpy as np

"""
Esta función implementa el algoritmo de umbralización IsoData para segmentar una imagen en dos clases, 
utilizando un umbral inicial (tau) y un criterio de convergencia (tol).

Parámetros:

image_data: La imagen a segmentar.
tau: El umbral inicial para la segmentación. Debe ser un valor entre el mínimo y el máximo de la imagen.
tol: El criterio de convergencia. Si la diferencia absoluta entre el umbral actual y el umbral posterior es menor que tol, se considera que el algoritmo ha convergido y se detiene la iteración.
Variables:

segmentation: Una matriz binaria que representa la segmentación de la imagen, donde los píxeles con intensidad
mayor o igual al umbral (tau) se etiquetan como 1 y los píxeles con intensidad menor se etiquetan como 0.

mBG: La media de los valores de intensidad en la región de fondo (píxeles etiquetados como 0) de la imagen.
mFG: La media de los valores de intensidad en la región de objeto (píxeles etiquetados como 1) de la imagen.
tau_post: El nuevo umbral calculado como el promedio entre las medias de fondo y objeto.

Descripción del proceso:

Se inicia un bucle while que se ejecuta hasta que se cumpla el criterio de convergencia.
Dentro del bucle, se realiza la umbralización binaria de la imagen utilizando el umbral actual (tau).
Se calculan las medias de los valores de intensidad en las regiones de fondo (mBG) y objeto (mFG).
Se calcula un nuevo umbral (tau_post) como el promedio entre las medias de fondo y objeto.
Se verifica si la diferencia absoluta entre el umbral actual y el umbral posterior es menor que el 
criterio de convergencia (tol).
Si la diferencia es menor que tol, se rompe el bucle y se considera que el algoritmo ha convergido.
De lo contrario, se actualiza el valor de tau con tau_post y se continúa con la siguiente iteración del bucle.
Al final del proceso, se devuelve la matriz binaria de segmentación.
"""
def isoData(image_data, tau, tol):
  #El tau es un valor entre el minimo y el maximo de la imagen
  
  while True:
    #umbralizar
    segmentation = image_data >= tau

    #calcular un umbral mas optimo
    ##excluyendo los valores del fondo que son muy pequeños
    mBG = image_data[segmentation == 0 ].mean()
    mFG = image_data[segmentation == 1 ].mean()

    tau_post = 0.5 *(mBG + mFG)

    if np.abs(tau-tau_post)<tol:
      break
    else:
      tau = tau_post

  return segmentation

"""
Esta función implementa el algoritmo de crecimiento de regiones para segmentar una región de interés en una
imagen tridimensional. El algoritmo se inicia en una posición de semilla (x, y, z) y se expande iterativamente
añadiendo píxeles vecinos que cumplen cierta condición de similitud.

Parámetros:

image: La imagen tridimensional en la que se realizará la segmentación.
x, y, z: Las coordenadas de la posición de semilla para iniciar el crecimiento de regiones.
tol: La tolerancia o umbral de similitud para determinar si un píxel vecino debe ser agregado a la región.
Variables:

segmentation: Una matriz binaria que representa la región segmentada, donde los píxeles pertenecientes a la 
región se etiquetan como 1 y los píxeles no pertenecientes se etiquetan como 0.
valor_medio_cluster: El valor medio del píxel de semilla utilizado como referencia para la similitud con los
píxeles vecinos.
vecinos: Una lista de coordenadas de píxeles vecinos que se deben explorar para expandir la región.

Descripción del proceso:
Se inicializa la matriz de segmentación como una matriz de ceros del mismo tamaño que la imagen de entrada.
Se verifica si el píxel de semilla ya ha sido etiquetado como parte de la región. Si es así, se retorna la 
segmentación actual y se finaliza la función.
Se asigna el valor medio del píxel de semilla como referencia para la similitud con los píxeles vecinos.
Se etiqueta el píxel de semilla como parte de la región en la matriz de segmentación.
Se inicializa la lista de vecinos con la coordenada del píxel de semilla.
Se inicia un bucle while que se ejecuta mientras la lista de vecinos no esté vacía.
Dentro del bucle, se extrae la última coordenada de vecinos y se asigna a las variables x, y, z.
Se realiza un bucle anidado para explorar los vecinos de la coordenada actual.
Se calcula la coordenada del vecino (nx, ny, nz) sumando las diferencias en las direcciones x, y, z.
Se verifica si el vecino está dentro de los límites de la imagen.
Se verifica si el valor absoluto de la diferencia entre el valor medio del píxel de semilla y el valor del 
vecino es menor que la tolerancia (tol).
Si se cumple la condición de similitud y el vecino no ha sido etiquetado previamente como parte de la región, 
se etiqueta el vecino como parte de la región en la matriz de segmentación y se agrega su coordenada a la 
lista de vecinos.
El bucle while continúa hasta que no queden más vecinos por explorar.
Al final del proceso, se devuelve la matriz de segmentación.
"""
def region_growing(image, x, y, z, tol):
    segmentation = np.zeros_like(image)
    if segmentation[x,y,z] == 1:
        return
    valor_medio_cluster = image[x,y,z]
    segmentation[x,y,z] = 1
    vecinos = [(x, y, z)]
    while vecinos:
        x, y, z = vecinos.pop()
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    #vecino
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx >= 0 and nx < image.shape[0] and \
                        ny >= 0 and ny < image.shape[1] and \
                        nz >= 0 and nz < image.shape[2]:
                        if np.abs(valor_medio_cluster - image[nx,ny,nz]) < tol and \
                            segmentation[nx,ny,nz] == 0:
                            segmentation[nx,ny,nz] = 1
                            vecinos.append((nx, ny, nz))
    return segmentation

"""
Esta función implementa el algoritmo de clustering mediante k-means para segmentar una imagen en ks clases 
o grupos. El algoritmo asigna cada píxel de la imagen a la clase más cercana en términos de similitud de 
valores de intensidad y actualiza iterativamente los centroides de las clases hasta que se alcanza la 
convergencia.

Parámetros:

image: La imagen que se desea segmentar.
ks: El número de clases o grupos en los que se desea segmentar la imagen.
Variables:

k_values: Un arreglo de longitud ks que representa los centroides iniciales de cada clase. Estos centroides 
se distribuyen uniformemente en el rango de valores de intensidad presentes en la imagen.
old_segmentation: La segmentación anterior, que se utiliza para verificar la convergencia del algoritmo.
d_values: Una lista de arreglos que almacena las diferencias absolutas entre los valores de intensidad de 
cada píxel y los centroides.
segmentation: Una matriz que almacena la asignación de cada píxel a una clase según su similitud de intensidad.

Descripción del proceso:
Se genera un arreglo k_values que contiene ks valores equidistantes en el rango de valores de intensidad 
presentes en la imagen.
Se inicializa old_segmentation como None.
Se inicia un bucle while que se repite hasta que se alcance la convergencia del algoritmo.
Dentro del bucle, se calcula d_values, que es una lista de arreglos que almacenan las diferencias absolutas
entre los valores de intensidad de cada píxel y los centroides en k_values.
Se asigna a segmentation la clase correspondiente a la mínima diferencia absoluta en cada píxel.
Para cada clase en k_values, se actualiza su valor en k_values calculando la media de los valores de 
intensidad de los píxeles asignados a esa clase en segmentation.
Se verifica si old_segmentation no es None y si la segmentación actual segmentation es igual a la segmentación
anterior old_segmentation. Si son iguales, se rompe el bucle y se finaliza el algoritmo.
Se asigna old_segmentation como la segmentación actual.
Al final del proceso, se devuelve la matriz de segmentación.
"""
def clustering(image, ks):

  k_values = np.linspace(np.amin(image), np.amax(image), ks)

  old_segmentation = None

  while True:
    d_values = [np.abs(k-image)for k in k_values]

    segmentation = np.argmin(d_values, axis=0)

    for k_id in range(ks):
      k_values[k_id] = np.mean(image[segmentation==k_id])

    if old_segmentation is not None and np.array_equal(segmentation, old_segmentation):
      break

    old_segmentation = segmentation

  return segmentation

"""
Esta función implementa el algoritmo de mezcla de modelos gaussianos (Gaussian Mixture Model - GMM) para 
realizar la segmentación de una imagen en ks clases o grupos. El algoritmo asume que los datos de intensidad 
de la imagen se distribuyen de acuerdo a una combinación de modelos gaussianos y estima los parámetros de 
estos modelos para asignar cada píxel a la clase más probable.

Parámetros:

image: La imagen que se desea segmentar.
ks: El número de clases o grupos en los que se desea segmentar la imagen.
Variables:

mu_values: Un arreglo de longitud ks que representa los valores iniciales de media de cada modelo gaussiano.
Estos valores se distribuyen uniformemente en el rango de valores de intensidad presentes en la imagen.

sd_values: Un arreglo de longitud ks que representa los valores iniciales de desviación estándar de cada 
modelo gaussiano. Todos los valores se inicializan en 1.

old_segmentation: La segmentación anterior, que se utiliza para verificar la convergencia del algoritmo.

d_values: Una lista de arreglos que almacena las probabilidades de pertenencia de cada píxel a cada clase 
según los modelos gaussianos.

segmentation: Una matriz que almacena la asignación de cada píxel a una clase según la probabilidad más alta.

Descripción del proceso:
Se genera un arreglo mu_values que contiene ks valores equidistantes en el rango de valores de intensidad 
presentes en la imagen.
Se crea un arreglo sd_values de longitud ks con todos los elementos inicializados en 1.
Se inicializa old_segmentation como None.
Se inicia un bucle while que se repite hasta que se alcance la convergencia del algoritmo.
Dentro del bucle, se calcula d_values, que es una lista de arreglos que almacenan las probabilidades de 
pertenencia de cada píxel a cada clase según los modelos gaussianos estimados con los parámetros actuales.
Se asigna a segmentation la clase correspondiente a la probabilidad más alta en cada píxel.
Para cada clase en mu_values, se actualiza su valor de media en mu_values calculando la media de los valores
de intensidad de los píxeles asignados a esa clase en segmentation.
Se verifica si old_segmentation no es None y si la segmentación actual segmentation es igual a la segmentación
anterior old_segmentation. Si son iguales y se han obtenido ks clases distintas, se rompe el bucle y se 
finaliza el algoritmo.
Se asigna old_segmentation como la segmentación actual.
Al final del proceso, se devuelve la matriz de segmentación.
"""

def gmm(image, ks):

  mu_values = np.linspace(np.amin(image), np.amax(image), ks)
  sd_values = np.ones_like(mu_values)

  old_segmentation = None

  while True:
    d_values = [np.exp(-0.5 * np.divide((image - mu_values[i])**2, np.maximum(sd_values[i]**2, 1e-10))) for i in range(ks)]

    segmentation = np.argmax(d_values, axis=0)
    for i in range(ks):
      mu_values[i] = np.mean(image[segmentation==i])
      #sd_values[i] = np.std(image[segmentation==i])

    if old_segmentation is not None and np.array_equal(segmentation, old_segmentation):
      if len(np.unique(segmentation)) == ks:
                break

    old_segmentation = segmentation

  return segmentation

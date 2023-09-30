import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import io
from algoritmos.estandarizacion import rescaling, zscore, white_stripe,histogram_matching, mean_filter_3d, median_filter_3d, meanwithBorder


# Configura el título de la página y el ícono que se mostrará en la pestaña del navegador

st.set_page_config(page_title="Estandarización & eliminación de ruido", page_icon="./icono/pagina.png")

#Muestra un encabezado en formato de markdown con el texto "Estandarización & eliminación de ruido

st.markdown("# Estandarización & eliminación de ruido")


#Muestra un encabezado en la barra lateral (sidebar) con el texto "Preprocesamiento".

st.sidebar.header("Preprocesamiento")


#Muestra un texto explicativo que indica al usuario que seleccione una imagen.

st.write(
    """Selecciona una imagen para comenzar a procesarla"""
)


nombreImagenes = []

folder_path = "uploaded_images"  # Ruta de la carpeta a listar

# Verificar si la carpeta existe
if os.path.exists(folder_path):

    # Enumerar los archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Iterar sobre cada archivo
    for filename in files:

        # Mostrar el nombre del archivo
        nombreImagenes.append(filename)
else:
    st.error(f"La carpeta '{folder_path}' no existe") #Si la carpeta no existe


#CAMBIOO
folder_path_seg = "seg_images" 
if st.sidebar.button("Borrar segmentaciones anteriores"):
    # Verifica si la carpeta existe
    if os.path.exists(folder_path_seg):
        # Enumera los archivos en la carpeta
        files = os.listdir(folder_path_seg)

        # Itera sobre cada archivo y lo borra
        for filename in files:
            file_path = os.path.join(folder_path_seg, filename)
            os.remove(file_path)
        
        st.success("Archivos borrados correctamente")
        st.experimental_rerun()
    else:
        st.error(f"La carpeta '{folder_path_seg}' no existe")




#Muestra un cuadro de selección en la barra lateral (sidebar) donde el usuario puede elegir una opción de
#una lista de imágenes disponibles. La opción seleccionada se guarda en la variable imagen.

imagen = st.sidebar.selectbox('Selecciona una opción', nombreImagenes, index=nombreImagenes.index('T1.nii.gz'))

#Crea una lista de opciones de algoritmos de estandarización disponibles.

opciones_algoritmos = ['Ninguno','Rescaling', 'Z-Score', 'White Stripe','Histogram Matching']


#Muestra una lista de opciones de algoritmos de estandarización como botones de radio en la barra lateral.
#El usuario puede seleccionar un algoritmo de estandarización y la opción seleccionada se guarda en la variable 
#algoritmo.

algoritmo = st.sidebar.radio('Selecciona un algortimo de estandarización', opciones_algoritmos)

#Crea una lista de opciones de algoritmos de eliminación de ruido disponibles.

opciones_ruido = ['Ninguno','Mean Filter', 'Median Filter', 'Median Filter with edges']

#Muestra una lista de opciones de algoritmos de eliminación de ruido como botones de radio en la barra lateral.
#El usuario puede seleccionar un algoritmo de eliminación de ruido y la opción seleccionada se guarda en la variable ruido.

ruido = st.sidebar.radio('(Opcional) Selecciona un algortimo de eliminación de ruido', opciones_ruido)

# Dividir el espacio en dos columnas
col1, col2 = st.columns(2)

#Se construye la ruta completa de la imagen seleccionada concatenando la carpeta folder_path con el nombre de la imagen imagen.
path = folder_path+"/"+imagen

#Se carga la imagen seleccionada utilizando la biblioteca nibabel. La función nib.load() carga la imagen desde 
#la ruta especificada y get_fdata() devuelve los datos de la imagen en forma de matriz numpy, que se asigna a 
#la variable image_data.

image_load = nib.load(folder_path+"/"+imagen)
image_data = image_load.get_fdata()  


#Se verifica si la variable 'imagen_preprocesada' no existe en el estado de la sesión de streamlit. Si no 
#existe, se asigna el valor de image_data a la variable 'imagen_preprocesada' en el estado de la sesión. 
#Esto se utiliza para almacenar la imagen preprocesada en el estado de la sesión y mantenerla disponible 
#durante la sesión de la aplicación.

if 'imagen_preprocesada' not in st.session_state:
    st.session_state.imagen_preprocesada = image_data


#Se verifica si la variable 'name_imagen_preprocesada' no existe en el estado de la sesión de streamlit. 
#Si no existe, se asigna el valor de imagen (nombre de la imagen seleccionada) a la variable 'name_imagen_preprocesada'
#en el estado de la sesión. Esto se utiliza para almacenar el nombre de la imagen preprocesada en el estado 
#de la sesión y mantenerlo disponible durante la sesión de la aplicación.

if 'name_imagen_preprocesada' not in st.session_state:
    st.session_state.name_imagen_preprocesada = imagen
    
if 'imagen_datos' not in st.session_state:
    st.session_state.imagen_datos = image_load


#####cargar histograma 


 #Se verifica si el valor de algoritmo es diferente de 'Ninguno', lo que significa que se ha seleccionado un algoritmo de estandarización.

if algoritmo != 'Ninguno':
   
    #Se inicia un bloque de código que se ejecutará en una columna (col1) dentro del diseño de la interfaz de la aplicación.
 
    with col1:
        
        #Se crea un arreglo hist1_data que contiene los valores de píxeles de la imagen original (image_data) que son mayores que 10. Los valores se aplanan en un arreglo unidimensional.
        
        hist1_data = image_data[image_data > 10].flatten()
        # Se crea una figura (fig1) y un objeto de ejes (ax1) utilizando la biblioteca matplotlib. Esto se utiliza para generar el gráfico del histograma.
        fig1, ax1 = plt.subplots()
        #Se traza el histograma de los datos hist1_data utilizando el objeto de ejes ax1. Se especifica el número de bins como 100.
        ax1.hist(hist1_data, bins=100)
        #Se muestra un título descriptivo en la interfaz de la aplicación utilizando la función st.write().
        st.write("Histograma sin estandarización")
        # muestra el gráfico del histograma en la interfaz de la aplicación utilizando la función st.pyplot(). El gráfico es generado por matplotlib y se pasa como argumento la figura fig1 creada anteriormente.
        st.pyplot(fig1)        


    with col2:
        #Se verifica si el valor de algoritmo es igual a 'Rescaling', lo que significa que se ha seleccionado el algoritmo de estandarización de rescaling.
        if algoritmo == 'Rescaling':

            #Se aplica el algoritmo de rescaling a la imagen original (image_data) utilizando la función rescaling(). El resultado se guarda en la variable image_data_rescaled.
            image_data_rescaled = rescaling(image_data)

            #Se crea un arreglo hist2_data que contiene los valores de píxeles de la imagen estandarizada (image_data_rescaled) que son mayores que 0.01. Los valores se aplanan en un arreglo unidimensional.
            hist2_data = image_data_rescaled[image_data_rescaled > 0.01].flatten()

            #e crea una figura (fig2) y un objeto de ejes (ax2) utilizando la biblioteca matplotlib. Esto se utiliza para generar el gráfico del histograma de la imagen estandarizada.
            fig2, ax2 = plt.subplots()

            #e traza el histograma de los datos hist2_data utilizando el objeto de ejes ax2. Se especifica el número de bins como 100.
            ax2.hist(hist2_data, bins=100)

        
        #Se verifica el valor de algoritmo para determinar qué tipo de algoritmo de estandarización se seleccionó.
        #Si algoritmo es igual a 'Z-Score', se aplica el algoritmo de Z-Score a la imagen original (image_data) utilizando la función zscore(). El resultado se guarda en la variable image_data_rescaled. Luego se crea y muestra el histograma de los datos estandarizados.
        #Si algoritmo es igual a 'White Stripe', se aplica el algoritmo de White Stripe a la imagen original utilizando la función white_stripe(). El resultado se guarda en la variable image_data_rescaled. Luego se crea y muestra el histograma de los datos estandarizados.
        #Si algoritmo es igual a 'Histogram Matching', se solicita al usuario que ingrese el número de percentiles (ks) y se aplica el algoritmo de Histogram Matching a la imagen original utilizando la función histogram_matching(). El resultado se guarda en la variable image_data_rescaled. Luego se crea y muestra el histograma de los datos estandarizados.
        
        #En cada caso, se crea un arreglo hist2_data que contiene los valores de píxeles de la imagen estandarizada que cumplen ciertas condiciones. Los valores se aplanan en un arreglo unidimensional.
        #Se crea una figura (fig2) y un objeto de ejes (ax2) utilizando la biblioteca matplotlib. Esto se utiliza para generar el gráfico del histograma de la imagen estandarizada.
        #Se traza el histograma de los datos hist2_data utilizando el objeto de ejes ax2. Se especifica el número de bins como 100.
        #Se muestra el mensaje "Histograma con estandarización" en la interfaz de la aplicación.
        #Se actualiza la variable st.session_state.imagen_preprocesada con la imagen estandarizada (image_data_rescaled) y la variable st.session_state.name_imagen_preprocesada con el nombre de la imagen. Esto se utiliza para almacenar la imagen estandarizada en el estado de la sesión de la aplicación.
        
            
        if algoritmo == 'Z-Score':
            image_data_rescaled = zscore(image_data)
            hist2_data = image_data_rescaled.flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        if algoritmo == 'White Stripe':
            image_data_rescaled = white_stripe(image_data, imagen)
            hist2_data = image_data_rescaled[image_data_rescaled>0.5 ].flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        if algoritmo == 'Histogram Matching':
            ks = st.number_input("Percentiles:", value=3)
            image_data_rescaled = histogram_matching(image_data, ks, imagen)
            hist2_data = image_data_rescaled[image_data_rescaled>10].flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        st.write("Histograma con estandarización")
        st.pyplot(fig2)
        
    st.session_state.imagen_preprocesada =  image_data_rescaled
    st.session_state.name_imagen_preprocesada = imagen
    st.session_state.imagen_datos = image_load


#Se verifica el valor de ruido para determinar qué tipo de algoritmo de eliminación de ruido se seleccionó.
#Si ruido no es igual a 'Ninguno', se crea una figura (fig) y un objeto de ejes (ax) utilizando la biblioteca matplotlib. El tamaño de la figura se establece en 50x50.
#Se verifican diferentes condiciones dependiendo del valor de imagen para seleccionar la orientación del corte y configurar los valores específicos y máximos para los sliders.

#Se muestra un slider al usuario para seleccionar un valor específico dentro del rango permitido. El valor seleccionado se guarda en la variable valor_seleccionado.
#Se muestra la imagen correspondiente al corte seleccionado utilizando el objeto de ejes ax. La imagen se obtiene de la matriz image_data_rescaled y se aplica un mapa de colores ('bone') para mejorar la visualización.

#Se crea un búfer (buffer) para guardar la figura como un archivo de imagen en formato PNG utilizando plt.savefig().
#Se busca la posición inicial del búfer utilizando buffer.seek(0).
#Se muestra la imagen guardada en el búfer utilizando st.image(). La imagen se muestra junto con el nombre de la imagen original como leyenda y se ajusta al ancho de la columna.


if ruido != 'Ninguno':
    with col1:
 
        fig, ax = plt.subplots(figsize=(50, 50))

        if(imagen == 'T1.nii.gz'):
            opciones = [ 'Axial', 'Sagital', 'Coronal']
            corte = st.selectbox('1.Selecciona un corte', opciones)

            if corte == 'Axial':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[1]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(image_data_rescaled[:, valor_seleccionado, :], cmap='bone')

            if corte == 'Sagital':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[2]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(np.rot90(image_data_rescaled[:, :, valor_seleccionado],k=-1), cmap='bone')

            if corte == 'Coronal':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[0]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(image_data_rescaled[valor_seleccionado, :, :], cmap='bone')


        if(imagen == 'IR.nii.gz' or imagen == 'FLAIR.nii.gz'):
            valor_especifico = 25
            valor_maximo = image_data_rescaled.shape[2]
            ###imagen
            valor_seleccionado = st.slider("Imagen sin eliminación de ruido", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(image_data_rescaled[:, :, valor_seleccionado], k=-1), cmap='bone')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Mostrar la imagen utilizando st.image
        st.image(buffer, caption = imagen, use_column_width=True)


    #Se crea una figura (fig) y un objeto de ejes (ax) utilizando la biblioteca matplotlib. El tamaño de la figura se establece en 50x50.
    #Se verifican diferentes condiciones dependiendo del valor de ruido para seleccionar el tipo de filtro de eliminación de ruido y aplicarlo a la imagen previamente procesada (image_data_rescaled). El resultado se guarda en la variable filtered_image.
    #Se verifican diferentes condiciones dependiendo del valor de imagen para seleccionar la orientación del corte y configurar los valores específicos y máximos para los sliders.
    #Se muestra un slider al usuario para seleccionar un valor específico dentro del rango permitido. El valor seleccionado se guarda en la variable valor_seleccionado.
    #Se muestra la imagen correspondiente al corte seleccionado después de aplicar el filtro de eliminación de ruido, utilizando el objeto de ejes ax. La imagen se obtiene de la matriz filtered_image y se aplica un mapa de colores ('bone') para mejorar la visualización.
    
    #Se crea un búfer (buffer) para guardar la figura como un archivo de imagen en formato PNG utilizando plt.savefig().
    #Se busca la posición inicial del búfer utilizando buffer.seek(0).
    #Se muestra la imagen guardada en el búfer utilizando st.image(). La imagen se muestra junto con el nombre de la imagen original como leyenda y se ajusta al ancho de la columna.
    
    #Se actualiza la variable de estado imagen_preprocesada en st.session_state con la imagen filtrada (filtered_image).
    #Se actualiza la variable de estado name_imagen_preprocesada en st.session_state con el nombre de la imagen original (imagen).
 

    with col2:
  
        fig, ax = plt.subplots(figsize=(50, 50))

        if ruido == 'Mean Filter':
            filtered_image = mean_filter_3d(image_data_rescaled)
        
        if ruido == 'Median Filter':
            filtered_image = median_filter_3d(image_data_rescaled)

        if ruido == 'Median Filter with edges':
            filtered_image = meanwithBorder(image_data_rescaled)

        if(imagen == 'T1.nii.gz'):
            opciones = ['Axial', 'Sagital', 'Coronal']
            corte = st.selectbox('2.Selecciona un corte', opciones)

            if corte == 'Axial':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[1]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(filtered_image[:, valor_seleccionado, :], cmap='bone')

            if corte == 'Sagital':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[2]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(np.rot90(filtered_image[:, :, valor_seleccionado],k=-1), cmap='bone')

            if corte == 'Coronal':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[0]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(filtered_image[valor_seleccionado, :, :], cmap='bone')
        if(imagen == 'IR.nii.gz' or imagen == 'FLAIR.nii.gz'):
            valor_especifico = 25
            valor_maximo = filtered_image.shape[2]
            ###imagen
            valor_seleccionado = st.slider("Imagen con eliminación de ruido", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(filtered_image[:, :, valor_seleccionado], k=-1), cmap='bone')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Mostrar la imagen utilizando st.image
        st.image(buffer, caption = imagen, use_column_width=True)
        

    st.session_state.imagen_preprocesada =  filtered_image
    st.session_state.name_imagen_preprocesada = imagen
    st.session_state.imagen_datos = image_load
    
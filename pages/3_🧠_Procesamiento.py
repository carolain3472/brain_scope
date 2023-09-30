import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import io 

from algoritmos.segmentacion import isoData, region_growing, clustering, gmm


st.set_page_config(page_title="Segmentación", page_icon="./icono/pagina.png")

image_data = st.session_state.imagen_preprocesada
name_imagen = st.session_state.name_imagen_preprocesada
image_load = st.session_state.imagen_datos

seg_path = "seg_images/"


st.markdown("# Segmentación")
st.sidebar.header("Procesamiento")
st.write(
    """Selecciona un algoritmo para segmentar la imagen preprocesada"""
)

opciones_algoritmos = ['Ninguno','isoData', 'Region Growing', 'Clustering','Gaussian Mixture Model']
algoritmo = st.sidebar.selectbox('Selecciona un algortimo de segmentación', opciones_algoritmos)


#Se crea una figura (fig) y un objeto de ejes (ax) utilizando la biblioteca matplotlib. El tamaño de la figura se establece en 50x50.
#Se verifican diferentes condiciones dependiendo del valor de algoritmo para seleccionar el algoritmo de segmentación correspondiente y aplicarlo a la imagen original (image_data). El resultado se guarda en la variable segmentacion.
#Para cada algoritmo de segmentación, se muestra un control deslizante (st.sidebar.number_input) en la barra lateral para permitir al usuario ajustar los parámetros del algoritmo.
#Se muestra la imagen segmentada utilizando el objeto de ejes ax. La imagen segmentada se obtiene de la matriz segmentacion.
#Se crea un búfer (buffer) para guardar la figura como un archivo de imagen en formato PNG utilizando plt.savefig().
#Se busca la posición inicial del búfer utilizando buffer.seek(0).
#Se muestra la imagen guardada en el búfer utilizando st.image(). La imagen se ajusta al ancho de la columna.

if algoritmo != 'Ninguno':

    fig, ax = plt.subplots()

    if algoritmo == 'isoData':
        tau = st.sidebar.number_input("Tau:")
        tol = st.sidebar.number_input("Tol:")
        segmentacion = isoData(image_data, tau, tol)
    
    if algoritmo == 'Region Growing':
        tol = st.sidebar.number_input("Tol:", value=142)
        segmentacion = region_growing(image_data, 142,142,142, tol)
    
    if algoritmo == 'Clustering':
        ks = st.sidebar.number_input("Clusters:", value=2)
        segmentacion = clustering(image_data, ks)
    
    if algoritmo == 'Gaussian Mixture Model':
        ks = st.sidebar.number_input("Clusters:", value=2)
        segmentacion = gmm(image_data, ks)
       
        #Se verifica el nombre de la imagen (name_imagen) para determinar si es 'T1.nii.gz' o 'IR.nii.gz' o 'FLAIR.nii.gz'.
        #Si el nombre de la imagen es 'T1.nii.gz', se muestra un menú desplegable (st.selectbox) en la barra lateral para que el usuario seleccione un tipo de corte (axial, sagital o coronal).
        #Dependiendo del tipo de corte seleccionado, se muestra un control deslizante (st.slider) para que el usuario seleccione una coordenada en el corte correspondiente.
        
        #Se muestra la imagen segmentada (segmentacion) utilizando el objeto de ejes ax. La imagen segmentada se obtiene de la matriz segmentacion.
        #Si el nombre de la imagen es 'IR.nii.gz' o 'FLAIR.nii.gz', se muestra un control deslizante (st.slider) para que el usuario seleccione una coordenada en la imagen segmentada.
        #Se crea un búfer (buffer) para guardar la figura como un archivo de imagen en formato PNG utilizando plt.savefig().
        #Se busca la posición inicial del búfer utilizando buffer.seek(0).
        #Se muestra la imagen guardada en el búfer utilizando st.image(). La imagen se ajusta al ancho de la columna.
        
      

    if name_imagen == 'T1.nii.gz':
        opciones = [ 'Axial', 'Sagital', 'Coronal']
        corte = st.selectbox('Selecciona un corte para ver', opciones)

        if corte == 'Axial':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[1]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(segmentacion[:, valor_seleccionado, :], cmap='bone')

        if corte == 'Sagital':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ##imagen
            ax.imshow(np.rot90(segmentacion[:, :, valor_seleccionado],k=-1), cmap='bone')

        if corte == 'Coronal':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[0]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(segmentacion[valor_seleccionado, :, :], cmap='bone')


    if(name_imagen == 'IR.nii.gz' or name_imagen == 'FLAIR.nii.gz'):
        valor_especifico = 25
        valor_maximo = segmentacion.shape[2]
        ###imagen
        valor_seleccionado = st.slider("Esta es su imagen segmentada", 0, valor_maximo, valor_especifico)
        ###imagen
        ax.imshow(np.rot90(segmentacion[:, :, valor_seleccionado], k=-1), cmap='bone')

    #buffer = io.BytesIO()
    #plt.savefig(buffer, format='png')
    #buffer.seek(0)
    
    # Mostrar la imagen utilizando st.image
    st.write(
        """## Resultado de la segmentación"""
    )
    fig.set_size_inches(2,2)
    st.pyplot(fig)
    #st.image(buffer, caption = name_imagen, use_column_width=True)

    nifti_image= nib.Nifti1Image(segmentacion,image_load.affine, image_load.header)
    nifti_image.to_filename(seg_path+"seg_"+name_imagen)
    st.write("Se ha guardado con éxito la imagen")


    
    #En este código se divide el espacio en dos columnas (col1 y col2) utilizando st.columns(2). Luego, se muestra contenido en cada columna.
    
    #En la columna col1:
    #Se crea una figura (fig) y ejes (ax) con un tamaño grande.
    #Si el nombre de la imagen es 'T1.nii.gz', se muestra un menú desplegable (st.selectbox) en el que el usuario puede seleccionar un tipo de corte (axial, sagital o coronal).
    #Dependiendo del tipo de corte seleccionado, se muestra un control deslizante (st.slider) para que el usuario seleccione una coordenada en el corte correspondiente.
    #Se muestra la imagen original (image_data) utilizando el objeto de ejes ax. La imagen se obtiene de la matriz image_data.
    #Si el nombre de la imagen es 'IR.nii.gz' o 'FLAIR.nii.gz', se muestra un control deslizante (st.slider) para que el usuario seleccione una coordenada en la imagen original.
    #Se crea un búfer (buffer) para guardar la figura como un archivo de imagen en formato PNG utilizando plt.savefig().
    #Se busca la posición inicial del búfer utilizando buffer.seek(0).
    #Se muestra la imagen guardada en el búfer utilizando st.image(). La imagen se ajusta al ancho de la columna.
    
    #En la columna col2:
    #Se muestra un texto indicando que se trata del histograma de la imagen.
    #Se calcula el histograma de la imagen original (image_data) y se muestra utilizando plt.hist() y st.pyplot().
    
    

# Dividir el espacio en dos columnas
col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(50, 50))

    if(name_imagen == 'T1.nii.gz'):
        opciones = [ 'Axial', 'Sagital', 'Coronal']
        corte = st.selectbox('Selecciona un corte', opciones)

        if corte == 'Axial':
            valor_especifico = 100
            valor_maximo = image_data.shape[1]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[:, valor_seleccionado, :], cmap='bone')

        if corte == 'Sagital':
            valor_especifico = 100
            valor_maximo = image_data.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ##imagen
            ax.imshow(np.rot90(image_data[:, :, valor_seleccionado],k=-1), cmap='bone')

        if corte == 'Coronal':
            valor_especifico = 100
            valor_maximo = image_data.shape[0]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[valor_seleccionado, :, :], cmap='bone')


    if(name_imagen == 'IR.nii.gz' or name_imagen == 'FLAIR.nii.gz'):
        valor_especifico = 25
        valor_maximo = image_data.shape[2]
        ###imagen
        valor_seleccionado = st.slider("Esta es su imagen", 0, valor_maximo, valor_especifico)
        ###imagen
        ax.imshow(np.rot90(image_data[:, :, valor_seleccionado], k=-1), cmap='bone')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Mostrar la imagen utilizando st.image
    st.image(buffer, caption = name_imagen, use_column_width=True)

with col2:
    st.write(
        """Este es el histograma de su imagen"""
    )
    hist_data3 = image_data.flatten()
    fig3, ax3 = plt.subplots()
    ax3.hist(hist_data3, bins=100)
    st.pyplot(fig3)    
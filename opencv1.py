import streamlit as st
import cv2
import numpy as np
from PIL import Image

class Page:
    def __init__(self):
        st.set_page_config(page_title = 'Visão Computacional com OpenCV - Filtros de imagem')

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    def pencil_sketch(self, image):
        self.gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #self.inv_gray = 255 - self.gray_img
        self.blurred_img = cv2.GaussianBlur(self.gray_img, (21,21), 0, 0) #Kernel size: 21x21
        self.gray_sketch = cv2.divide(self.gray_img, self.blurred_img, scale = 256)
        return self.gray_sketch

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    def cartoon(self, rgb_img, num_pyr_downs = 2, num_bilaterals = 7):
        #1. Aplicar o filtro bilateral para reduzir a paleta de cores da imagem
        self.downsampled_img = rgb_img
        for _ in range(num_pyr_downs):
            self.downsampled_img = cv2.pyrDown(self.downsampled_img)
        for _ in range(num_bilaterals):
            self.filtered_small_img = cv2.bilateralFilter(self.downsampled_img, 9, 9, 7)
        self.filtered_normal_img = self.filtered_small_img
        for _ in range(num_pyr_downs):
            self.filtered_normal_img = cv2.pyrUp(self.filtered_normal_img)
        if self.filtered_normal_img.shape != rgb_img.shape:
            self.filtered_normal_img = cv2.resize(self.filtered_normal_img, rgb_img.shape[:2])
        #2. Converter a imagem colorida em tons de cinza
        self.img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        #3. Aplicar blur (borragem) médio para reduzir ruído de imagem
        self.img_blur = cv2.medianBlur(self.img_gray, 7)
        #4. Aplicar limitação adaptativa para detectar e destacar bordas
        self.gray_edges = cv2.adaptiveThreshold(
            self.img_blur, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 2)
        #5. Combinar a imagem colorida do passo 1. com as bordas destacadas do passo 4.
        self.rgb_edges = cv2.cvtColor(self.gray_edges, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(self.filtered_normal_img, self.rgb_edges)
        

    def page(self):
        st.title('Visão Computacional com OpenCV - Filtros de imagem')
        st.write('_Autor: Rafael Silva Coelho_')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto.""")
        st.write(
            """O intuito deste app é usar diversas funcionalidades do pacote de 
            visão computacional do Python OpenCV.""")

        with st.expander('1.Exemplo de filtros aplicados em imagens com OpenCV', expanded = True):
            st.header('Exemplo de filtros aplicados em imagens com OpenCV')
            #self.example_image = Image.open('images/image5.jpg')
            self.selectbox_ = st.selectbox(
                'Selecione um filtro', [
                    'Imagem original',
                    'Desenho a lápis',
                    'Cartoon'])
            if self.selectbox_ == 'Imagem original':
                st.image(Image.open('images/image5.jpg'))
            elif self.selectbox_ == 'Desenho a lápis':
                st.image(Image.open('images/image6.jpg'))
            elif self.selectbox_ == 'Cartoon':
                st.image(Image.open('images/image7.jpg'))
            st.write('---')
            st.header('Faça você mesmo')
            self.file_uploader = st.file_uploader(
                'Envie uma imagem',
                type = ['jpg'],
                accept_multiple_files = False)
            if self.file_uploader is not None:
                self.uploaded_image = np.asarray(Image.open(self.file_uploader))
                self.sketch_image = self.pencil_sketch(self.uploaded_image)
                self.cartoon = self.cartoon(self.uploaded_image)
                self.selectbox = st.selectbox(
                    'Selecione o filtro', [
                        'Desenho a lápis',
                        'Cartoon'])
                if self.selectbox == 'Desenho a lápis':
                    st.image(self.sketch_image, caption = 'Desenho a lápis')
                elif self.selectbox == 'Cartoon':
                    st.image(self.cartoon, caption = 'Cartoon')

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.file = open('opencv1.py', 'r').read()
            st.code(self.file, language = 'python')



page = Page().page()
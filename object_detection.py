import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import time
from PIL import Image
from object_detection.utils import visualization_utils as viz
from object_detection.utils.label_map_util import create_category_index_from_labelmap


class ObjectDetectionPage:
    def __init__(self):
        st.set_page_config(
            page_title = 'Detecção de Objetos com Tensorflow Object Detection API')

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    def _load_model(self):
        self.model = hub.load('https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1')
        return self.model

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    @tf.function
    def _detector_output(self, model, image):
        self.detector_output = model(image)
        return self.detector_output

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    @tf.function
    def _return_predictions(self, detector_output, image):
        self.model_output = {k: v.numpy() for k, v in detector_output.items()}
        self.boxes = self.model_output['detection_boxes'][0]
        self.classes = self.model_output['detection_classes'][0].astype('int')
        self.scores = self.model_output['detection_scores'][0]
        self.image_with_mask = viz.visualize_boxes_and_labels_on_image_array(
            image = image[0].numpy(),
            boxes = self.boxes,
            classes = self.classes,
            scores = self.scores,
            category_index = create_category_index_from_labelmap(
                'object_detection/data/mscoco_label_map.pbtxt'),
            use_normalized_coordinates = True,
            agnostic_mode = False,
            line_thickness = 5)
        return self.image_with_mask

    def page(self):
        st.title('Detecção de Objetos em Imagem com Tensorflow Object Detection API')
        st.write('_Autor: Rafael Silva Coelho_')
        st.caption('Caso você acesse alguma dessas páginas em um smartphone, use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto.')
        st.write(
            """Este app tem como objetivo detectar e classificar objetos presentes em uma imagem 
            com uma determinada taxa de precisão usando Tensorflow Object Detection API.""")

        with st.expander('1. Exemplo de detecção de objetos em uma imagem', expanded = True):
            st.header('Exemplo de detecção de objetos em uma imagem')
            self.example_image1 = tf.keras.utils.load_img('images/image1.jpg')
            st.image(self.example_image1, caption = 'Imagem de exemplo')
            self.example_image2 = tf.keras.utils.load_img('images/image2.jpg')
            st.image(
                self.example_image2, 
                caption = 'Imagem com objetos detectados e classificados com % de acurácia')

        with st.expander('2. Faça você mesmo', expanded = True):
            self.model = self._load_model()
            st.header('Faça você mesmo')
            self.col00, self.col01 = st.columns(2)
            self.file_uploaded = st.file_uploader(
                'Envie uma imagem',
                type = ['jpg'],
                accept_multiple_files = False)
            st.caption(
                'O resultado final será processado e exibido assim que você enviar a imagem.')
            st.caption(
                'Você pode enviar imagens direto do seu computador ou arrastar de algum site.')
            if self.file_uploaded is not None:
                self.uploaded_image = Image.open(self.file_uploaded)
                st.image(self.uploaded_image, caption = 'Imagem enviada')
                self.image = tf.convert_to_tensor(np.asarray(self.uploaded_image))
                self.image = tf.expand_dims(self.image, axis = 0) #[1, width, height, 3]
                self.start = time.time()
                self.detector_output = self._detector_output(self.model, self.image)
                self.objects_image = self._return_predictions(self.detector_output, self.image)
                self.end = time.time()
                self.time = self.end - self.start
                st.image(self.objects_image, caption = 'Objetos detectados e classificados')
                st.metric('Processamento', '{:.2f}s'.format(self.time))
                
        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.file = open('object_detection.py', 'r').read()
            st.code(self.file, language = 'python')
                
page = ObjectDetectionPage().page()


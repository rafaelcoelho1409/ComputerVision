import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st
import numpy as np
import time
from object_detection.utils import ops, visualization_utils as viz
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from PIL import Image

class ImageSegmentationPage:
    def __init__(self):
        st.set_page_config(page_title = 'Segmentação de Imagem com Tensorflow Hub')

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    def _load_model(self):
        self.model = hub.load('https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1')
        return self.model

    @st.cache(
        hash_funcs = {st.delta_generator.DeltaGenerator: lambda _: None}, 
        show_spinner = True)
    def _show_segmentation(self, output, image_output):
        self.output = output
        self.image_output = image_output
        self.model_output = {k: v.numpy() for k, v in self.output.items()}
        self.detection_masks = tf.convert_to_tensor(self.model_output['detection_masks'][0])
        self.detection_boxes = tf.convert_to_tensor(self.model_output['detection_boxes'][0])
        self.detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
            self.detection_masks,
            self.detection_boxes,
            self.image_output.shape[1],
            self.image_output.shape[2])
        self.detection_masks_reframed = tf.greater(self.detection_masks_reframed, 0.5)
        self.model_output['detection_masks_reframed'] = self.detection_masks_reframed.numpy().astype(np.uint8)
        self.boxes = self.model_output['detection_boxes'][0]
        self.classes = self.model_output['detection_classes'][0].astype('int')
        self.scores = self.model_output['detection_scores'][0]
        self.masks = self.model_output['detection_masks_reframed']
        self.image_with_mask = viz.visualize_boxes_and_labels_on_image_array(
            image = self.image_output[0].numpy(),
            boxes = self.boxes,
            classes = self.classes,
            scores = self.scores,
            category_index = create_category_index_from_labelmap(
                'object_detection/data/mscoco_label_map.pbtxt'),
            use_normalized_coordinates = True,
            max_boxes_to_draw = 200,
            min_score_thresh = 0.30,
            agnostic_mode = False,
            instance_masks = self.masks,
            line_thickness = 5)
        return self.image_with_mask

    def page(self):
        st.title('Segmentação de Imagem com Tensorflow Hub')
        st.write('_Autor: Rafael Silva Coelho_')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto.""")
        st.write(
            """Este app tem como objetivo segmentar imagem, ou seja, dividir uma imagem em 
            diversas regiões para facilitar análises a partir deste processo.""")

        with st.expander('1. Exemplo de segmentação de imagem', expanded = True):
            st.header('Exemplo de segmentação de imagem')
            self.example_image1 = tf.keras.utils.load_img('images/image3.jpg')
            st.image(self.example_image1, caption = 'Imagem de exemplo')
            self.example_image2 = tf.keras.utils.load_img('images/image4.jpg')
            st.image(self.example_image2, caption = 'Segmentação da imagem de exemplo')

        with st.expander('2. Faça você mesmo', expanded = True):
            st.header('Faça você mesmo')
            self.uploaded_image = st.file_uploader(
                'Envie uma imagem',
                type = ['jpg'],
                accept_multiple_files = False)
            st.caption(
                'O resultado final será processado e exibido assim que você enviar a imagem.')
            st.caption(
                'Você pode enviar imagens direto do seu computador ou arrastar de algum site.')
            st.caption(
                'O modelo leva aproximadamente 1 minuto para retornar o resultado pronto.')
            if self.uploaded_image is not None:
                self.uploaded_image = np.asarray(Image.open(self.uploaded_image))
                st.image(self.uploaded_image, caption = 'Imagem enviada')
                self.image = tf.convert_to_tensor(self.uploaded_image)
                self.image = tf.expand_dims(self.image, axis = 0)
                self.start = time.time()
                self.model = self._load_model()
                self.output = self.model(self.image)
                self.segmented_image = self._show_segmentation(self.output, self.image)
                self.end = time.time()
                self.time = self.end - self.start
                st.image(self.segmented_image, caption = 'Imagem segmentada')
                st.metric('Processamento', '{:.2f}s'.format(self.time))

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.file = open('image_segmentation.py', 'r').read()
            st.code(self.file, language = 'python')
            
page = ImageSegmentationPage().page()


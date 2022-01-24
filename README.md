# Visão Computacional com Tensorflow, Keras e OpenCV
<p float="left">
  <img src="images/image2.jpg" width="400" />
  <img src="images/image4.jpg" width="400" /> 
</p>

## Objetivo
Disponibilizar aplicações de modelos de Visão Computacional em formato de web app. Foram abordados os seguintes tópicos:
- Detecção e classificação de objetos em imagem
- Segmentação de objetos em imagem
- AutoEncoder (modelo de rede neural aplicado para redução de ruídos em imagem)
- Aplicação de filtros de imagens com OpenCV

## Links deste projeto
- https://share.streamlit.io/rafaelcoelho1409/computervision/object_detection.py (Detecção e classificação de objetos em imagem)
- https://share.streamlit.io/rafaelcoelho1409/computervision/image_segmentation.py (Segmentação de objetos em imagem)
- https://share.streamlit.io/rafaelcoelho1409/computervision/opencv1.py (Aplicação de Visão Computacional com OpenCV)  
OBS: devido a limitações do Streamlit Cloud, este repositório fica muito grande para o armazenamento interno do Streamlit Cloud, logo a máquina que processa este repositório para disponibilizar as aplicações, com sua memória limitada, não aguenta e dá erro.  
O único app que funciona sem limitações no Streamlit Cloud é o terceiro (Aplicação de Visão Computacional com OpenCV), portanto todos os apps só funcionam corretamente localmente (instruções abaixo de como rodar localmente na sua máquina).

## Recursos utilizados
- Visual Studio Code
- python3.8.8
- virtualenv
- pip3: gerenciador de pacotes python3.x

## Pacotes do Python
- streamlit
- tensorflow
- tensorflow.keras
- tensorflow_hub
- plotly
- time
- PIL
- cv2 (OpenCV)
- object_detection (Tensorflow Object Detection API)

## Tópicos abordados neste projeto
- Detecção e classificação de objetos com Tensorflow Object Detection API
- Segmentação de imagem
- Convolutional AutoEncoder (modelo de rede neural)
- Filtros de imagem do OpenCV
- Detecção e classificação de objetos com OpenCV (repostado)

## Para executar esse arquivo localmente em sua máquina
- baixe esse repositório em sua máquina:
> git clone https://github.com/rafaelcoelho1409/DeepLearning.git
- instale os pacotes necessários que estão no arquivo requirements.txt:
> pip3 install -r requirements.txt
- escolha seu interpretador python (python3, python3.x)  
- execute os seguintes comandos (para Linux):
> cd DeepLearning  
> streamlit run {nome_do_app}.py (substitua {nome_do_app} por object_detection.py, image_segmentation.py, opencv1.py)  
- Com esses comandos, a página será aberta automaticamente. Caso não abra, vá até seu navegador e digite:
> http://localhost:8501  

## Screenshots do projeto construído
<img src="images/screenshot01.png" />
<img src="images/screenshot02.png" />
<img src="images/screenshot03.png" />
<img src="images/screenshot04.png" />
<img src="images/screenshot05.png" />
<img src="images/screenshot06.png" />
<img src="images/screenshot07.png" />
<img src="images/screenshot08.png" />
<img src="images/screenshot09.png" />

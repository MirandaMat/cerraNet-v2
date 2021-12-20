# Aumento de dados estático

# Biblioteca

## Gerencia arquivos
from glob import glob
import numpy as np
import pandas as pd
import os

## Transformacao na imagem
import imageio
import imageio as imageio
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa


## Gráficos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
#%matplotlib inline

# Variaveis
i = 0
j = 0
k = 0
l = 0

img_names = glob(os.path.join(os.getcwd(), 'dataset/Dataset_original/forest/*.tiff'))

for imagem in img_names:
    #Ler todas as imagens na pasta e plotar
    img = imageio.imread(imagem)

    # Corta imagens no centro
    crop = iaa.CenterCropToFixedSize(height=256, width=256)
    crop_img = crop.augment_image(img)

    # Salvar imagem
    nomeArquivo = os.path.basename(imagem)
    Image.fromarray(crop_img).save('dataset/data_DA/forest/'+nomeArquivo)

    # Rotação
    rotate = iaa.Affine(rotate=(-45, 30))
    rotated_image = rotate.augment_image(img)

    # Corta imagens no centro
    crop = iaa.CenterCropToFixedSize(height=256, width=256)
    crop_rotate = crop.augment_image(rotated_image)

    # Salvar imagem
    nomeArquivo = os.path.basename(imagem)
    Image.fromarray(crop_rotate).save('dataset/data_DA/forest/DARot_'+nomeArquivo)

    # Flip horizontal
    flip_hr = iaa.Fliplr(p=1.0)
    flip_hr_image = flip_hr.augment_image(img)

    # Corta imagens no centro
    crop = iaa.CenterCropToFixedSize(height=256, width=256)
    crop_fh = crop.augment_image(img)

    # Salvar imagem
    nomeArquivo = os.path.basename(imagem)
    Image.fromarray(crop_fh).save('dataset/data_DA/forest/DAFlipH_'+nomeArquivo)

    # Flip vertical
    flip_vr = iaa.Flipud(p=1.0)
    flip_vr_image = flip_vr.augment_image(img)

    # Corta imagens no centro
    crop = iaa.CenterCropToFixedSize(height=256, width=256)
    crop_fv = crop.augment_image(flip_vr_image)

    # Salvar imagem
    nomeArquivo = os.path.basename(imagem)
    Image.fromarray(crop_fv).save('dataset/data_DA/savanna/DAFlipV_'+nomeArquivo)

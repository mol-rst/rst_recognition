import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
from PIL import Image

#画像を拡張する関数(水増し)
def draw_images(generator, x, dir_name, index):
    save_name = str(index)
    g = generator.flow(x, batch_size=1, save_to_dir = output_dir,save_prefix=save_name, save_format='jpg')

    for i in range(1):#1枚に拡張
        bach = g.next()

#output_dir = "extend_img"
categories = ["mana","sayu","kae","kasumi","mizuha","mii",
"yukari","haruka",
"aone","ruka","sango",
"amaha","kanade","nagisa",
"mikuru","akari","kuroha","haku"]

#if not(os.path.exists(output_dir)):
    #os.mkdir(output_dir)

    #拡張の元画像を読み込む
#images = glob.glob(os.path.join("image","*.jpg"))

#ImageDataGeneratorの設定
datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0,
                            height_shift_range=0,zoom_range=0,fill_mode="wrap",
                            horizontal_flip=False, vertical_flip=False)
    
    #読み込んだ画像を拡張
output_dir = "検証データ_160"
if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)
images = glob.glob(os.path.join("検証データ","*jpg"))
for i in range(len(images)):
    img = load_img(images[i])
    #img = img.resize((160,160))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    draw_images(datagen, x, output_dir, i)
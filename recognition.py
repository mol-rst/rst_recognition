from keras import models
from keras.models import model_from_json
from keras.preprocessing.image import load_img,img_to_array,array_to_img
import numpy as np
from PIL import Image
import os

#モデルの読み込み
model = model_from_json(open('model/chara_predict.json').read())

#重みの読み込み
model.load_weights("weight/chara_predict.hdf5")
#model.summary()

#モデルのコンパイル
from keras import optimizers

model.compile(loss = "binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-3),
              metrics=["acc"])


categories = ["舞菜","紗由","かえ","香澄","瑞葉","みい",
"紫","陽花",
"碧音","瑠夏","珊瑚",
"天葉","奏","那岐咲",
"美久龍","朱莉","玄刃","ハク"]

#認識したい画像の読み込み
img_name = os.listdir("検証データ")

for j in range(len(img_name)):
    #img = load_img("検証データ/"+str(j)+".jpg",target_size=(80,80,3))
    img = load_img("検証データ/"+img_name[j],target_size=(80,80,3))
    x = img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
#予測
    features = model.predict(x)
    temp = np.zeros(18)
    for i in range(0,18):
        temp[i] = features[0,i]
    sortfea = sorted(temp, reverse=True)
#予測の結果から処理を分ける
    for i in range(0,18):
        if features[0,i]== sortfea[0]:
            cat = categories[i]
            print(img_name[j]+"は「"+categories[i]+"]と認識されました(認識率[%]=", end='')
            print(np.array(sortfea[0]*100,dtype=int))
    
    for i in range(0,18):
        if features[0,i]== sortfea[1]:
            cat = categories[i]
            print("もし違うのであれば「"+categories[i]+"]ですか？(認識率[%]=", end='')
            print(np.array(sortfea[1]*100,dtype=int))

    for i in range(0,18):
        if features[0,i]== sortfea[2]:
            cat = categories[i]
            print("それでも違うのであれば「"+categories[i]+"]ですか？(認識率[%]=", end='')
            print(np.array(sortfea[2]*100,dtype=int))

    print(" ")

        
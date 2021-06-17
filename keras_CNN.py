#モデルの構築
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(80,80,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256,(2,2),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(180,activation="relu"))
model.add(layers.Dense(18,activation="sigmoid"))

model.summary()#モデル構成の確認


#モデルのコンパイル
from keras import optimizers

model.compile(loss = "binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=["acc"])


#学習・検証データの準備
from keras.utils import np_utils
import numpy as np

categories = ["mana","sayu","kae","kasumi","mizuha","mii",
                "yukari","haruka",
                "aone","ruka","sango",
                "amaha","kanade","nagisa",
                "mikuru","akari","kuroha","haku"]

nb_classes = len(categories)

X_train, X_test, y_train, y_test = np.load('data/chara_data.npy', allow_pickle=True)

#データの正規化
X_train = X_train.astype("float")/255
X_test = X_test.astype("float")/255

#kerasを用いるためにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


#モデルの学習

model = model.fit(X_train,
                  y_train,
                  epochs=10,
                  batch_size=6,
                  validation_data=(X_test,y_test))

import matplotlib.pyplot as plt

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss')


#モデルの保存
json_string = model.to_json()
open('model/chara_predict.json','w').write(json_string)

#重みの保存
hdf5_file = "weight/chara_predict.hdf5"
model.save_weights(hdf5_file)

test_X = np.load("testdata/chara_data_test_X_80.npy")
test_Y = np.load("testdata/chara_data_test_Y_80.npy")

from keras.utils import np_utils

test_Y = np_utils.to_categorical(test_Y,18)

score = model.model.evaluate(x=test_X,y=test_Y)

print('loss=', score[0])
print('accuracy=', score[1])


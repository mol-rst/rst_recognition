from PIL import Image
import os, glob
import numpy as np
import random,math

root_dir = "image"#画像が保存されているルートディレクトリのパス

categories = ["mana","sayu","kae","kasumi","mizuha","mii",
"yukari","haruka",
"aone","ruka","sango",
"amaha","kanade","nagisa",
"mikuru","akari","kuroha","haku"]

X = []#画像データ用配列
Y = []#ラベルデータ用配列

def make_sample(files):#画像データごとにadd_sampleを呼び出す
    global X,Y
    X=[]
    Y=[]
    for cat,fname in files:
        add_sample(cat,fname)
    return np.array(X), np.array(Y)

    
def add_sample(cat,fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    #img = img.resize((160,160))
    data = np.asarray(img)#np.asarrayはnp.arrayとほとんど同じ。asarrayは今回の場合imgと同じidを取る
    X.append(data)#appendは配列Xの一番後ろにdataを追加するもの
    Y.append(cat)

allfiles = []#全データ格納用配列

for idx, cat in enumerate(categories):#enumerateは()内に配列をいれることで、今回の場合、0 mana 1 sayuと番号付きのリストを作製できる(catにmanaなどが入る)
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx,f)) 

random.shuffle(allfiles)#全データをシャッフルしてcd、そのうちの8割を訓練データとする
th = math.floor(len(allfiles)*0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train,y_train = make_sample(train)
X_test,y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)
np.save('data/chara_data.npy',xy)
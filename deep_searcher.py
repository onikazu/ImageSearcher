"""
python deep_searcher.py index

python deep_searcher.py search query_img/rasp3_1.jpg inception_v3
"""


from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
import faiss

import numpy as np
import sys, os
import os.path
import pickle
import glob


def make_index():
    """インデックス作成"""
    model = VGG16(include_top=False, weights='imagenet')
    file_names = glob.glob("./img/*")
    index = {}
    for file_name in file_names:
        preds = _make_vector(file_name, model)
        index[file_name] = preds

    # インデックスの保存
    with open('vgg.pickle', mode='wb') as f:
        pickle.dump(index, f)

    # パラメータ・モデルの保存
    # モデルの保存
    open('vgg.json',"w").write(model.to_json())
    # 学習済みの重みを保存
    model.save_weights('vgg.h5')

def _make_vector(file_name, model):
    img = image.load_img(file_name, target_size=(224, 224))
    image.save_img("./temp/{}".format(file_name.split("/")[-1]), img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    print("{} is done".format(file_name))
    return preds


def search_index(file_name):
    if sys.argv[3] == "inception_v3":
        model_name = 'inception_v3.json'
        param_name = "inception_v3.h5"
        pickle_name = "inception_v3.pickle"
    elif sys.argv[3] == "vgg16":
        model_name = 'vgg.json'
        param_name = "vgg.h5"
        pickle_name = "vgg.pickle"
    else:
        print("select model_name")
    model = model_from_json(open(model_name, 'r').read())
    model.load_weights(param_name)
    # index = _ready_faiss()
    # target_vec = _make_vector(file_name, model)
    # D, I = index.search(target_vec, 3)
    # print(I)
    target_vec = _make_vector(file_name, model)
    datas = {}
    with open(pickle_name, mode='rb') as f:
        datas = pickle.load(f)
    file_names, distances = calc_dist(target_vec, datas)
    distances = np.array(distances)
    min_index = distances.argmin()
    print()
    print("クエリ画像")
    print(file_name)
    print()
    print("結果リスト")
    print(file_names)
    print(distances)
    print()
    print("結果")
    print("1位　{}".format(file_names[distances.argsort()[0]]))
    print("2位　{}".format(file_names[distances.argsort()[1]]))
    print("3位　{}".format(file_names[distances.argsort()[2]]))



def calc_dist(target, database):
    file_names = []
    distances = []
    for k, v in database.items():
        distances.append(np.linalg.norm(target-v))
        file_names.append(k)
    return file_names, distances


def _ready_faiss():
    print("start indexing")
    datas = {}
    with open('inception_v3.pickle', mode='rb') as f:
        datas = pickle.load(f)
    # databese配列の作成
    image_names = []
    vectors = []
    for k in datas:
        image_names.append(k)
        vectors.append(datas[k])
    vectors = np.array(vectors).astype("float32")

    # faissを用いたPQ
    nlist = 100
    m = 8
    d = 2048  # 顔特徴ベクトルの次元数
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    print(vectors.shape)
    index.train(vectors)
    index.add(vectors)
    print("indexing is end")
    return index


if __name__ == "__main__":
    if sys.argv[1] == "index":
        make_index()
    elif sys.argv[1] == "search":
        file_name = sys.argv[2]
        search_index(file_name)
    else:
        print("argument is lacked")

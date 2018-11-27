# -*- encoding: utf-8 -*-
"""
python similar-image-retrieval.py search sample.pickle query_img/rasp3_1.jpg
"""


import cv2
import numpy as np
import os.path
import sys
import pickle

detector = cv2.KAZE_create()
color_model = cv2.IMREAD_GRAYSCALE

def calc_clusters(files, k):
  """
  指定された画像ファイルから局所特徴量を抽出し K 個のクラスタに分割した重心を算出する。
  :param files: クラスタを算出する画像ファイル
  :param k: クラス多数
  :return: [クラスタ重心, ファイルごとの特徴点]
  """
  trainer = cv2.BOWKMeansTrainer(k)
  keypoints = []
  for i, file in enumerate(files):
    image = _read_image(file, color_model)
    if image is not None:
      ks, ds = detector.detectAndCompute(image, None)
      if ds is not None:
        trainer.add(ds.astype(np.float32))
      keypoints.append(ks)
  return [trainer.cluster(), keypoints]

def calc_probabilities(files, centroids, keypoints = None):
  """
  指定されたクラスタ重心を使用してそれぞれのファイルのクラスタ所属確率(ヒストグラム)を算出する。
  :param files: クラスタ所属確率を計算する画像ファイル
  :param centroids: クラスタごとの重心
  :param keypoints: 算出済みの特徴点 (None の場合は内部で算出)
  :return: 画像ファイルごとのクラスタ所属確率
  """
  matcher = cv2.BFMatcher()
  extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
  extractor.setVocabulary(centroids)
  probs = []
  for i, file in enumerate(files):
    descriptor = None
    image = _read_image(file, color_model)
    if image is not None:
      ks = None
      if keypoints is not None and keypoints[i] is not None:
        ks = keypoints[i]
      else:
        ks = detector.detect(image, None)
      if ks is not None:
        descriptor = extractor.compute(image, ks)[0]
    probs.append(descriptor)
  return probs

def calc_similarity(prob1, prob2):
  """
  指定されたクラスタ所属確率の類似度を算出する。
  :param prob1: クラスタ所属確率1
  :param prob2 クラスタ所属確率1
  :return: クラスタ所属確率の類似度
  """
  return sum(map(lambda x: min(x[0], x[1]), zip(prob1, prob2)))

def _read_image(file, flag = color_model):
  """
  指定された画像を読み込んで画像として返す。cv2.imread() は Windows で日本語のパスに対応していないため。
  :param file: 読み込む画像ファイル
  :param color_model 画像のカラーチャネル
  :return: 画像
  """
  with open(file, "rb") as f:
    binary = np.fromfile(file, dtype=np.uint8)
    return cv2.imdecode(binary, flag)

if __name__ == "__main__":
  command = sys.argv[1]
  k = 128

  if command == "index":
    # インデックス作成
    index = sys.argv[2]
    files = sys.argv[3:]
    centroids, _ = calc_clusters(files, k)
    with open(index, "wb") as f:
      pickle.dump({"centroids": centroids, "files": files}, f)

  elif command == "search":
    index = sys.argv[2]
    file = sys.argv[3]
    centroids = None
    files = None
    with open(index, "rb") as f:
      obj = pickle.load(f)
      centroids = obj["centroids"]
      files = obj["files"]
    # 類似画像検索
    prob = calc_probabilities([file], centroids)[0]
    probs = calc_probabilities(files, centroids)
    rank = []
    for f, p in zip(files, probs):
      if p is not None:
        sim = calc_similarity(prob, p)
        rank.append([f, sim])
    rank = sorted(rank, key = lambda x: - x[1])
    for f, sim in rank:
      print("%.3f %s" % (sim, f))

  elif command == "similarity":
    files = sys.argv[2:]
    centroids, keypoints = calc_clusters(files, k)
    probs = calc_probabilities(files, centroids, keypoints)
    for i, f1 in enumerate(files):
      sys.stdout.write(("," if i is not 0 else "") + f1)
    sys.stdout.write("\n")
    for i, f1 in enumerate(files):
      sys.stdout.write(f1)
      for j, f2 in enumerate(files):
        sim = calc_similarity(probs[i], probs[j])
        sys.stdout.write("," + ("%.3f" % sim))
      sys.stdout.write("\n")

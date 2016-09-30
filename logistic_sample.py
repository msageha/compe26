# coding=utf-8
import pandas as pd
import os
import numpy as np
from time import clock
from PIL import Image
import sys
from skimage.feature import hog
from sklearn.cross_validation import train_test_split

def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
  classes = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
  df = pd.read_csv(file_name)
  n = len(df)
  Y = np.zeros((n, len(classes)))
  print('loading...')
  s = clock()
  for i, row in df.iterrows():
    f, l, t, r, b = row.filename, row.left, row.top, row.right, row.bottom
    img = Image.open(os.path.join(img_dir, f)).crop((l,t,r,b)) #項目領域画像を切り出す
    if img.size[0]<img.size[1]:                                #縦長の画像に関しては90度回転して横長の画像に統一する
      img = img.transpose(Image.ROTATE_90)
        
    # preprocess
    img_gray = img.convert('L') #グレースケール化
    img_gray = np.array(img_gray.resize(img_shape))/255.       # img_shapeに従った大きさにそろえる

    # feature extraction hog化
    img = np.array(hog(img_gray,orientations = orientations,
                  pixels_per_cell = pixels_per_cell,
                  cells_per_block = cells_per_block))
    if i == 0:
      feature_dim = len(img)
      print('feature dim:', feature_dim)
      X = np.zeros((n, feature_dim))
    if i%1000==0:
      print('load:', i)
    X[i,:] = np.array([img])
    y = list(row[classes])
    Y[i,:] = np.array(y)

  print('Done. Took', clock()-s, 'seconds.')
  return X, Y

# df = pd.read_csv('train.csv')
# 学習データを1:1に分けて改めて学習用データと検証用データを作ります.
img_shape = (216,72)
orientations = 9
pixels_per_cell = (6,6)
cells_per_block = (3, 3)
X, Y = load_data('train.csv', 'train_images', img_shape, orientations, pixels_per_cell, cells_per_block)
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1234)
x_train, y_train = X, Y
X, Y = load_data('test.csv', 'test_images', img_shape, orientations, pixels_per_cell, cells_per_block)
x_test, y_test = X, Y
print('学習データの数:', len(x_train))
print('検証データの数:', len(x_test))



from sklearn.linear_model import LogisticRegression

class MultiLabelLogistic():
  def __init__(self, n_out):
    self.n_out = n_out
    model_list = []
    for l in range(self.n_out):
      model_list.append(LogisticRegression())
    self.models = model_list

  def fit(self, X, Y):
    i = 0
    start_overall = clock()
    for model in self.models:
      start = clock()
      print('training model No.%s...'%(i+1))
      model.fit(X, Y[:,i])
      print('Done. Took', clock()-start, 'seconds.')
      i += 1
    print('Done. Took', clock()-start_overall, 'seconds.')

  def predict(self, X):
    i = 0
    predictions = np.zeros((len(X), self.n_out))
    start = clock()
    print('predicting...')
    for model in self.models:
      predictions[:,i] = model.predict(X)
      print(str(i+1),'/',str(self.n_out))
      i += 1
    print('Done. Took', clock()-start, 'seconds.')

    return predictions

#学習データでモデルを学習させる
model = MultiLabelLogistic(n_out = 9)  # 今回は9項目あるため, クラス数は9個に設定
model.fit(x_train, y_train)

#予測値の出力
predictions = model.predict(x_test)

np.save('predictions.npy', predictions)

# from sklearn.externals import joblib
# # 予測モデルをシリアライズ
# joblib.dump(model, 'model.pkl') 

#平均絶対誤差（Mean Absolute Error）
# def mae(y, yhat):
#   return np.mean(np.abs(y - yhat))
# print('MAE:', mae(y_test, predictions))

# single = np.where(y_test.sum(axis=1)==1)
# print('num of samples (single label):', len(single[0]))
# print('MAE:', mae(y_test[single], predictions[single]))

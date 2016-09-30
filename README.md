# compe26

https://deepanalytics.jp/compe/26?tab=detail
用のソースコード

https://deepanalytics.jp/compe/26/download
からtrainingデータとtestデータをダウンロードし，test_images, test.csv, train_images, train.csvを作成すること．

deepanalyticsフォルダは，cnnによる学習コード．

image_to_txt.py
は，ocrを使って，画像からテキストを抽出するためのコード．
train_text.pickleと，test_text.pickleに保存されている．


logistic回帰によるコード．基本的に使用しているのは，hog特徴量．
その他に，ocrなどによって抽出してきたテキストデータや，画像の位置など．

スコアは，
ベンチマーク 0.1124
hog特徴量の特徴周出をチューニングした結果 0.0230
画像のサイズを256*100にした 0.0207
ocrを入れてサイズを216*72 0.00766
ocrを入れてサイズを300*100に 0.006456
ocrを入れてサイズを750*250に 0.008288
ディープラーニング(cnn) 256サイズocrを乗っけた 0.0165

となった．
# coding=utf-8
import pandas as pd
import os
import numpy as np
from time import clock
from PIL import Image
import sys
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import ipdb
import sys
import pyocr
import pyocr.builders
import pickle

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
text = []


def load_data(file_name, img_dir):
  df = pd.read_csv(file_name)
  n = len(df)
  print('loading...')
  s = clock()

  for i, row in df.iterrows():
    f, l, t, r, b = row.filename, row.left, row.top, row.right, row.bottom
    img = Image.open(os.path.join(img_dir, f)).crop((l,t,r,b)) #項目領域画像を切り出す

    try:
      txt = tool.image_to_string(
        Image.open(os.path.join(img_dir, f)).crop((l,t,r,b)), #項目領域画像を切り出す
        lang="jpn+eng",
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
      )

    except:
      txt = ""

    if i%100==0:
      print('load:', i)

    text.append(txt.encode("utf-8"))

  print('Done. Took', clock()-s, 'seconds.')
  
load_data('train.csv', 'train_images')
with open("train_text.pickle", 'wb') as f:
  pickle.dump(text, f)

# tools = pyocr.get_available_tools()
# if len(tools) == 0:
#     print("No OCR tool found")
#     sys.exit(1)
# # The tools are returned in the recommended order of usage
# tool = tools[0]

# txt = tool.image_to_string(
#     Image.open(os.path.join('train_images', df.ix[0].filename)),
#     lang="jpn+eng",
#     builder=pyocr.builders.TextBuilder(tesseract_layout=6)
# )
# print txt

# tool.image_to_string(Image.open('/Users/MacBookAir/Downloads/ocr_sample.png'),
#                          lang="jpn",
#                          builder=pyocr.builders.TextBuilder(tesseract_layout=6))
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time

from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

start = time.time()

# Load X_train from the text file
X_train = np.loadtxt('../data_prep/prep_X_train.txt', delimiter=' ', dtype=int)

# Load X_test from the text file
X_test = np.loadtxt('../data_prep/prep_X_test.txt', delimiter=' ', dtype=int)

# Load y_train from the text file
y_train = np.loadtxt('../data_prep/prep_y_train.txt', dtype=int)

# Load y_test from the text file
y_test = np.loadtxt('../data_prep/prep_y_test.txt', dtype=int)

# 파일에서 저장된 정보 불러오기
with open('../data_prep/prep_info.txt', 'r', encoding='utf-8') as file:
    loaded_info = file.readlines()

# 불러온 정보 처리
max_len = int(loaded_info[0].split(': ')[1].strip())
vocab_size = int(loaded_info[1].split(': ')[1].strip())
stopwords_str = loaded_info[2].split(': ')[1].strip()

# 문자열로 저장된 리스트를 eval 함수를 사용하여 리스트로 변환
stopwords = eval(stopwords_str)

with open('../data_prep/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)


embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=10, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')


print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

def sentiment_predict(new_sentence):
  okt = Okt()
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')
sentiment_predict('이딴게 영화냐 ㅉㅉ')

print(f"{time.time()-start:.4f} sec")
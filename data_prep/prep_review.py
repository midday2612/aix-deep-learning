import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##1. 데이터 로드
# https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt에서 다운로드 받은 "ratings_train.txt", "ratings_test.txt" 파일 로드

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력

train_data[:5] # 상위 5개 출력

print('테스트용 리뷰 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력

test_data[:5]

##2. 데이터 정제
# document 열과 label 열의 중복을 제외한 값의 개수
train_data['document'].nunique(), train_data['label'].nunique()

# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)

print('총 샘플의 수 :',len(train_data))

train_data['label'].value_counts().plot(kind = 'bar')
plt.show()

print(train_data.groupby('label').size().reset_index(name = 'count'))
print(train_data.isnull().values.any())
print(train_data.isnull().sum())

train_data.loc[train_data.document.isnull()]
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
print(len(train_data))

#알파벳과 공백을 제외하고 모두 제거
eng_text = 'do!!! you expect... people~ to~ read~ the FAQ, etc. and actually accept hard~! atheism?@@'
print(re.sub(r'[^a-zA-Z ]', '', eng_text))

# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data[:5]
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)

print(train_data.isnull().sum())

train_data.loc[train_data.document.isnull()][:5]
train_data = train_data.dropna(how = 'any')

print(len(train_data))


test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거

print('전처리 후 테스트용 샘플의 개수 :',len(test_data))


##3. 토큰화
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

okt = Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)

X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)

print(X_train[:3])

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)

##4. 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(X_train[:3])

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

## 5. 빈 샘플 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# X_train과 y_train에서 drop_train에 해당하는 인덱스를 삭제
X_train = [X_train[i] for i in range(len(X_train)) if i not in drop_train]
y_train = [y_train[i] for i in range(len(y_train)) if i not in drop_train]

# 리스트를 다시 넘파이 배열로 변환
#X_train = np.array(X_train)
#y_train = np.array(y_train)

print(len(X_train))
print(len(y_train))

## 6. 패딩
print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 35
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


##전처리 데이터 저장
# Save X_train to a text file
with open('prep_X_train.txt', 'w', encoding='utf-8') as file:
    for seq in X_train:
        line = ' '.join(map(str, seq))
        file.write(line + '\n')

# Save X_test to a text file
with open('prep_X_test.txt', 'w', encoding='utf-8') as file:
    for seq in X_test:
        line = ' '.join(map(str, seq))
        file.write(line + '\n')

with open('prep_y_train.txt', 'w', encoding='utf-8') as file:
    for label in y_train:
        file.write(str(label) + '\n')

# Save y_test to a text file
with open('prep_y_test.txt', 'w', encoding='utf-8') as file:
    for label in y_test:
        file.write(str(label) + '\n')

with open('prep_info.txt', 'w', encoding='utf-8') as file:
    file.write(f'max_len: {max_len}\n')
    file.write(f'vocab_size: {vocab_size}\n')
    file.write(f'stopwords: {stopwords}\n')

with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle)
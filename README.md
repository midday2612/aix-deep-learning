# aix-deep-learning
aix 팀프로젝트-영화리뷰 감성 분석

## 주제
영화 리뷰 감성 분류 모델

## Members
      교육학과 김지민
      ICT융합학부 손민지
      전자공학부 윤여준

## 목표(option A)
영화 리뷰에 대한 텍스트를 분석 후 해당 리뷰가 긍정인지, 부정인지 감성분류를 수행한다.

## 동기
온라인 상의 리뷰 데이터를 효율적으로 분석하여 시청자들의 반응을 이해한다.
_________________________________________________________________________________________

## 데이터 로드 및 전처리

   1) ### 데이터 로드
      여기에 작성해주세요

   2) ### 데이터 정제하기
      여기에 작성해주세요
      
   3) ### 토큰화
      여기에 작성해주세요
      
   4) ### 정수 인코딩
      여기에 작성해주세요
      
   5) ### 빈 샘플 제거

      4번까지의 과정을 통해  빈도수가 낮은 단어들을 삭제했고, 그 결과 몇몇 샘플들이 완전히 비어있게 되었다. 빈 샘플을 제거하여 모델이 효과적으로 학습하고 일반화할 수 있도록 도와주며, 불필요한 계산을 줄여 효율성을 높일 수 있다. 
      ```python
      drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
      ```

      각 샘플들의 길이를 확인하면서 길이가 0인 샘플들을 찾아내고 인덱스(위치)를 추출한다. drop_train에는 빈 샘플들의 인덱스가 저장된다. 
       ```python
       # 빈 샘플들을 제거
       X_train = np.delete(X_train, drop_train, axis=0)
       y_train = np.delete(y_train, drop_train, axis=0)
       print(len(X_train))
       print(len(y_train))
      ```
       <img width="43" alt="image" src="https://github.com/midday2612/aix-deep-learning/assets/109676875/fa5639af-3603-4257-8909-06017e6567f9">

      빈 샘플을 제거하는 코드이다. 훈련데이터 샘플의 개수가 145881개로 줄어들었다. 빈 샘플이 잘 제거되었음을 확인할 수 있다. 
       
   6) ### 패딩
      패딩은 시퀀스 데이터의 길이를 맞춰주는 작업이다. 텍스트 데이터들의 길이가 각각 다른경우 모델이 효과적으로 학습하기 위해서는 패딩을 통해 입력 데이터의 길이를 일정하게 맞춰줄 수 있다.

      일반적으로는 가장 긴 시퀀스의 길이에 맞춰 패딩을 진행하지만 해당 데이터 셋에서는 데이터 셋 간의 길이 차이가 크기 때문에 리소스 효율성을 고려하여 적절한 패딩 길이를 찾을 필요가 있다.

      ```python
      print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
      print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
      plt.hist([len(review) for review in X_train], bins=50)
      plt.xlabel('length of samples')
      plt.ylabel('number of samples')      
      plt.show()
      ```
      코드를 실행시켜 보면 다음과 같은 그래프를 얻을 수 있다.
      <img width="477" alt="image" src="https://github.com/midday2612/aix-deep-learning/assets/109676875/5726da5e-1d70-4c3d-a25d-a8a9cf760f79">


      가장 긴 리뷰의 길이는 69다. 대다수의 샘플이 잘리지 않도록 적절한 max_len값을 선택해야 하는데, 아래의 코드를 실행하여 전체 샘플 중에서 길이가 max_len 이하인 샘플의 비율을 계산하여 얼마나 많        은 샘플이 해당 길이 이하로 잘리지 않을지 확인한다.

      max_len의 값을 바꿔보며 테스트한 결과 최종적으로 max_len = 30으로 설정하였다.  


      ```python
      max_len = 30
      below_threshold_len(max_len, X_train)
      ```
      이 코드를 실행 시키면 전체 샘플 중 길이가 30 이하인 샘플의 비율이 92.68444828318972임을 확인할 수 있다. 
      <img width="447" alt="image" src="https://github.com/midday2612/aix-deep-learning/assets/109676875/ccfd190d-f2a4-43f1-bb4b-b65a0a508a07">

      ```python
      X_train = pad_sequences(X_train, maxlen=max_len)
      X_test = pad_sequences(X_test, maxlen=max_len)
      ```
      마지막으로 패딩을 수행한다. 
      pad_sequences 함수는 텐서플로(TensorFlow) 라이브러리의 keras.preprocessing.sequence 모듈에 있다. 

      
_________________________________________________________________________________________

## LSTM을 이용한 예측 모델

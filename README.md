### 판매량을 예측하고 결과를 MySQL 데이터베이스에 저장하기  

판매량 예측은 앞으로의 투자, 홍보 전략, 사업 진행, 재고 관리 등 사실상 비즈니스 결정을 내릴 때 가장 중요한 예측 중의 하나입니다.  
여기서는 다양한 머신러닝과 시계열 모델로 판매량을 예상하고 실제 판매량과 비교하여,  
그 중에 가장 오답이 적은 그래디언트부스트 방식으로 판매량을 예측한 결과를 데이터베이스로 저장할 수 있게 하였습니다.  
구축된 데이터베이스는 공유 등을 통하여 웹 등에서 활용이 가능합니다.  

  
### 사용된 프로그램과 웹사이트  
  
설치해야 하는 프로그램 : 파이썬, MySQL  

  
### 원본 데이터 출처  
  
[Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)  
![image](https://github.com/SungUk/futuresales/assets/5809062/7ac7219c-6068-44bd-b61b-0d625f4089e1)  
  
  
### 설치 및 실행  
  
파이썬을 실행시키기 이전에 MySQL을 이용하여 현재 컴퓨터에 predictions라는 데이터베이스를 만들어줍니다.  
아니면 다른 장소와 방식으로 데이터베이스를 만든 뒤에 파이썬 코드를 알맞게 편집하시면 됩니다.  
공유를 원하시면 외부 계정을 만드신 뒤에 적절히 설정해주시면 됩니다.  

main.py 파일의 아래의 코드에서 secretnumber라고 되어있는 부분에 root 비번을 입력합니다.  
engine = create_engine("mysql+mysqlconnector://root:secretnumber@localhost:3306/predictions")  
  
main.py를 실행하시면 predictions 데이터베이스에 predictions 테이블이 생성되어 원하는 자료가 저장된 것을 확인 가능합니다.  
  
  
### 데이터 전처리와 EDA  

가격이 0이거나 다른 여러가지 이유로 비정상적인 자료는 파이썬 코드를 통해서 제거하고,  
자료의 형식이 맞지 않는 경우는 입력 파일 자체를 수정하였습니다.  
매월 판매량이 기록된 충분한 자료가 있는 경우만 선별합니다.  
가게와 월에 따른 판매량 변화를 조사하고 이를 반영하여 파생 변수 생성 등 적합한 조취를 취해주도록 코드를 짰습니다.  
![image](https://github.com/SungUk/futuresales/assets/5809062/1bb5e0ed-2e8d-4fc0-9ba5-df61daab4b79)

  


### MySQL에 데이터베이스로 저장된 이미지  

아래의 그림은 파이썬 코드를 실행하여 예측된 판매량이 자동으로 테이블을 만들어 저장이된 이미지입니다.  
  
![image](https://github.com/SungUk/futuresales/assets/5809062/0ffc4e52-d682-47ae-9611-39c600c48f5d)


    

  

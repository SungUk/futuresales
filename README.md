### 판매량을 예측하고 결과를 MySQL 데이터베이스에 저장하기  

판매량 예측은 앞으로의 투자, 홍보 전략, 사업 진행, 재고 관리 등 사실상 비즈니스 결정을 내릴 때 가장 중요한 예측 중의 하나입니다.  
여기서는 다양한 머신러닝과 시계열 모델로 판매량을 예상하고 실제 판매량과 비교하여,  
그 중에 가장 오답이 적은 그래디언트부스트 방식으로 판매량을 예측한 결과를 데이터베이스로 저장할 수 있게 하였습니다.  
구축된 데이터베이스는 공유 등을 통하여 웹 등에서 활용이 가능합니다.  

  
### 사용된 프로그램과 웹사이트  
  
설치해야 하는 프로그램 : 파이썬, MySQL  

  
### 설치 및 실행  
  
catboost 폴더는 Python으로 CatboostClassifier 모델을 이용하여 품질 예측을 한 결과를 R의 프로젝트 폴더로 출력해줍니다.  
기존의 Random Forest 모델과 DNN 모델 결과 파일도 들어가 있어서 이들의 예측 결과도 CatboostClassifier 모델과 비교하게 만들었습니다.  
먼저 main.py 코드에서 R 프로젝트 폴더를 알맞게 변경합니다.  
  
예) <C:/Users/ssw/Documents/test2/>를 <C:/Users/hong/Documents/RShiny/>로 변경.
  
RShiny 폴더는 R 프로젝트 폴더로 품질예측 결과 파일을 입력받아 웹에서 interactive하게 보여주는 역할을 합니다.  

  
작업하실 R 프로젝트 폴더에 파일들을 복사해서 붙여넣기 해주세요.  
실행하실 때는 RStudio에서 아래의 명령어를 실행하시면 됩니다.  
```
> #install.packages("shiny")
> #install.packages('rsconnect')
> #Shinyapps.io에 회원가입하시고 Account > Token에서 Show 버튼을 누르면 나타나는 문자를 아래에 양식에 맞게 입력합니다.
> rsconnect::setAccountInfo(name='xxxxxxxxxxxxxxxxxx', token='xxxxxxxxxxxxxxxxxxx', secret='xxxxxxxxxxxxxxxxxxxxxxxxx')
> #자신의 컴퓨터에서 먼저 보기
> library(shiny)
> runApp()
> #웹에 퍼블리쉬하기
> library(rsconnect)
> deployApp()
```
  
### 최종 결과물 예시  
  
결과 페이지 : 제가 만들 최종 결과물의 주소는 아래와 같습니다.  
  
https://u7s2pv-sunguk-shin.shinyapps.io/test2/  
  
  



  

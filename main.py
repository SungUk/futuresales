#pip install pandas
#pip install catboost
#pip3 install -U scikit-learn
#pip install sqlalchemy
#pip install mysql-connector-python
import numpy as np
import pandas as pd
import warnings
import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect
from sqlalchemy import exists

pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")

test = pd.read_csv('./test.csv', dtype={'ID': 'int32', 'shop_id': 'int32','item_id': 'int32'})
item_categories = pd.read_csv('./item_categories.tsv',sep='\t',dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('./items.csv', dtype={'item_name': 'str', 'item_id': 'int32','item_category_id': 'int32'})
shops = pd.read_csv('./shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('./sales_train.csv', parse_dates=['date'], dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32','item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})
sales['date']=pd.to_datetime(sales['date'],format='%d.%m.%Y')
train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)

#train 자료에서 가격이 0초과인 자료만 남김
print('Train rows: ', train.shape[0])
print('Train columns: ', train.shape[1])
train = train.query('item_price > 0')
print('Train with general prices rows: ', train.shape[0])
print('Train with general prices columns: ', train.shape[1])

#test자료에서 train에 없는 자료 제거
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
train_shop_ids = train['shop_id'].unique()
train_item_ids = train['item_id'].unique()
# test_shop_ids에 train_shop_ids에 없는 값을 찾습니다.
missing_shop_ids = set(test_shop_ids) - set(train_shop_ids)
# missing_shop_ids에 있는 값을 출력합니다.
print("test_shop_ids에 있는 train_shop_ids에 없는 값들:", missing_shop_ids)
missing_item_ids = set(test_item_ids) - set(train_item_ids)
print("test_item_ids에 있는 train_item_ids에 없는 값들:", missing_item_ids)
filtered_test = test[~test['item_id'].isin(missing_item_ids)]
filtered_test = filtered_test.iloc[:, 1:]
filtered_test.reset_index(drop=True, inplace=True)
#train 자료에서 test에 있는 자료만 남김
test_shop_ids = filtered_test['shop_id'].unique()
test_item_ids = filtered_test['item_id'].unique()
lk_train = train[train['shop_id'].isin(test_shop_ids)]
# Only items that exist in test set.
lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]
print('Data set size before leaking:', train.shape[0])
print('Data set size after leaking:', lk_train.shape[0])

# shop_id와 item_id, 두 개의 컬럼을 함께 고려하여 조합이 34번 이상 나오는 조합 찾기
combination_counts = lk_train.groupby(['shop_id', 'item_id']).size().reset_index(name='count')
valid_combinations = combination_counts[combination_counts['count'] >= 34]
filtered_train = lk_train.merge(valid_combinations, on=['shop_id', 'item_id'], how='inner').drop(columns='count')
#filtered_train.to_csv('sales_train_filtered.txt')
print('Data set size after filtering:', filtered_train.shape[0])
train_item_ids = filtered_train['item_id'].unique()
missing_item_ids = set(test_item_ids) - set(train_item_ids)
print("test_item_ids에 있는 train_item_ids에 없는 값들:", missing_item_ids)
filtered_test = test[~test['item_id'].isin(missing_item_ids)]
filtered_test = filtered_test.iloc[:, 1:]
filtered_test.reset_index(drop=True, inplace=True)
filtered_test.to_csv('test_filtered.csv')
print("==========================================================")

# Select only useful features. Time-consuming
train_monthly = filtered_train[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
# Group by month in this case "date_block_num" and aggregate features.
train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})
# Rename features.
train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']
# Build a data set with all the possible combinations of ['date_block_num','shop_id','item_id'] so we won't have missing records.
shop_ids = train_monthly['shop_id'].unique()
item_ids = train_monthly['item_id'].unique()
empty_df = []
#date_block_num : 0~33
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i, shop, item])
empty_df = pd.DataFrame(empty_df, columns=['date_block_num', 'shop_id', 'item_id'])
# Merge the train set with the complete set (missing records will be filled with 0).
train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num','shop_id','item_id'], how='left')
train_monthly.fillna(0, inplace=True)

# Extract time based features.
train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x: ((x//12) + 2013))
train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x: (x % 12))
train_monthly['month'] = train_monthly['month']+1
#train_monthly.to_csv('sales_train_monthly.txt')

#add derived variables shop_item_mean
# `train_monthly` 데이터프레임을 이용하여 shop과 item별 item_cnt 평균을 계산합니다.
shop_item_mean = train_monthly.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt'].mean()
shop_item_mean.rename(columns={'item_cnt': 'shop_item_mean'}, inplace=True)
# `shop_item_mean` 데이터프레임을 기존 `train_monthly`에 병합합니다.
train_monthly = train_monthly.merge(shop_item_mean, on=['shop_id', 'item_id'], how='left')
# 결측치(NaN)를 0으로 대체합니다.
train_monthly['shop_item_mean'].fillna(0, inplace=True)
# `train_monthly`에 shop_item_mean 열이 추가된 것을 확인합니다.
#item_cnt_before1month,item_cnt_before2month,item_cnt_before3month
train_monthly['item_cnt_before1month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)
train_monthly['item_cnt_before1month'].fillna(0, inplace=True)
train_monthly['item_cnt_before2month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-2)
train_monthly['item_cnt_before2month'].fillna(0, inplace=True)
train_monthly['item_cnt_before3month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-3)
train_monthly['item_cnt_before3month'].fillna(0, inplace=True)
#add derived variables year_mean
year_mean = train_monthly.groupby(['shop_id', 'item_id','year'], as_index=False)['item_cnt'].mean()
year_mean.rename(columns={'item_cnt': 'year_mean'}, inplace=True)
train_monthly = train_monthly.merge(year_mean, on=['shop_id', 'item_id','year'], how='left')
# 결측치(NaN)를 0으로 대체합니다.
train_monthly['year_mean'].fillna(0, inplace=True)
#add derived variables month_mean
month_mean = train_monthly.groupby(['shop_id', 'item_id','month'], as_index=False)['item_cnt'].mean()
month_mean.rename(columns={'item_cnt': 'month_mean'}, inplace=True)
train_monthly = train_monthly.merge(month_mean, on=['shop_id', 'item_id','month'], how='left')
# 결측치(NaN)를 0으로 대체합니다.
train_monthly['month_mean'].fillna(0, inplace=True)
#add derived variables item_mean
item_mean = train_monthly.groupby(['item_id'], as_index=False)['item_cnt'].mean()
item_mean.rename(columns={'item_cnt': 'item_mean'}, inplace=True)
train_monthly = train_monthly.merge(item_mean, on=['item_id'], how='left')
# 결측치(NaN)를 0으로 대체합니다.
train_monthly['item_mean'].fillna(0, inplace=True)
#add derived variables shop_mean
shop_mean = train_monthly.groupby(['shop_id'], as_index=False)['item_cnt'].mean()
shop_mean.rename(columns={'item_cnt': 'shop_mean'}, inplace=True)
train_monthly = train_monthly.merge(shop_mean, on=['shop_id'], how='left')
# 결측치(NaN)를 0으로 대체합니다.
train_monthly['shop_mean'].fillna(0, inplace=True)

X_train=train_monthly[['shop_id', 'item_id','year','month','shop_item_mean','mean_item_cnt','item_cnt_before1month','item_cnt_before2month','item_cnt_before3month','year_mean','month_mean','mean_item_price','shop_mean','item_mean']]
Y_train=train_monthly['item_cnt']
# 특정 조건 (year가 2015이고 month가 10인 경우)을 만족하는 행 선택
condition = train_monthly['date_block_num'] == 33
selected_rows = train_monthly[condition]
#print(selected_rows)
merged_test = filtered_test.merge(selected_rows, on=['shop_id', 'item_id'], how='left')
#print(merged_test)
filtered_test['item_cnt'] = merged_test['item_cnt']
filtered_test['item_cnt'].fillna(0, inplace=True)
filtered_test['shop_item_mean'] = merged_test['shop_item_mean']
filtered_test['shop_item_mean'].fillna(0, inplace=True)
filtered_test['mean_item_cnt'] = merged_test['mean_item_cnt']
filtered_test['mean_item_cnt'].fillna(0, inplace=True)
filtered_test['item_cnt_before1month'] = merged_test['item_cnt_before1month']
filtered_test['item_cnt_before1month'].fillna(0, inplace=True)
filtered_test['item_cnt_before2month'] = merged_test['item_cnt_before2month']
filtered_test['item_cnt_before2month'].fillna(0, inplace=True)
filtered_test['item_cnt_before3month'] = merged_test['item_cnt_before3month']
filtered_test['item_cnt_before3month'].fillna(0, inplace=True)
filtered_test['year_mean'] = merged_test['year_mean']
filtered_test['year_mean'].fillna(0, inplace=True)
filtered_test['month_mean'] = merged_test['month_mean']
filtered_test['month_mean'].fillna(0, inplace=True)
filtered_test['mean_item_price'] = merged_test['mean_item_price']
filtered_test['mean_item_price'].fillna(0, inplace=True)
filtered_test['shop_mean'] = merged_test['shop_mean']
filtered_test['shop_mean'].fillna(0, inplace=True)
filtered_test['item_mean'] = merged_test['item_mean']
filtered_test['item_mean'].fillna(0, inplace=True)
#print(filtered_test)
X_test=filtered_test[['shop_id','item_id','shop_item_mean','mean_item_cnt','item_cnt_before1month','item_cnt_before2month','item_cnt_before3month','year_mean','month_mean','mean_item_price','shop_mean','item_mean']]
X_test['year']=2015
X_test['month']=10
Y_test=filtered_test['item_cnt']

# 'year'과 'month' 컬럼을 문자열로 변경
X_train['year'] = X_train['year'].astype(str)
X_train['month'] = X_train['month'].astype(str)
X_test['year'] = X_test['year'].astype(str)
X_test['month'] = X_test['month'].astype(str)
# 'shop_id'와 'item_id' 컬럼도 문자열로 변경
X_train['shop_id'] = X_train['shop_id'].astype(str)
X_train['item_id'] = X_train['item_id'].astype(str)
X_test['shop_id'] = X_test['shop_id'].astype(str)
X_test['item_id'] = X_test['item_id'].astype(str)

model = CatBoostRegressor(iterations=10000,  # 반복 횟수 설정
                          depth=10,          # 트리 깊이 설정
                          learning_rate=0.2, # 학습률 설정
                          loss_function='RMSE', # 손실 함수 설정
                          random_seed=42,
                          l2_leaf_reg=0.2
                          )
model.fit(X_train,Y_train,verbose=200)
Y_pred = model.predict(X_test)
np.savetxt('test_predicted.csv',Y_pred)
rmse = mean_squared_error(Y_test, Y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)

filtered_test['Y_pred'] = Y_pred
# 결과를 텍스트 파일로 저장
filtered_test.to_csv('test_with_predictions_i10000_d10_lr0.2.csv', index=False)
print("Done.")

# CatBoost 모델에서 피처 중요도 가져오기
feature_importance = model.get_feature_importance(type="FeatureImportance")

# 피처 이름 가져오기
feature_names = X_train.columns

# 중요도를 피처 이름에 연결
feature_importance_dict = dict(zip(feature_names, feature_importance))

# 중요도를 내림차순으로 정렬
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 피처 중요도 그래프 그리기
#plt.figure(figsize=(12, 6))
#plt.barh([x[0] for x in sorted_feature_importance], [x[1] for x in sorted_feature_importance])
#plt.xlabel('Feature Importance')
#plt.title('CatBoost Feature Importance')
# 여백을 주는 코드 추가
#plt.tight_layout()
#plt.gca().invert_yaxis()  # 중요도가 높은 피처가 위에 표시되도록 y축 뒤집기
#plt.show()

#save to database
engine = create_engine("mysql+mysqlconnector://root:secretnumber@localhost:3306/predictions")
# Define the SQLAlchemy base class and create a table for predictions
Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    shop_id = Column(Integer)
    item_id = Column(Integer)
    real_item_cnt = Column(Integer)
    predicted_item_cnt = Column(Float)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

inspector = inspect(engine)
if not inspector.has_table("predictions"):
    # If the table doesn't exist, create it
    Base.metadata.create_all(engine)

# Save the predictions to the database
for index, row in filtered_test.iterrows():
    shop_id = int(row['shop_id'])
    item_id = int(row['item_id'])
    real_item_cnt = int(row['item_cnt'])
    predicted_item_cnt = float(row['Y_pred'])

    # Check if the record already exists
    record_exists = session.query(exists().where(
        Prediction.shop_id == shop_id,
        Prediction.item_id == item_id
    )).scalar()

    if record_exists:
        # Update the existing record
        session.query(Prediction).filter(
            Prediction.shop_id == shop_id,
            Prediction.item_id == item_id
        ).update({"real_item_cnt": real_item_cnt, "predicted_item_cnt": predicted_item_cnt})
    else:
        # Insert a new record
        prediction = Prediction(shop_id=shop_id, item_id=item_id, real_item_cnt=real_item_cnt, predicted_item_cnt=predicted_item_cnt)
        session.add(prediction)

# Commit the changes to the database
session.commit()
# Close the session
session.close()
print("Predictions saved to the database.")



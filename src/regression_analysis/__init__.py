# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 11:33
# @Function: 回归分析
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

final_df = pd.read_csv('../../data/processed/final_df.csv')

X=final_df.drop('Price',axis=1)
y=final_df['Price']
# print(X,y)
# print(pd.isnull(X).sum())

# 数据集划分成训练集8和测试集2
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42) # 随机种子确保每次运行代码时划分的结果都是相同的
# print(X_train,X_test,y_train,y_test)

# 计算特征与目标变量之间的互信息，查看哪些特征对于预测目标变量的影响较大
from sklearn.feature_selection import mutual_info_classif
imp = pd.DataFrame(mutual_info_classif(X,y),
                  index=X.columns)

imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)
print(666)


# 评价指标
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def predict(ml_model):
    print("Model is: ", ml_model)

    model = ml_model.fit(X_train, y_train)

    print("Training score: ", model.score(X_train, y_train))

    predictions = model.predict(X_test)
    print("Predictions: ", predictions)
    print('\n输出预测值和标签之间的r2score,MAE,MSE,RMSE等指标数据')
    r2score = r2_score(y_test, predictions)
    print("r2 score is: ", r2score)

    print('MAE:', mean_absolute_error(y_test, predictions))
    print('MSE:', mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
    print('\n\n')

    # 真实值和预测值的差值
    sns.displot(y_test - predictions)
    plt.title(f'{ml_model.__class__.__name__}')
    plt.show()


# 03支持向量回归
from sklearn.svm import SVR
predict(SVR())

# 04 决策树回归
from sklearn.tree import DecisionTreeRegressor
predict(DecisionTreeRegressor())

# 05 随机森林回归
from sklearn.ensemble import RandomForestRegressor
predict(RandomForestRegressor())




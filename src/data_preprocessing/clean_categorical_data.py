# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 12:34
# @Function: 非数值型数据清洗
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
# 设定字体为微软雅黑,解决中文汉字乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']

df = pd.read_csv('../../data/raw/after_numeric_clean.csv')
pd.set_option('display.max_columns', None)

object_columns = df.select_dtypes(include='object').columns
print("非数值型特征值：")
print(object_columns)

categorical = df.select_dtypes(include='object')
# (3)用LabelEncoder()编码将字符形式的航线转换为数值形式
# 字符串拆分的方式提取出不同的航线信息
categorical['Route1']=categorical['Route'].str.split('→').str[0]
categorical['Route2']=categorical['Route'].str.split('→').str[1]
categorical['Route3']=categorical['Route'].str.split('→').str[2]
categorical['Route4']=categorical['Route'].str.split('→').str[3]
categorical['Route5']=categorical['Route'].str.split('→').str[4]
categorical.drop('Route', axis=1, inplace=True)

# (4)因为Route3,Route4,Route5中存在的空值NaN较多，将其统一替换为None
categorical[['Route3', 'Route4', 'Route5']] = categorical[['Route3', 'Route4', 'Route5']].replace({pd.NA: None})
# print(categorical.head())

encoder = LabelEncoder()
categorical['Route1'] = encoder.fit_transform(categorical['Route1'])
categorical['Route2'] = encoder.fit_transform(categorical['Route2'])
categorical['Route3'] = encoder.fit_transform(categorical['Route3'])
categorical['Route4'] = encoder.fit_transform(categorical['Route4'])
categorical['Route5'] = encoder.fit_transform(categorical['Route5'])
# print(categorical.head())
categorical.drop('Additional_Info', axis=1, inplace=True)

# (7)对Total_Stops进行硬编码。编码方式是用字典形式赋值
stop_dict = {'non-stop': 0, '2 stops': 3, '1 stop': 1, '3 stops': 3, '4 stops': 4}
categorical['Total_Stops'] = categorical['Total_Stops'].map(stop_dict)

# (8\9\10)对"Airline"、"Source"和"Destination"列进行one-hot编码
test_data=categorical
Airline=pd.get_dummies(test_data[["Airline"]])
categorical.drop('Airline', axis=1, inplace=True)
Airline.to_csv('../../data/processed/Airline.csv', index=False)

Source=pd.get_dummies(test_data[["Source"]])
categorical.drop('Source', axis=1, inplace=True)
Source.to_csv('../../data/processed/Source.csv', index=False)

Destination=pd.get_dummies(test_data[["Destination"]])
categorical.drop('Destination', axis=1, inplace=True)
Destination.to_csv('../../data/processed/Destination.csv', index=False)
# print(categorical.head())


# 拼接final_dfz
categorical = pd.concat([categorical, Airline, Source, Destination], axis=1)
df_plane_Numeric = pd.read_csv('../../data/processed/df_plane_numeric.csv')
final_df = pd.concat([categorical, df_plane_Numeric], axis=1)
print(final_df.head(),final_df.shape)
final_df.to_csv('../../data/processed/final_df.csv', index=False)


# 处理Price离群点
plt.figure(figsize=(10, 6))
sns.boxplot(x=final_df['Price'])
plt.title('Price 特征的箱体图')
plt.savefig('final_df_price.png')
plt.show()

outliers = final_df[final_df['Price'] > 40000]

if not outliers.empty:
    median_price = final_df['Price'].median()
    final_df.loc[final_df['Price'] > 40000, 'Price'] = median_price

    final_df.to_csv('../../data/processed/final_df_replace.csv', index=False)
else:
    print("没有离群点需要替换。")

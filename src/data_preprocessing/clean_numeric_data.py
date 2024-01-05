# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 12:34
# @Function: 数值型数据清洗

import pandas as pd

df = pd.read_csv('../../data/raw/after_load.csv')
pd.set_option('display.max_columns', None)

# 1. 显示所有特征列的数据类型
print(df.dtypes)

# 2. 提取特征列“Date_of_Journey”中的年月日
df['journey_day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
df['journey_month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
df['journey_year'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.year
df.drop('Date_of_Journey', axis=1, inplace=True)
# print(df.head())

# 3. 提取特征列“Dept_time”中的小时和分钟
df['Dept_Time_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dept_Time_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
df.drop('Dep_Time', axis=1, inplace=True)
# print(df.head())

# 4. 提取特征列“Arrival_time”中小时和分钟
df['Arrival_Time_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_Time_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop('Arrival_Time', axis=1, inplace=True)
# print(df.head())

# 5. 对Duration特征列进行预处理
duration=list(df["Duration"])
# print(len(duration1))
for i in range(len(duration)):
    # 如果不包含空格，说明该元素的格式不符合"Xh Ym"的标准形式。
    if len(duration[i].split()) != 2:
        # 如果包含"h"，则表示只有小时数而没有分钟数，将其格式改为"Xh 0m"。
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        # 如果不包含"h"，则表示只有分钟数而没有小时数，将其格式改为"0h Ym"。
        else:
            duration[i] = "0h " + duration[i]

duration_hours = []
duration_minutes = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split("h")[0]))
    duration_minutes.append(int(duration[i].split("m")[0].split()[-1]))

df["dur_hour"] = duration_hours
df["dur_min"] = duration_minutes
df.drop("Duration",axis=1,inplace=True)

# print(df.dtypes)

output_path1 = '../../data/raw/after_numeric_clean.csv'
df.to_csv(output_path1, index=False)

numeric_columns = ['Price', 'journey_day', 'journey_month', 'journey_year', 'Dept_Time_hour', 'Dept_Time_min',
                   'Arrival_Time_hour', 'Arrival_Time_min', 'dur_hour', 'dur_min']

new_df = df[numeric_columns]

output_path2 = '../../data/processed/df_plane_numeric.csv'
new_df.to_csv(output_path2, index=False)


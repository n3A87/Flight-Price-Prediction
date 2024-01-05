# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 11:38
# @Function: 加载数据集，认识数据集

import pandas as pd
import matplotlib.pyplot as plt
# 设定字体为微软雅黑,解决中文汉字乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Microsoft Yahei']


def show_data():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 设置value的显示长度,更加美观
    pd.set_option('max_colwidth', 100)
    # 设置1000列时才换行,不加这行代码，结果会自动换行用\连接
    pd.set_option('display.width', 1000)
    print(df_plane.head())

    # 使用describe(), shape(), info()等函数查阅数据集的相关信息
    print("\n数据集描述信息：")
    print(df_plane.describe())
    print("\n数据集形状：")
    print(df_plane.shape)
    print("\n数据集信息：")
    print(df_plane.info())

# (2) 检查每个特征值是否存在空值
def is_null():
    print("\n有空值的特征值的空值个数：")
    null_counts = df_plane.isnull().sum()
    # 打印存在空值的特征及其空值个数
    for column, count in null_counts.items():
        if count > 0:
            print(f"特征 {column} 中有 {count} 个空值.")

# (4) 绘图显示每个特征数据值个数
def display():
    plt.figure(figsize=(10, 6))
    feature_counts = df_plane.nunique()
    # print(feature_counts)
    percentage = (feature_counts / len(df_plane)) * 100
    # print(percentage)
    percentage.plot(kind='bar', color='skyblue')
    plt.title('每个特征数据值个数百分比')
    plt.xlabel('特征')
    plt.ylabel('数据量百分比')
    plt.xticks(rotation=45)
    for i, v in enumerate(percentage):
        plt.text(i, v + 0.2, f'{v:.2f}%', ha='center', va='bottom')

    save_path = '../../Images/1_5_数据分布.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    # 显示绘图
    plt.show()


if __name__ == '__main__':
    file_path = "../../data/raw/Data_Train8.csv"
    df_plane = pd.read_csv(file_path)
    show_data()
    is_null()

    # (3) 显示存在空值的行，如果小于5个则删除
    print("\n存在空值的行：")
    null_rows = df_plane[df_plane.isnull().any(axis=1)]
    print(null_rows)
    if len(null_rows) < 5:
        df_plane = df_plane.dropna()
        print("\n空值小于5个，已删除。")
    print(pd.isnull(df_plane).sum())

    display()

    df_plane.to_csv('../../data/raw/after_load.csv', index=False)



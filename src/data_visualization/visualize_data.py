# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 12:45
# @Function: 分别可视化Airline、Total_Stops、Total_Stops、Destination和Price的关系的箱型图

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设定字体为微软雅黑,解决中文汉字乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']

df = pd.read_csv('../../data/raw/after_load.csv')


def draw_airline_price():
    plt.figure(figsize=(15,8))
    plt.xticks(fontsize=8)
    # 将航空公司名字分行显示，避免名字错叠遮挡
    df['Airline'] = df['Airline'].str.replace(' ', '\n')

    sns.boxplot(x='Airline', y='Price', data=df.sort_values('Price', ascending=False),
                palette='Set3',linewidth=1.5, width=0.7,
                flierprops = {'marker':'o',#异常值形状
                              'markerfacecolor':'red',#形状填充色
                              'color':'black',#形状外廓颜色
                             },
                capprops={'linestyle':'--','color':'blue'},
                whiskerprops={'linestyle':'--','color':'blue'})

    # 添加标签和标题
    plt.xlabel('Airline',size=15)
    plt.ylabel('Price',size=15)
    plt.title('航空公司机票价格统计信息箱型图',size=20)

    save_path = '../../Images/2_1_airline_price.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()



def draw_total_stops_price():
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Total_Stops',y='Price',data=df.sort_values('Price',ascending=False),palette='Set3')
    # 添加标签和标题
    plt.xlabel('Total_Stops',size=15)
    plt.ylabel('Price',size=15)
    plt.title('Total_Stops和Price关系箱型图',size=20)

    save_path = '../../Images/2_2_total_stops_price.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()


def draw_source_price():
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x='Source',y='Price',data=df.sort_values('Price',ascending=False), palette='Set3')
    # 添加标签和标题
    plt.xlabel('Source',size=15)
    plt.ylabel('Price',size=15)
    plt.title('Source和Price关系箱型图',size=20)

    save_path = '../../Images/2_3_source_price.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()


def draw_destination_price():
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Destination',y='Price',data=df.sort_values('Price',ascending=False),palette='Set3')
    # 添加标签和标题
    plt.xlabel('Destination',size=15)
    plt.ylabel('Price',size=15)
    plt.title('Destination和Price关系箱型图',size=20)

    save_path = '../../Images/2_4_destination_price.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    draw_airline_price()
    draw_total_stops_price()
    draw_source_price()
    draw_destination_price()
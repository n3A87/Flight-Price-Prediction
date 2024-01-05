# -*- coding: utf-8 -*-
# @Author  : 3A87
# @Time    : 2023/12/30 11:33
# @Function: 回归模型可视化
import sys
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc
import ui.new_ui as ui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # 导入 Matplotlib 库中与 Qt5 集成的模块

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# 设定字体为微软雅黑,解决中文汉字乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

class myMainWindow(qw.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('回归可视化界面')

        final_df = pd.read_csv('../../data/processed/final_df.csv')

        self.X = final_df.drop('Price', axis=1)
        self.y = final_df['Price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20,
                                                                                random_state=42)
        # 按钮
        self.SVR.clicked.connect(self.run_svr)
        self.CART.clicked.connect(self.btn_CART)
        self.RF.clicked.connect(self.btn_RF)
        self.pushButton.clicked.connect(self.show_comparison_metrics)

        # 为将matplotlib生成的图嵌入到界面中，自定义画布类
        self.canvas = MplCanvas(self.groupBox)
        self.canvas.setGeometry(qc.QRect(30, 38, 860, 420))

        self.data = {}

        # 状态提示
        self.statusBar = self.statusBar()

    def run_svr(self):
        self.statusBar.showMessage('使用支持向量回归模型训练和预测中...')
        qw.QApplication.processEvents() # 处理事件，确保消息显示 （模型的训练和预测会阻塞状态栏的消息

        # SVR model
        svr = SVR(kernel='linear') # 指定核函数的类型，线性
        svr.fit(self.X_train, self.y_train)

        y_pred = svr.predict(self.X_test)

        # 拼接model_info
        model_info = f'Model is: SVR()\n'
        ts = svr.score(self.X_train, self.y_train)
        model_info += f'Training score: {ts}\n'
        model_info += f'Predictions: {y_pred}\n'
        model_info += f'输出预测值和标签之间的r2score,MAE,MSE,RMSE等指标数据\n'
        rs = r2_score(self.y_test, y_pred)
        model_info += f'r2 score: {rs:.4f}\n'
        mae = mean_absolute_error(self.y_test, y_pred)
        model_info += f'MAE: {mae:.4f}\n'
        mse = mean_squared_error(self.y_test, y_pred)
        model_info += f'MSE: {mse:.4f}\n'
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        model_info += f'RMSE: {rmse:.4f}\n'

        self.data.update({'SVR': [ts, rs, mae, mse, rmse]})
        self.textEdit.setPlainText(model_info)
        self.canvas.plot_difference_distribution(self.y_test, y_pred)
        self.statusBar.showMessage('使用支持向量回归模型训练和预测完成!!!')

    def btn_CART(self):
        self.statusBar.showMessage('使用决策树回归模型训练和预测中...')
        qw.QApplication.processEvents()

        cart = DecisionTreeRegressor()
        cart.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = cart.predict(self.X_test)

        # # Display model information
        model_info = f'Model is: DecisionTreeRegressor()\n'
        ts = cart.score(self.X_train, self.y_train)
        model_info += f'Training score: {ts}\n'
        model_info += f'Predictions: {y_pred}\n'
        model_info += f'输出预测值和标签之间的r2score,MAE,MSE,RMSE等指标数据\n'

        rs = r2_score(self.y_test, y_pred)
        model_info += f'r2 score: {rs:.4f}\n'
        mae = mean_absolute_error(self.y_test, y_pred)
        model_info += f'MAE: {mae:.4f}\n'
        mse = mean_squared_error(self.y_test, y_pred)
        model_info += f'MSE: {mse:.4f}\n'
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        model_info += f'RMSE: {rmse:.4f}\n'

        self.data.update({'cart': [ts, rs, mae, mse, rmse]})
        self.textEdit.setPlainText(model_info)
        self.canvas.plot_difference_distribution(self.y_test, y_pred)
        self.statusBar.showMessage('使用决策树回归模型训练和预测完成!!!')

    def btn_RF(self):
        self.statusBar.showMessage('使用随机森林回归模型训练和预测中...')
        qw.QApplication.processEvents()

        rf = RandomForestRegressor()
        rf.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = rf.predict(self.X_test)

        # # Display model information
        model_info = f'Model is: RandomForestRegressor()\n'
        ts = rf.score(self.X_train, self.y_train)
        model_info += f'Training score: {ts}\n'
        model_info += f'Predictions: {y_pred}\n'

        model_info += f'输出预测值和标签之间的r2score,MAE,MSE,RMSE等指标数据\n'
        rs = r2_score(self.y_test, y_pred)
        model_info += f'r2 score: {rs:.4f}\n'
        mae = mean_absolute_error(self.y_test, y_pred)
        model_info += f'MAE: {mae:.4f}\n'
        mse = mean_squared_error(self.y_test, y_pred)
        model_info += f'MSE: {mse:.4f}\n'
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        model_info += f'RMSE: {rmse:.4f}\n'

        self.data.update({'rf': [ts, rs, mae, mse, rmse]})
        self.textEdit.setPlainText(model_info)
        self.canvas.plot_difference_distribution(self.y_test, y_pred)
        self.statusBar.showMessage('使用随机森林回归模型训练和预测完成!!!')

    def show_comparison_metrics(self):
        self.statusBar.showMessage('用户点击了`对比分析`按钮...')
        qw.QApplication.processEvents()

        df = pd.DataFrame(self.data)
        self.tableWidget.setColumnCount(len(df.columns))
        self.tableWidget.setRowCount(len(df.index))
        self.tableWidget.setHorizontalHeaderLabels(['SVR', 'DecisionTreeRegressor','RandomForestRegressor'])
        self.tableWidget.setVerticalHeaderLabels(['Training Score', 'R2 Score', 'MAE', 'MSE', 'RMSE'])

        self.tableWidget.horizontalHeader().setSectionResizeMode(qw.QHeaderView.Stretch)  # 自适应窗口宽度

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                self.tableWidget.setItem(i, j, qw.QTableWidgetItem(str(df.iloc[i, j])))

# 自定义的 Matplotlib 图形画布类
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding, qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_difference_distribution(self, y_test, predictions):
        # 清空当前图形
        self.axes.clear()

        # 计算差异
        diff = y_test - predictions

        # 使用 seaborn 绘制直方图
        sns.histplot(diff, kde=True, ax=self.axes)

        # 设置标题和标签
        self.axes.set_title("Distribution of Differences (真实值和预测值的差值)")
        self.axes.set_xlabel("Difference")
        self.axes.set_ylabel("Frequency")

        # 重新绘制
        self.draw()


if __name__ == "__main__":
    app = qw.QApplication(sys.argv)
    w1 = myMainWindow()
    w1.show()

    app.exec_()

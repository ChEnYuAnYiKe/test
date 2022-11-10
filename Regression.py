import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.datasets import load_boston  # 导入数据集
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 导入线性模型
from sklearn.metrics import r2_score  # 使用r2_score对模型评估
from sklearn.model_selection import train_test_split  # 导入数据集划分模块
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

boston = load_boston()
x = boston['data']  # 影响房价的特征信息数据
y = boston['target']  # 房价
name = boston['feature_names']

# 将数据进行拆分，一份用于训练，一份用于测试和验证,add
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# 线性回归模型
lf = Ridge()
lf.fit(x_train, y_train)  # 训练数据,学习模型参数
y_predict = lf.predict(x_test)  # 预测
print('回归的系数为:\n w = %s \n b = %s' % (lf.coef_, lf.intercept_))

# 与验证值作比较
error = mean_squared_error(y_test, y_predict).round(5)  # 平方差
score = r2_score(y_test, y_predict).round(5)  # 相关系数

# 绘制真实值和预测值的对比图
fig = plt.figure(figsize=(13, 7))
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False  # 绘图
plt.plot(range(y_test.shape[0]), y_test, color='red', linewidth=1, linestyle='-')
plt.plot(range(y_test.shape[0]), y_predict, color='blue', linewidth=1, linestyle='dashdot')
plt.legend(['真实值', '预测值'])
error = "标准差d=" + str(error) + "\n" + "相关指数R^2=" + str(score)
plt.xlabel(error, size=18, color="black")
plt.grid()
plt.show()

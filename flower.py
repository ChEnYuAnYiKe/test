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
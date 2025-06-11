#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
# import pandas as pd
# import os
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.cm as cm

# import sklearn 
# from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC,LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier

# import xgboost as xgb
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
# from sklearn.ensemble import VotingClassifier

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# import warnings
# warnings.filterwarnings("ignore")

# sns.set(style='darkgrid',font_scale=1.3)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保中文显示正常
# plt.rcParams['axes.unicode_minus'] = False   # 确保负号显示正常

# %matplotlib inline



# telcom = pd.read_csv('Customer-Churn.csv')
# print(telcom.head())

# #查找缺失值
# pd.isnull(telcom).sum()

# #查看数据类型
# telcom.info()

# # totalcharges总费用，需要转换为float类型
# telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors='coerce')
# telcom['TotalCharges'].dtypes

# # 查看缺失值
# pd.isnull(telcom["TotalCharges"]).sum() 

# # 中位数填充
# telcom.fillna({'TotalCharges':telcom['TotalCharges'].median()},inplace=True)

# #数据归一化

# #对Churn一列的值用1 0 代替，方便处理
# telcom['Churn'].replace(to_replace = "Yes",value=1,inplace=True)
# telcom['Churn'].replace(to_replace = "No",value=0,inplace=True)
# telcom['Churn'].head()

# #1、提取特征
# churn_var=telcom.iloc[:,2:20]
# churn_var.drop("PhoneService",axis=1, inplace=True)
# churn_var.head()

# #2、处理量纲差异大
# # “MonthlyCharges"、"TotalCharges"两个特征跟其他特征相比，量纲差异大
# #特征离散化 模型易于快速迭代，且模型更稳定
# #查看'MonthlyCharges'列的4分位
# churn_var['MonthlyCharges'].describe() 

# #用四分位数进行离散
# churn_var['MonthlyCharges']=pd.qcut(churn_var['MonthlyCharges'],4,labels=['1','2','3','4'])
# churn_var['MonthlyCharges'].head()

# #查看'TotalCharges'列的4分位
# churn_var['TotalCharges'].describe()

# #用四分位数进行离散 
# churn_var['TotalCharges']=pd.qcut(churn_var['TotalCharges'],4,labels=['1','2','3','4'])
# churn_var['TotalCharges'].head()

# # 3、分类数据转换成“整数编码”
# # 查看churn_var中分类变量的label标签

# #自定义函数获取分类变量中的label
# def Label(x):
#     print(x,"--" ,churn_var[x].unique()) 
# #筛选出数据类型为“object”的数据点
# df_object=churn_var.select_dtypes(['object']) 
# print(list(map(Label,df_object)))

# #通过同行百分比的“交叉分析”发现，label “No internetserive”的人数占比在以下特征
# # [OnlineSecurity，OnlineBackup，DeviceProtection，TechSupport，StreamingTV，StreamingTV]都是惊人的一致，
# # 故我们可以判断label “No internetserive”不影响流失率。
# # 因为这6项增值服务，都是需要开通“互联网服务”的基础上才享受得到的。不开通“互联网服务”视为没开通这6项增值服务，
# # 故可以将 6个特正中的“No internetserive” 并到 “No”里面。


# churn_var.replace(to_replace='No internet service',value='No',inplace=True)

# churn_var.replace(to_replace='No phone service',value='No',inplace=True)
# df_object=churn_var.select_dtypes(['object']) 
# print(list(map(Label,df_object.columns)))

# # 整数编码 sklearn中的LabelEncoder()
# def labelencode(x):
#     churn_var[x] = LabelEncoder().fit_transform(churn_var[x])
# for i in range(0,len(df_object.columns)):
#     labelencode(df_object.columns[i])
# print(list(map(Label,df_object.columns)))

# # 4、处理“样本不均衡”
# #分拆变量
# x=churn_var
# y=telcom['Churn'].values
# print('抽样前的数据特征',x.shape)
# print('抽样前的数据标签',y.shape)


# # 检查特征列数据类型
# print(x.dtypes)

# # 检查目标列类型
# print(y.dtype)

# # 确保所有特征为数值型
# x = x.apply(pd.to_numeric, errors='coerce')  # 字符串自动转NaN
# x = x.dropna(axis=1)  # 删除无法转换的列

# # 确保目标变量为整数型
# y = y.astype(int)

# from imblearn.over_sampling import SMOTE
# model_smote = SMOTE(random_state=42)  # 必须添加random_state
# x_resampled, y_resampled = model_smote.fit_resample(x, y)  # 改用fit_resample()

# # 转换回DataFrame（安全写法）
# x_resampled = pd.DataFrame(x_resampled, columns=[col for col in churn_var.columns if col != 'Churn'])

# # 分拆数据集（新版推荐参数）
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x_resampled, 
#     y_resampled,
#     test_size=0.3,
#     random_state=42,  # 避免使用0
#     stratify=y_resampled  # 新增分层抽样
# )
# print('过抽样数据特征：', x.shape,
#       '训练数据特征：',x_train.shape,
#       '测试数据特征：',x_test.shape)

# print('过抽样后数据标签：', y.shape,
#       '   训练数据标签：',y_train.shape,
#       '   测试数据标签：',y_test.shape)

# # XGB算法
# model_xgb= XGBClassifier()
# model_xgb.fit(x_train,y_train)
# from xgboost import plot_importance
# plot_importance(model_xgb,height=0.5)


# import joblib
# # 保存训练好的XGBoost模型
# joblib.dump(model_xgb, 'xgb_model.pkl')

# # 保存特征列名（用于Streamlit输入匹配）
# feature_names = x_train.columns.tolist()  # 假设x_train是训练数据
# joblib.dump(feature_names, 'feature_names.pkl')

# # 保存分类字段选项（适配你的离散化/编码逻辑）
# category_options = {
#     'Contract': ['Month-to-month', 'One year', 'Two years'],
#     'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
#     'MonthlyCharges': ['1', '2', '3', '4'],  # 你的四分位标签
#     'TotalCharges': ['1', '2', '3', '4']
# }
# joblib.dump(category_options, 'category_options.pkl')











# In[ ]:


# # -*- coding: utf-8 -*-
# """
# 电信客户流失预测模型训练脚本
# 功能：数据预处理、特征工程、模型训练与保存
# """

# # ===== 1. 库导入 =====
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# # 可视化设置
# sns.set(style='darkgrid', font_scale=1.2)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
# plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# # ===== 2. 数据加载与清洗 =====
# def load_and_clean_data(filepath):
#     """数据加载和清洗函数"""
#     print("⏳ 正在加载数据...")
#     telcom = pd.read_csv(filepath)
    
#     # 数据验证
#     print("\n✅ 数据完整性检查:")
#     print(pd.isnull(telcom).sum())
    
#     # 类型转换
#     telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors='coerce')
#     telcom.fillna({'TotalCharges': telcom['TotalCharges'].median()}, inplace=True)
    
#     # 目标变量编码
#     telcom['Churn'] = telcom['Churn'].map({'Yes': 1, 'No': 0})
    
#     print("\n📊 流失分布:")
#     print(telcom["Churn"].value_counts())
#     return telcom

# # ===== 3. 特征工程 =====
# def feature_engineering(df):
#     """特征处理函数"""
#     print("\n🔧 正在进行特征工程...")
#     churn_var = df.iloc[:, 2:20].drop("PhoneService", axis=1)
    
#     # 量纲处理（四分位离散化）
#     for col in ['MonthlyCharges', 'TotalCharges']:
#         churn_var[col] = pd.qcut(churn_var[col], 4, labels=['1','2','3','4']).astype(int)
    
#     # 分类变量处理
#     churn_var.replace({
#         'No internet service': 'No',
#         'No phone service': 'No'
#     }, inplace=True)
    
#     # 整数编码
#     cat_cols = churn_var.select_dtypes(['object']).columns
#     for col in cat_cols:
#         churn_var[col] = LabelEncoder().fit_transform(churn_var[col])
    
#     return churn_var

# # ===== 4. 主执行流程 =====
# if __name__ == "__main__":
#     # 数据加载
#     data = load_and_clean_data('Customer-Churn.csv')
    
#     # 特征工程
#     features = feature_engineering(data)
#     target = data['Churn']
    
#     # 处理样本不均衡
#     print("\n⚖️ 正在处理样本不均衡...")
#     X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(features, target)
    
#     # 数据集划分
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_resampled, y_resampled, 
#         test_size=0.3, 
#         random_state=42,
#         stratify=y_resampled
#     )
    
#     # 模型训练
#     print("\n🤖 正在训练XGBoost模型...")
#     model = XGBClassifier()
#     model.fit(X_train, y_train)
    
#     # 模型保存
#     joblib.dump(model, 'xgb_model.pkl')
#     joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
    
#     # 保存分类选项（用于Streamlit）
#     category_options = {
#         'Contract': ['Month-to-month', 'One year', 'Two years'],
#         'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
#         'MonthlyCharges': ['1', '2', '3', '4'],
#         'TotalCharges': ['1', '2', '3', '4']
#     }
#     joblib.dump(category_options, 'category_options.pkl')
    
#     print("\n🎉 模型训练完成！已保存以下文件：")
#     print("- xgb_model.pkl (训练好的模型)")
#     print("- feature_names.pkl (特征名称)")
#     print("- category_options.pkl (分类选项)")






# """
# 电信客户流失预测模型训练脚本（修正版）
# 主要修改：
# 1. 使用ColumnTransformer替代手动编码
# 2. 保存完整的预处理管道
# 3. 确保特征处理一致性
# """

# # ===== 1. 库导入 =====
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import joblib

# ===== 2. 数据预处理 =====
def preprocess_data(filepath):
    """数据加载和基础清洗"""
    df = pd.read_csv(filepath)
    
    # 处理缺失值和类型转换
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # 目标变量编码
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

# ===== 3. 特征工程与模型训练 =====
if __name__ == "__main__":
    # 数据加载
    data = preprocess_data('Customer-Churn.csv')
    
    # 定义特征列
    numerical_features = ['tenure', 'TotalCharges']
    categorical_features = ['Contract', 'PaymentMethod', 'InternetService']
    
    # 构建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # 创建完整管道（包含SMOTE和模型）
    pipeline = make_pipeline(
        preprocessor,
        SMOTE(random_state=42),
        XGBClassifier(enable_categorical=True)  # 启用分类支持
    )
    
    # 准备数据
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # 训练模型
    pipeline.fit(X, y)
    
    # 保存完整管道
    joblib.dump(pipeline, 'churn_pipeline.pkl')
    
    # 保存特征名称（用于Streamlit输入验证）
    feature_metadata = {
        'numerical': numerical_features,
        'categorical': {
            'Contract': ['Month-to-month', 'One year', 'Two years'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            'InternetService': ['DSL', 'Fiber optic', 'No']
        }
    }
    joblib.dump(feature_metadata, 'feature_metadata.pkl')
    
    print("✅ 模型训练完成！已保存：")
    print("- churn_pipeline.pkl (完整预处理+模型管道)")
    print("- feature_metadata.pkl (特征元数据)")








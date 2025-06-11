# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt

# # 加载模型和数据
# model = joblib.load('xgb_model.pkl')
# feature_names = joblib.load('feature_names.pkl')
# category_options = joblib.load('category_options.pkl')

# # 页面设置
# st.title("电信用户流失预测系统")
# st.markdown("""
# **关键发现**：  
# - 在网时长（`tenure`）是最重要特征  
# - 使用**电子支票**的用户流失率更高  
# - 月费用≥第3分位的用户风险高  
# """)

# # --- 用户输入区域 ---
# st.sidebar.header("输入特征")
# inputs = {}

# # 1. 关键特征（根据你的分析重点）
# inputs['tenure'] = st.sidebar.slider("在网时长（月）", 0, 72, 12)
# inputs['Contract'] = st.sidebar.selectbox("合同类型", category_options['Contract'])
# inputs['PaymentMethod'] = st.sidebar.selectbox("付款方式", category_options['PaymentMethod'])
# inputs['MonthlyCharges'] = st.sidebar.selectbox("月费用分位", category_options['MonthlyCharges'])

# # 2. 其他特征设默认值（非重点字段）
# for col in feature_names:
#     if col not in inputs:
#         inputs[col] = 0  # 数值型默认0，如果是分类字段需调整

# # --- 预测与结果展示 ---
# if st.sidebar.button("预测"):
#     # 转换输入格式
#     input_df = pd.DataFrame([inputs])[feature_names]
    
#     # 预测
#     proba = model.predict_proba(input_df)[0][1]
#     st.success(f"流失概率: {proba:.1%}")

#     # 高风险告警（基于你的分析）
#     if inputs['PaymentMethod'] == 'Electronic check' or inputs['MonthlyCharges'] in ['3', '4']:
#         st.error("⚠️ 高风险用户组合：电子支票 + 高月费")

#     # 特征重要性可视化
#     st.subheader("模型认为最重要的特征")
#     fig, ax = plt.subplots()
#     ax.barh(feature_names, model.feature_importances_)
#     st.pyplot(fig)





# 第二次修改
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 加载模型和特征元数据
pipeline = joblib.load('churn_pipeline.pkl')
feature_meta = joblib.load('feature_metadata.pkl')

# 页面设置
st.title("电信用户流失预测系统")

# --- 用户输入区域 ---
st.sidebar.header("输入特征")
inputs = {}

# 1. 数值型特征
for col in feature_meta['numerical']:
    if col == 'tenure':
        inputs[col] = st.sidebar.slider("在网时长（月）", 0, 72, 12)
    else:
        inputs[col] = st.sidebar.number_input(col, value=0.0)

# 2. 分类型特征
for col, options in feature_meta['categorical'].items():
    inputs[col] = st.sidebar.selectbox(col, options)

# --- 预测执行 ---
if st.sidebar.button("预测"):
    # 创建输入DataFrame（保持与训练相同结构）
    input_df = pd.DataFrame([inputs])
    
    try:
        # 使用管道进行预测（自动执行预处理）
        proba = pipeline.predict_proba(input_df)[0][1]
        st.success(f"流失概率: {proba:.1%}")
        

        # 高风险规则（根据业务需求调整）
        high_risk_conditions = [
            inputs['PaymentMethod'] == 'Electronic check',
            inputs['Contract'] == 'Month-to-month',
            inputs['tenure'] < 12
        ]
        if any(high_risk_conditions):
            st.error("⚠️ 高风险用户：建议优先跟进")
            
        # 特征重要性可视化
        st.subheader("关键影响因素")
        fig, ax = plt.subplots()
        features = pipeline.named_steps['columntransformer'].get_feature_names_out()
        importances = pipeline.named_steps['xgbclassifier'].feature_importances_
        ax.barh(features[-10:], importances[-10:])  # 显示最重要的10个特征
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        st.text("请确保所有输入字段符合要求")





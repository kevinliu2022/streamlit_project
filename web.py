import joblib
import streamlit as st
import datetime
import time
import base64
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

# st.set_page_config(page_title="Page Title", layout="wide")
# st.snow()      #打开界面飘出雪花
# st.markdown("""
#     <style>
#         .reportview-container {
#             margin-top: -2em;
#         }
#         #MainMenu {visibility: hidden;}
#         .stDeployButton {display:none;}
#         footer {visibility: hidden;}
#         #stDecoration {display:none;}
#     </style>
# """, unsafe_allow_html=True)

def process_data(data):
    features_columns = [col for col in data.columns]
    min_max_scaler = preprocessing.MinMaxScaler()
    test_data_scaler = min_max_scaler.fit_transform(data[features_columns])
    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns
    pca = PCA(n_components=16)
    new_test_pca_16 = pca.fit_transform(test_data_scaler)
    new_test_pca_16 = pd.DataFrame(new_test_pca_16)
    return new_test_pca_16

def sidebar_bg(side_bg):       #背景
    side_bg_ext = 'png'
    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )
# 调用
sidebar_bg('./data/背景2.jpg')
def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

# 调用
main_bg('./data/背景1.jpeg')

st.sidebar.subheader("项目背景:")
st.sidebar.markdown("汽在工业生产中扮演着重要的角色，无论是化工、制药、食品加工还是纺织等行业，都离不开蒸汽的支持。精准估算蒸汽用量对于企业的生产效率和成本控制具有重要意义。")
st.sidebar.markdown("数据集来源:")
st.sidebar.markdown("经脱敏后的锅炉传感器采集的数据（采集频率是分钟级别）来源于网络搜索查找。")
model_name=st.sidebar.selectbox("请选择用于预测的模型...",["随机森林","决策树","多元线性回归","GBDT","KNN"])
audio_file = open('./data/music.mp3', 'rb')
st.sidebar.audio(audio_file, format='audio/mp3')
date1 = st.sidebar.date_input("今天的日期", datetime.date(2024, 11, 4))
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

st.title("工业蒸汽量预测系统")
st.subheader("-----------集宁师范学院   电科专升本一班 萱萱-----------")

file_upload=st.file_uploader("请上传用于预测的数据....",type=['txt','csv'])

if file_upload is not None:
    global model_file
    test_data = pd.read_csv(file_upload, sep='\t', encoding='utf-8')
    st.write("你上传的数据如下：")
    st.write(test_data)
    data_processed=process_data(test_data)
    if model_name == "多元线性回归":
        model_file = "./model/steamPrediction_lr_model.pkl"
    elif model_name == "随机森林":
        model_file = "./model/steamPrediction_randomforest_model.pkl"
    elif model_name == "决策树":
        model_file = "./model/steamPrediction_decisiontree_model.pkl"
    elif model_name == "GBDT":
        model_file = "./model/steamPrediction_gbdt_model.pkl"
    elif model_name == "KNN":
        model_file = "./model/steamPrediction_knn_model.pkl"
    model=joblib.load(model_file)
    with st.spinner("正在模型预测中...."):
        start=time.time()
        prediction = model.predict(data_processed)
        end=time.time()
    st.info("模型预测成功！！！！！")
    st.write("结果如下：",prediction,":rose:总共花了{:.2f} seconds".format(end-start))
    st.balloons()#如果模型预测成功，就飘出气球




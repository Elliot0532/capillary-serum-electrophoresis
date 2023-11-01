import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array

# 创建一个映射，将数字转换为标签
label_mapping = {
    0: 'AK',
    1: 'AL',
    2: 'GK',
    3: 'GL',
    4: 'K',
    5: 'L',
    6: 'MK',
    7: 'ML',
    8: 'negative',
}

# 加载模型（这需要根据您的模型路径进行修改）
model = load_model('/Users/longzhao/Documents/Academic documents/my paper/2306Deep learning-based classification of capillary serum protein electrophoresis images and its application in immunophenotyping/my_model_0731resnet(val_0.904).h5')

# 创建一个侧边栏，用户可以通过这个侧边栏上传图像
uploaded_file = st.sidebar.file_uploader("请拖拽或选择一张图像", type=['png', 'jpg', 'jpeg'])

# 如果用户上传了图像，就进行预测
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 显示上传的图像
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # 预处理图像（这需要根据您的模型的需求进行修改）
    image = image.resize((280, 40))
    image = img_to_array(image)
    image = image / 255.0

    # 如果图像是灰度的，将它转换为 RGB
    if image.shape[-1] != 3:
        image = np.repeat(image, 3, axis=-1)

    image = np.expand_dims(image, axis=0)

    # 进行预测
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    predicted_label = label_mapping[predicted_class]

    # 显示预测结果
    st.write(f"Predicted Label: {predicted_label}")
    st.write(f"Confidence: {np.max(predictions)}")

    # 创建一个选项，让用户纠正预测的结果
    correct_label = st.selectbox('Is this correct? If not, please select the correct label:', list(label_mapping.values()))

    # 创建一个提交按钮
    if st.button('Submit Correction'):
        # 如果用户点击了提交按钮，并且他们纠正了预测的结果，保存这个信息
        if correct_label != predicted_label:
            with open('feedback.txt', 'a') as f:
                f.write(f"{uploaded_file.name},{correct_label}\n")

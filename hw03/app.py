import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os

# 页面标题
st.set_page_config(page_title="人脸识别系统", layout="centered")
st.title("👤 人脸识别系统")
st.write("上传一张图片，系统会自动检测人脸并识别（如果配置了已知人脸库）")

# 加载已知人脸库的函数
@st.cache_resource
def load_known_faces(known_dir="known_people"):
    """加载 known_people 文件夹中的所有图片，返回编码和名字列表"""
    known_encodings = []
    known_names = []
    if not os.path.exists(known_dir):
        return known_encodings, known_names
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_dir, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                # 去掉扩展名作为人名
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

# 加载已知人脸
known_encodings, known_names = load_known_faces()
if known_names:
    st.success(f"✅ 已加载 {len(known_names)} 个已知人脸：{', '.join(known_names)}")
else:
    st.info("ℹ️ 未加载已知人脸，仅进行人脸检测。")

# 图片上传组件
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

# 示例图片选项（如果 examples 文件夹存在）
example_images = []
if os.path.exists("examples"):
    example_images = [f for f in os.listdir("examples") if f.lower().endswith(('.jpg','.jpeg','.png'))]
if example_images:
    selected_example = st.selectbox("或选择一张示例图片", [""] + example_images)
    if selected_example:
        # 模拟上传文件
        with open(os.path.join("examples", selected_example), "rb") as f:
            uploaded_file = f

# 处理图片的函数
def process_image(image, known_encodings, known_names):
    # 将 PIL 图片转为 numpy 数组
    img = np.array(image.convert('RGB'))
    # 检测人脸位置
    face_locations = face_recognition.face_locations(img)
    # 提取人脸编码
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # 在图片上绘制结果
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 画矩形框
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # 识别
        name = "Unknown"
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
        # 在人脸框上方写名字
        cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return img

# 如果有上传文件，进行处理
if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_container_width=True)

    # 处理
    with st.spinner("正在检测人脸..."):
        result_img = process_image(image, known_encodings, known_names)

    # 显示结果
    st.image(result_img, caption="检测结果", use_container_width=True, channels="RGB")
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
import io

# Создание директорий, если их нет
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Загрузка модели YOLOv8
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8s.pt')
        print(f"Model loaded successfully, type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Функция для сохранения истории запросов в JSON
def save_history(date, file_name, media_type, detection_data):
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    entry = {
        "date": date,
        "file_name": file_name,
        "type": media_type,
    }
    if media_type == "image":
        entry["num_people"] = detection_data
    elif media_type == "video":
        entry["frames"] = detection_data  
    
    history.append(entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

# Функция для обнаружения людей на фото
def detect_people_image(image_path):
    if model is None:
        st.error("Model failed to load. Check the console for details.")
        return 0, None
    img = cv2.imread(image_path)
    results = model(img)
    
    # Фильтрация: оставляем только людей (класс 0 - 'person')
    filtered_results = []
    for result in results:
        boxes = result.boxes
        person_indices = [i for i, box in enumerate(boxes) if box.cls == 0]
        filtered_boxes = boxes[person_indices]
        result.boxes = filtered_boxes
        filtered_results.append(result)
    
    # Получение изображения с bounding boxes только для людей
    img_with_boxes = filtered_results[0].plot()
    
    # Подсчет людей
    num_people = len(filtered_results[0].boxes)
    
    # Сохранение результата
    output_path = 'results/output_image.jpg'
    cv2.imwrite(output_path, img_with_boxes)
    
    # Сохранение истории
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_history(date, os.path.basename(image_path), "image", num_people)
    
    return num_people, output_path

# Функция для обнаружения людей в видео
def detect_people_video(video_path):
    if model is None:
        st.error("Model failed to load. Check the console for details.")
        return 0, None
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    max_people = 0
    frame_counts = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        
        # Фильтрация: оставляем только людей (класс 0 - 'person')
        filtered_results = []
        for result in results:
            boxes = result.boxes
            person_indices = [i for i, box in enumerate(boxes) if box.cls == 0]
            filtered_boxes = boxes[person_indices]
            result.boxes = filtered_boxes
            filtered_results.append(result)
        
        # Получение кадра с bounding boxes только для людей
        frame_with_boxes = filtered_results[0].plot()
        
        # Подсчет людей в текущем кадре
        num_people = len(filtered_results[0].boxes)
        max_people = max(max_people, num_people)
        frame_counts.append({"frame": frame_number, "num_people": num_people})
        frame_number += 1
        
        # Добавление текста на кадр
        cv2.putText(frame_with_boxes, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame_with_boxes)
    
    cap.release()
    out.release()
    
    # Сохранение истории
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_history(date, os.path.basename(video_path), "video", frame_counts)
    
    return max_people, 'results/output_video.mp4'

# Функция для генерации данных отчета
def generate_report_data():
    history_file = "history.json"
    if not os.path.exists(history_file):
        return pd.DataFrame()
    
    with open(history_file, "r") as f:
        history = json.load(f)
    
    data = []
    for entry in history:
        date = entry["date"]
        file_name = entry["file_name"]
        media_type = entry["type"]
        if media_type == "image":
            num_people = entry["num_people"]
            data.append([date, file_name, media_type, "N/A", num_people])
        elif media_type == "video":
            for frame in entry["frames"]:
                frame_number = frame["frame"]
                num_people = frame["num_people"]
                data.append([date, file_name, media_type, frame_number, num_people])
    
    df = pd.DataFrame(data, columns=["Date", "File Name", "Type", "Frame Number", "Number of People"])
    return df


st.title("Подсчет гостей за столом в кафе")

# Загрузка файла (фото или видео)
uploaded_file = st.file_uploader("Загрузите фото или видео", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_path = f"uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if "image" in file_type:
        num_people, output_file = detect_people_image(file_path)
        if output_file:
            st.image(output_file, caption=f"Количество гостей: {num_people}", use_column_width=True)
            with open(output_file, "rb") as f:
                st.download_button("Скачать обработанное фото", f, file_name="result.jpg", mime="image/jpeg")
    elif "video" in file_type:
        max_people, output_file = detect_people_video(file_path)
        if output_file:
            st.video(output_file)
            st.write(f"Максимальное количество гостей: {max_people}")
            with open(output_file, "rb") as f:
                st.download_button("Скачать обработанное видео", f, file_name="result.mp4", mime="video/mp4")

# Кнопка для генерации отчета
if st.button("Генерация отчета"):
    df = generate_report_data()
    if not df.empty:
        excel_file = io.BytesIO()
        df.to_excel(excel_file, index=False)
        excel_file.seek(0)
        st.download_button(
            label="Скачать отчет Exel",
            data=excel_file,
            file_name="detection_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("Нет данных для создания отчета.")
from datetime import datetime
from flask import Flask, render_template, Response,request,redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from collections import Counter
import cv2
import csv
import pandas as pd
import os
import threading

app = Flask(__name__)

app.static_folder = 'assets'

face_classifier = cv2.CascadeClassifier(r'D:\emotionProject\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\emotionProject\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_counts = {label: 0 for label in emotion_labels}
total_emotion_counts = Counter()
most_common_emotion = None
most_common_count = 0

stop_streaming = True
streaming = False
emotion_scans = []
current_scan_info = None
face_filename = None

def analyze_emotions(frame):
    global most_common_count
    global most_common_emotion
    
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    emotion_counts = Counter()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_counts[label] += 1
            total_emotion_counts[label] += 1

            if emotion_counts[label] > most_common_count:
                most_common_count = emotion_counts[label]
                most_common_emotion = label

            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def capture_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            folder_path = 'D:\emotionProject\static\face_images'  
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            now = datetime.now()
            timestamp = now.strftime('%Y%m%d%H%M%S')
            face_filename = f'face_{timestamp}.jpg'
            face_path = os.path.join(folder_path, face_filename)
            cv2.imwrite(face_path, roi_gray)
            return face_filename
    return None

def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if stop_streaming:
            capture_face(frame)
            break

        if streaming:
            frame = analyze_emotions(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            emotion_scan_info = {'time': datetime.now(), 'emotion_results': dict(emotion_counts)}
            emotion_scans.append(emotion_scan_info)

            # Cập nhật total_emotion_counts
            total_emotion_counts.update(emotion_counts)
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/delete_scan', methods=['POST'])
def delete_scan():
    row_index = int(request.form['row_index'])  # Lấy chỉ mục hàng cần xóa
    df = pd.read_csv('emotion_scans.csv')
    
    if row_index in df.index:
        df = df.drop(row_index)  # Xóa hàng từ DataFrame
        df.to_csv('emotion_scans.csv', index=False)  # Ghi lại DataFrame vào tệp CSV
        return redirect(url_for('read_emotion_scans_from_csv'))
    else:
        return 'Error: Row index not found'

@app.route('/list')
def read_emotion_scans_from_csv():
    try:
        df = pd.read_csv('emotion_scans.csv')
        print(df)
        data = df.values.tolist()
        return render_template('list.html', data=data)
    except FileNotFoundError:
        return render_template('list.html', data="")

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global streaming
    streaming = True

    global stop_streaming
    stop_streaming = False

    global total_emotion_counts
    total_emotion_counts = Counter()

    return cam_detect()

@app.route('/stop_detection')
def stop_detection():
    global stop_streaming
    stop_streaming = True

    global streaming
    streaming = False

    global current_scan_info, face_filename
    result_text = ""
    for label, count in total_emotion_counts.items():
        result_text += f'{label}: {count} '
        
    return result_text

@app.route('/save_scan', methods=['POST'])
def save_scan():
    global face_filename
    result = request.form['result']
    now = datetime.now().strftime('%H:%M %d-%m-%Y')
    print(f"Saving result at {now}")
    with open('emotion_scans.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, result])
    return "Đã lưu."

@app.route('/cam_detect')
def cam_detect():
    return render_template('emotion-detect.html')

if __name__ == "__main__":
    app.run(debug=True,port=5000)

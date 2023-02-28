import cv2
import numpy as np
from math import pow, sqrt
import os
from flask import Flask, render_template, request, Response
app = Flask(__name__)
from playsound import playsound
import sounddevice as sd
import soundfile as sf
from pygame import mixer
mixer.init()
#sound = mixer.Sound('006.wav')
# for file upload
app.config['UPLOAD_FOLDER'] = 'upload'

labels = [line.strip() for line in open("class_labels.txt")]
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))
network = cv2.dnn.readNetFromCaffe("Yolov5-Prototype.txt", "SSD_MobileNet.caffemodel")


def livesocialdistancedetector():
    app = Flask(__name__)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        # Capture one frame after another
        ret, frame = cap.read()

        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()

        # Focal length
        F = 615

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Filtering only persons detected in the frame. Class Id of 'person' is 15
                if class_id == 15.0:
                    obj="human"
                    # Draw bounding box for the object
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                    label = "{}: {:.2f}%".format(labels[class_id], confidence* 100)
                    print("{}".format(label))

                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX + endX) / 2, 4)
                    y_mid = round((startY + endY) / 2, 4)

                    height = round(endY - startY, 4)

                    # Distance from camera based on triangle similarity
                    distance = (165 * F) / height
                    print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm, y_mid_cm, distance)

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0] - pos_dict[j][0], 2) + pow(pos_dict[i][1] - pos_dict[j][1], 2) + pow(
                        pos_dict[i][2] - pos_dict[j][2], 2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 400:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0, 0, 255)
                sound.play()
            else:
                COLOR = (0, 255, 0)
            (startX, startY, endX, endY) = coordinates[i]
           # COLOR = (int(COLOR[0]), int(COLOR[1]), int(COLOR[2]))
            print(type(frame), type(startX), type(startY), type(endX), type(endY), type(COLOR))
            cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), COLOR, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame,obj,(startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def uploadvideodistancedetector(path):
    app = Flask(__name__)
    cap = cv2.VideoCapture(path)
    obj="human"
    while cap.isOpened():

        # Capture one frame after another
        ret, frame = cap.read()

        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()

        # Focal length
        F = 615

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Filtering only persons detected in the frame. Class Id of 'person' is 15
                if (class_id == 15.00):
                    # Draw bounding box for the object
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                    label = "{}: {:.2f}%".format(labels[class_id], confidence* 100)
                    print("{}".format(label))

                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX + endX) / 2, 4)
                    y_mid = round((startY + endY) / 2, 4)

                    height = round(endY - startY, 4)

                    # Distance from camera based on triangle similarity
                    distance = (165 * F) / height
                    print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm, y_mid_cm, distance)

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0] - pos_dict[j][0], 2) + pow(pos_dict[i][1] - pos_dict[j][1], 2) + pow(
                        pos_dict[i][2] - pos_dict[j][2], 2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 400:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0, 0, 255)
            else:
                COLOR = (0, 255, 0)
            (startX, startY, endX, endY) = coordinates[i]
            #COLOR = (int(COLOR[0]), int(COLOR[1]), int(COLOR[2]))
            print(type(frame), type(startX), type(startY), type(endX), type(endY), type(COLOR))
            cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), COLOR, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame,obj,(startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin":
            return render_template('home.html')
        else:
            return render_template('index.html', message="Invalid Username or Password")

    return render_template('index.html', message="Invalid Request")


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == "POST":
        files = request.files.getlist("file")
        for file in files:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            return Response(uploadvideodistancedetector(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return render_template('index.html', message="Invalid Request")

@app.route('/live')
def live():
    return render_template('livetracking.html')

@app.route('/livetracking')
def livetracking():
    return Response(livesocialdistancedetector(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
	return render_template("index.html")

if __name__ == "__main__":
    app.run()

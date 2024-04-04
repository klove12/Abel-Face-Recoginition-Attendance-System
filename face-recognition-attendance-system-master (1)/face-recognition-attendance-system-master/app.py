import cv2
import os
from flask import Flask, request, render_template
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from twilio.rest import Client

#### Defining Flask App
app = Flask(__name__)

#### Setting timezone to East Africa Time (EAT)
east_africa = pytz.timezone('Africa/Nairobi')

#### Saving Date today in 2 different formats
datetoday = datetime.now(east_africa).strftime("%m_%d_%y")
datetoday2 = datetime.now(east_africa).strftime("%d-%B-%Y")

# Your Twilio account SID and auth token
account_sid = 'AC16f1480c918f21c853049a2830bb3d65'
auth_token = '2e4adfe7a183bcad62eb2d2f683a2a21'
client = Client(account_sid, auth_token)

# Dictionary to keep track of absent students
absent_students = {}

# Function to send messages to parents of absent students
def send_absent_messages():
    for name, roll in absent_students.items():
        message = client.messages.create(
            body=f"Your child, {name} with roll number {roll}, was absent today.",
            from_='+12512741219',
            to='+251717857977'
        )
        print(message.sid)

# Scheduler to run the send_absent_messages function at 3:00 AM every day
scheduler = BackgroundScheduler()
scheduler.add_job(send_absent_messages, 'cron', hour=20, minute=18, timezone='Africa/Nairobi')
scheduler.start()

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    dates = pd.to_datetime(df['Time']).dt.date
    times = pd.to_datetime(df['Time']).dt.time
    l = len(df)
    return names, rolls, dates, times, l

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now(east_africa).strftime("%I:%M:%S %p")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']) and datetoday not in list(pd.to_datetime(df['Time']).dt.date):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        absent_students[username] = userid  # Mark student as absent


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, dates, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, dates=dates, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2) 

#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    i = 0  # Counter for captured attendance
    while ret and i < 50:  # Capture attendance 50 times
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if faces is not None and len(faces) > 0:  # Check if faces is not None and not empty
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            i += 1
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, dates, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, dates=dates, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i = 0  # Counter for captured images
    while i < 50:  # Capture 50 photos
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if i % 1 == 0:  # Capture every frame
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, dates, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, dates=dates, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
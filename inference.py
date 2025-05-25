import tensorflow as tf
from deepface import DeepFace
from deepface.models.demography import Emotion, Age
from deepface.modules import modeling
import numpy as np
import cv2
import os
import socketio
import base64
from aiohttp import web 
import asyncio


def l2_norm(x):
    return tf.math.l2_normalize(x, axis=-1)

face_img = None
APP_INSTANCE = None

# ---------- SETUP MODEL ---------- #
tf.keras.config.enable_unsafe_deserialization()
model_path = "metric_learning_model.keras"
model = tf.keras.models.load_model(model_path, custom_objects={"l2_norm": l2_norm})
with open("threshold.txt", "r") as f:
    threshold = float(f.readline())

# ---------- SETUP SOCKETIO ---------- #
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)


# Get image function
def get_img(path):
    img = tf.io.read_file(path)
    # Get image as tensor data
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize image. Currently will be resized to (224, 224)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3]) 
    return img

# Compute metric
def compute_similarity(emb1, emb2, metric):
    if metric == "euclidean":
        return tf.norm(emb1-emb2, ord="euclidean").numpy()
    elif metric == "cosine":
        distance = tf.keras.losses.cosine_similarity(emb1, emb2)
        return -distance.numpy()[0]
    else:
        raise Exception("Metric unrecognized")

# Perform face recognition
def process_capture(frame, loop):
    # Check if faces dir is empty
    if len(os.listdir("faces")) < 1:
        cv2.imwrite(f"faces/initial.jpg", frame)
        return
    # Use temporary image to ensure consistent formatting of images when feeding into model
    cv2.imwrite("tmp.jpg", frame)
    
    # Check with each faces
    for file in os.listdir("faces"):
        # Get both images from folder and current frame
        im1 = get_img(os.path.join("faces", file))
        im2 = get_img("tmp.jpg")

        # Get embeddings
        emb1, emb2 = model(tf.stack([im1, im2]), training=False)

        # Compute distance
        distance = compute_similarity(emb1, emb2, "euclidean")
        # Check with threshold
        if distance < threshold:
            # Emit 
            asyncio.run_coroutine_threadsafe(
                sio.emit("detected", file[:-4]),
                loop
            )
            return
        else:
            asyncio.run_coroutine_threadsafe(
                sio.emit("detected", "Unknown person"),
                loop
            )
        


# Emotion and age classification model
emotion_clf = modeling.build_model(task="facial_attribute", model_name="Emotion")
age_clf = modeling.build_model(task ="facial_attribute", model_name="Age")


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()
def stream_video(loop):
    global face_img
    print("Streaming video...")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")

        # Get face coordinate, anti-spoofing flag, and confidence    
        face_objs = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False, align=True, anti_spoofing=True)

        # For all faces detected
        for obj in face_objs:
            # If no face was detected, break
            if obj["confidence"] == 0:
                break   
            # Get face coordinates
            x = obj["facial_area"]["x"]
            y = obj["facial_area"]["y"]
            w = obj["facial_area"]["w"]
            h = obj["facial_area"]["h"]

            # Crop face image
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            # Build rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            
            # Emotion classification
            emotion_predictions = emotion_clf.predict([face_img])
            dominant_emotion = Emotion.labels[np.argmax(emotion_predictions)]

            # Age prediction
            age_pred = int(age_clf.predict(np.expand_dims(face_img, axis=0)))
            
            # Anti-spoofing
            real = "Real"
            if obj["is_real"] == False:
                real = "FAKE PERSON"

            # Show as text
            cv2.putText(frame, f"{dominant_emotion}, {age_pred}: {real}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            # Perform face detection
            process_capture(face_img, loop)


        # Send frames as video data
        _, encoded_frame = cv2.imencode(".jpg", frame)
        frame_bytes = encoded_frame.tobytes()
        frame_bytes_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        asyncio.run_coroutine_threadsafe(
            sio.emit("frame", frame_bytes_b64),
            loop
        )
    
# ---------- SocketIO ---------- #
@sio.event
async def connect(sid, environ):
    print(f"Connected to {sid}")

# ----- Events ----- #
@sio.on("register")
async def on_register(data, name):
    cv2.imwrite(os.path.join("faces", name + ".jpg"), face_img)

@sio.on("stop")
async def on_stop(data):
    global APP_INSTANCE
    cam.release()
    cv2.destroyAllWindows()
    APP_INSTANCE["video_task"].cancel()
    await APP_INSTANCE.shutdown()
    await APP_INSTANCE.cleanup()
    asyncio.get_event_loop().stop()


@app.on_startup.append
async def start_background_task(app):
    global APP_INSTANCE
    try:
        loop = asyncio.get_event_loop()
    except Exception as e:
        print("Error: ", e)
    APP_INSTANCE = app
    app["video_task"] = loop.run_in_executor(None, stream_video, loop)

@app.on_shutdown.append
async def cleanup_background_task(app):
    app["video_task"].cancel()


web.run_app(app, port=5000)
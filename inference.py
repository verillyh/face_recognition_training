import tensorflow as tf
from deepface import DeepFace
from deepface.models.demography import Emotion, Gender, Age
from deepface.modules import modeling
import numpy as np
import cv2
import os

def l2_norm(x):
    return tf.math.l2_normalize(x, axis=-1)

# Load model and threshold level
tf.keras.config.enable_unsafe_deserialization()
model_path = "metric_learning_model.keras"
model = tf.keras.models.load_model(model_path, custom_objects={"l2_norm": l2_norm})
with open("threshold.txt", "r") as f:
    threshold = float(f.readline())


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
def process_capture(frame, name="Vincent"):
    # Check if faces dir is empty
    if len(os.listdir("faces")) != 1:
        cv2.imwrite(f"faces/{name}.jpg", frame)
        return
    cv2.imwrite("tmp.jpg", frame)
    
    for file in os.listdir("faces"):
        im1 = get_img(os.path.join("faces", file))
        im2 = get_img("tmp.jpg")

        emb1, emb2 = model(tf.stack([im1, im2]), training=False)

        distance = compute_similarity(emb1, emb2, "euclidean")
        if distance < threshold:
            print("A match with", file)
            return
        os.remove("tmp.jpg")



# Emotion and age classification model
emotion_clf = modeling.build_model(task="facial_attribute", model_name="Emotion")
age_clf = modeling.build_model(task ="facial_attribute", model_name="Age")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, frame = cam.read()

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


    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    if cv2.waitKey(1) ==  ord(" "):
        process_capture(face_img)
    
    

cam.release()
cv2.destroyAllWindows()
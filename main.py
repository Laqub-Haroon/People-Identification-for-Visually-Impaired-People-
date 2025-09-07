import cv2
import numpy as np
from pathlib import Path
from gtts import gTTS
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import speech_recognition as sr
import os
from collections import defaultdict
import csv

# ---------------- Voice Functions ----------------

# Voice output
def speak(text):
    print(f"🔊 Speaking: {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    # Use platform-appropriate audio play
    if os.name == "nt":  # Windows
        os.system("start response.mp3")
    else:  # Linux/Mac
        os.system("mpg123 response.mp3")

# Voice input
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ Heard: {text}")
        return text
    except sr.UnknownValueError:
        speak("Sorry, I did not catch that. Please repeat.")
        return listen()
    except sr.RequestError:
        speak("Sorry, voice recognition service is unavailable.")
        return ""

# ---------------- Similarity Function ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ---------------- Face Analysis Model (CPU only) ----------------
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))  # force CPU with ctx_id=-1

# ---------------- Variables ----------------
known_faces = defaultdict(list)  # name -> list of embeddings
person_info = {}  # name -> description
image_folder = "dataset"
description_file = "description.csv"
threshold = 0.5

# ---------------- Load Known Data ----------------
if Path(description_file).exists():
    with open(description_file, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_raw = row.get("Name")
            description = row.get("Description")
            image_file = row.get("Image File Path")

            if not (name_raw and description and image_file):
                continue

            name = name_raw.strip().lower().replace(" ", "_")
            person_info[name] = description.strip()

            img_path = os.path.join(image_folder, image_file.strip())
            if not os.path.exists(img_path):
                print(f"❌ Image not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = app.get(img)
            if not faces:
                print(f"❌ No face found in {img_path}")
                continue

            known_faces[name].append(faces[0].embedding)
            print(f"✅ Loaded: {name}")
else:
    print(f"❌ Description file not found: {description_file}")

# ---------------- Webcam Capture ----------------
print("📷 Capturing image from webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to grab frame from camera")
    exit()

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
faces = app.get(img)

if not faces:
    speak("No faces detected in the room.")
    print("❌ No faces detected.")
    exit()

recognized_people = []

# ---------------- Face Recognition ----------------
for face in faces:
    input_embedding = face.embedding
    best_match = "Unknown"
    best_score = -1

    # Compare with known embeddings
    for name, embeddings in known_faces.items():
        for emb in embeddings:
            score = cosine_similarity(input_embedding, emb)
            if score > best_score:
                best_score = score
                best_match = name

    if best_score > threshold:
        full_name = best_match.replace("_", " ").title()
        description = person_info.get(best_match, "No description available.")
        print(f"✅ {full_name} is in the room. Description: {description}")
        speak(f"{full_name} is in the room. {description}")
        recognized_people.append(full_name)
    else:
        # ---------------- Handle Unknown Person ----------------
        speak("Unknown person detected. Please tell me your name.")
        new_name_text = listen()
        new_name = new_name_text.strip().lower().replace(" ", "_")

        speak("Tell me something about yourself.")
        what_you_do = listen()

        full_name = new_name.replace("_", " ").title()
        description = f"{full_name} is known for: {what_you_do}."

        # Save face crop
        bbox = face.bbox.astype(int)
        face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        save_path = f"{image_folder}/{new_name}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        # Save embedding and description
        known_faces[new_name].append(input_embedding)
        person_info[new_name] = description

        # Save to CSV (no category column)
        file_exists = Path(description_file).exists()
        with open(description_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Description", "Image File Path"])
            writer.writerow([full_name, description, f"{new_name}.jpg"])

        speak(f"{full_name} has been saved. I will remember them.")
        print(f"✅ New person saved: {full_name}")
        recognized_people.append(full_name)

# ---------------- Final Announcement ----------------
if recognized_people:
    known_names = [name for name in recognized_people if name != "Unknown"]
    if known_names:
        joined_names = ", ".join(known_names)
        speak(f"{joined_names} {'is' if len(known_names)==1 else 'are'} in the room.")
    else:
        speak("No known person is in the room.")

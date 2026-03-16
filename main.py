import os
import torch
import cv2
import shutil
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
import yt_dlp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TORCH_CUDA_ARCH_LIST'] ='8.9' 

USE_LOCAL_VIDEO = False # True: direct file, False: Download from YT
YOUTUBE_URL = "https://youtu.be/AUs0JFY57Wc?si=Ri7RWSzLRY8XqvY6"
LOCAL_VIDEO_PATH = "video.mp4" 
REFERENCE_IMG = "reference.jpg" 
OUTPUT_DIR = "ai_perfect_shots"
FRAME_SKIP = 30 


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[SUCCESS] RTX 5080 Activated!")
    except Exception as e:
        print(f"[GPU ERROR] {e}")

def is_pixel_perfect(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                if landmark.visibility < 0.8:
                    return False
                if landmark.y < 0.7:
                    return False
        
    face_results = face_mesh.process(rgb_frame)
    if not face_results.multi_face_landmarks:
        return False
    
    for landmarks in face_results.multi_face_landmarks:
        mouth_gap = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        eye_gap = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
        
        if mouth_gap < 0.012 and eye_gap > 0.012:
            return True
    return False

def download_youtube_video(url):
    print(f"[YT-DLP] Downloading video for stable processing...")
    output_filename = "yt_download.mp4"
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_filename,
        'quiet': False,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_filename

def run_project():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    video_source = LOCAL_VIDEO_PATH if USE_LOCAL_VIDEO else download_youtube_video(YOUTUBE_URL)
    
    cap = cv2.VideoCapture(video_source)
    saved = 0
    frame_idx = 0

    print(f"\n[GPU MINING] Scanning for Studio-Quality Shots...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % FRAME_SKIP == 0:
            try:
                if is_pixel_perfect(frame):
                    verify = DeepFace.verify(
                        img1_path = frame, 
                        img2_path = REFERENCE_IMG, 
                        model_name = "Facenet", 
                        detector_backend = "retinaface",
                        enforce_detection = False,
                        silent = True
                    )

                    if verify['distance'] < 0.25:
                        saved += 1
                        print(f"[MATCH] Perfect Shot {saved} at frame {frame_idx}")
                        cv2.imwrite(f"{OUTPUT_DIR}/perfect_shot_{saved}.jpg", frame)
            except:
                pass
        frame_idx += 1
    
    cap.release()
    print(f"\n[DONE] Total {saved} Professional Shots saved.")

if __name__ == "__main__":
    run_project()
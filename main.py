import os
import torch
import cv2
import shutil
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TORCH_CUDA_ARCH_LIST'] ='8.9' 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[SUCCESS] RTX 5080 Activated!")
    except Exception as e:
        print(f"[GPU ERROR] {e}")

OUTPUT_DIR = "ai_perfect_shots"
LOCAL_VIDEO_PATH = "video.mp4" 
REFERENCE_IMG = "video.jpg" 
FRAME_SKIP = 30 

def is_pixel_perfect(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return False
    
    for landmarks in results.multi_face_landmarks:
        mouth_gap = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        eye_gap = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
        
        if mouth_gap < 0.012 and eye_gap > 0.012:
            return True
    return False

def run_project():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
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
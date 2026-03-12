import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] ='8.9' 

import cv2
import shutil
from deepface import DeepFace
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[SUCCESS] RTX  Activated! Loading models to VRAM...")
    except Exception as e:
        print(f"[GPU ERROR] {e}")
else:
    print("[WARNING] NVIDIA GPU not detected by TensorFlow. Check CUDA drivers.")


USE_LOCAL_VIDEO = True
LOCAL_VIDEO_PATH = "video.mp4" 
REFERENCE_IMG = "video.jpg" 
OUTPUT_DIR = "ai_perfect_shots"
FRAME_SKIP = 200

def setup_clean_folders():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

def run_project():
    setup_clean_folders()
    cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
    saved = 0
    frame_idx = 0

    print(f"\n[GPU MINING] Scanning video using NVIDIA Cuda Cores...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % FRAME_SKIP == 0:
            try:
                verify = DeepFace.verify(
                    img1_path = frame, 
                    img2_path = REFERENCE_IMG, 
                    model_name = "Facenet", 
                    detector_backend = "retinaface",
                    enforce_detection = False
                )

                if verify['distance'] < 0.25:
                    analysis = DeepFace.analyze(
                        img_path = frame, 
                        actions = ['emotion'], 
                        detector_backend = 'retinaface', 
                        enforce_detection = False,
                        silent = True
                    )
                    
                    if analysis[0]['dominant_emotion'] in ['neutral', 'happy']:
                        saved += 1
                        print(f"[MATCH] Shot {saved} found at frame {frame_idx}")
                        cv2.imwrite(f"{OUTPUT_DIR}/perfect_shot_{saved}.jpg", frame)
            except:
                pass
        frame_idx += 1
    
    cap.release()
    print(f"\n[DONE] Total {saved} shots saved.")

if __name__ == "__main__":
    run_project()
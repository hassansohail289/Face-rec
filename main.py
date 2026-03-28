import os
import sys
import torch
import cv2
import shutil
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TORCH_CUDA_ARCH_LIST'] ='8.9' 

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "ai_perfect_shots")
SCANNED_DIR = os.path.join(BASE_DIR, "all_scanned_frames")

FRAME_SKIP = 30 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
                if landmark.visibility < 0.8 or landmark.y < 0.7: return False
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        ls = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if abs(ls.y - rs.y) > 0.05: return False
    face_results = face_mesh.process(rgb_frame)
    if not face_results.multi_face_landmarks: return False
    for landmarks in face_results.multi_face_landmarks:
        mouth_gap = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        eye_gap = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
        if mouth_gap < 0.012 and eye_gap > 0.012: return True
    return False

def run_project():
    if len(sys.argv) < 3:
        print("\n[USAGE] FindDecentFace.exe <video_file> <reference_image>")
        print("Example: FindDecentFace.exe video.mp4 boss.jpg")
        return

    video_source = sys.argv[1]
    reference_img = sys.argv[2]

    if not os.path.exists(video_source) or not os.path.exists(reference_img):
        print("[ERROR] File not found. Check if names are correct."); return

    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] Could not open video."); return

    for folder in [OUTPUT_DIR, SCANNED_DIR]:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
    
    saved = 0
    scanned_count = 0
    frame_idx = 0
    print(f"\n[CLI START] Video: {video_source} | Fixed Skip: 1s")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % FRAME_SKIP == 0:
            scanned_count += 1
            cv2.imwrite(os.path.join(SCANNED_DIR, f"scanned_{scanned_count}.jpg"), frame)
            try:
                if is_pixel_perfect(frame):
                    verify = DeepFace.verify(
                        img1_path=frame, 
                        img2_path=reference_img, 
                        model_name="Facenet", 
                        detector_backend="retinaface", 
                        enforce_detection=False, 
                        silent=True
                    )
                    if verify['distance'] < 0.25:
                        saved += 1
                        print(f"[MATCH] Perfect Shot {saved} at frame {frame_idx}")
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"perfect_shot_{saved}.jpg"), frame)
            except: pass
        frame_idx += 1
    
    cap.release()
    print(f"\n[DONE] Total Scanned: {scanned_count} | Total Perfect: {saved}")

if __name__ == "__main__":
    run_project()
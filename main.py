import os
import sys
import torch
import cv2
import shutil
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
import yt_dlp
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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
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

def select_file(file_list, file_type):
    if not file_list:
        return None
    file_list.sort(key=lambda x: os.path.basename(x).lower())
    if len(file_list) == 1:
        print(f"[AUTO-SELECT] {file_type}: {os.path.basename(file_list[0])}")
        return os.path.normpath(os.path.abspath(file_list[0]))
    print(f"\nSelect {file_type}:")
    for i, file in enumerate(file_list):
        print(f"{i+1}. {os.path.basename(file)}")
    while True:
        try:
            choice = input(f"Enter number (1-{len(file_list)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(file_list):
                return os.path.normpath(os.path.abspath(file_list[idx]))
        except ValueError:
            pass

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
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.05:
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
    print(f"[YT-DLP] Downloading video...")
    output_filename = os.path.join(BASE_DIR, "yt_download.mp4")
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
    return os.path.normpath(os.path.abspath(output_filename))

def run_project():
    print("\n--- AI SYSTEM STARTUP (AUDIT MODE) ---")
    print("1. Local Video")
    print("2. YouTube Video")
    mode = input("Select Mode (1 or 2): ")

    video_source = ""
    if mode == '2':
        url = input("Enter YouTube URL: ")
        video_source = download_youtube_video(url)
    else:
        v_exts = ['*.mp4', '*.mkv', '*.avi', '*.mov']
        v_files = []
        for ext in v_exts:
            v_files.extend(glob.glob(os.path.join(BASE_DIR, ext)))
        video_source = select_file(v_files, "Video")

    if not video_source or not os.path.exists(video_source):
        print("No video found.")
        return

    i_exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    i_files = []
    for ext in i_exts:
        i_files.extend(glob.glob(os.path.join(BASE_DIR, ext)))
    reference_img = select_file(i_files, "Reference Image")

    if not reference_img or not os.path.exists(reference_img):
        print("No image found.")
        return

    for folder in [OUTPUT_DIR, SCANNED_DIR]:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
    
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source.")
        return

    saved = 0
    scanned_count = 0
    frame_idx = 0

    print(f"\n[GPU MINING] Scanning and Auditing Frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % FRAME_SKIP == 0:
            scanned_count += 1
            cv2.imwrite(os.path.join(SCANNED_DIR, f"scanned_frame_{scanned_count}.jpg"), frame)
            
            try:
                if is_pixel_perfect(frame):
                    verify = DeepFace.verify(
                        img1_path = frame, 
                        img2_path = reference_img, 
                        model_name = "Facenet", 
                        detector_backend = "retinaface",
                        enforce_detection = False,
                        silent = True
                    )

                    if verify['distance'] < 0.25:
                        saved += 1
                        print(f"[MATCH] Perfect Shot {saved} at frame {frame_idx}")
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"perfect_shot_{saved}.jpg"), frame)
            except:
                pass
        
        frame_idx += 1

    
    cap.release()
    print(f"\n[DONE] Total Scanned: {scanned_count} | Total Perfect: {saved}")
    
    os.startfile(BASE_DIR)
    input("Press Enter to exit...")

if __name__ == "__main__":
    run_project()
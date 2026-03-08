import cv2
import os
import shutil
from deepface import DeepFace
import yt_dlp

# --- CONFIG ---
VIDEO_URL = "https://youtu.be/RsAKKF2-_Kg?si=1L8rYJZMBKMB5tSw" 
REFERENCE_IMG = "sundar.webp" 
OUTPUT_DIR = "ai_perfect_shots"
TEMP_VIDEO = "temp_interview.mp4"

def setup_clean_folders():
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Deleting old shots from '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    if os.path.exists(TEMP_VIDEO):
        print("[INFO] Removing old temporary video...")
        os.remove(TEMP_VIDEO)

def download_video(url):
    print("[INFO] Downloading new video... (Progress will be shown below)")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': TEMP_VIDEO,
        'quiet': False,
        'no_warnings': True,
        'download_ranges': lambda info_dict, ydl: [{'start_time': 680, 'end_time': 780}], 
        'force_keyframes_at_cuts': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def run_project():
    setup_clean_folders()
    
    download_video(VIDEO_URL)

    if not os.path.exists(TEMP_VIDEO):
        print("[ERROR] Video download failed!")
        return

    cap = cv2.VideoCapture(TEMP_VIDEO)
    saved = 0
    frame_idx = 0

    print("\n[INFO] Starting AI Scanning (Facenet + OpenCV)...")

    while cap.isOpened() and saved < 5:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % 80 == 0:
            try:
                result = DeepFace.verify(
                    img1_path = frame, 
                    img2_path = REFERENCE_IMG, 
                    model_name = "Facenet", 
                    detector_backend = "opencv",
                    distance_metric = "cosine",
                    enforce_detection = False, 
                    align = False
                )
                
                dist = result['distance']
                if dist < 0.25: 
                    print(f"[MATCH] Frame {frame_idx}: Score {dist:.4f}")
                    cv2.imwrite(f"{OUTPUT_DIR}/shot_{saved+1}.jpg", frame)
                    saved += 1
            except:
                pass
        
        frame_idx += 1
        if frame_idx % 800 == 0:
            print(f"[PROGRESS] Scanning frames... {frame_idx}")

    cap.release()
    print(f"\n[DONE] New shots have been saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    run_project()
import cv2
import os
import shutil
from deepface import DeepFace
import yt_dlp


USE_LOCAL_VIDEO = False 
LOCAL_VIDEO_PATH = "my_video.mp4" 

VIDEO_URL = "https://youtu.be/QE3QwTA5ujE?si=JiiX0X6aQpYZtOEt" 
REFERENCE_IMG = "reference.jpg" 
OUTPUT_DIR = "ai_perfect_shots"
TEMP_VIDEO = "temp_interview.mp4"

def setup_clean_folders():
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Deleting old shots from '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if not USE_LOCAL_VIDEO and os.path.exists(TEMP_VIDEO):
        print("[INFO] Removing old temporary video...")
        os.remove(TEMP_VIDEO)

def download_video(url):
    print("[INFO] Downloading new video segment from YouTube...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': TEMP_VIDEO,
        'quiet': False,
        'no_warnings': True,
        'download_ranges': lambda info_dict, ydl: [{'start_time': 300, 'end_time': 360}], 
        'force_keyframes_at_cuts': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def run_project():
    setup_clean_folders()
    
    
    video_source = ""
    if USE_LOCAL_VIDEO:
        if os.path.exists(LOCAL_VIDEO_PATH):
            video_source = LOCAL_VIDEO_PATH
            print(f"[INFO] Using local video: {video_source}")
        else:
            print(f"[ERROR] Local video '{LOCAL_VIDEO_PATH}' not found!")
            return
    else:
        download_video(VIDEO_URL)
        video_source = TEMP_VIDEO

    cap = cv2.VideoCapture(video_source)
    saved = 0
    frame_idx = 0

    print(f"\n[INFO] Starting AI Scanning on: {video_source}")

    while cap.isOpened() and saved < 10:
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
                
                if result['distance'] < 0.25: 
                    
                    analysis = DeepFace.analyze(
                        img_path = frame, 
                        actions = ['emotion'], 
                        detector_backend = 'opencv', 
                        enforce_detection = False,
                        silent = True
                    )
                    
                    emotion = analysis[0]['dominant_emotion']
                    
                    if emotion in ['neutral', 'happy']:
                       
                        print(f"[PERFECT MATCH] Frame {frame_idx}: Score {result['distance']:.4f} | Emotion: {emotion}")
                        cv2.imwrite(f"{OUTPUT_DIR}/perfect_shot_{saved+1}.jpg", frame)
                        saved += 1
            except:
                pass
        
        frame_idx += 1
        if frame_idx % 400 == 0:
            print(f"[PROGRESS] Scanning frames... {frame_idx}")

    cap.release()
    print(f"\n[DONE] Professional FULL shots saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    run_project()
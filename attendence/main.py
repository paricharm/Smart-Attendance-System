"""
Smart Attendance System - Main Application
Integrated version for VSCode with webcam support
"""

import os
import time
import uuid
import glob
import pickle
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
import faiss

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Paths
    ROOT = os.path.dirname(os.path.abspath(__file__))
    YOLO_WEIGHTS = os.path.join(ROOT, "best.pt")
    ENROLLMENT_DIR = os.path.join(ROOT, "enrollment")
    DB_PATH = os.path.join(ROOT, "attendance.db")
    FAISS_INDEX_PATH = os.path.join(ROOT, "faiss_index.bin")
    FAISS_MAP_PATH = os.path.join(ROOT, "faiss_map.pkl")
    SNAPSHOT_DIR = Path(ROOT) / "snapshots"
    OUTPUT_DIR = Path(ROOT) / "output"
    
    # Parameters
    EMBEDDING_SIZE = 512
    COSINE_SIM_THRESHOLD = 0.65
    ATTENDANCE_DEBOUNCE_SECONDS = 300  # 5 minutes
    MAX_PROTOTYPES_PER_USER = 10
    
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

# Create directories
Config.SNAPSHOT_DIR.mkdir(exist_ok=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True)
os.makedirs(Config.ENROLLMENT_DIR, exist_ok=True)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading YOLO model...")
        if not os.path.exists(Config.YOLO_WEIGHTS):
            raise FileNotFoundError(f"YOLO weights not found: {Config.YOLO_WEIGHTS}")
        self.yolo = YOLO(Config.YOLO_WEIGHTS)
        self.yolo.fuse()
        
        print("Loading FaceNet model...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        print("Loading MTCNN...")
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device, post_process=False)
        
        print("✅ All models loaded\n")


# ============================================================
# DATABASE MANAGEMENT
# ============================================================

class DatabaseManager:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self.conn = self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = conn.cursor()
        
        # Users table
        c.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TEXT
                    )""")
        
        # Attendance table
        c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        name TEXT,
                        timestamp TEXT,
                        cam_id TEXT,
                        confidence REAL,
                        snapshot_path TEXT
                    )""")
        
        # Last seen table
        c.execute("""CREATE TABLE IF NOT EXISTS last_seen (
                        user_id TEXT PRIMARY KEY,
                        last_ts REAL
                    )""")
        
        conn.commit()
        return conn
    
    def add_user(self, user_id, name):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                  (user_id, name, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()
    
    def get_last_seen(self, user_id):
        c = self.conn.cursor()
        c.execute("SELECT last_ts FROM last_seen WHERE user_id=?", (user_id,))
        row = c.fetchone()
        return row[0] if row else None
    
    def update_last_seen(self, user_id, timestamp):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO last_seen (user_id, last_ts) VALUES (?, ?)",
                  (user_id, timestamp))
        self.conn.commit()
    
    def log_attendance(self, user_id, name, cam_id, confidence, snapshot_path):
        # Check duplicate
        last_ts = self.get_last_seen(user_id)
        current_ts = time.time()
        
        if last_ts is not None:
            if current_ts - last_ts <= Config.ATTENDANCE_DEBOUNCE_SECONDS:
                return False
        
        # Log attendance
        c = self.conn.cursor()
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        c.execute("""INSERT INTO attendance (id, user_id, name, timestamp, cam_id, confidence, snapshot_path)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (entry_id, user_id, name, timestamp, cam_id, float(confidence), str(snapshot_path)))
        
        self.update_last_seen(user_id, current_ts)
        self.conn.commit()
        return True
    
    def get_attendance_records(self):
        import pandas as pd
        query = """SELECT id, user_id, name, timestamp, cam_id, confidence, snapshot_path 
                   FROM attendance ORDER BY timestamp DESC"""
        return pd.read_sql_query(query, self.conn)


# ============================================================
# FACE RECOGNITION ENGINE
# ============================================================

class FaceRecognitionEngine:
    def __init__(self, model_manager, db_manager):
        self.models = model_manager
        self.db = db_manager
        
        # FAISS index
        self.faiss_index = faiss.IndexFlatIP(Config.EMBEDDING_SIZE)
        self.faiss_id_map = []
        self.faiss_name_map = []
    
    def get_face_embedding(self, face_rgb):
        """Extract face embedding"""
        aligned = None
        
        try:
            aligned = self.models.mtcnn(face_rgb)
        except:
            aligned = None
        
        try:
            if aligned is None:
                pil_img = Image.fromarray(face_rgb)
                pil_img = pil_img.resize((160, 160), Image.BILINEAR)
                arr = np.array(pil_img).astype(np.float32)
                arr = (arr - 127.5) / 128.0
                aligned = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            
            if aligned.dim() == 3:
                aligned = aligned.unsqueeze(0)
            
            aligned = aligned.to(self.models.device)
            
            with torch.no_grad():
                emb = self.models.facenet(aligned).cpu().numpy().flatten()
            
            return emb.astype('float32')
        except:
            return np.zeros(Config.EMBEDDING_SIZE, dtype='float32')
    
    def normalize_embeddings(self, emb_array):
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_array /= norms
    
    def enroll_from_folder(self, enroll_dir=Config.ENROLLMENT_DIR, min_conf=0.3):
        """Enroll users from folder structure"""
        if not os.path.exists(enroll_dir):
            print(f"❌ Enrollment folder not found: {enroll_dir}")
            return
        
        user_folders = [d for d in os.listdir(enroll_dir) 
                        if os.path.isdir(os.path.join(enroll_dir, d))]
        
        if not user_folders:
            print("❌ No user folders found")
            return
        
        print(f"\n{'='*60}")
        print(f"ENROLLING USERS")
        print(f"{'='*60}")
        print(f"Found {len(user_folders)} users: {user_folders}\n")
        
        for user_name in tqdm(user_folders, desc="Enrolling"):
            user_path = os.path.join(enroll_dir, user_name)
            user_id = str(uuid.uuid4())
            
            self.db.add_user(user_id, user_name)
            
            img_files = glob.glob(f"{user_path}/*.jpg") + glob.glob(f"{user_path}/*.png")
            embeddings = []
            
            for img_path in img_files[:Config.MAX_PROTOTYPES_PER_USER]:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                results = self.models.yolo(img_rgb, verbose=False)
                boxes = results[0].boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                bb = boxes.xyxy[0].cpu().numpy().astype(int)
                conf = boxes.conf[0].cpu().item()
                
                if conf < min_conf:
                    continue
                
                x1, y1, x2, y2 = bb
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(img_rgb.shape[1], x2)
                y2 = min(img_rgb.shape[0], y2)
                
                face = img_rgb[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                emb = self.get_face_embedding(face)
                embeddings.append(emb)
            
            if embeddings:
                emb_array = np.stack(embeddings, axis=0)
                self.normalize_embeddings(emb_array)
                self.faiss_index.add(emb_array)
                
                for _ in embeddings:
                    self.faiss_id_map.append(user_id)
                    self.faiss_name_map.append(user_name)
                
                print(f"  ✅ {user_name}: {len(embeddings)} embeddings")
        
        print(f"\n✅ Enrollment complete! Total: {self.faiss_index.ntotal} embeddings\n")
        self.save_faiss()
    
    def save_faiss(self):
        """Save FAISS index"""
        faiss.write_index(self.faiss_index, Config.FAISS_INDEX_PATH)
        with open(Config.FAISS_MAP_PATH, 'wb') as f:
            pickle.dump({'id_map': self.faiss_id_map, 'name_map': self.faiss_name_map}, f)
        print(f"✅ FAISS index saved: {self.faiss_index.ntotal} embeddings")
    
    def load_faiss(self):
        """Load FAISS index"""
        if os.path.exists(Config.FAISS_INDEX_PATH) and os.path.exists(Config.FAISS_MAP_PATH):
            self.faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
            with open(Config.FAISS_MAP_PATH, 'rb') as f:
                data = pickle.load(f)
                self.faiss_id_map = data['id_map']
                self.faiss_name_map = data['name_map']
            print(f"✅ FAISS index loaded: {self.faiss_index.ntotal} embeddings")
            return True
        return False
    
    def process_frame(self, frame_bgr, cam_id="webcam", min_conf=0.3):
        """Process frame and detect faces"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        results = self.models.yolo(frame_rgb, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None or len(boxes) == 0:
            return []
        
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        
        faces = []
        boxes_for_draw = []
        
        for bb, cf in zip(xyxy, confs):
            if cf < min_conf:
                continue
            
            x1, y1, x2, y2 = bb.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame_rgb.shape[1] - 1, x2)
            y2 = min(frame_rgb.shape[0] - 1, y2)
            
            face = frame_rgb[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
            
            faces.append(face)
            boxes_for_draw.append([x1, y1, x2, y2, cf])
        
        if len(faces) == 0:
            return []
        
        # Get embeddings
        embs = [self.get_face_embedding(f) for f in faces]
        emb_batch = np.stack(embs, axis=0).astype('float32')
        self.normalize_embeddings(emb_batch)
        
        # Search
        if self.faiss_index.ntotal == 0:
            return [("Unknown", 0.0, box[:4]) for box in boxes_for_draw]
        
        D, I = self.faiss_index.search(emb_batch, 1)
        
        results = []
        
        for i, box in enumerate(boxes_for_draw):
            similarity = float(D[i][0])
            idx = I[i][0]
            
            if idx < len(self.faiss_id_map) and similarity >= Config.COSINE_SIM_THRESHOLD:
                user_id = self.faiss_id_map[idx]
                name = self.faiss_name_map[idx]
                
                # Save snapshot
                face_bgr = cv2.cvtColor(faces[i], cv2.COLOR_RGB2BGR)
                snap_path = Config.SNAPSHOT_DIR / f"{name}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(snap_path), face_bgr)
                
                # Log attendance
                logged = self.db.log_attendance(user_id, name, cam_id, similarity, snap_path)
                
                if logged:
                    print(f"✅ Attendance: {name} (sim={similarity:.3f})")
                
                display_name = name
            else:
                display_name = "Unknown"
                similarity = 0.0
            
            results.append((display_name, similarity, box[:4]))
        
        return results


# ============================================================
# MAIN APPLICATION
# ============================================================

class AttendanceApp:
    def __init__(self):
        print("\n" + "="*60)
        print("SMART ATTENDANCE SYSTEM")
        print("="*60 + "\n")
        
        self.models = ModelManager()
        self.db = DatabaseManager()
        self.engine = FaceRecognitionEngine(self.models, self.db)
    
    def run_live_attendance(self, duration_seconds=None):
        """Run live attendance from webcam"""
        print("\n" + "="*60)
        print("LIVE ATTENDANCE MODE")
        print("="*60)
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(Config.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration
                if duration_seconds:
                    if time.time() - start_time >= duration_seconds:
                        break
                
                # Process every 3rd frame
                if frame_count % 3 == 0:
                    results = self.engine.process_frame(frame, cam_id="live_webcam")
                    
                    # Draw results
                    for name, conf, (x1, y1, x2, y2) in results:
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        label = f"{name} ({conf:.2f})" if name != "Unknown" else "Unknown"
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                frame_count += 1
                
                # Show frame
                cv2.imshow("Live Attendance", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Live attendance ended\n")
    
    def menu(self):
        """Interactive menu"""
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Enroll users from folder")
            print("2. Run live attendance")
            print("3. View attendance records")
            print("4. Register new person (Face Registration)")
            print("5. Exit")
            print("="*60)
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.engine.enroll_from_folder()
            
            elif choice == '2':
                if self.engine.faiss_index.ntotal == 0:
                    print("\n❌ No users enrolled. Please enroll users first.\n")
                    continue
                self.run_live_attendance()
            
            elif choice == '3':
                df = self.db.get_attendance_records()
                if len(df) > 0:
                    print("\n" + "="*60)
                    print("ATTENDANCE RECORDS")
                    print("="*60)
                    print(df[['name', 'timestamp', 'confidence', 'cam_id']].head(20))
                    print(f"\nTotal records: {len(df)}")
                else:
                    print("\n❌ No attendance records found\n")
            
            elif choice == '4':
                print("\n⚠️  Please run 'face_registration.py' separately")
                print("   python face_registration.py\n")
            
            elif choice == '5':
                print("\n✅ Goodbye!\n")
                break
            
            else:
                print("\n❌ Invalid choice\n")


if __name__ == "__main__":
    app = AttendanceApp()
    app.menu()
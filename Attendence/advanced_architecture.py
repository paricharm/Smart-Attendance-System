"""
Smart Attendance System - Main Application (Integrated with Advanced Architecture)
Direct run version with anti-overfitting features
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
from sklearn.decomposition import PCA
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Paths
    ROOT = os.path.dirname(os.path.abspath(__file__))
    YOLO_WEIGHTS = os.path.join(ROOT, "best.pt")
    ENROLLMENT_DIR = os.path.join(ROOT, "enrollment")
    DB_PATH = os.path.join(ROOT, "attendance.db")
    MODEL_PATH = os.path.join(ROOT, "face_model.pkl")
    SNAPSHOT_DIR = Path(ROOT) / "snapshots"
    OUTPUT_DIR = Path(ROOT) / "output"
    
    # Advanced Architecture Parameters
    EMBEDDING_SIZE = 512
    REDUCED_DIM = 128  # PCA reduction
    
    # Recognition Thresholds
    COSINE_THRESHOLD = 0.70
    L2_THRESHOLD = 0.85
    ENSEMBLE_THRESHOLD = 0.65
    MARGIN_THRESHOLD = 0.10  # Between 1st and 2nd place
    
    # Attendance
    ATTENDANCE_DEBOUNCE_SECONDS = 300  # 5 minutes
    
    # Quality Control
    MIN_FACE_CONFIDENCE = 0.6
    MIN_FACE_SIZE = 80
    MIN_BLUR_SCORE = 100
    
    # Multi-frame Voting
    USE_MULTI_FRAME_VOTING = True
    CONFIRMATION_FRAMES = 3
    
    # Enrollment
    MAX_IMAGES_PER_USER = 20
    MIN_IMAGES_PER_USER = 3

# Create directories
Config.SNAPSHOT_DIR.mkdir(exist_ok=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True)
os.makedirs(Config.ENROLLMENT_DIR, exist_ok=True)


# ============================================================
# ADVANCED FACE RECOGNITION ENGINE
# ============================================================

class AdvancedFaceRecognition:
    """
    Advanced architecture with anti-overfitting features:
    - PCA dimensionality reduction
    - Ensemble methods (cosine + L2)
    - Template averaging with outlier removal
    - Multi-metric validation
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load models
        print("📦 Loading FaceNet model...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        print("📦 Loading MTCNN...")
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device, post_process=False)
        
        # PCA for dimensionality reduction
        self.pca = None
        
        # Dual FAISS indices
        self.faiss_cosine = faiss.IndexFlatIP(Config.REDUCED_DIM)
        self.faiss_l2 = faiss.IndexFlatL2(Config.REDUCED_DIM)
        
        # User data
        self.user_templates = {}   # user_id -> averaged template
        self.user_names = {}       # user_id -> name
        self.faiss_id_map = []
        
        # Multi-frame voting
        self.frame_counter = 0
        self.face_history = {}
        
        print("✅ Advanced recognition engine initialized\n")
    
    def extract_embedding(self, face_rgb):
        """Extract 512-d embedding from face"""
        try:
            aligned = self.mtcnn(face_rgb)
            
            if aligned is None:
                pil_img = Image.fromarray(face_rgb)
                pil_img = pil_img.resize((160, 160), Image.BILINEAR)
                arr = np.array(pil_img).astype(np.float32)
                arr = (arr - 127.5) / 128.0
                aligned = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            
            if aligned.dim() == 3:
                aligned = aligned.unsqueeze(0)
            
            aligned = aligned.to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet(aligned).cpu().numpy().flatten()
            
            return embedding.astype('float32')
        
        except Exception as e:
            return None
    
    def check_face_quality(self, face_rgb):
        """Check face image quality (blur detection)"""
        if face_rgb.size == 0:
            return False, 0
        
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return blur_score > Config.MIN_BLUR_SCORE, blur_score
    
    def create_template(self, embeddings):
        """Create robust template by averaging with outlier removal"""
        if len(embeddings) <= 3:
            return np.mean(embeddings, axis=0)
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Remove outliers (keep 80% closest to centroid)
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        threshold_idx = int(len(distances) * 0.8)
        sorted_indices = np.argsort(distances)
        inlier_indices = sorted_indices[:threshold_idx]
        
        # Average inliers
        template = np.mean(embeddings[inlier_indices], axis=0)
        template = template / (np.linalg.norm(template) + 1e-8)
        
        return template
    
    def enroll_user(self, user_id, name, face_images):
        """Enroll user with quality checks"""
        embeddings = []
        
        for img in face_images[:Config.MAX_IMAGES_PER_USER]:
            # Quality check
            is_good, blur_score = self.check_face_quality(img)
            if not is_good:
                continue
            
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) < Config.MIN_IMAGES_PER_USER:
            return False, f"Only {len(embeddings)} good quality images"
        
        # Create template
        template = self.create_template(embeddings)
        self.user_templates[user_id] = template
        self.user_names[user_id] = name
        
        return True, f"{len(embeddings)} embeddings"
    
    def build_index(self):
        """Build FAISS indices with PCA"""
        if len(self.user_templates) == 0:
            print("❌ No users enrolled")
            return False
        
        print("\n" + "="*60)
        print("🔨 BUILDING OPTIMIZED INDEX")
        print("="*60)
        
        # Collect templates
        templates = []
        user_ids = []
        
        for user_id, template in self.user_templates.items():
            templates.append(template)
            user_ids.append(user_id)
        
        templates = np.array(templates).astype('float32')
        
        # Apply PCA
        print(f"📊 Original dimension: {templates.shape[1]}")
        self.pca = PCA(n_components=Config.REDUCED_DIM, whiten=True)
        templates_reduced = self.pca.fit_transform(templates).astype('float32')
        print(f"📊 Reduced dimension: {templates_reduced.shape[1]}")
        print(f"📊 Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Normalize for cosine
        templates_normalized = templates_reduced.copy()
        norms = np.linalg.norm(templates_normalized, axis=1, keepdims=True)
        templates_normalized = templates_normalized / (norms + 1e-8)
        
        # Build indices
        self.faiss_cosine = faiss.IndexFlatIP(Config.REDUCED_DIM)
        self.faiss_cosine.add(templates_normalized)
        
        self.faiss_l2 = faiss.IndexFlatL2(Config.REDUCED_DIM)
        self.faiss_l2.add(templates_reduced)
        
        self.faiss_id_map = user_ids
        
        print(f"✅ Index built with {len(user_ids)} users")
        print("="*60 + "\n")
        return True
    
    def recognize(self, face_rgb):
        """
        Recognize face with ensemble validation
        Returns: (user_id, name, confidence, status) or None
        """
        # Quality check
        is_good, blur_score = self.check_face_quality(face_rgb)
        if not is_good:
            return None
        
        # Extract embedding
        embedding = self.extract_embedding(face_rgb)
        if embedding is None:
            return None
        
        # Apply PCA
        if self.pca is None:
            return None
        
        embedding_reduced = self.pca.transform(embedding.reshape(1, -1)).astype('float32')
        
        # Normalize for cosine
        embedding_normalized = embedding_reduced.copy()
        norm = np.linalg.norm(embedding_normalized)
        embedding_normalized = embedding_normalized / (norm + 1e-8)
        
        # Search both indices (top 5)
        k = min(5, self.faiss_cosine.ntotal)
        
        D_cos, I_cos = self.faiss_cosine.search(embedding_normalized, k)
        D_l2, I_l2 = self.faiss_l2.search(embedding_reduced, k)
        
        # Get top matches
        top_cos_score = float(D_cos[0][0])
        top_cos_idx = int(I_cos[0][0])
        
        top_l2_dist = float(D_l2[0][0])
        top_l2_idx = int(I_l2[0][0])
        top_l2_score = 1.0 / (1.0 + top_l2_dist)
        
        # Check if both metrics agree
        if top_cos_idx != top_l2_idx:
            return None  # Disagreement
        
        # Check thresholds
        if top_cos_score < Config.COSINE_THRESHOLD:
            return None
        
        if top_l2_score < Config.L2_THRESHOLD:
            return None
        
        # Ensemble score
        ensemble_score = 0.6 * top_cos_score + 0.4 * top_l2_score
        
        if ensemble_score < Config.ENSEMBLE_THRESHOLD:
            return None
        
        # Check margin (1st vs 2nd place)
        if len(D_cos[0]) > 1:
            margin = top_cos_score - float(D_cos[0][1])
            if margin < Config.MARGIN_THRESHOLD:
                return None  # Too close
        
        # Check top-k consistency
        top_k_names = []
        for idx in I_cos[0][:3]:
            if idx < len(self.faiss_id_map):
                user_id = self.faiss_id_map[idx]
                top_k_names.append(self.user_names[user_id])
        
        if len(set(top_k_names)) > 1:
            return None  # Top-k disagree
        
        # Match confirmed
        user_id = self.faiss_id_map[top_cos_idx]
        name = self.user_names[user_id]
        
        return (user_id, name, ensemble_score, "matched")
    
    def recognize_with_voting(self, face_rgb, bbox):
        """Multi-frame voting for temporal consistency"""
        if not Config.USE_MULTI_FRAME_VOTING:
            return self.recognize(face_rgb)
        
        self.frame_counter += 1
        
        # Get face ID based on position
        x1, y1, x2, y2 = bbox
        face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        face_id = self._get_face_id(face_center)
        
        # Initialize history
        if face_id not in self.face_history:
            self.face_history[face_id] = {
                'matches': [],
                'last_frame': self.frame_counter
            }
        
        # Clean old tracks
        self.face_history = {
            fid: data for fid, data in self.face_history.items()
            if self.frame_counter - data['last_frame'] < 30
        }
        
        # Get current match
        result = self.recognize(face_rgb)
        
        if result is None:
            return None
        
        user_id, name, score, status = result
        
        # Add to history
        history = self.face_history[face_id]
        history['matches'].append(name)
        history['last_frame'] = self.frame_counter
        
        # Keep only recent frames
        history['matches'] = history['matches'][-Config.CONFIRMATION_FRAMES:]
        
        # Check if enough frames
        if len(history['matches']) < Config.CONFIRMATION_FRAMES:
            return (user_id, name, score, "verifying")
        
        # Vote
        name_counts = Counter(history['matches'])
        most_common_name, count = name_counts.most_common(1)[0]
        
        # Require majority
        if count >= Config.CONFIRMATION_FRAMES:
            return (user_id, most_common_name, score, "confirmed")
        
        return (user_id, name, score, "verifying")
    
    def _get_face_id(self, face_center):
        """Get face ID for tracking"""
        x, y = face_center
        grid_size = 50
        return f"{int(x/grid_size)}_{int(y/grid_size)}"
    
    def save_model(self):
        """Save model"""
        data = {
            'user_templates': self.user_templates,
            'user_names': self.user_names,
            'faiss_id_map': self.faiss_id_map,
            'pca': self.pca
        }
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Model saved to {Config.MODEL_PATH}")
    
    def load_model(self):
        """Load model"""
        if not os.path.exists(Config.MODEL_PATH):
            return False
        
        with open(Config.MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        
        self.user_templates = data['user_templates']
        self.user_names = data['user_names']
        self.faiss_id_map = data['faiss_id_map']
        self.pca = data['pca']
        
        self.build_index()
        print(f"💾 Model loaded from {Config.MODEL_PATH}")
        return True


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
        
        c.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TEXT
                    )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        name TEXT,
                        timestamp TEXT,
                        cam_id TEXT,
                        confidence REAL,
                        snapshot_path TEXT
                    )""")
        
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
        last_ts = self.get_last_seen(user_id)
        current_ts = time.time()
        
        if last_ts is not None:
            if current_ts - last_ts <= Config.ATTENDANCE_DEBOUNCE_SECONDS:
                return False
        
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
# MAIN APPLICATION
# ============================================================

class AttendanceApp:
    def __init__(self):
        print("\n" + "="*60)
        print("🎓 SMART ATTENDANCE SYSTEM (ADVANCED)")
        print("="*60 + "\n")
        
        # Check YOLO weights
        if not os.path.exists(Config.YOLO_WEIGHTS):
            print(f"❌ YOLO weights not found: {Config.YOLO_WEIGHTS}")
            print("   Please place 'best.pt' in the project directory")
            exit(1)
        
        # Load YOLO
        print("📦 Loading YOLO model...")
        self.yolo = YOLO(Config.YOLO_WEIGHTS)
        self.yolo.fuse()
        
        # Initialize components
        self.face_engine = AdvancedFaceRecognition()
        self.db = DatabaseManager()
        
        # Try to load existing model
        if self.face_engine.load_model():
            print("✅ Loaded existing face recognition model\n")
        else:
            print("ℹ️  No existing model found - need to enroll users\n")
    
    def enroll_from_folder(self):
        """Enroll users from folder"""
        if not os.path.exists(Config.ENROLLMENT_DIR):
            print(f"❌ Enrollment folder not found: {Config.ENROLLMENT_DIR}")
            return
        
        user_folders = [d for d in os.listdir(Config.ENROLLMENT_DIR) 
                        if os.path.isdir(os.path.join(Config.ENROLLMENT_DIR, d))]
        
        if not user_folders:
            print("❌ No user folders found")
            return
        
        print("="*60)
        print("👥 ENROLLING USERS")
        print("="*60)
        print(f"Found {len(user_folders)} users: {user_folders}\n")
        
        for user_name in tqdm(user_folders, desc="Enrolling"):
            user_path = os.path.join(Config.ENROLLMENT_DIR, user_name)
            user_id = str(uuid.uuid4())
            
            # Load images
            img_files = glob.glob(f"{user_path}/*.jpg") + glob.glob(f"{user_path}/*.png") + glob.glob(f"{user_path}/*.jpeg")
            
            face_images = []
            for img_path in img_files:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Detect face
                results = self.yolo(img_rgb, verbose=False)
                boxes = results[0].boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                bb = boxes.xyxy[0].cpu().numpy().astype(int)
                conf = boxes.conf[0].cpu().item()
                
                if conf < Config.MIN_FACE_CONFIDENCE:
                    continue
                
                x1, y1, x2, y2 = bb
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(img_rgb.shape[1], x2)
                y2 = min(img_rgb.shape[0], y2)
                
                face = img_rgb[y1:y2, x1:x2]
                
                if face.size > 0 and min(face.shape[:2]) >= Config.MIN_FACE_SIZE:
                    face_images.append(face)
            
            # Enroll
            if len(face_images) >= Config.MIN_IMAGES_PER_USER:
                success, msg = self.face_engine.enroll_user(user_id, user_name, face_images)
                if success:
                    self.db.add_user(user_id, user_name)
                    print(f"  ✅ {user_name}: {msg}")
                else:
                    print(f"  ⚠️  {user_name}: {msg}")
            else:
                print(f"  ❌ {user_name}: Only {len(face_images)} valid faces (need {Config.MIN_IMAGES_PER_USER}+)")
        
        # Build index
        self.face_engine.build_index()
        self.face_engine.save_model()
        
        print("\n✅ Enrollment complete!\n")
    
    def run_live_attendance(self):
        """Run live attendance"""
        if self.face_engine.faiss_cosine.ntotal == 0:
            print("❌ No users enrolled. Please enroll users first.\n")
            return
        
        print("="*60)
        print("📹 LIVE ATTENDANCE MODE")
        print("="*60)
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame
                if frame_count % 3 == 0:
                    results = self.yolo(frame, verbose=False)
                    boxes = results[0].boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            bbox = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().item()
                            
                            if conf < Config.MIN_FACE_CONFIDENCE:
                                continue
                            
                            x1, y1, x2, y2 = bbox
                            
                            # Check face size
                            if (x2 - x1) < Config.MIN_FACE_SIZE or (y2 - y1) < Config.MIN_FACE_SIZE:
                                continue
                            
                            face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                            
                            # Recognize with voting
                            result = self.face_engine.recognize_with_voting(face_rgb, bbox)
                            
                            if result is not None:
                                user_id, name, score, status = result
                                
                                # Color based on status
                                if status == "confirmed":
                                    color = (0, 255, 0)  # Green
                                    label = f"{name} ({score:.2f})"
                                    
                                    # Log attendance
                                    snap_path = Config.SNAPSHOT_DIR / f"{name}_{int(time.time()*1000)}.jpg"
                                    cv2.imwrite(str(snap_path), frame[y1:y2, x1:x2])
                                    
                                    logged = self.db.log_attendance(user_id, name, "live_camera", score, snap_path)
                                    if logged:
                                        print(f"✅ Attendance logged: {name} ({score:.3f})")
                                
                                elif status == "verifying":
                                    color = (255, 165, 0)  # Orange
                                    label = "Verifying..."
                                else:
                                    color = (0, 255, 0)
                                    label = f"{name} ({score:.2f})"
                            else:
                                color = (0, 0, 255)  # Red
                                label = "Unknown"
                            
                            # Draw
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                frame_count += 1
                cv2.imshow("Live Attendance (Advanced)", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Live attendance ended\n")
    
    def view_records(self):
        """View attendance records"""
        df = self.db.get_attendance_records()
        if len(df) > 0:
            print("\n" + "="*60)
            print("📊 ATTENDANCE RECORDS")
            print("="*60)
            print(df[['name', 'timestamp', 'confidence', 'cam_id']].head(20))
            print(f"\nTotal records: {len(df)}")
            print(f"Unique attendees: {df['name'].nunique()}")
        else:
            print("\n❌ No attendance records found\n")
    
    def menu(self):
        """Main menu"""
        while True:
            print("\n" + "="*60)
            print("📋 MAIN MENU")
            print("="*60)
            print("1. Enroll users from folder")
            print("2. Run live attendance")
            print("3. View attendance records")
            print("4. System info")
            print("5. Exit")
            print("="*60)
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.enroll_from_folder()
            
            elif choice == '2':
                self.run_live_attendance()
            
            elif choice == '3':
                self.view_records()
            
            elif choice == '4':
                self.show_info()
            
            elif choice == '5':
                print("\n✅ Goodbye!\n")
                break
            
            else:
                print("\n❌ Invalid choice\n")
    
    def show_info(self):
        """Show system info"""
        print("\n" + "="*60)
        print("ℹ️  SYSTEM INFORMATION")
        print("="*60)
        print(f"Enrolled users: {len(self.face_engine.user_names)}")
        print(f"FAISS index size: {self.face_engine.faiss_cosine.ntotal}")
        print(f"PCA reduction: {Config.EMBEDDING_SIZE} → {Config.REDUCED_DIM}")
        print(f"Cosine threshold: {Config.COSINE_THRESHOLD}")
        print(f"L2 threshold: {Config.L2_THRESHOLD}")
        print(f"Ensemble threshold: {Config.ENSEMBLE_THRESHOLD}")
        print(f"Multi-frame voting: {Config.USE_MULTI_FRAME_VOTING}")
        print(f"Confirmation frames: {Config.CONFIRMATION_FRAMES}")
        print("="*60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        app = AttendanceApp()
        app.menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
"""
Smart Face Attendance System
Workflow: Face Image → FaceNet → 512-D → PCA → 128-D → FAISS Search
Reduces overfitting while maintaining accuracy
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
import time
from datetime import datetime
import os

# FaceNet for embeddings
from facenet_pytorch import InceptionResnetV1, MTCNN

# YOLO for face detection
from ultralytics import YOLO

# PCA for dimensionality reduction
from sklearn.decomposition import PCA

# FAISS for similarity search
import faiss

# Pandas for attendance logging
import pandas as pd


class FaceAttendanceSystem:
    """
    Complete Face Attendance System with:
    - YOLO face detection
    - FaceNet 512-D embeddings
    - PCA reduction to 128-D
    - FAISS similarity search
    """
    
    def __init__(
        self,
        yolo_model_path="best.pt",
        enrollment_dir="enrollment",
        database_path="face_database.pkl",
        attendance_log="attendance_log.csv",
        embedding_dim=512,
        reduced_dim=128,
        similarity_threshold=0.6
    ):
        """
        Initialize the attendance system
        
        Args:
            yolo_model_path: Path to YOLO face detection model
            enrollment_dir: Directory with enrolled face images
            database_path: Path to save/load face database
            attendance_log: CSV file for attendance records
            embedding_dim: FaceNet embedding dimension (512)
            reduced_dim: PCA reduced dimension (128)
            similarity_threshold: Threshold for face matching (0-1)
        """
        print("\n" + "="*70)
        print("🚀 INITIALIZING SMART ATTENDANCE SYSTEM")
        print("="*70)
        
        self.enrollment_dir = Path(enrollment_dir)
        self.database_path = database_path
        self.attendance_log = attendance_log
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.similarity_threshold = similarity_threshold
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {self.device}")
        
        # Load models
        print("\n🔧 Loading models...")
        self._load_models(yolo_model_path)
        
        # Initialize database
        self.face_database = {
            'names': [],
            'embeddings_512d': [],
            'embeddings_reduced': [],
            'pca_model': None,
            'faiss_index': None,
            'actual_dim': 0
        }
        
        # Load existing database if available
        self._load_database()
        
        print("\n✅ System initialized successfully!")
        print("="*70 + "\n")
    
    def _load_models(self, yolo_model_path):
        """Load YOLO and FaceNet models"""
        
        # 1. YOLO for face detection
        print("  └─ Loading YOLO face detector...")
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        print("     ✓ YOLO loaded")
        
        # 2. FaceNet for 512-D embeddings
        print("  └─ Loading FaceNet (512-D embeddings)...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print("     ✓ FaceNet loaded")
        
        # 3. MTCNN for face alignment (optional but recommended)
        print("  └─ Loading MTCNN for face alignment...")
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            keep_all=False,
            device=self.device
        )
        print("     ✓ MTCNN loaded")
    
    def extract_face_embedding(self, face_img):
        """
        Extract 512-D embedding from face image using FaceNet
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            512-D numpy array or None if extraction fails
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Resize to FaceNet input size (160x160)
            face_resized = cv2.resize(face_rgb, (160, 160))
            
            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            # Normalize (L2 normalization)
            embedding_np = embedding_np / np.linalg.norm(embedding_np)
            
            return embedding_np
            
        except Exception as e:
            print(f"⚠️  Embedding extraction failed: {e}")
            return None
    
    def build_database_from_enrollment(self):
        """
        Build face database from enrollment directory
        Workflow: Images → FaceNet → 512-D → PCA → 128-D → FAISS
        """
        print("\n" + "="*70)
        print("🔨 BUILDING FACE DATABASE")
        print("="*70)
        
        if not self.enrollment_dir.exists():
            print(f"❌ Enrollment directory not found: {self.enrollment_dir}")
            return False
        
        names = []
        embeddings_512d = []
        
        # Get all registered users
        user_folders = [d for d in self.enrollment_dir.iterdir() if d.is_dir()]
        
        if not user_folders:
            print("❌ No registered users found")
            return False
        
        print(f"\n📂 Found {len(user_folders)} registered users")
        print(f"🎯 Target: {self.embedding_dim}D → {self.reduced_dim}D embeddings\n")
        
        # Process each user
        for user_folder in user_folders:
            user_name = user_folder.name
            image_files = list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.png"))
            
            if not image_files:
                print(f"⚠️  No images found for {user_name}")
                continue
            
            print(f"👤 Processing: {user_name} ({len(image_files)} images)")
            user_embeddings = []
            
            for img_path in image_files:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Extract 512-D embedding
                embedding = self.extract_face_embedding(img)
                if embedding is not None:
                    user_embeddings.append(embedding)
            
            if user_embeddings:
                # Average all embeddings for this user (reduces overfitting)
                avg_embedding = np.mean(user_embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Re-normalize
                
                names.append(user_name)
                embeddings_512d.append(avg_embedding)
                print(f"  ✓ Generated {self.embedding_dim}D embedding (from {len(user_embeddings)} images)")
            else:
                print(f"  ✗ Failed to generate embeddings")
        
        if not embeddings_512d:
            print("\n❌ No valid embeddings generated")
            return False
        
        # Convert to numpy array
        embeddings_512d = np.array(embeddings_512d).astype('float32')
        
        print(f"\n📊 Database Statistics:")
        print(f"  • Total users: {len(names)}")
        print(f"  • Embedding shape: {embeddings_512d.shape}")
        
        # Apply PCA: 512-D → 128-D (reduces overfitting)
        print(f"\n🔬 Applying PCA dimensionality reduction...")
        
        # Determine actual number of PCA components
        # PCA components must be <= min(n_samples, n_features)
        n_samples = embeddings_512d.shape[0]
        n_features = embeddings_512d.shape[1]
        max_components = min(n_samples, n_features)
        
        # Use the smaller of: desired dim or max possible components
        actual_components = min(self.reduced_dim, max_components)
        
        if actual_components < self.reduced_dim:
            print(f"  ⚠️  Limited samples: Using {actual_components}D instead of {self.reduced_dim}D")
            print(f"     (Need at least {self.reduced_dim} users for full {self.reduced_dim}D reduction)")
        
        print(f"  {self.embedding_dim}D → {actual_components}D")
        
        if actual_components == n_samples and n_samples < 10:
            # Very few samples - consider not using PCA at all
            print(f"  ℹ️  Using minimal PCA with only {n_samples} users")
            print(f"     Recommendation: Add more users for better performance")
        
        pca = PCA(n_components=actual_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings_512d)
        
        # Normalize after PCA - ensure C-contiguous array for FAISS
        embeddings_reduced = np.ascontiguousarray(embeddings_reduced, dtype='float32')
        faiss.normalize_L2(embeddings_reduced)
        
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"  ✓ PCA complete")
        print(f"  • Variance retained: {explained_variance:.2f}%")
        print(f"  • Reduced shape: {embeddings_reduced.shape}")
        
        # Build FAISS index
        print(f"\n🔍 Building FAISS index...")
        dimension = embeddings_reduced.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings_reduced)
        
        print(f"  ✓ FAISS index built")
        print(f"  • Index type: IndexFlatIP (cosine similarity)")
        print(f"  • Total vectors: {faiss_index.ntotal}")
        
        # Update database
        self.face_database = {
            'names': names,
            'embeddings_512d': embeddings_512d,
            'embeddings_reduced': embeddings_reduced,
            'pca_model': pca,
            'faiss_index': faiss_index,
            'actual_dim': actual_components  # Store actual dimension used
        }
        
        # Save database
        self._save_database()
        
        print(f"\n✅ Database built successfully!")
        print("="*70 + "\n")
        
        return True
    
    def _save_database(self):
        """Save face database to disk"""
        print(f"💾 Saving database to {self.database_path}...")
        
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
        
        print("  ✓ Database saved")
    
    def _load_database(self):
        """Load face database from disk"""
        if os.path.exists(self.database_path):
            print(f"\n📥 Loading existing database from {self.database_path}...")
            
            try:
                with open(self.database_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                
                if self.face_database['faiss_index'] is not None:
                    print(f"  ✓ Database loaded")
                    print(f"  • Users: {len(self.face_database['names'])}")
                    print(f"  • FAISS vectors: {self.face_database['faiss_index'].ntotal}")
                else:
                    print("  ⚠️  Database exists but is empty")
            
            except Exception as e:
                print(f"  ⚠️  Failed to load database: {e}")
                print("  → Will create new database")
        else:
            print(f"\n📭 No existing database found")
            print(f"  → Run 'Build Database' to create one")
    
    def recognize_face(self, face_img):
        """
        Recognize face using FAISS search
        Workflow: Face → FaceNet → 512-D → PCA → reduced-D → FAISS Search
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            (name, confidence) or (None, 0) if not recognized
        """
        if self.face_database['faiss_index'] is None:
            return None, 0
        
        # Extract 512-D embedding
        embedding_512d = self.extract_face_embedding(face_img)
        if embedding_512d is None:
            return None, 0
        
        # Apply PCA: 512-D → reduced-D
        embedding_reduced = self.face_database['pca_model'].transform(
            embedding_512d.reshape(1, -1)
        )
        
        # Ensure C-contiguous array for FAISS
        embedding_reduced = np.ascontiguousarray(embedding_reduced, dtype='float32')
        
        # Normalize
        faiss.normalize_L2(embedding_reduced)
        
        # FAISS search (k=1 for top match)
        distances, indices = self.face_database['faiss_index'].search(embedding_reduced, k=1)
        
        # Get results
        distance = distances[0][0]
        idx = indices[0][0]
        
        # Convert distance to confidence (cosine similarity)
        confidence = float(distance)
        
        if confidence >= self.similarity_threshold:
            name = self.face_database['names'][idx]
            return name, confidence
        else:
            return None, confidence
    
    def mark_attendance(self, name):
        """
        Mark attendance for a person
        
        Args:
            name: Person's name
            
        Returns:
            True if attendance marked, False if already marked today
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Load existing attendance log
        if os.path.exists(self.attendance_log):
            df = pd.read_csv(self.attendance_log)
            
            # Check if already marked today
            if len(df[(df['Name'] == name) & (df['Date'] == current_date)]) > 0:
                return False
        else:
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
        # Add new entry
        new_entry = pd.DataFrame({
            'Name': [name],
            'Date': [current_date],
            'Time': [current_time]
        })
        
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.attendance_log, index=False)
        
        return True
    
    def run_attendance(self, camera_id=0, show_fps=True):
        """
        Run real-time face attendance system
        
        Args:
            camera_id: Camera device ID
            show_fps: Show FPS counter
        """
        if self.face_database['faiss_index'] is None:
            print("\n❌ No database found! Please build database first.")
            return
        
        print("\n" + "="*70)
        print("📹 STARTING REAL-TIME ATTENDANCE SYSTEM")
        print("="*70)
        print(f"📊 Database: {len(self.face_database['names'])} users")
        print(f"🎯 Threshold: {self.similarity_threshold}")
        
        actual_dim = self.face_database.get('actual_dim', self.reduced_dim)
        print(f"🔧 Pipeline: YOLO → FaceNet → 512D → PCA → {actual_dim}D → FAISS")
        
        print("\nControls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save screenshot")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        # Cooldown for attendance marking (5 seconds)
        attendance_cooldown = {}
        cooldown_duration = 5
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                current_time = time.time()
                
                # Detect faces every 3 frames (optimization)
                if frame_count % 3 == 0:
                    results = self.yolo(frame, verbose=False)
                    boxes = results[0].boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            bbox = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])
                            
                            if conf < 0.5:
                                continue
                            
                            x1, y1, x2, y2 = bbox
                            
                            # Expand bbox slightly for better face extraction
                            h, w = frame.shape[:2]
                            margin = 10
                            x1 = max(0, x1 - margin)
                            y1 = max(0, y1 - margin)
                            x2 = min(w, x2 + margin)
                            y2 = min(h, y2 + margin)
                            
                            face = frame[y1:y2, x1:x2]
                            
                            # Recognize face
                            name, confidence = self.recognize_face(face)
                            
                            # Draw bounding box
                            if name:
                                color = (0, 255, 0)  # Green for recognized
                                label = f"{name} ({confidence:.2f})"
                                
                                # Mark attendance (with cooldown)
                                if name not in attendance_cooldown or \
                                   (current_time - attendance_cooldown[name]) > cooldown_duration:
                                    
                                    if self.mark_attendance(name):
                                        print(f"✅ Attendance marked: {name} ({confidence:.2f})")
                                        attendance_cooldown[name] = current_time
                                    else:
                                        # Already marked today
                                        if name not in attendance_cooldown:
                                            print(f"ℹ️  Already present: {name}")
                                        attendance_cooldown[name] = current_time
                            else:
                                color = (0, 0, 255)  # Red for unknown
                                label = f"Unknown ({confidence:.2f})"
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Label background
                            (label_w, label_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            cv2.rectangle(
                                display_frame, 
                                (x1, y1 - label_h - 10), 
                                (x1 + label_w, y1), 
                                color, 
                                -1
                            )
                            
                            cv2.putText(
                                display_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                            )
                
                # FPS calculation
                fps_frame_count += 1
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Display info
                if show_fps:
                    cv2.putText(
                        display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                cv2.putText(
                    display_frame, 
                    f"Users: {len(self.face_database['names'])}", 
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                cv2.putText(
                    display_frame, "Q: Quit | S: Screenshot", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                cv2.imshow("Smart Attendance System", display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Stopping attendance system...")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("✅ Attendance system stopped")
            print("="*70 + "\n")
    
    def view_attendance_log(self, date=None):
        """
        View attendance log
        
        Args:
            date: Specific date (YYYY-MM-DD) or None for all
        """
        if not os.path.exists(self.attendance_log):
            print("📭 No attendance records found")
            return
        
        df = pd.read_csv(self.attendance_log)
        
        if df.empty:
            print("📭 No attendance records found")
            return
        
        print("\n" + "="*70)
        print("📊 ATTENDANCE LOG")
        print("="*70)
        
        if date:
            df_filtered = df[df['Date'] == date]
            print(f"Date: {date}")
        else:
            df_filtered = df
            print("All Records")
        
        if df_filtered.empty:
            print("No records found")
        else:
            print(f"\nTotal entries: {len(df_filtered)}\n")
            print(df_filtered.to_string(index=False))
        
        print("="*70 + "\n")
    
    def get_database_info(self):
        """Display database information"""
        print("\n" + "="*70)
        print("📊 DATABASE INFORMATION")
        print("="*70)
        
        if self.face_database['faiss_index'] is None:
            print("❌ No database loaded")
            print("\nPlease run 'Build Database' from the menu")
        else:
            print(f"\n✅ Database Status: Active")
            print(f"\nRegistered Users: {len(self.face_database['names'])}")
            
            for i, name in enumerate(self.face_database['names'], 1):
                print(f"  [{i}] {name}")
            
            actual_dim = self.face_database.get('actual_dim', self.reduced_dim)
            
            print(f"\nEmbedding Details:")
            print(f"  • Original dimension: {self.embedding_dim}D (FaceNet)")
            print(f"  • Reduced dimension: {actual_dim}D (PCA)")
            
            if actual_dim < self.reduced_dim:
                print(f"    ⚠️  Target was {self.reduced_dim}D, limited by {len(self.face_database['names'])} users")
            
            print(f"  • Total vectors in FAISS: {self.face_database['faiss_index'].ntotal}")
            
            if self.face_database['pca_model']:
                variance = np.sum(self.face_database['pca_model'].explained_variance_ratio_) * 100
                print(f"  • Variance retained: {variance:.2f}%")
            
            print(f"\nSearch Configuration:")
            print(f"  • Similarity threshold: {self.similarity_threshold}")
            print(f"  • FAISS index type: IndexFlatIP (cosine similarity)")
        
        print("="*70 + "\n")
    
    def list_registered_users(self):
        """List all registered users with image counts"""
        enrollment_path = Path(self.enrollment_dir)
        
        if not enrollment_path.exists():
            print("❌ Enrollment directory not found")
            return []
        
        user_folders = [d for d in enrollment_path.iterdir() if d.is_dir()]
        
        if not user_folders:
            print("\n📭 No registered users found")
            return []
        
        print("\n" + "="*70)
        print(f"REGISTERED USERS ({len(user_folders)})")
        print("="*70)
        
        users_info = []
        for i, user_folder in enumerate(sorted(user_folders, key=lambda x: x.name), 1):
            user_name = user_folder.name
            img_count = len(list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.png")))
            emoji = "🔥" if img_count >= 10 else "📸" if img_count >= 5 else "🌱"
            print(f"  [{i}] {user_name:20s} - {img_count} images {emoji}")
            users_info.append({'name': user_name, 'images': img_count, 'path': user_folder})
        
        print("="*70 + "\n")
        return users_info
    
    def delete_user(self, name):
        """Delete a registered user"""
        import shutil
        
        user_folder = Path(self.enrollment_dir) / name
        
        if not user_folder.exists():
            print(f"❌ User '{name}' not found")
            return False
        
        img_count = len(list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.png")))
        print(f"\n⚠️  About to delete user: {name} ({img_count} images)")
        print(f"💀 This action cannot be undone! Gone forever! Poof! 💨")
        confirm = input("Are you sure? Type 'DELETE' to confirm: ")
        
        if confirm == 'DELETE':
            try:
                shutil.rmtree(user_folder)
                print(f"✅ User '{name}' deleted successfully")
                print(f"👋 Goodbye, {name}! May we meet again in another database...")
                print(f"\n🔨 Remember to rebuild the database to update the system")
                return True
            except Exception as e:
                print(f"❌ Error deleting user: {e}")
                return False
        else:
            print("❌ Deletion cancelled")
            print("😅 Phew! That was close!")
            return False
    
    def view_user_details(self, name):
        """View detailed information about a user"""
        user_folder = Path(self.enrollment_dir) / name
        
        if not user_folder.exists():
            print(f"❌ User '{name}' not found")
            return
        
        image_files = list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.png"))
        
        print("\n" + "="*70)
        print(f"USER DETAILS: {name}")
        print("="*70)
        print(f"\n📂 Location: {user_folder}")
        print(f"📸 Total Images: {len(image_files)}")
        
        if image_files:
            print(f"\n🖼️  Image Files:")
            for i, img_path in enumerate(image_files, 1):
                file_size = img_path.stat().st_size / 1024  # KB
                print(f"  [{i}] {img_path.name:20s} ({file_size:.1f} KB)")
        
        print("="*70 + "\n")


def main_menu():
    """Interactive menu for face attendance system"""
    
    # ASCII Art Banner
    print("\n" + "="*70)
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🎓 SMART FACE ATTENDANCE SYSTEM 🎓                    ║
    ║                                                               ║
    ║     Face → FaceNet → 512D → PCA → 128D → FAISS Search       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    print("="*70 + "\n")
    
    # Initialize system
    system = FaceAttendanceSystem(
        yolo_model_path="best.pt",
        enrollment_dir="enrollment",
        database_path="face_database.pkl",
        attendance_log="attendance_log.csv",
        embedding_dim=512,
        reduced_dim=128,
        similarity_threshold=0.6
    )
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. 🔨 Build/Rebuild Face Database")
        print("2. 📹 Start Attendance System")
        print("3. 👥 Manage Users (View/Delete)")
        print("4. 📊 View Attendance Log (Today)")
        print("5. 📅 View Attendance Log (All)")
        print("6. 📈 Database Information")
        print("7. ⚙️  Settings")
        print("8. 👋 Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == '1':
            # Build database
            system.build_database_from_enrollment()
        
        elif choice == '2':
            # Start attendance
            if system.face_database['faiss_index'] is None:
                print("\n❌ No database found!")
                print("Please build the database first (Option 1)")
            else:
                system.run_attendance(camera_id=0, show_fps=True)
        
        elif choice == '3':
            # Manage users
            user_management_menu(system)
        
        elif choice == '4':
            # View today's attendance
            today = datetime.now().strftime("%Y-%m-%d")
            system.view_attendance_log(date=today)
        
        elif choice == '5':
            # View all attendance
            system.view_attendance_log()
        
        elif choice == '6':
            # Database info
            system.get_database_info()
        
        elif choice == '7':
            # Settings menu
            settings_menu(system)
        
        elif choice == '8':
            # Exit
            print("\n" + "="*70)
            print("👋 Thank you for using Smart Attendance System!")
            print("🎉 Stay safe and keep learning!")
            print("="*70 + "\n")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-8")




def user_management_menu(system):
    """User management submenu"""
    while True:
        print("\n" + "="*70)
        print("👥 USER MANAGEMENT")
        print("="*70)
        
        users = system.list_registered_users()
        
        if not users:
            print("No users to manage")
            input("\nPress Enter to return to main menu...")
            break
        
        print("\nOptions:")
        print("1. View user details")
        print("2. Delete user")
        print("3. Back to main menu")
        print("="*70)
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            name = input("\nEnter user name: ").strip()
            system.view_user_details(name)
        
        elif choice == '2':
            name = input("\nEnter user name to delete: ").strip()
            if system.delete_user(name):
                print("\n✅ User deleted successfully")
                print("🔨 Please rebuild the database (Main Menu > Option 1)")
                input("\nPress Enter to continue...")
        
        elif choice == '3':
            break
        
        else:
            print("❌ Invalid choice")


def settings_menu(system):
    """Settings submenu"""
    while True:
        print("\n" + "="*70)
        print("⚙️  SETTINGS")
        print("="*70)
        print(f"Current Configuration:")
        print(f"  • Similarity Threshold: {system.similarity_threshold}")
        print(f"  • PCA Dimensions: {system.reduced_dim}D")
        print(f"  • Database Path: {system.database_path}")
        print(f"  • Attendance Log: {system.attendance_log}")
        print("\nOptions:")
        print("1. Change similarity threshold")
        print("2. Back to main menu")
        print("="*70)
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            try:
                new_threshold = float(input(f"Enter new threshold (0.0-1.0, current: {system.similarity_threshold}): "))
                if 0.0 <= new_threshold <= 1.0:
                    system.similarity_threshold = new_threshold
                    print(f"✅ Threshold updated to {new_threshold}")
                else:
                    print("❌ Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == '2':
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
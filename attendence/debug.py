"""
Face Recognition Debug Tool
Run this to diagnose why faces show as "Unknown"
"""

import os
import cv2
import numpy as np
import torch
import pickle
import faiss
from pathlib import Path
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(ROOT, "best.pt")
ENROLLMENT_DIR = os.path.join(ROOT, "enrollment")
FAISS_INDEX_PATH = os.path.join(ROOT, "faiss_index.bin")
FAISS_MAP_PATH = os.path.join(ROOT, "faiss_map.pkl")

# ============================================================
# DEBUG FUNCTIONS
# ============================================================

def load_models():
    """Load YOLO and FaceNet"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}\n")
    
    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.fuse()
    
    print("Loading FaceNet...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    print("Loading MTCNN...")
    mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=False)
    
    print("✅ Models loaded\n")
    return yolo, facenet, mtcnn, device


def get_face_embedding(face_rgb, facenet, mtcnn, device):
    """Extract embedding from face"""
    aligned = None
    method = "none"
    
    # Try MTCNN
    try:
        aligned = mtcnn(face_rgb)
        if aligned is not None:
            method = "mtcnn"
    except:
        aligned = None
    
    # Fallback to manual
    if aligned is None:
        try:
            pil_img = Image.fromarray(face_rgb)
            pil_img = pil_img.resize((160, 160), Image.BILINEAR)
            arr = np.array(pil_img).astype(np.float32)
            arr = (arr - 127.5) / 128.0
            aligned = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            method = "manual"
        except:
            return None, "failed"
    
    if aligned.dim() == 3:
        aligned = aligned.unsqueeze(0)
    
    aligned = aligned.to(device)
    
    try:
        with torch.no_grad():
            emb = facenet(aligned).cpu().numpy().flatten()
        
        if np.linalg.norm(emb) < 0.1:
            return None, "zero_vector"
        
        return emb.astype('float32'), method
    except:
        return None, "failed"


def normalize_embedding(emb):
    """Normalize embedding for cosine similarity"""
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def test_live_capture(yolo, facenet, mtcnn, device, faiss_index, faiss_name_map):
    """Test live webcam capture and compare with enrolled faces"""
    print("\n" + "="*60)
    print("🎥 LIVE CAPTURE TEST")
    print("="*60)
    print("Press SPACE to capture and test")
    print("Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = yolo(frame_rgb, verbose=False)
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().item()
                    
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"Conf: {conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display, "SPACE: Test | Q: Quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Debug - Live Capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print("\n" + "="*60)
                print("📸 TESTING CURRENT FRAME")
                print("="*60)
                
                if boxes is None or len(boxes) == 0:
                    print("❌ No face detected!")
                    continue
                
                # Get best face
                best_box = boxes[0]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
                conf = best_box.conf[0].cpu().item()
                
                print(f"\n1️⃣ Face Detection:")
                print(f"   Confidence: {conf:.3f}")
                print(f"   BBox: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"   Size: {x2-x1}x{y2-y1} pixels")
                
                # Expand bbox
                img_h, img_w = frame_rgb.shape[:2]
                expand = 0.15
                w = x2 - x1
                h = y2 - y1
                x1 = max(0, int(x1 - w * expand))
                y1 = max(0, int(y1 - h * expand))
                x2 = min(img_w, int(x2 + w * expand))
                y2 = min(img_h, int(y2 + h * expand))
                
                face = frame_rgb[y1:y2, x1:x2]
                
                # Check blur
                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                print(f"\n2️⃣ Face Quality:")
                print(f"   Expanded size: {x2-x1}x{y2-y1} pixels")
                print(f"   Blur score: {blur_score:.1f}")
                print(f"   Quality: {'✅ GOOD' if blur_score > 50 else '⚠️ BLURRY'}")
                
                # Get embedding
                emb, method = get_face_embedding(face, facenet, mtcnn, device)
                
                if emb is None:
                    print(f"\n❌ Embedding extraction failed: {method}")
                    continue
                
                emb_norm = np.linalg.norm(emb)
                print(f"\n3️⃣ Embedding Extraction:")
                print(f"   Method: {method}")
                print(f"   Embedding norm: {emb_norm:.3f}")
                print(f"   Status: {'✅ VALID' if emb_norm > 0.1 else '❌ INVALID'}")
                
                # Normalize
                emb = normalize_embedding(emb)
                
                # Search FAISS
                if faiss_index.ntotal == 0:
                    print(f"\n❌ FAISS index is EMPTY! No enrolled faces.")
                    continue
                
                print(f"\n4️⃣ FAISS Search:")
                print(f"   Index size: {faiss_index.ntotal} embeddings")
                
                emb_search = emb.reshape(1, -1).astype('float32')
                k = min(5, faiss_index.ntotal)
                D, I = faiss_index.search(emb_search, k)
                
                print(f"\n   🎯 TOP {k} MATCHES:")
                print(f"   {'Rank':<6} {'Name':<20} {'Similarity':<12} {'Pass 0.30?':<12}")
                print(f"   {'-'*50}")
                
                for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
                    if idx < len(faiss_name_map):
                        name = faiss_name_map[idx]
                        passes = "✅ YES" if dist >= 0.30 else "❌ NO"
                        print(f"   {rank:<6} {name:<20} {dist:<12.3f} {passes}")
                
                top_score = D[0][0]
                top_name = faiss_name_map[I[0][0]] if I[0][0] < len(faiss_name_map) else "Invalid"
                
                print(f"\n5️⃣ FINAL RESULT:")
                print(f"   Best match: {top_name}")
                print(f"   Similarity: {top_score:.3f}")
                print(f"   Threshold 0.30: {'✅ PASS' if top_score >= 0.30 else '❌ FAIL'}")
                print(f"   Threshold 0.35: {'✅ PASS' if top_score >= 0.35 else '❌ FAIL'}")
                print(f"   Threshold 0.40: {'✅ PASS' if top_score >= 0.40 else '❌ FAIL'}")
                
                # Check margin
                if len(D[0]) >= 2:
                    margin = D[0][0] - D[0][1]
                    print(f"\n6️⃣ Match Confidence:")
                    print(f"   Top score: {D[0][0]:.3f}")
                    print(f"   2nd score: {D[0][1]:.3f}")
                    print(f"   Margin: {margin:.3f}")
                    print(f"   Margin >= 0.02: {'✅ PASS' if margin >= 0.02 else '❌ FAIL'}")
                
                print("="*60 + "\n")
            
            elif key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def analyze_enrollment():
    """Analyze enrolled faces"""
    print("\n" + "="*60)
    print("📂 ENROLLMENT ANALYSIS")
    print("="*60)
    
    if not os.path.exists(ENROLLMENT_DIR):
        print(f"❌ Enrollment directory not found: {ENROLLMENT_DIR}")
        return None
    
    user_folders = [d for d in os.listdir(ENROLLMENT_DIR)
                   if os.path.isdir(os.path.join(ENROLLMENT_DIR, d))]
    
    if not user_folders:
        print("❌ No user folders found!")
        return None
    
    print(f"\nFound {len(user_folders)} users:")
    for user in user_folders:
        user_path = os.path.join(ENROLLMENT_DIR, user)
        img_files = [f for f in os.listdir(user_path) if f.endswith(('.jpg', '.png'))]
        print(f"  📁 {user}: {len(img_files)} images")
    
    return user_folders


def check_faiss_index():
    """Check FAISS index"""
    print("\n" + "="*60)
    print("🗂️ FAISS INDEX CHECK")
    print("="*60)
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"❌ FAISS index not found: {FAISS_INDEX_PATH}")
        print("   👉 You need to enroll users first!")
        return None, None
    
    if not os.path.exists(FAISS_MAP_PATH):
        print(f"❌ FAISS map not found: {FAISS_MAP_PATH}")
        return None, None
    
    # Load index
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_MAP_PATH, 'rb') as f:
        data = pickle.load(f)
        faiss_name_map = data['name_map']
    
    print(f"\n✅ FAISS Index loaded:")
    print(f"   Total embeddings: {faiss_index.ntotal}")
    print(f"   Dimension: {faiss_index.d}")
    
    # Count per user
    from collections import Counter
    name_counts = Counter(faiss_name_map)
    
    print(f"\n   Embeddings per user:")
    for name, count in name_counts.items():
        print(f"     {name}: {count}")
    
    return faiss_index, faiss_name_map


# ============================================================
# MAIN MENU
# ============================================================

def main():
    print("\n" + "="*60)
    print("🔍 FACE RECOGNITION DEBUGGER")
    print("="*60)
    print("This tool helps diagnose why faces show as 'Unknown'\n")
    
    # Check enrollment
    users = analyze_enrollment()
    
    # Check FAISS
    faiss_index, faiss_name_map = check_faiss_index()
    
    if faiss_index is None:
        print("\n⚠️ Cannot proceed without FAISS index")
        print("   Run main.py and choose option 1 to enroll users first!")
        return
    
    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    yolo, facenet, mtcnn, device = load_models()
    
    # Main menu
    while True:
        print("\n" + "="*60)
        print("DEBUG MENU")
        print("="*60)
        print("1. Test live webcam capture")
        print("2. Re-analyze enrollment")
        print("3. Re-check FAISS index")
        print("4. Exit")
        print("="*60)
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            test_live_capture(yolo, facenet, mtcnn, device, faiss_index, faiss_name_map)
        
        elif choice == '2':
            analyze_enrollment()
        
        elif choice == '3':
            faiss_index, faiss_name_map = check_faiss_index()
        
        elif choice == '4':
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
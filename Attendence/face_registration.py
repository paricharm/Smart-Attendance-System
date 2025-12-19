"""
Face Registration System - Auto Capture Edition
Captures faces automatically with a special surprise! 🎉
"""

import cv2
import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import random

class FaceRegistration:
    def __init__(self, yolo_model, enrollment_dir="enrollment"):
        """
        Initialize Face Registration System
        
        Args:
            yolo_model: Loaded YOLO model for face detection
            enrollment_dir: Directory to store enrolled faces
        """
        self.yolo = yolo_model
        self.enrollment_dir = Path(enrollment_dir)
        self.enrollment_dir.mkdir(exist_ok=True)
        
        # Quality thresholds (relaxed for better capture)
        self.min_face_size = 60  # Smaller minimum size
        self.min_confidence = 0.4  # Lower confidence threshold
        self.blur_threshold = 50  # Much more forgiving blur threshold
        
        # 🎭 Easter Egg: Secret messages
        self.secret_messages = [
            "👀 I see you...",
            "Looking good today!",
            "Smile detected! 😊",
            "You're doing great!",
            "Are you a developer? You look like one!",
            "Best face I've seen all day!",
            "💯 Perfect shot!",
            "Nailed it!",
            "10/10 would capture again",
            "You vs the code you wrote at 3 AM",
        ]
        
        # 🎮 Easter Egg: Achievement unlocked counter
        self.achievements_unlocked = 0
        
    def check_image_quality(self, face_img):
        """Check if face image has good quality (relaxed thresholds)"""
        h, w = face_img.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return False, "Face too small"
        
        # Blur check is now optional - most faces will pass
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Much more lenient - only reject extremely blurry images
        if blur_score < self.blur_threshold:
            return False, f"Too blurry ({blur_score:.0f})"
        
        return True, f"✓ Ready ({blur_score:.0f})"
    
    def get_secret_message(self):
        """🎭 Easter Egg: Get a random encouraging message"""
        return random.choice(self.secret_messages)
    
    def check_for_special_achievement(self, captured_count):
        """🎮 Easter Egg: Check for special milestones"""
        achievements = {
            5: "🏆 HALFWAY THERE! Keep going!",
            7: "🌟 LUCKY NUMBER 7! You're on fire!",
            10: "🎉 PERFECT 10! Registration complete!",
        }
        return achievements.get(captured_count)
    
    def register_new_person(self, name, num_images=10, camera_id=0):
        """
        Register a new person by capturing faces automatically from webcam
        
        Args:
            name: Person's name
            num_images: Number of face images to capture
            camera_id: Camera device ID (default 0)
        """
        user_folder = self.enrollment_dir / name
        user_folder.mkdir(exist_ok=True)
        
        existing_images = list(user_folder.glob("*.jpg"))
        if existing_images:
            print(f"\n⚠️  User '{name}' already has {len(existing_images)} images")
            choice = input("Do you want to add more images? (y/n): ")
            if choice.lower() != 'y':
                return False
            start_idx = len(existing_images) + 1
        else:
            start_idx = 1
        
        print(f"\n{'='*60}")
        print(f"🎬 AUTO-CAPTURE MODE: {name}")
        print(f"{'='*60}")
        print(f"Target: {num_images} images")
        print(f"Saving to: {user_folder}")
        print(f"\nInstructions:")
        print("  - Look at the camera from different angles")
        print("  - Images will capture AUTOMATICALLY")
        print("  - Move your head slightly between captures")
        print("  - Press 'q' to quit early")
        print(f"{'='*60}\n")
        
        # 🎭 Easter Egg: Start message
        print(f"💫 {self.get_secret_message()}\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        captured_count = 0
        frame_count = 0
        last_capture_time = 0
        capture_interval = 2.0  # Auto-capture every 2 seconds
        countdown = capture_interval
        
        try:
            while captured_count < num_images:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                current_time = time.time()
                
                # Update countdown
                time_since_last = current_time - last_capture_time
                countdown = max(0, capture_interval - time_since_last)
                
                # Auto-capture logic
                should_capture = countdown <= 0 and time_since_last >= capture_interval
                
                # Detect faces
                if frame_count % 3 == 0:
                    results = self.yolo(frame, verbose=False)
                    boxes = results[0].boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        confidences = boxes.conf.cpu().numpy()
                        best_idx = np.argmax(confidences)
                        
                        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                        conf = confidences[best_idx]
                        
                        if conf >= self.min_confidence:
                            x1, y1, x2, y2 = bbox
                            face = frame[y1:y2, x1:x2]
                            
                            is_good, quality_msg = self.check_image_quality(face)
                            
                            # Draw bounding box - flash green when capturing
                            if should_capture and is_good:
                                color = (0, 255, 255)  # Yellow flash
                                thickness = 4
                            elif is_good:
                                color = (0, 255, 0)
                                thickness = 2
                            else:
                                color = (0, 165, 255)
                                thickness = 2
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            cv2.putText(display_frame, quality_msg, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # AUTO-CAPTURE
                            if should_capture and is_good:
                                img_name = f"img{start_idx + captured_count}.jpg"
                                img_path = user_folder / img_name
                                cv2.imwrite(str(img_path), face)
                                
                                captured_count += 1
                                last_capture_time = current_time
                                
                                # 🎭 Easter Egg: Fun message
                                secret_msg = self.get_secret_message()
                                print(f"✅ [{captured_count}/{num_images}] {img_name} - {secret_msg}")
                                
                                # 🎮 Easter Egg: Check for achievements
                                achievement = self.check_for_special_achievement(captured_count)
                                if achievement:
                                    print(f"   {achievement}")
                                    self.achievements_unlocked += 1
                
                # Progress overlay
                progress_text = f"Captured: {captured_count}/{num_images}"
                cv2.putText(display_frame, progress_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Countdown timer
                timer_text = f"Next capture in: {countdown:.1f}s"
                timer_color = (0, 255, 255) if countdown < 1.0 else (255, 255, 255)
                cv2.putText(display_frame, timer_text, (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
                
                cv2.putText(display_frame, "Q: Quit", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f"Auto-Register: {name}", display_frame)
                
                # Q to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️  Registration cancelled by user")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # 🎊 Easter Egg: Special completion message
        print(f"\n{'='*60}")
        if captured_count >= num_images:
            print(f"🎉 REGISTRATION COMPLETE! 🎉")
            print(f"   Name: {name}")
            print(f"   Images: {captured_count}")
            print(f"   Achievements Unlocked: {self.achievements_unlocked} 🏆")
            print(f"   Location: {user_folder}")
            print(f"\n   🌟 You're now officially part of the system!")
            print(f"   💪 Your face game is STRONG!")
            return True
        else:
            print(f"⚠️  REGISTRATION INCOMPLETE")
            print(f"   Captured: {captured_count}/{num_images}")
            print(f"   Location: {user_folder}")
            return False
        print(f"{'='*60}\n")
    
    def list_registered_users(self):
        """List all registered users"""
        users = [d.name for d in self.enrollment_dir.iterdir() if d.is_dir()]
        
        if not users:
            print("No registered users found")
            # 🎭 Easter Egg: Empty state message
            print("🌵 It's lonely in here... Time to register someone!")
            return []
        
        print(f"\n{'='*60}")
        print(f"REGISTERED USERS ({len(users)})")
        print(f"{'='*60}")
        
        for i, user in enumerate(users, 1):
            user_folder = self.enrollment_dir / user
            img_count = len(list(user_folder.glob("*.jpg")))
            # 🎭 Easter Egg: Fun emoji based on image count
            emoji = "🔥" if img_count >= 10 else "📸" if img_count >= 5 else "🌱"
            print(f"  [{i}] {user:20s} - {img_count} images {emoji}")
        
        print(f"{'='*60}\n")
        return users
    
    def delete_user(self, name):
        """Delete a registered user"""
        user_folder = self.enrollment_dir / name
        
        if not user_folder.exists():
            print(f"❌ User '{name}' not found")
            return False
        
        img_count = len(list(user_folder.glob("*.jpg")))
        print(f"\n⚠️  About to delete user: {name} ({img_count} images)")
        # 🎭 Easter Egg: Dramatic deletion warning
        print(f"💀 This action cannot be undone! Gone forever! Poof! 💨")
        confirm = input("Are you sure? (yes/no): ")
        
        if confirm.lower() == 'yes':
            import shutil
            shutil.rmtree(user_folder)
            print(f"✅ User '{name}' deleted")
            print(f"👋 Goodbye, {name}! May we meet again in another database...")
            return True
        else:
            print("❌ Deletion cancelled")
            print("😅 Phew! That was close!")
            return False


def registration_menu():
    """Interactive menu for face registration"""
    from ultralytics import YOLO
    
    # 🎭 Easter Egg: ASCII art banner
    print("\n" + "="*60)
    print("   _____ __  __   _   ___ _____   ___ ___ ___ ")
    print("  / ____|  \/  | /_\ | _ |_   _| | _ | __/ __|")
    print("  \__ \| |\/| |/ _ \|   / | |   |   /|  _\__ \\")
    print("  |___/|_|  |_/_/ \_|_|_\ |_|   |_|_|___|___/")
    print("       FACE REGISTRATION - AUTO MODE 🎬")
    print("="*60)
    
    print("\nLoading YOLO model...")
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path in the code")
        return
    
    yolo = YOLO(model_path)
    print("✅ Model loaded\n")
    print("🎮 Easter Egg Mode: ACTIVATED!\n")
    
    reg_system = FaceRegistration(yolo, enrollment_dir="enrollment")
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Register new person 📸")
        print("2. List registered users 📋")
        print("3. Delete user 🗑️")
        print("4. Exit 👋")
        print("="*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            name = input("\nEnter person's name: ").strip()
            if not name:
                print("❌ Name cannot be empty")
                continue
            
            if not name.replace(" ", "").replace("_", "").isalnum():
                print("❌ Name can only contain letters, numbers, spaces, and underscores")
                continue
            
            try:
                num_images = int(input("Number of images to capture (default 10): ") or "10")
                if num_images < 5 or num_images > 50:
                    print("❌ Number of images must be between 5 and 50")
                    continue
            except ValueError:
                print("❌ Invalid number")
                continue
            
            reg_system.register_new_person(name, num_images=num_images)
        
        elif choice == '2':
            reg_system.list_registered_users()
        
        elif choice == '3':
            users = reg_system.list_registered_users()
            if users:
                name = input("\nEnter name to delete: ").strip()
                reg_system.delete_user(name)
        
        elif choice == '4':
            print("\n" + "="*60)
            print("👋 Thanks for using SMART ATTENDANCE!")
            print("🎉 You unlocked the auto-capture Easter egg!")
            print("✨ May your captures be swift and your faces be sharp!")
            print("="*60 + "\n")
            break
        
        else:
            print("❌ Invalid choice")
            # 🎭 Easter Egg: Fun error message
            print("🤔 I'm a simple menu, I only understand 1-4!")


if __name__ == "__main__":
    registration_menu()
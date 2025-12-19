"""
Streamlit UI for Smart Face Attendance System
Features:
- Face Registration
- Live Camera Attendance
- CCTV Video Upload & Processing
- Attendance Reports & Analytics
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime, timedelta
import tempfile
import os
from PIL import Image
import io

# Import your main system
from main import FaceAttendanceSystem

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        background-color: #48bb78;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f56565;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #4299e1;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.database_built = False
    st.session_state.camera_active = False
    st.session_state.registration_images = []
    st.session_state.processing_video = False

def initialize_system():
    """Initialize the face attendance system"""
    if st.session_state.system is None:
        with st.spinner("🔧 Initializing system..."):
            try:
                system = FaceAttendanceSystem(
                    yolo_model_path="best.pt",
                    enrollment_dir="enrollment",
                    database_path="face_database.pkl",
                    attendance_log="attendance_log.csv",
                    embedding_dim=512,
                    reduced_dim=128,
                    similarity_threshold=0.6
                )
                st.session_state.system = system
                
                # Check if database exists
                if system.face_database['faiss_index'] is not None:
                    st.session_state.database_built = True
                
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                return False
    return True

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Smart Face Attendance System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Face Recognition • Real-time Detection • AI-Powered</p>', unsafe_allow_html=True)
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    system = st.session_state.system
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "👤 Face Registration", "👥 User Management", "📹 Live Attendance", "🎥 Video Processing", "📊 Reports & Analytics", "⚙️ Settings"]
    )
    
    # Sidebar system info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    if st.session_state.database_built:
        num_users = len(system.face_database['names'])
        st.sidebar.success(f"✅ Database Active")
        st.sidebar.info(f"👥 {num_users} Registered Users")
    else:
        st.sidebar.warning("⚠️ No Database")
        st.sidebar.info("Build database to start")
    
    st.sidebar.markdown("---")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(system)
    elif page == "👤 Face Registration":
        show_registration(system)
    elif page == "👥 User Management":
        show_user_management(system)
    elif page == "📹 Live Attendance":
        show_live_attendance(system)
    elif page == "🎥 Video Processing":
        show_video_processing(system)
    elif page == "📊 Reports & Analytics":
        show_reports(system)
    elif page == "⚙️ Settings":
        show_settings(system)

def show_dashboard(system):
    """Dashboard page"""
    st.header("🏠 Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.database_built:
            num_users = len(system.face_database['names'])
        else:
            num_users = 0
        st.metric("👥 Registered Users", num_users)
    
    with col2:
        if os.path.exists(system.attendance_log):
            df = pd.read_csv(system.attendance_log)
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = len(df[df['Date'] == today])
        else:
            today_count = 0
        st.metric("✅ Today's Attendance", today_count)
    
    with col3:
        if os.path.exists(system.attendance_log):
            total_records = len(df)
        else:
            total_records = 0
        st.metric("📊 Total Records", total_records)
    
    with col4:
        if st.session_state.database_built:
            actual_dim = system.face_database.get('actual_dim', 0)
        else:
            actual_dim = 0
        st.metric("🔬 PCA Dimensions", f"{actual_dim}D")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔨 Build/Rebuild Database", use_container_width=True):
            with st.spinner("Building database..."):
                success = system.build_database_from_enrollment()
                if success:
                    st.session_state.database_built = True
                    st.success("✅ Database built successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to build database")
    
    with col2:
        if st.button("📹 Start Live Camera", use_container_width=True):
            if st.session_state.database_built:
                st.info("👉 Go to 'Live Attendance' page")
            else:
                st.warning("⚠️ Please build database first")
    
    with col3:
        if st.button("👤 Register New Person", use_container_width=True):
            st.info("👉 Go to 'Face Registration' page")
    
    st.markdown("---")
    
    # Recent attendance
    st.subheader("📅 Recent Attendance")
    
    if os.path.exists(system.attendance_log):
        df = pd.read_csv(system.attendance_log)
        if not df.empty:
            # Show last 10 records
            recent_df = df.tail(10).iloc[::-1]
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("📭 No attendance records yet")
    else:
        st.info("📭 No attendance records yet")
    
    # Database info
    if st.session_state.database_built:
        st.markdown("---")
        st.subheader("🗄️ Database Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Registered Users:**")
            for i, name in enumerate(system.face_database['names'], 1):
                st.write(f"{i}. {name}")
        
        with col2:
            st.write("**Technical Details:**")
            st.write(f"- Original Embedding: {system.embedding_dim}D")
            actual_dim = system.face_database.get('actual_dim', 0)
            st.write(f"- Reduced Embedding: {actual_dim}D")
            st.write(f"- FAISS Vectors: {system.face_database['faiss_index'].ntotal}")
            if system.face_database['pca_model']:
                variance = np.sum(system.face_database['pca_model'].explained_variance_ratio_) * 100
                st.write(f"- Variance Retained: {variance:.2f}%")

def show_registration(system):
    """Face registration page"""
    st.header("👤 Face Registration")
    
    st.info("📸 Register new people to the system by capturing or uploading face images")
    
    # Registration method selection
    method = st.radio(
        "Choose Registration Method:",
        ["📸 Webcam Capture", "📁 Upload Images"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if method == "📸 Webcam Capture":
        show_webcam_registration(system)
    else:
        show_upload_registration(system)

def show_webcam_registration(system):
    """Webcam registration interface"""
    st.subheader("📸 Webcam Registration")
    
    # User details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        person_name = st.text_input("👤 Person's Name", placeholder="Enter full name")
    
    with col2:
        num_images = st.number_input("📷 Number of Images", min_value=5, max_value=20, value=10)
    
    if person_name and st.button("🎬 Start Webcam Registration", use_container_width=True):
        # Check if user exists
        user_folder = Path(system.enrollment_dir) / person_name
        if user_folder.exists():
            st.warning(f"⚠️ User '{person_name}' already exists!")
            if st.button("➕ Add More Images"):
                st.info("Running external registration script...")
                st.code(f"Run: python face_registration.py")
        else:
            st.info("🎬 Starting webcam registration...")
            st.info("This will open a separate window. Please allow camera access.")
            st.code(f"Run: python face_registration.py\nThen select option 1 and enter name: {person_name}")
            
            st.warning("⚠️ Note: Webcam capture opens in a separate window. Use the terminal interface.")

def show_upload_registration(system):
    """Upload images registration interface"""
    st.subheader("📁 Upload Images Registration")
    
    # User details
    person_name = st.text_input("👤 Person's Name", placeholder="Enter full name")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Face Images (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Preview images
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:5]):
            with cols[idx]:
                image = Image.open(file)
                st.image(image, caption=f"Image {idx+1}", use_container_width=True)
        
        if len(uploaded_files) > 5:
            st.info(f"... and {len(uploaded_files) - 5} more images")
    
    if st.button("💾 Save Registration", disabled=not (person_name and uploaded_files), use_container_width=True):
        if not person_name:
            st.error("❌ Please enter a name")
            return
        
        if len(uploaded_files) < 5:
            st.error("❌ Please upload at least 5 images")
            return
        
        # Create user folder
        user_folder = Path(system.enrollment_dir) / person_name
        user_folder.mkdir(parents=True, exist_ok=True)
        
        # Save images
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        saved_count = 0
        for idx, file in enumerate(uploaded_files):
            try:
                # Read image
                image = Image.open(file)
                img_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                
                # Save image
                img_path = user_folder / f"img{idx+1}.jpg"
                cv2.imwrite(str(img_path), img_bgr)
                saved_count += 1
                
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Saving... {idx+1}/{len(uploaded_files)}")
                
            except Exception as e:
                st.error(f"❌ Error saving {file.name}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        if saved_count > 0:
            st.success(f"✅ Successfully saved {saved_count} images for {person_name}!")
            st.info("🔨 Remember to rebuild the database to include this person")
        else:
            st.error("❌ Failed to save images")

def show_user_management(system):
    """User management page - view, edit, and delete registered users"""
    st.header("👥 User Management")
    
    st.info("📂 Manage registered users - view details, add images, or remove users")
    
    enrollment_path = Path(system.enrollment_dir)
    
    if not enrollment_path.exists():
        st.warning("📭 No enrollment directory found")
        return
    
    # Get all registered users
    user_folders = [d for d in enrollment_path.iterdir() if d.is_dir()]
    
    if not user_folders:
        st.warning("📭 No registered users found")
        st.info("👉 Go to 'Face Registration' to add users")
        return
    
    # Display total users
    st.success(f"✅ Total Registered Users: {len(user_folders)}")
    
    st.markdown("---")
    
    # User list with actions
    st.subheader("📋 Registered Users")
    
    # Search/filter
    search_term = st.text_input("🔍 Search users", placeholder="Type to filter...")
    
    # Filter users
    if search_term:
        filtered_users = [u for u in user_folders if search_term.lower() in u.name.lower()]
    else:
        filtered_users = user_folders
    
    if not filtered_users:
        st.warning(f"No users found matching '{search_term}'")
        return
    
    # Display users in a grid
    for user_folder in sorted(filtered_users, key=lambda x: x.name):
        user_name = user_folder.name
        image_files = list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.png"))
        num_images = len(image_files)
        
        # Create expandable section for each user
        with st.expander(f"👤 {user_name} ({num_images} images)", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**User Information:**")
                st.write(f"- Name: {user_name}")
                st.write(f"- Total Images: {num_images}")
                st.write(f"- Location: `{user_folder}`")
                
                # Show image previews
                if image_files:
                    st.write("**Image Previews:**")
                    preview_cols = st.columns(5)
                    for idx, img_path in enumerate(image_files[:5]):
                        with preview_cols[idx]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                    
                    if num_images > 5:
                        st.info(f"... and {num_images - 5} more images")
            
            with col2:
                st.write("**Actions:**")
                
                # View all images
                if st.button(f"🖼️ View All Images", key=f"view_{user_name}"):
                    st.session_state[f'show_all_{user_name}'] = True
                
                # Add more images
                st.write("---")
                add_images = st.file_uploader(
                    "➕ Add More Images",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key=f"add_{user_name}"
                )
                
                if add_images:
                    if st.button(f"💾 Save {len(add_images)} Images", key=f"save_{user_name}"):
                        saved_count = 0
                        for idx, file in enumerate(add_images):
                            try:
                                image = Image.open(file)
                                img_array = np.array(image)
                                
                                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                else:
                                    img_bgr = img_array
                                
                                # Find next available number
                                existing_nums = [int(f.stem.replace('img', '')) for f in image_files if f.stem.startswith('img') and f.stem[3:].isdigit()]
                                next_num = max(existing_nums) + 1 + idx if existing_nums else idx + 1
                                
                                img_path = user_folder / f"img{next_num}.jpg"
                                cv2.imwrite(str(img_path), img_bgr)
                                saved_count += 1
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
                        if saved_count > 0:
                            st.success(f"✅ Added {saved_count} images!")
                            st.info("🔨 Rebuild database to update recognition")
                            time.sleep(1)
                            st.rerun()
                
                # Delete user section
                st.write("---")
                st.write("**⚠️ Danger Zone:**")
                
                # Two-step deletion
                delete_key = f"delete_confirm_{user_name}"
                if delete_key not in st.session_state:
                    st.session_state[delete_key] = False
                
                if not st.session_state[delete_key]:
                    if st.button(f"🗑️ Delete User", key=f"delete_{user_name}", type="secondary"):
                        st.session_state[delete_key] = True
                        st.rerun()
                else:
                    st.warning(f"⚠️ Delete {user_name}?")
                    st.write(f"This will remove {num_images} images")
                    
                    col_yes, col_no = st.columns(2)
                    
                    with col_yes:
                        if st.button("✅ Yes, Delete", key=f"confirm_{user_name}", type="primary"):
                            try:
                                import shutil
                                shutil.rmtree(user_folder)
                                st.success(f"✅ Deleted {user_name}")
                                st.info("🔨 Rebuild database to update system")
                                st.session_state[delete_key] = False
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting user: {e}")
                    
                    with col_no:
                        if st.button("❌ Cancel", key=f"cancel_{user_name}"):
                            st.session_state[delete_key] = False
                            st.rerun()
            
            # Show all images modal
            if st.session_state.get(f'show_all_{user_name}', False):
                st.write("---")
                st.write("**📸 All Images:**")
                
                # Display all images in a grid
                cols_per_row = 4
                for i in range(0, len(image_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, img_path in enumerate(image_files[i:i+cols_per_row]):
                        with cols[j]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=img_path.name, use_container_width=True)
                                
                                # Delete individual image
                                if st.button(f"🗑️ Delete", key=f"del_img_{img_path.name}_{i}_{j}"):
                                    try:
                                        img_path.unlink()
                                        st.success(f"✅ Deleted {img_path.name}")
                                        time.sleep(0.5)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            except Exception as e:
                                st.error(f"Error loading {img_path.name}")
                
                if st.button("✖️ Close Gallery", key=f"close_{user_name}"):
                    st.session_state[f'show_all_{user_name}'] = False
                    st.rerun()
    
    # Bulk actions
    st.markdown("---")
    st.subheader("⚙️ Bulk Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔨 Rebuild Database", use_container_width=True, type="primary"):
            with st.spinner("Rebuilding database..."):
                success = system.build_database_from_enrollment()
                if success:
                    st.session_state.database_built = True
                    st.success("✅ Database rebuilt successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to rebuild database")
    
    with col2:
        st.metric("Total Users", len(user_folders))
    
    with col3:
        total_images = sum(len(list(u.glob("*.jpg")) + list(u.glob("*.png"))) for u in user_folders)
        st.metric("Total Images", total_images)

def show_live_attendance(system):
    """Live camera attendance page"""
    st.header("📹 Live Camera Attendance")
    
    if not st.session_state.database_built:
        st.warning("⚠️ Please build the database first!")
        if st.button("🔨 Build Database Now"):
            with st.spinner("Building database..."):
                success = system.build_database_from_enrollment()
                if success:
                    st.session_state.database_built = True
                    st.success("✅ Database built!")
                    st.rerun()
        return
    
    st.info("📸 Real-time face recognition using your webcam")
    
    # Camera settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        camera_id = st.number_input("📷 Camera ID", min_value=0, max_value=5, value=0)
    
    with col2:
        confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        system.similarity_threshold = confidence_threshold
    
    with col3:
        show_fps = st.checkbox("📊 Show FPS", value=True)
    
    st.markdown("---")
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("▶️ Start Camera", use_container_width=True, type="primary")
    
    with col2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    
    # Placeholder for video feed
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if start_button:
        st.session_state.camera_active = True
    
    if stop_button:
        st.session_state.camera_active = False
        video_placeholder.empty()
        status_placeholder.info("📷 Camera stopped")
    
    # Run camera
    if st.session_state.camera_active:
        run_live_camera(system, camera_id, show_fps, video_placeholder, status_placeholder)

def run_live_camera(system, camera_id, show_fps, video_placeholder, status_placeholder):
    """Run live camera with face recognition"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        status_placeholder.error("❌ Could not open camera")
        st.session_state.camera_active = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    attendance_cooldown = {}
    cooldown_duration = 5
    
    frame_count = 0
    
    status_placeholder.success("✅ Camera is running...")
    
    try:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Detect faces every 3 frames
            if frame_count % 3 == 0:
                results = system.yolo(frame, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        
                        if conf < 0.5:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        
                        # Expand bbox
                        h, w = frame.shape[:2]
                        margin = 10
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        face = frame[y1:y2, x1:x2]
                        
                        # Recognize face
                        name, confidence = system.recognize_face(face)
                        
                        # Draw bounding box
                        if name:
                            color = (0, 255, 0)
                            label = f"{name} ({confidence:.2f})"
                            
                            # Mark attendance
                            if name not in attendance_cooldown or \
                               (current_time - attendance_cooldown[name]) > cooldown_duration:
                                if system.mark_attendance(name):
                                    attendance_cooldown[name] = current_time
                        else:
                            color = (0, 0, 255)
                            label = f"Unknown ({confidence:.2f})"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS calculation
            fps_frame_count += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
            
            # Display info
            if show_fps:
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Small delay
            time.sleep(0.03)
    
    finally:
        cap.release()
        st.session_state.camera_active = False

def show_video_processing(system):
    """Video processing page for CCTV footage"""
    st.header("🎥 CCTV Video Processing")
    
    if not st.session_state.database_built:
        st.warning("⚠️ Please build the database first!")
        return
    
    st.info("📹 Upload and process CCTV footage for attendance tracking")
    
    # Video uploader
    uploaded_video = st.file_uploader(
        "Upload Video File (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video:
        st.success(f"✅ Video uploaded: {uploaded_video.name}")
        
        # Processing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_every_n_frames = st.number_input("Process Every N Frames", min_value=1, max_value=30, value=5)
        
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
            system.similarity_threshold = confidence_threshold
        
        with col3:
            save_annotated = st.checkbox("💾 Save Annotated Video", value=False)
        
        # Process button
        if st.button("🚀 Process Video", use_container_width=True, type="primary"):
            process_video(system, uploaded_video, process_every_n_frames, save_annotated)

def process_video(system, uploaded_video, process_every_n, save_annotated):
    """Process uploaded video for attendance"""
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"📊 Video Info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        
        # Detection tracking
        detected_persons = set()
        frame_detections = []
        
        # Video writer for annotated video
        out = None
        if save_annotated:
            output_path = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process only every N frames
            if frame_count % process_every_n == 0:
                processed_count += 1
                
                # Detect faces
                results = system.yolo(frame, verbose=False)
                boxes = results[0].boxes
                
                frame_persons = []
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        
                        if conf < 0.5:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        
                        # Expand bbox
                        h, w = frame.shape[:2]
                        margin = 10
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        face = frame[y1:y2, x1:x2]
                        
                        # Recognize face
                        name, confidence = system.recognize_face(face)
                        
                        if name:
                            detected_persons.add(name)
                            frame_persons.append(name)
                            color = (0, 255, 0)
                            label = f"{name} ({confidence:.2f})"
                        else:
                            color = (0, 0, 255)
                            label = f"Unknown ({confidence:.2f})"
                        
                        # Draw on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store detection info
                frame_detections.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'persons': frame_persons
                })
                
                # Show progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: Frame {frame_count}/{total_frames} | Detected: {len(detected_persons)} persons")
                
                # Display current frame
                if processed_count % 5 == 0:  # Update display every 5 processed frames
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Write annotated frame
            if save_annotated and out is not None:
                out.write(frame)
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        st.success("✅ Video processing complete!")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎬 Total Frames", total_frames)
        
        with col2:
            st.metric("🔍 Processed Frames", processed_count)
        
        with col3:
            st.metric("👥 Unique Persons", len(detected_persons))
        
        # Detected persons
        if detected_persons:
            st.subheader("✅ Detected Persons")
            for person in sorted(detected_persons):
                if st.button(f"📝 Mark Attendance: {person}", key=f"mark_{person}"):
                    if system.mark_attendance(person):
                        st.success(f"✅ Attendance marked for {person}")
                    else:
                        st.info(f"ℹ️ {person} already marked today")
        
        # Download annotated video
        if save_annotated and os.path.exists(output_path):
            st.subheader("💾 Download Annotated Video")
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f,
                    file_name=output_path,
                    mime="video/mp4"
                )
        
    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            os.unlink(video_path)

def show_reports(system):
    """Reports and analytics page"""
    st.header("📊 Reports & Analytics")
    
    if not os.path.exists(system.attendance_log):
        st.info("📭 No attendance records yet")
        return
    
    df = pd.read_csv(system.attendance_log)
    
    if df.empty:
        st.info("📭 No attendance records yet")
        return
    
    # Date range selector
    st.subheader("📅 Select Date Range")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Filter data
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df[mask]
    
    # Summary metrics
    st.markdown("---")
    st.subheader("📈 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", len(filtered_df))
    
    with col2:
        unique_persons = filtered_df['Name'].nunique()
        st.metric("👥 Unique Persons", unique_persons)
    
    with col3:
        unique_days = filtered_df['Date'].dt.date.nunique()
        st.metric("📅 Days Covered", unique_days)
    
    with col4:
        if unique_days > 0:
            avg_per_day = len(filtered_df) / unique_days
            st.metric("📊 Avg/Day", f"{avg_per_day:.1f}")
    
    # Attendance table
    st.markdown("---")
    st.subheader("📋 Attendance Records")
    
    # Format for display
    display_df = filtered_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df = display_df.sort_values(['Date', 'Time'], ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download CSV
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=f"attendance_{start_date}_{end_date}.csv",
        mime="text/csv"
    )
    
    # Analytics
    st.markdown("---")
    st.subheader("📊 Analytics")
    
    # Attendance by person
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Attendance by Person**")
        person_counts = filtered_df['Name'].value_counts()
        st.bar_chart(person_counts)
    
    with col2:
        st.write("**Attendance by Date**")
        date_counts = filtered_df.groupby(filtered_df['Date'].dt.date).size()
        st.line_chart(date_counts)
    
    # Detailed person stats
    st.markdown("---")
    st.subheader("👥 Individual Statistics")
    
    for person in sorted(filtered_df['Name'].unique()):
        person_df = filtered_df[filtered_df['Name'] == person]
        
        with st.expander(f"📊 {person} ({len(person_df)} days)"):
            person_display = person_df.copy()
            person_display['Date'] = person_display['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(person_display[['Date', 'Time']], use_container_width=True)

def show_settings(system):
    """Settings page"""
    st.header("⚙️ Settings")
    
    # System configuration
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_threshold = st.slider(
            "🎯 Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=system.similarity_threshold,
            step=0.05,
            help="Higher values = stricter matching"
        )
        
        if st.button("💾 Update Threshold"):
            system.similarity_threshold = new_threshold
            st.success(f"✅ Threshold updated to {new_threshold}")
    
    with col2:
        st.write("**Current Settings:**")
        st.write(f"- Similarity Threshold: {system.similarity_threshold}")
        st.write(f"- Target PCA Dimension: {system.reduced_dim}D")
        st.write(f"- FaceNet Embedding: {system.embedding_dim}D")
    
    # Database management
    st.markdown("---")
    st.subheader("🗄️ Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔨 Rebuild Database", use_container_width=True):
            with st.spinner("Rebuilding database..."):
                success = system.build_database_from_enrollment()
                if success:
                    st.session_state.database_built = True
                    st.success("✅ Database rebuilt successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to rebuild database")
    
    with col2:
        if st.button("📊 Database Info", use_container_width=True):
            if st.session_state.database_built:
                st.info(f"👥 {len(system.face_database['names'])} registered users")
            else:
                st.warning("⚠️ No database built")
    
    with col3:
        if st.button("🗑️ Clear Attendance Log", use_container_width=True):
            if os.path.exists(system.attendance_log):
                if st.checkbox("⚠️ Confirm deletion"):
                    os.remove(system.attendance_log)
                    st.success("✅ Attendance log cleared")
            else:
                st.info("📭 No attendance log found")
    
    # File paths
    st.markdown("---")
    st.subheader("📁 File Paths")
    
    st.code(f"""
Enrollment Directory: {system.enrollment_dir}
Database File: {system.database_path}
Attendance Log: {system.attendance_log}
YOLO Model: best.pt
    """)
    
    # About
    st.markdown("---")
    st.subheader("ℹ️ About")
    
    st.info("""
    **Smart Face Attendance System**
    
    Version: 1.0.0
    
    Features:
    - FaceNet 512-D embeddings
    - PCA dimensionality reduction
    - FAISS similarity search
    - Real-time detection
    - Video processing
    - Attendance tracking
    
    Technology Stack:
    - YOLO (Face Detection)
    - FaceNet (Face Recognition)
    - FAISS (Similarity Search)
    - Streamlit (UI)
    - OpenCV (Video Processing)
    """)

if __name__ == "__main__":
    main()
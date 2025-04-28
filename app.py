import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
import io
import pandas as pd
from datetime import datetime
from db_utils import create_tables, add_student, get_student_by_id, get_all_students, get_active_class_sessions, create_class_session, mark_attendance, get_attendance_for_session, get_all_class_sessions, close_class_session

# Set page configuration
st.set_page_config(page_title="Student Exam Verification System", layout="wide")
# Try to import scikit-image for SSIM, but provide fallback if not available
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    # Use a more subtle info message instead of a warning
    st.info("Using alternative comparison method based on Mean Squared Error. For better accuracy, consider installing scikit-image package.")
    # This info will only show once when the app starts



# Initialize database
create_tables()

# Function to detect faces using OpenCV
def detect_face(image_array):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Preprocessing to improve detection accuracy
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces with more conservative parameters to reduce false positives
    # Increasing minNeighbors to require more evidence for a face detection
    # Using a slightly larger minimum face size to avoid detecting small patterns as faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If multiple faces are detected, keep only the largest one
    # This helps prevent false positives from background elements
    if len(faces) > 1:
        # Convert faces to a list of tuples for easier handling
        faces_list = [(x, y, w, h) for (x, y, w, h) in faces]
        # Sort by area (width * height) in descending order
        faces_list.sort(key=lambda face: face[2] * face[3], reverse=True)
        # Keep only the largest face
        faces = np.array([faces_list[0]], dtype=np.int32)
    
    return faces

# Function to compare faces using multiple comparison methods for better accuracy
def compare_faces(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Resize to same dimensions
    gray1 = cv2.resize(gray1, (100, 100))
    gray2 = cv2.resize(gray2, (100, 100))
    
    # Method 1: Histogram comparison
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare histograms using correlation method
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Method 2: Structural similarity index (SSIM) or enhanced alternative
    # This measures the similarity between two images in terms of luminance, contrast and structure
    if HAVE_SKIMAGE:
        ssim_score = ssim(gray1, gray2)
    else:
        # If skimage is not available, use an enhanced alternative method
        # Apply Gaussian blur to reduce noise sensitivity
        gray1_blur = cv2.GaussianBlur(gray1, (3, 3), 0)
        gray2_blur = cv2.GaussianBlur(gray2, (3, 3), 0)
        
        # Apply Laplacian edge detection to focus on facial features
        gray1_edge = cv2.Laplacian(gray1_blur, cv2.CV_64F)
        gray2_edge = cv2.Laplacian(gray2_blur, cv2.CV_64F)
        
        # Calculate mean squared error as a similarity measure
        mse = np.mean((gray1_blur - gray2_blur) ** 2)
        
        # Calculate edge similarity
        edge_similarity = 1.0 / (1.0 + np.mean(np.abs(gray1_edge - gray2_edge)))
        
        # Calculate normalized cross-correlation for additional comparison
        norm_corr = np.mean(np.multiply(gray1_blur, gray2_blur)) / (np.std(gray1_blur) * np.std(gray2_blur) + 1e-5)
        
        # Combine MSE, edge similarity, and correlation for a more robust score
        ssim_score = (1 / (1 + mse) * 0.4) + (edge_similarity * 0.2) + (norm_corr * 0.4)
    
    # Combine both scores (giving more weight to histogram comparison)
    combined_score = 0.7 * hist_similarity + 0.3 * ssim_score
    
    # Return True if the combined score exceeds the threshold
    return combined_score > 0.4  # Adjusted threshold for combined metrics

# Main title
st.title("Student Exam Verification System")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Register Student", "Verify Student", "Attendance", "View Database"])

# Home page
if page == "Home":
    st.header("Welcome to Student Exam Verification System")
    st.write("""
    This application helps verify students for examinations using facial recognition.
    
    ### Features:
    - Register new students with their ID and photo
    - Verify student identity before exams
    - View and manage student database
    
    Select an option from the sidebar to get started.
    """)
    
    st.image("https://img.freepik.com/free-vector/face-recognition-scanning-concept-illustration_114360-7962.jpg", 
             caption="Facial Recognition System", width=None, use_container_width=True)

# Register Student page
elif page == "Register Student":
    st.header("Register New Student")
    
    # Input fields
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student Name")
    
    # Photo upload
    uploaded_file = st.file_uploader("Upload Student Photo (Passport size recommended)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Convert to bytes for storage
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Process the image to detect face
        try:
            # Convert PIL Image to numpy array for OpenCV
            img_array = np.array(image)
            
            # Find face locations
            faces = detect_face(img_array)
            
            # Create a copy of the image for visualization
            debug_img = img_array.copy()
            
            # Draw rectangles around detected faces for debugging
            for (x, y, w, h) in faces:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show the debug image with face detections
            if len(faces) > 0:
                st.image(debug_img, caption=f"Face Detection Results: {len(faces)} face(s) detected", width=300)
            
            if len(faces) == 0:
                st.error("No face detected in the image. Please upload a clear photo with a face.")
                st.info("Tips: Ensure good lighting, face is clearly visible, and try a different angle if needed.")
            elif len(faces) > 1:
                st.warning(f"System detected {len(faces)} faces in the image. This may be incorrect - the system sometimes detects parts of the background or shadows as faces.")
                st.info("The green rectangles show what the system is detecting as faces. Try uploading a different photo with: Better lighting, plain background, face centered in the frame, and no accessories (if possible).")
                
                # Add override option
                override = st.checkbox("I confirm there is only one face in the image (override detection)")
                
                if override:
                    # Allow registration with override
                    if st.button("Register Student (Override)"):
                        if student_id and student_name:
                            # Save to database - we'll store the image directly
                            add_student(student_id, student_name, img_byte_arr, [])
                            st.success(f"Student {student_name} (ID: {student_id}) registered successfully!")
                        else:
                            st.error("Please fill in all fields.")
            else:
                # Save button
                if st.button("Register Student"):
                    if student_id and student_name:
                        # Save to database - we'll store the image directly
                        add_student(student_id, student_name, img_byte_arr, [])
                        st.success(f"Student {student_name} (ID: {student_id}) registered successfully!")
                    else:
                        st.error("Please fill in all fields.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Verify Student page
elif page == "Verify Student":
    st.header("Verify Student for Examination")
    
    # Two columns for the interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Automatic Face Recognition**")
        st.write("The system will automatically scan your face and verify your identity against the database.")
        
        # Button to start camera verification
        run_verification = st.button("Start Camera for Verification")
        
        if run_verification:
            # Access webcam
            st.write("Looking for camera...")
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access webcam. Please check your camera connection.")
                else:
                    # Create placeholders for the camera feed and results
                    camera_placeholder = st.empty()
                    verification_result = st.empty()
                    verification_details = st.empty()
                    
                    # Get a frame from the camera
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB (for display)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(rgb_frame, caption="Camera Feed", channels="RGB")
                        
                        # Find faces in the frame
                        faces = detect_face(rgb_frame)
                        
                        if len(faces) == 0:
                            verification_result.warning("No face detected. Please position yourself clearly in front of the camera.")
                        elif len(faces) > 1:
                            verification_result.warning("Multiple faces detected. Please ensure only one person is in the frame.")
                        else:
                            # Get all students from database
                            students = get_all_students()
                            
                            # Initialize variables for best match
                            best_match_student = None
                            best_match_score = 0
                            
                            # Compare with all stored images
                            for student in students:
                                stored_img = np.array(Image.open(io.BytesIO(student['photo'])))
                                
                                # Convert images to grayscale
                                gray1 = cv2.cvtColor(stored_img, cv2.COLOR_RGB2GRAY)
                                gray2 = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                                
                                # Resize to same dimensions
                                gray1 = cv2.resize(gray1, (100, 100))
                                gray2 = cv2.resize(gray2, (100, 100))
                                
                                # Calculate histograms
                                hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                                hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                                
                                # Normalize histograms
                                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                                
                                # Method 1: Histogram comparison
                                hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                                
                                # Method 2: Structural similarity index (SSIM) or MSE-based alternative
                                if HAVE_SKIMAGE:
                                    ssim_score = ssim(gray1, gray2)
                                else:
                                    # If skimage is not available, use a simpler method
                                    mse = np.mean((gray1 - gray2) ** 2)
                                    ssim_score = 1 / (1 + mse) if mse > 0 else 1.0
                                
                                # Combine both scores
                                combined_score = 0.7 * hist_similarity + 0.3 * ssim_score
                                
                                # Update best match if this is better
                                if combined_score > best_match_score:
                                    best_match_score = combined_score
                                    best_match_student = student
                            
                            # Check if we found a good match
                            if best_match_score > 0.4:  # Lower threshold for better matching
                                verification_result.success(f"✅ Verification Successful! Student identified: {best_match_student['name']}")
                                verification_details.write(f"**Name:** {best_match_student['name']}")
                                verification_details.write(f"**ID:** {best_match_student['id']}")
                                verification_details.write(f"**Match Confidence:** {best_match_score:.2f}")
                                
                                # Display stored photo
                                stored_image = Image.open(io.BytesIO(best_match_student['photo']))
                                verification_details.image(stored_image, caption="Stored Photo", width=300)
                                
                                st.balloons()
                            else:
                                verification_result.error("❌ Verification Failed! No matching student found in database.")
                    else:
                        st.error("Failed to capture image from camera.")
                    
                    # Release the camera
                    cap.release()
            except Exception as e:
                st.error(f"Error accessing camera: {e}")
    
    with col2:
        # Manual verification option
        st.write("**Manual Verification**")
        st.write("If automatic verification fails, you can manually enter a student ID.")
        
        # Input student ID for verification
        verify_id = st.text_input("Enter Student ID")
        
        if verify_id:
            student = get_student_by_id(verify_id)
            if student:
                st.write(f"**Name:** {student['name']}")
                st.write(f"**ID:** {student['id']}")
                
                # Display stored photo
                stored_image = Image.open(io.BytesIO(student['photo']))
                st.image(stored_image, caption="Stored Photo", width=300)
            else:
                st.error("Student not found. Please check the ID.")

# Attendance page
elif page == "Attendance":
    st.header("Class Attendance Tracking")
    
    # Tabs for different attendance functions
    attendance_tab, sessions_tab, reports_tab = st.tabs(["Take Attendance", "Manage Sessions", "View Reports"])
    
    # Take Attendance tab
    with attendance_tab:
        st.subheader("Take Attendance for Class Session")
        
        # Get active class sessions
        active_sessions = get_active_class_sessions()
        
        if not active_sessions:
            st.warning("No active class sessions found. Please create a new session first.")
            
            # Quick create session
            with st.form("create_session_form"):
                session_name = st.text_input("Session Name (e.g., 'Math Exam 101')")
                session_desc = st.text_area("Session Description (optional)")
                submit_btn = st.form_submit_button("Create Session")
                
                if submit_btn and session_name:
                    session_id = create_class_session(session_name, session_desc)
                    if session_id:
                        st.success(f"Session '{session_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create session. Please try again.")
        else:
            # Select active session
            session_options = {f"{session['name']} (ID: {session['id']})": session['id'] for session in active_sessions}
            selected_session = st.selectbox("Select Class Session", list(session_options.keys()))
            selected_session_id = session_options[selected_session]
            
            st.write(f"Taking attendance for: **{selected_session.split(' (ID:')[0]}**")
            
            # Button to start camera for attendance
            take_attendance = st.button("Start Camera for Attendance")
            
            if take_attendance:
                # Access webcam
                st.write("Looking for camera...")
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Could not access webcam. Please check your camera connection.")
                    else:
                        # Create placeholders for the camera feed and results
                        camera_placeholder = st.empty()
                        recognition_result = st.empty()
                        student_details = st.empty()
                        
                        # Get a frame from the camera
                        ret, frame = cap.read()
                        if ret:
                            # Convert BGR to RGB (for display)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(rgb_frame, caption="Camera Feed", channels="RGB")
                            
                            # Find faces in the frame
                            faces = detect_face(rgb_frame)
                            
                            if len(faces) == 0:
                                recognition_result.warning("No face detected. Please position yourself clearly in front of the camera.")
                            elif len(faces) > 1:
                                recognition_result.warning("Multiple faces detected. Please ensure only one student is in the frame.")
                            else:
                                # Get all students from database
                                students = get_all_students()
                                
                                # Initialize variables for best match
                                best_match_student = None
                                best_match_score = 0
                                
                                # Compare with all stored images
                                for student in students:
                                    stored_img = np.array(Image.open(io.BytesIO(student['photo'])))
                                    
                                    # Convert images to grayscale
                                    gray1 = cv2.cvtColor(stored_img, cv2.COLOR_RGB2GRAY)
                                    gray2 = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                                    
                                    # Resize to same dimensions
                                    gray1 = cv2.resize(gray1, (100, 100))
                                    gray2 = cv2.resize(gray2, (100, 100))
                                    
                                    # Calculate histograms
                                    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                                    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                                    
                                    # Normalize histograms
                                    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                                    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                                    
                                    # Method 1: Histogram comparison
                                    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                                    
                                    # Method 2: Structural similarity index (SSIM) or MSE-based alternative
                                    if HAVE_SKIMAGE:
                                        ssim_score = ssim(gray1, gray2)
                                    else:
                                        # If skimage is not available, use a simpler method
                                        mse = np.mean((gray1 - gray2) ** 2)
                                        ssim_score = 1 / (1 + mse) if mse > 0 else 1.0
                                    
                                    # Combine both scores
                                    combined_score = 0.7 * hist_similarity + 0.3 * ssim_score
                                    
                                    # Update best match if this is better
                                    if combined_score > best_match_score:
                                        best_match_score = combined_score
                                        best_match_student = student
                                
                                # Check if we found a good match
                                if best_match_score > 0.4:  # Lower threshold for better matching
                                    # Mark attendance in database
                                    success = mark_attendance(best_match_student['id'], selected_session_id)
                                    
                                    if success:
                                        recognition_result.success(f"✅ Attendance Marked! Student identified: {best_match_student['name']}")
                                        student_details.write(f"**Name:** {best_match_student['name']}")
                                        student_details.write(f"**ID:** {best_match_student['id']}")
                                        student_details.write(f"**Match Confidence:** {best_match_score:.2f}")
                                        
                                        # Display stored photo
                                        stored_image = Image.open(io.BytesIO(best_match_student['photo']))
                                        student_details.image(stored_image, caption="Stored Photo", width=300)
                                        
                                        st.balloons()
                                    else:
                                        recognition_result.error("❌ Failed to mark attendance. Please try again.")
                                else:
                                    recognition_result.error("❌ Recognition Failed! No matching student found in database.")
                        else:
                            st.error("Failed to capture image from camera.")
                        
                        # Release the camera
                        cap.release()
                except Exception as e:
                    st.error(f"Error accessing camera: {e}")
            
            # Display current attendance for this session
            st.subheader("Current Attendance")
            attendance_records = get_attendance_for_session(selected_session_id)
            
            if attendance_records:
                attendance_df = pd.DataFrame(attendance_records)
                attendance_df = attendance_df[["student_id", "student_name", "timestamp"]]
                attendance_df.columns = ["Student ID", "Student Name", "Time Recorded"]
                st.dataframe(attendance_df)
                
                st.write(f"Total students present: {len(attendance_records)}")
            else:
                st.info("No attendance records for this session yet.")
    
    # Manage Sessions tab
    with sessions_tab:
        st.subheader("Manage Class Sessions")
        
        # Create new session form
        with st.expander("Create New Class Session"):
            with st.form("new_session_form"):
                session_name = st.text_input("Session Name (e.g., 'Math Exam 101')")
                session_desc = st.text_area("Session Description (optional)")
                submit_btn = st.form_submit_button("Create Session")
                
                if submit_btn and session_name:
                    session_id = create_class_session(session_name, session_desc)
                    if session_id:
                        st.success(f"Session '{session_name}' created successfully!")
                    else:
                        st.error("Failed to create session. Please try again.")
        
        # List all sessions
        st.subheader("All Class Sessions")
        sessions = get_all_class_sessions()
        
        if sessions:
            for session in sessions:
                with st.expander(f"{session['name']} - {session['date']} (Status: {session['status']})"):
                    st.write(f"**ID:** {session['id']}")
                    st.write(f"**Date:** {session['date']}")
                    st.write(f"**Description:** {session['description'] or 'No description'}")
                    st.write(f"**Status:** {session['status']}")
                    
                    # Close session button (only for active sessions)
                    if session['status'] == 'active':
                        if st.button(f"Close Session", key=f"close_{session['id']}"):
                            if close_class_session(session['id']):
                                st.success("Session closed successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to close session. Please try again.")
                    
                    # Show attendance for this session
                    attendance_records = get_attendance_for_session(session['id'])
                    if attendance_records:
                        st.write(f"**Attendance:** {len(attendance_records)} students present")
                        
                        # Display attendance records
                        attendance_df = pd.DataFrame(attendance_records)
                        attendance_df = attendance_df[["student_id", "student_name", "timestamp"]]
                        attendance_df.columns = ["Student ID", "Student Name", "Time Recorded"]
                        st.dataframe(attendance_df)
                    else:
                        st.info("No attendance records for this session.")
        else:
            st.info("No class sessions found. Create a new session to get started.")
    
    # View Reports tab
    with reports_tab:
        st.subheader("Attendance Reports")
        
        # Get all sessions for reporting
        sessions = get_all_class_sessions()
        
        if sessions:
            # Select session for report
            session_options = {f"{session['name']} - {session['date']}": session['id'] for session in sessions}
            selected_report = st.selectbox("Select Session for Report", list(session_options.keys()))
            selected_report_id = session_options[selected_report]
            
            # Get attendance for selected session
            attendance_records = get_attendance_for_session(selected_report_id)
            
            if attendance_records:
                # Display attendance summary
                st.write(f"**Session:** {selected_report.split(' - ')[0]}")
                st.write(f"**Date:** {selected_report.split(' - ')[1]}")
                st.write(f"**Total Students Present:** {len(attendance_records)}")
                
                # Create DataFrame for display
                attendance_df = pd.DataFrame(attendance_records)
                attendance_df = attendance_df[["student_id", "student_name", "timestamp"]]
                attendance_df.columns = ["Student ID", "Student Name", "Time Recorded"]
                
                # Display as table
                st.dataframe(attendance_df)
                
                # Download CSV option
                csv = attendance_df.to_csv(index=False)
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name=f"attendance_report_{selected_report.split(' - ')[0]}_{selected_report.split(' - ')[1]}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No attendance records found for this session.")
        else:
            st.info("No class sessions found. Create a session and take attendance first.")

# View Database page
elif page == "View Database":
    st.header("Student Database")
    
    # Get all students from database
    students = get_all_students()
    
    if students:
        st.write(f"Total students in database: {len(students)}")
        
        # Display students in a table
        student_table = []
        for student in students:
            student_table.append({
                "ID": student['id'],
                "Name": student['name'],
                "Registration Date": student['registration_date']
            })
        
        st.table(student_table)
        
        # Display individual student details
        selected_student_id = st.selectbox("Select student to view details", 
                                         [student['id'] for student in students])
        
        if selected_student_id:
            student = get_student_by_id(selected_student_id)
            if student:
                st.write(f"### {student['name']} (ID: {student['id']})")
                st.write(f"Registration Date: {student['registration_date']}")
                
                # Display stored photo
                stored_image = Image.open(io.BytesIO(student['photo']))
                st.image(stored_image, caption="Stored Photo", width=300)
    else:
        st.info("No students in the database yet. Please register students first.")
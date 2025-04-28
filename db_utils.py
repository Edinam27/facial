import sqlite3
import json
from datetime import datetime

# Database file path
DB_FILE = 'students.db'

def create_tables():
    """
    Create the necessary tables in the database if they don't exist
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        photo BLOB NOT NULL,
        face_encoding TEXT,
        registration_date TEXT NOT NULL
    )
    ''')
    
    # Create attendance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        class_session_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id),
        FOREIGN KEY (class_session_id) REFERENCES class_sessions(id)
    )
    ''')
    
    # Create class sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS class_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        description TEXT,
        status TEXT DEFAULT 'active'
    )
    ''')
    
    conn.commit()
    conn.close()

def add_student(student_id, name, photo_blob, face_encoding=None):
    """
    Add a new student to the database
    
    Args:
        student_id (str): Unique student ID
        name (str): Student's full name
        photo_blob (bytes): Binary data of student's photo
        face_encoding (list, optional): Face encoding data as a list of floats
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Convert face encoding list to JSON string for storage if provided
        face_encoding_json = json.dumps(face_encoding) if face_encoding else '[]'
        
        # Get current date and time
        registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert student data
        cursor.execute(
            "INSERT INTO students (id, name, photo, face_encoding, registration_date) VALUES (?, ?, ?, ?, ?)",
            (student_id, name, photo_blob, face_encoding_json, registration_date)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding student: {e}")
        return False

def get_student_by_id(student_id):
    """
    Retrieve a student by their ID
    
    Args:
        student_id (str): Student ID to search for
    
    Returns:
        dict: Student data including id, name, photo, face_encoding, and registration_date
              or None if student not found
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        student_row = cursor.fetchone()
        
        if student_row:
            # Convert row to dictionary
            student = dict(student_row)
            # Convert JSON string back to list if it exists
            if student['face_encoding']:
                student['face_encoding'] = json.loads(student['face_encoding'])
            return student
        else:
            return None
    except Exception as e:
        print(f"Error retrieving student: {e}")
        return None
    finally:
        conn.close()

def get_all_students():
    """
    Retrieve all students from the database
    
    Returns:
        list: List of dictionaries containing student data
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM students ORDER BY name")
        student_rows = cursor.fetchall()
        
        students = []
        for row in student_rows:
            student = dict(row)
            if student['face_encoding']:
                student['face_encoding'] = json.loads(student['face_encoding'])
            students.append(student)
        
        return students
    except Exception as e:
        print(f"Error retrieving students: {e}")
        return []
    finally:
        conn.close()

def update_student(student_id, name=None, photo_blob=None, face_encoding=None):
    """
    Update an existing student's information
    
    Args:
        student_id (str): Student ID to update
        name (str, optional): New name
        photo_blob (bytes, optional): New photo data
        face_encoding (list, optional): New face encoding data
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Build update query based on provided parameters
        update_parts = []
        params = []
        
        if name is not None:
            update_parts.append("name = ?")
            params.append(name)
        
        if photo_blob is not None:
            update_parts.append("photo = ?")
            params.append(photo_blob)
        
        if face_encoding is not None:
            update_parts.append("face_encoding = ?")
            params.append(json.dumps(face_encoding))
        
        if not update_parts:
            return False  # Nothing to update
        
        # Add student_id to params
        params.append(student_id)
        
        # Execute update query
        cursor.execute(
            f"UPDATE students SET {', '.join(update_parts)} WHERE id = ?",
            params
        )
        
        if cursor.rowcount == 0:
            return False  # No rows updated, student might not exist
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating student: {e}")
        return False
    finally:
        conn.close()

def delete_student(student_id):
    """
    Delete a student from the database
    
    Args:
        student_id (str): Student ID to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        
        if cursor.rowcount == 0:
            return False  # No rows deleted, student might not exist
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting student: {e}")
        return False
    finally:
        conn.close()

# Class Session Functions
def create_class_session(name, description=None):
    """
    Create a new class session for attendance tracking
    
    Args:
        name (str): Name of the class session (e.g., 'Math Exam 101')
        description (str, optional): Description of the class session
    
    Returns:
        int: ID of the created class session, or None if failed
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get current date and time
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute(
            "INSERT INTO class_sessions (name, date, description) VALUES (?, ?, ?)",
            (name, date, description)
        )
        
        # Get the ID of the inserted row
        session_id = cursor.lastrowid
        
        conn.commit()
        return session_id
    except Exception as e:
        print(f"Error creating class session: {e}")
        return None
    finally:
        conn.close()

def get_active_class_sessions():
    """
    Get all active class sessions
    
    Returns:
        list: List of dictionaries containing class session data
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM class_sessions WHERE status = 'active' ORDER BY date DESC")
        sessions = [dict(row) for row in cursor.fetchall()]
        
        return sessions
    except Exception as e:
        print(f"Error retrieving class sessions: {e}")
        return []
    finally:
        conn.close()

def get_all_class_sessions():
    """
    Get all class sessions
    
    Returns:
        list: List of dictionaries containing class session data
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM class_sessions ORDER BY date DESC")
        sessions = [dict(row) for row in cursor.fetchall()]
        
        return sessions
    except Exception as e:
        print(f"Error retrieving class sessions: {e}")
        return []
    finally:
        conn.close()

def close_class_session(session_id):
    """
    Close a class session (mark as inactive)
    
    Args:
        session_id (int): ID of the class session to close
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE class_sessions SET status = 'closed' WHERE id = ?", (session_id,))
        
        if cursor.rowcount == 0:
            return False  # No rows updated, session might not exist
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error closing class session: {e}")
        return False
    finally:
        conn.close()

# Attendance Functions
def mark_attendance(student_id, class_session_id):
    """
    Mark a student as present for a class session
    
    Args:
        student_id (str): Student ID
        class_session_id (int): Class session ID
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if attendance already marked
        cursor.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND class_session_id = ?",
            (student_id, class_session_id)
        )
        
        if cursor.fetchone():
            return True  # Already marked, consider it successful
        
        # Get current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute(
            "INSERT INTO attendance (student_id, class_session_id, timestamp) VALUES (?, ?, ?)",
            (student_id, class_session_id, timestamp)
        )
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False
    finally:
        conn.close()

def get_attendance_for_session(class_session_id):
    """
    Get all attendance records for a class session
    
    Args:
        class_session_id (int): Class session ID
    
    Returns:
        list: List of dictionaries containing attendance data with student info
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.id, a.student_id, a.timestamp, s.name as student_name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.class_session_id = ?
            ORDER BY a.timestamp
        """, (class_session_id,))
        
        attendance = [dict(row) for row in cursor.fetchall()]
        return attendance
    except Exception as e:
        print(f"Error retrieving attendance: {e}")
        return []
    finally:
        conn.close()

def find_student_by_face(face_image):
    """
    Find a student by comparing their face with all stored faces
    This is a placeholder function - the actual implementation will be in app.py
    using the compare_faces function
    
    Args:
        face_image: The face image to compare
    
    Returns:
        dict: Student data if found, None otherwise
    """
    # This is implemented in app.py using the compare_faces function
    # This is just a placeholder to document the function
    pass
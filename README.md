# Student Exam Verification System

A facial recognition application for verifying students for examinations by comparing their faces to an existing database of pictures.

## Features

- Register students with their ID, name, and photo
- Verify student identity using facial recognition
- View and manage the student database
- Real-time camera verification

## Requirements

- Python 3.8 or higher
- Webcam for verification
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone or download this repository

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. The application will open in your default web browser

3. Use the sidebar to navigate between different pages:

   - **Home**: Overview of the application
   - **Register Student**: Add new students to the database
   - **Verify Student**: Verify student identity using facial recognition
   - **View Database**: Browse and manage the student database

## Registration Process

1. Navigate to the "Register Student" page
2. Enter the student's ID and name
3. Upload a clear passport-sized photo of the student
4. Click "Register Student" to save to the database

## Verification Process

1. Navigate to the "Verify Student" page
2. Enter the student ID to retrieve their record
3. Click "Start Camera for Verification"
4. The system will compare the live camera feed with the stored photo
5. The verification result will be displayed

## Technical Implementation

- The application uses OpenCV's Haar Cascade Classifier for face detection
- Face comparison is done using histogram comparison techniques
- All student data is stored in a SQLite database
- The user interface is built with Streamlit for a responsive experience

## Database Management

The application uses SQLite to store student information. The database file (`students.db`) will be created automatically when you first run the application.

## Security Considerations

- All data is stored locally in the SQLite database
- Consider implementing additional security measures for production use

## Troubleshooting

- **Camera not working**: Ensure your webcam is properly connected and not being used by another application
- **Face not detected**: Make sure there is good lighting and the face is clearly visible
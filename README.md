# Smart Attendance System for Students

**Author:** Fatouma Chirfi Haidara  
**Degree:** BSc Honours — School of Computing, Engineering and Technology  
**University:** Robert Gordon University, Aberdeen  
**Supervisor:** Dr. Eyad Elyan  
**Date:** April 2026

---

## Overview

This project is a facial recognition-based smart attendance system built using Python and open-source tools. It allows students to register their faces and mark attendance automatically via webcam or image upload, without requiring any specialist hardware. The system is deployed as a Streamlit web application and stores all attendance records in a CSV file.

---

## Project Structure

```
smart-attendance/
│
├── app.py                        # Main Streamlit web application
├── encodings_full-4.pickle       # Stored face encodings database
├── attendance.csv                # Attendance log (auto-generated)
├── requirements.txt              # Python dependencies
│
├── Capstone_Project.ipynb        # CNN model training notebook (Google Colab)
├── Data_cleaning_-6.ipynb        # Dataset preparation and cleaning notebook
│
└── README.md                     # This file
```

---

## Features

- **Student Registration** — register a new student by capturing their face via webcam or uploading a photo
- **Real-time Face Identification** — identify registered students via webcam snapshot or uploaded image
- **Attendance Logging** — automatically records name, date, time, and confidence score to a CSV file
- **Duplicate Prevention** — prevents the same student from being logged more than once per day
- **Attendance Log Viewer** — view, filter by date, and download attendance records as CSV
- **Adjustable Match Threshold** — control recognition sensitivity via a sidebar slider

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Core programming language |
| Streamlit | Web application framework |
| DeepFace + FaceNet | Face embedding generation |
| OpenCV | Image processing and face detection |
| face_recognition | Facial encoding and matching |
| MTCNN | Multi-task face detection |
| MobileNetV2 | CNN transfer learning model |
| NumPy / Pandas | Data handling |
| Pickle | Encoding storage |
| Google Colab | Model training environment |

---

## Dataset

The system was trained using the **Labelled Faces in the Wild (LFW)** dataset.  
The dataset was preprocessed as follows:
- Individuals with fewer than 4 images were removed to reduce class imbalance
- Data was split into 80% training and 20% testing
- A balanced six-class subset was used for CNN transfer learning training

The LFW dataset can be downloaded from:  
http://vis-www.cs.umass.edu/lfw/

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/your-username/smart-attendance.git
cd smart-attendance
```

### 2. Create a virtual environment 

```bash
python3 -m venv venv
source venv/bin/activate        # macOS 


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install manually:

```bash
pip install streamlit deepface tf-keras opencv-python-headless numpy pandas pillow scipy face_recognition
```

### 4. Add the encodings file

Place the `encodings_full-4.pickle` file in the same directory as `app.py`.  
This file contains the pre-generated face encodings for the system.  
If the file is not present, the system will still run but you will need to register students first via the Register tab.

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## How to Use

### Registering a Student
1. Open the **Register Student** tab
2. Enter the student's full name
3. Choose to upload a photo or use your webcam
4. Click **Register Student**
5. The face encoding is saved to the pickle file automatically

### Marking Attendance
1. Open the **Upload Photo** or **Webcam** tab
2. Upload an image or take a webcam snapshot
3. The system will detect and identify faces automatically
4. Recognised students are logged to `attendance.csv`

### Viewing Attendance
1. Open the **Attendance Log** tab
2. Filter records by date using the dropdown
3. Download the full log as a CSV file using the download button

## CNN Model Training

The CNN model was trained separately in Google Colab using the `Capstone_Project.ipynb` notebook. The training pipeline includes:

1. Loading and cleaning the LFW dataset metadata
2. MTCNN face detection on sample images
3. Binary CNN classifier (George W. Bush vs all others) using a Sequential architecture
4. Six-class MobileNetV2 transfer learning model trained in two phases:
   - **Phase 1:** Base model frozen, classification head trained for up to 15 epochs
   - **Phase 2:** Top 30 layers unfrozen and fine-tuned at learning rate 1×10⁻⁵
5. Model evaluated using accuracy, precision, recall, and confusion matrix

To retrain the model, open `Capstone_Project.ipynb` in Google Colab and mount your Google Drive with the LFW dataset.

## Data Cleaning

The `Data_cleaning_-6.ipynb` notebook handles dataset preparation:
- Filters individuals with fewer than 4 images from the LFW dataset
- Copies selected faces into a clean directory (`selected_faces`)
- Exports the cleaned dataset as a zip file for use in training


## References
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
LFW Dataset: http://vis-www.cs.umass.edu/lfw/

from pathlib import Path
from datetime import datetime
import pickle
import os
import tempfile
 
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.spatial.distance import cosine
 
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    DEEPFACE_ERROR = ""
except Exception as e:
    DeepFace = None
    DEEPFACE_OK = False
    DEEPFACE_ERROR = str(e)
 
# Paths
BASE_DIR       = Path(__file__).resolve().parent
ENCODINGS_FILE = BASE_DIR / "encodings_full-4.pickle"
CSV_FILE       = BASE_DIR / "attendance.csv"
 
MODEL_NAME = "Facenet"
DETECTOR   = "opencv"
 
# Page config
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
)
 
st.title("🎓 Smart Attendance System for Students")
st.caption("Upload a photo or use your webcam to mark attendance automatically.")
 
# Startup check
if not DEEPFACE_OK:
    st.error("DeepFace could not be loaded.")
    st.code(DEEPFACE_ERROR)
    st.info("Run:  pip install deepface tf-keras")
    st.stop()
 
# Load encodings
@st.cache_resource
def load_encodings():
    if not ENCODINGS_FILE.exists():
        return [], []
    try:
        with ENCODINGS_FILE.open("rb") as f:
            data = pickle.load(f)

        # ✅ support both possible keys (safe)
        encodings = data.get("encodings", data.get("encodings_full", []))
        encodings = [np.asarray(e, dtype=np.float64) for e in encodings]

        names = list(data.get("names", []))

        return encodings, names

    except Exception as e:
        st.sidebar.error(f"Could not read encodings file: {e}")
        return [], []
 
 
known_encodings, known_names = load_encodings()
st.sidebar.success(f"✅ Loaded {len(known_names)} face encodings")
 
match_threshold = st.sidebar.slider(
    "Match threshold (lower = stricter)",
    min_value=0.20,
    max_value=0.60,
    value=0.35,        
    step=0.01,
    help="Cosine distance cutoff. Faces further than this are marked Unrecognised.",
)
 
 
def save_encodings(encodings, names):
    data = {"encodings": encodings, "names": names}
    with ENCODINGS_FILE.open("wb") as f:
        pickle.dump(data, f)
 
 
# Embedding helpers
def get_embedding(img_array: np.ndarray):
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        Image.fromarray(img_array).save(tmp_path)

        result = DeepFace.represent(
            img_path=tmp_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )

        return np.asarray(result[0]["embedding"], dtype=np.float64)

    except Exception as e:
        print("Embedding error:", e)
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
 
def match_embedding(query_emb, threshold):
    if len(known_encodings) == 0 or query_emb is None:
        return "Unrecognised", 0.0
    distances = [cosine(query_emb, k) for k in known_encodings]
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]
    if best_dist <= threshold:
        confidence = max(0.0, (1.0 - best_dist)) * 100
        return known_names[best_idx], confidence
    return "Unrecognised", 0.0
 
# Attendance log
def load_attendance():
    if CSV_FILE.exists():
        try:
            return pd.read_csv(CSV_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Name", "Date", "Time", "Confidence"])
 
 
def log_attendance(name, confidence):
    df    = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
 
    if "(" in name and name.endswith(")"):
        display_name = name[:name.rfind("(")].strip()
       
    else:
        display_name = name.replace("_", " ")
        student_id   = "N/A"
 
    already = (
        (df["Name"] == display_name) &
 
        (df["Date"] == today)
    ).any()
 
    if not already:
        new_row = pd.DataFrame([{
            "Name":       display_name,
            "Date":       today,
            "Time":       now,
            "Confidence": f"{confidence:.1f}%",
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        return True
    return False
 
 
#  Face recognition on full image
def recognise_faces(image_array: np.ndarray, threshold: float):
    annotated = image_array.copy()
    results   = []
 
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(image_array).save(tmp_path)
        detections = DeepFace.extract_faces(
            img_path=tmp_path,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
    except Exception:
        detections = []
    finally:
        os.unlink(tmp_path)
 
    for det in detections:
        region = det.get("facial_area", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
 
        if w == 0 or h == 0:
            continue
 
        face_crop = image_array[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
 
        query_emb        = get_embedding(face_crop)
        name, confidence = match_embedding(query_emb, threshold)
 
        color = (0, 200, 0) if name != "Unrecognised" else (200, 0, 0)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(annotated, (x, y+h-30), (x+w, y+h), color, cv2.FILLED)
        cv2.putText(
            annotated,
            name.replace("_", " "),
            (x+4, y+h-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
 
        results.append({"name": name, "confidence": confidence})
 
        if name != "Unrecognised" and log_attendance(name, confidence):
            st.toast(f"✅ {name.replace('_', ' ')} marked present!", icon="🎓")
 
    return annotated, results, len(results)
 
 
#  Warning if no encodings file
if not ENCODINGS_FILE.exists():
    st.warning("encodings_full-4.pickle not found — register students in the Register tab first.")
 
# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Upload Photo",
    "📹 Webcam",
    "📋 Attendance Log",
    "📝 Register Student",
])
 
#  Tab 1 : Upload
with tab1:
    st.subheader("Upload a photo to identify faces")
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="identify_upload"
    )
 
    if uploaded_file:
        image_pil   = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image_pil)
 
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="Original", use_container_width=True)
 
        with st.spinner("Recognising faces..."):
            annotated, results, detected_faces = recognise_faces(image_array, match_threshold)
 
        st.write(f"Detected faces: {detected_faces}")
        with col2:
            st.image(annotated, caption="Result", use_container_width=True)
 
        st.divider()
        if results:
            st.subheader(f"Found {len(results)} face(s):")
            for r in results:
                if r["name"] != "Unrecognised":
                    st.success(f"✅ **{r['name'].replace('_', ' ')}** — {r['confidence']:.1f}% confidence")
                else:
                    st.error("❌ Unknown person")
        else:
            st.warning("No faces detected in this image.")
 
#  Tab 2 : Webcam
with tab2:
    st.subheader("Use webcam to capture and identify")
    st.info("Allow camera access in your browser, then take a snapshot.")
 
    snap = st.camera_input("Take a photo", key="identify_camera")
 
    if snap:
        image_pil   = Image.open(snap).convert("RGB")
        image_array = np.array(image_pil)
 
        with st.spinner("Recognising faces..."):
            annotated, results, detected_faces = recognise_faces(image_array, match_threshold)
 
        st.image(annotated, caption="Result", use_container_width=True)
 
        if results:
            for r in results:
                if r["name"] != "Unrecognised":
                    st.success(f"✅ **{r['name'].replace('_', ' ')}** — {r['confidence']:.1f}% confidence")
                else:
                    st.error("❌ Unknown person")
        else:
            st.warning("No faces detected.")
 
# Tab 3 : Attendance log
with tab3:
    st.subheader("Attendance Records")
 
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_attendance"):
            st.rerun()
        if st.button("🗑️ Clear Today", key="clear_today"):
            df    = load_attendance()
            today = datetime.now().strftime("%Y-%m-%d")
            df    = df[df["Date"] != today]
            df.to_csv(CSV_FILE, index=False)
            st.rerun()
 
    df = load_attendance()
 
    if df.empty:
        st.info("No attendance recorded yet.")
    else:
        today    = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]
 
        m1, m2, m3 = st.columns(3)
        m1.metric("Present Today",  len(today_df))
        m2.metric("Total Records",  len(df))
        m3.metric("Unique People",  df["Name"].nunique())
 
        st.divider()
        dates         = sorted(df["Date"].unique(), reverse=True)
        selected_date = st.selectbox("Filter by date", ["All"] + list(dates), key="date_filter")
        if selected_date != "All":
            df = df[df["Date"] == selected_date]
 
        st.dataframe(df, use_container_width=True, hide_index=True)
 
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_attendance",
        )
 
# Tab 4 : Register
with tab4:
    st.subheader("Register New Student")
 
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", key="student_name",
                             placeholder="e.g. Fatouma Haidara")
    
 
    option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"],
                      key="register_method")
 
    image = None
 
    if option == "Upload Image":
        uploaded = st.file_uploader(
            "Upload student photo", type=["jpg", "jpeg", "png"], key="register_upload"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        snap = st.camera_input("Take a photo", key="register_camera")
        if snap:
            image = Image.open(snap).convert("RGB")
 
    if image is not None:
        st.image(image, caption="Captured Image", use_container_width=True)
 
        if st.button("Register Student", key="register_student"):
            if name.strip() == "":
                st.warning("⚠️ Please enter the student's full name.")
            else:
                if name.strip() in known_names:
                    st.error(f"❌ {name.strip()} is already registered.")
                else:
                    with st.spinner("Generating face embedding..."):
                        emb = get_embedding(np.array(image))
                    if emb is None:
                        st.error("No face detected. Please try a clearer photo.")
                    else:
                        known_encodings.append(emb)
                        known_names.append(name.strip())
                        save_encodings(known_encodings, known_names)
                        st.success(f"✅ {name.strip()} registered successfully!")
                        st.cache_resource.clear()
                        st.rerun()
 
 
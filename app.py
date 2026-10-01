import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import math
import time


st.set_page_config(
    page_title="Gesture RAG Assistant",
    layout="wide"
)

st.title("🤖 Gesture RAG Assistant")

st.write(
    "Upload a PDF, ask a question, and use hand gestures "
    "to control the assistant."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "gesture" not in st.session_state:
    st.session_state.gesture = "NO HAND"

if "search_requested" not in st.session_state:
    st.session_state.search_requested = False

if "next_requested" not in st.session_state:
    st.session_state.next_requested = False

if "clear_requested" not in st.session_state:
    st.session_state.clear_requested = False


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

pdf_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if pdf_file is not None:

    st.success(
        f"PDF uploaded: {pdf_file.name}"
    )

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    chunk_size = 500

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


    st.write(
        f"Total characters: {len(text)}"
    )

    st.write(
        f"Total chunks: {len(chunks)}"
    )


    if chunks:

        # --------------------------------------------------
        # EMBEDDINGS
        # --------------------------------------------------

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        embeddings = model.encode(chunks)

        embedding_dimension = embeddings.shape[1]


        # --------------------------------------------------
        # FAISS
        # --------------------------------------------------

        index = faiss.IndexFlatL2(
            embedding_dimension
        )

        vectors = np.array(
            embeddings
        ).astype("float32")

        index.add(vectors)


        st.success(
            f"FAISS database ready — {index.ntotal} chunks"
        )


        # --------------------------------------------------
        # QUESTION
        # --------------------------------------------------

        st.subheader("Ask a Question")

        question = st.text_input(
            "Enter your question:"
        )


        # --------------------------------------------------
        # MANUAL SEARCH
        # --------------------------------------------------

        search_button = st.button(
            "🔎 Search / Generate Answer"
        )


        # --------------------------------------------------
        # GESTURE SEARCH
        # --------------------------------------------------

        if st.session_state.search_requested:

            search_button = True

            st.session_state.search_requested = False


        # --------------------------------------------------
        # RAG
        # --------------------------------------------------

        if search_button and question:

            question_embedding = model.encode(
                [question]
            )

            question_vector = np.array(
                question_embedding
            ).astype("float32")


            distances, indices = index.search(
                question_vector,
                k=min(3, len(chunks))
            )


            relevant_text = ""


            for index_number in indices[0]:

                relevant_text += (
                    chunks[index_number]
                    + "\n\n"
                )


            # --------------------------------------------------
            # LLAMA
            # --------------------------------------------------

            prompt = f"""
Answer the question using only the information
provided in the context below.

Context:
{relevant_text}

Question:
{question}

If the answer is not available in the context,
say:

"I could not find this information in the uploaded PDF."

Give a clear and concise answer.
"""


            with st.spinner("Llama 3.2 is generating the answer..."):

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


            st.subheader("🤖 AI Answer")

            st.write(
                response["message"]["content"]
            )


            # --------------------------------------------------
            # RETRIEVED INFORMATION
            # --------------------------------------------------

            with st.expander(
                "View Retrieved Information"
            ):

                for i, index_number in enumerate(indices[0]):

                    st.write(
                        f"Result {i + 1}"
                    )

                    st.info(
                        chunks[index_number]
                    )


# ==================================================
# GESTURE FUNCTIONS
# ==================================================


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def count_fingers(hand):

    fingers = 0

    wrist = hand[0]

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, pip in finger_pairs:

        tip_distance = distance(
            hand[tip],
            wrist
        )

        pip_distance = distance(
            hand[pip],
            wrist
        )

        if tip_distance > pip_distance * 1.10:

            fingers += 1

    return fingers


# ==================================================
# CAMERA CONTROLS
# ==================================================

st.subheader("📷 Gesture Control")

col1, col2 = st.columns(2)


with col1:

    if st.button("▶ Start Camera"):

        st.session_state.camera_running = True


with col2:

    if st.button("⏹ Stop Camera"):

        st.session_state.camera_running = False


# ==================================================
# CAMERA FRAGMENT
# ==================================================


@st.fragment(run_every=0.05)
def camera_fragment():

    if not st.session_state.camera_running:

        st.info(
            "Camera is OFF. Click Start Camera."
        )

        return


    if "camera" not in st.session_state:

        st.session_state.camera = cv2.VideoCapture(0)


    cap = st.session_state.camera


    if not cap.isOpened():

        st.error(
            "Camera could not be opened."
        )

        st.session_state.camera_running = False

        return


    ret, frame = cap.read()


    if not ret:

        st.error(
            "Could not read camera frame."
        )

        cap.release()

        st.session_state.camera_running = False

        return


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    if "detector" not in st.session_state:

        model_path = (
            "models/hand_landmarker.task"
        )

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE
        )

        st.session_state.detector = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )


    detector = st.session_state.detector


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = detector.detect(
        mp_image
    )


    gesture = "NO HAND"


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        finger_count = count_fingers(
            hand
        )


        if finger_count == 0:

            gesture = "CLEAR"


        elif finger_count == 1:

            gesture = "SEARCH"


        elif finger_count == 2:

            gesture = "NEXT"


        elif finger_count == 4:

            gesture = "STOP"


        else:

            gesture = f"{finger_count} FINGERS"


        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    st.session_state.gesture = gesture


    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    st.image(
        frame,
        channels="BGR"
    )


    # --------------------------------------------------
    # GESTURE ACTIONS
    # --------------------------------------------------

    if gesture == "SEARCH":

        st.session_state.search_requested = True


    elif gesture == "CLEAR":

        st.session_state.clear_requested = True


    elif gesture == "STOP":

        st.session_state.camera_running = False

        cap.release()

        del st.session_state.camera

        if "detector" in st.session_state:

            st.session_state.detector.close()

            del st.session_state.detector


    time.sleep(0.05)


camera_fragment()


# ==================================================
# GESTURE STATUS
# ==================================================

st.subheader("Current Gesture")

st.write(
    f"### {st.session_state.gesture}"
)


if st.session_state.gesture == "SEARCH":

    st.success(
        "☝️ SEARCH detected — enter your question "
        "and the RAG system will generate the answer."
    )


elif st.session_state.gesture == "NEXT":

    st.info(
        "✌️ NEXT detected"
    )


elif st.session_state.gesture == "CLEAR":

    st.warning(
        "✊ CLEAR detected"
    )


elif st.session_state.gesture == "STOP":

    st.warning(
        "✋ STOP detected — camera stopped."
    )
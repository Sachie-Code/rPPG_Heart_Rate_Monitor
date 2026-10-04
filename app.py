import streamlit as st
from streamlit_webrtc import ( webrtc_streamer, WebRtcMode,)
from rppg_processor import RPPGProcessor

st.set_page_config(
    page_title="rPPG Monitor",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <style>
    iframe {
        max-width: 550px !important;
        height: 550px !important;
    }

    [data-testid="column"] {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title(
    "❤️ rPPG Heart Rate Monitor"
)

st.write(
    "Real-time heart-rate estimation using webcam video and "
    "remote photoplethysmography (rPPG)."
)

if "processor" not in st.session_state:
    st.session_state.processor = (
        RPPGProcessor()
    )

processor = st.session_state.processor

left, right = st.columns([2, 2])

with left:
    st.subheader("📷 Live Camera")

    def video_frame_callback(frame):
        try:
            image = frame.to_ndarray(
                format="bgr24"
            )

            frame_time = getattr(
                frame,
                "time",
                None
            )

            processed_image = (
                processor.process_frame(
                    image,
                    timestamp=frame_time
                )
            )

            return frame.from_ndarray(
                processed_image,
                format="bgr24"
            )

        except Exception:
            return frame


    webrtc_streamer(
        key="rppg-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

with right:
    st.subheader("How it works")

    st.write(
        "Webcam → Face Detection → Forehead & Cheek ROIs → "
        "RGB Signal Extraction → rPPG Signal Processing → "
        "Heart Rate Estimation → Waveform"
    )
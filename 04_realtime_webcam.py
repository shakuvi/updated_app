import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import joblib
import os
import time
import pandas as pd
from datetime import datetime
import altair as alt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from collections import deque

# ───────── CONFIG ─────────
MODEL_DIR = 'run_mlp'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_geom_mlp.pth')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
AXIS_LEN = 50  # mm for drawing head‐pose axes
SIGMOID_TH = 0.5  # threshold on sigmoid(logit)
SMOOTHING_ALPHA = 0.8  # EMA smoothing factor (0=no smoothing, 1=infinite smoothing)
DATA_HISTORY_LENGTH = 1000  # Maximum number of data points to keep in memory
# Frame processing rate limiter (process 1 out of FRAME_SKIP frames)
FRAME_SKIP = 2
# Maximum points to display in charts
MAX_CHART_POINTS = 200
# Data update interval (seconds)
DATA_UPDATE_INTERVAL = 0.5
# PnP model points (nose tip, chin, eye corners, mouth corners)
PNP_IDS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)
# ────────────────────────────

# RTC Configuration for webrtc_streamer
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ── Trained classifier architecture ────────────────────────
class GazeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ────────────────────────────────────────────────────────────

# ───────── HELPER FUNCTIONS ─────────
@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler with caching for better performance"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            clf = GazeClassifier().to(device)
            clf.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            clf.eval()
            return scaler, clf, device, True
        else:
            return None, None, None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False


@st.cache_resource
def load_face_mesh():
    """Create and cache the face mesh detector"""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def draw_axes(img, rvec, tvec, cam_mat):
    axes = np.float32([[AXIS_LEN, 0, 0], [0, AXIS_LEN, 0], [0, 0, AXIS_LEN]])
    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, cam_mat, None)
    ori, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, cam_mat, None)
    o = tuple(ori.ravel().astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for pt, col in zip(imgpts.reshape(-1, 2).astype(int), colors):
        cv2.line(img, o, tuple(pt), col, 2)


def compute_head_pose(lm, w, h):
    pts2d = np.array([(lm[i].x * w, lm[i].y * h) for i in PNP_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_3D, pts2d, cam, None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, None, None, None, None
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        return yaw, pitch, rvec, tvec, cam
    except Exception:
        # Silently handle errors in head pose computation
        return None, None, None, None, None


def compute_iris_ratios(lm, w, h):
    iris_ids = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_IRISES for i in edge})
    half = len(iris_ids) // 2
    left_ids, right_ids = iris_ids[:half], iris_ids[half:]

    def one(ids):
        xs = np.array([lm[i].x * w for i in ids])
        ys = np.array([lm[i].y * h for i in ids])
        cx, cy = xs.mean(), ys.mean()
        h_ratio = (cx - xs.min()) / (xs.max() - xs.min() + 1e-6)
        v_ratio = (cy - ys.min()) / (ys.max() - ys.min() + 1e-6)
        return float(h_ratio), float(v_ratio), xs, ys

    l_h, l_v, lx, ly = one(left_ids)
    r_h, r_v, rx, ry = one(right_ids)
    return l_h, l_v, r_h, r_v, lx, ly, rx, ry


def downsample_data(df, max_points=MAX_CHART_POINTS):
    """Downsample the dataframe to a manageable number of points for charting"""
    if len(df) <= max_points:
        return df

    # Calculate the sampling interval
    sample_interval = max(1, len(df) // max_points)

    # For very large datasets, use systematic sampling
    return df.iloc[::sample_interval].copy()


def create_attention_time_chart(df):
    if df.empty:
        return None

    # Downsample data for faster chart rendering
    chart_df = downsample_data(df)

    chart = alt.Chart(chart_df).mark_line().encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('smoothed_prob:Q', title='Attention Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(
            alt.datum.attention == 'LOOKING',
            alt.value('green'),
            alt.value('red')
        )
    ).properties(
        title='Attention Level Over Time',
        width=600,
        height=300
    )

    return chart


def create_attention_distribution_chart(df):
    if df.empty:
        return None

    attention_counts = df['attention'].value_counts().reset_index()
    attention_counts.columns = ['attention', 'count']

    chart = alt.Chart(attention_counts).mark_bar().encode(
        x=alt.X('attention:N', title='Attention Status'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.condition(
            alt.datum.attention == 'LOOKING',
            alt.value('green'),
            alt.value('red')
        )
    ).properties(
        title='Attention Distribution',
        width=300,
        height=300
    )

    return chart


def create_distraction_chart(df):
    if df.empty:
        return None

    # Downsample data for faster chart rendering
    chart_df = downsample_data(df)

    # Make a copy of the dataframe with just the columns we need
    chart_data = pd.DataFrame({
        'timestamp': chart_df['timestamp'],
        'status': chart_df['attention'].apply(lambda x: 0 if x == 'LOOKING' else 1)
    })

    # Create simple rect mark chart with explicit encoding
    chart = alt.Chart(chart_data).mark_rect().encode(
        x=alt.X('timestamp:T', title='Time'),
        color=alt.Color('status:N',
                        scale=alt.Scale(domain=[0, 1], range=['green', 'red']),
                        legend=None)
    ).properties(
        title=None,
        height=80
    )

    return chart


@st.cache_data
def compute_attention_statistics(df):
    """Compute attention statistics with caching"""
    if df.empty:
        return {}

    # Calculate attention ratio
    total_records = len(df)
    looking_records = len(df[df['attention'] == 'LOOKING'])
    attention_ratio = looking_records / total_records if total_records > 0 else 0

    # Calculate attention streak
    df_copy = df.copy()
    attention_changes = df_copy['attention'].ne(df_copy['attention'].shift()).cumsum()
    df_copy['change_group'] = attention_changes

    # Group by attention and change group, avoiding reset_index naming conflicts
    grouped = df_copy.groupby(['attention', 'change_group']).size()
    # Convert to dataframe with explicit column names
    attention_streaks = pd.DataFrame({
        'streak_length': grouped
    }).reset_index()

    max_attention_streak = 0
    looking_streaks = attention_streaks[attention_streaks['attention'] == 'LOOKING']
    if not looking_streaks.empty:
        max_attention_streak = looking_streaks['streak_length'].max()

    max_distraction_streak = 0
    not_looking_streaks = attention_streaks[attention_streaks['attention'] == 'NOT LOOKING']
    if not not_looking_streaks.empty:
        max_distraction_streak = not_looking_streaks['streak_length'].max()

    # Calculate current streak
    current_streak_type = df['attention'].iloc[-1] if not df.empty else None
    current_streak = df.iloc[::-1]['attention'].eq(current_streak_type).cumsum().iloc[0]

    # Calculate attention transitions
    attention_transitions = sum(df['attention'].ne(df['attention'].shift()).fillna(0))

    # Calculate duration
    if len(df) >= 2:
        duration_seconds = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    else:
        duration_seconds = 0

    return {
        'attention_ratio': attention_ratio,
        'max_attention_streak': max_attention_streak,
        'max_distraction_streak': max_distraction_streak,
        'current_streak_type': current_streak_type,
        'current_streak': current_streak,
        'attention_transitions': attention_transitions,
        'duration_seconds': duration_seconds
    }


# WebRTC Video Processor with optimizations
class AttentionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Load cached resources
        self.face_mesh = load_face_mesh()
        self.scaler, self.clf, self.device, self.model_loaded = load_model_and_scaler()

        # Initialize state
        self.smoothed_prob = None
        self.results_queue = []
        self.frame_count = 0
        self.last_update_time = time.time()

    def recv(self, frame):
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")

        # Process only every Nth frame for performance
        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)
        update_data = (current_time - self.last_update_time) >= DATA_UPDATE_INTERVAL

        if not self.model_loaded:
            cv2.putText(img, "Model not loaded!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame.from_ndarray(img, format="bgr24")

        if process_this_frame:
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            res = self.face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                # head‐pose
                yaw, pitch, rvec, tvec, cam = compute_head_pose(lm, w, h)
                if rvec is not None:
                    draw_axes(img, rvec, tvec, cam)

                # iris
                l_h, l_v, r_h, r_v, lx, ly, rx, ry = compute_iris_ratios(lm, w, h)
                for xs, ys in ((lx, ly), (rx, ry)):
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    pad = 6
                    cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                                  (255, 255, 0), thickness=3)
                    for x, y in zip(xs.astype(int), ys.astype(int)):
                        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)

                # build & scale features
                if yaw is not None and pitch is not None:
                    raw = np.array([[yaw, pitch, l_h, l_v, r_h, r_v]], dtype=np.float32)
                    scaled = self.scaler.transform(raw)
                    feat = torch.from_numpy(scaled).to(self.device)

                    # classify
                    with torch.no_grad():
                        logit = self.clf(feat).item()
                    prob = 1 / (1 + np.exp(-logit))

                    # EMA smoothing
                    if self.smoothed_prob is None:
                        self.smoothed_prob = prob
                    else:
                        self.smoothed_prob = SMOOTHING_ALPHA * self.smoothed_prob + (1 - SMOOTHING_ALPHA) * prob

                    status = "LOOKING" if self.smoothed_prob > SIGMOID_TH else "NOT LOOKING"
                    col = (0, 255, 0) if self.smoothed_prob > SIGMOID_TH else (0, 0, 255)

                    # overlay
                    cv2.putText(img, f"{status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
                    cv2.putText(img, f"P={prob:.2f}, S={self.smoothed_prob:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                    # Record data in queue - but only update at specific intervals
                    if update_data:
                        self.last_update_time = current_time
                        self.results_queue.append({
                            'timestamp': datetime.now(),
                            'yaw': yaw,
                            'pitch': pitch,
                            'left_h': l_h,
                            'left_v': l_v,
                            'right_h': r_h,
                            'right_v': r_v,
                            'prob': prob,
                            'smoothed_prob': self.smoothed_prob,
                            'attention': status
                        })
            else:
                # No face detected
                cv2.putText(img, "NO FACE DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="Attention Tracker", layout="wide")

    # Create session states if they don't exist
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()
    if 'attention_data' not in st.session_state:
        st.session_state.attention_data = []
    if 'webrtc_context' not in st.session_state:
        st.session_state.webrtc_context = None
    if 'last_chart_update' not in st.session_state:
        st.session_state.last_chart_update = time.time()

    # Title
    st.title("Real-time Attention Tracking Dashboard")

    # Sidebar options
    st.sidebar.title("Settings")

    # Sidebar controls
    if st.sidebar.button("Reset Session Data"):
        st.session_state.attention_data = []
        st.session_state.start_time = datetime.now()
        st.session_state.last_chart_update = time.time()
        st.rerun()

    # Add a manual refresh button
    refresh = st.sidebar.button("Refresh Statistics")

    # Main layout
    col1, col2 = st.columns([3, 2])

    # Camera view
    with col1:
        st.subheader("Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="attention-tracker",
            video_processor_factory=AttentionVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Store the webrtc context in session state
        st.session_state.webrtc_context = webrtc_ctx

    # Stats and charts
    with col2:
        st.subheader("Attention Statistics")
        stats_placeholder = st.empty()

    # Charts
    charts_col1, charts_col2 = st.columns(2)

    with charts_col1:
        attention_chart_placeholder = st.empty()

    with charts_col2:
        distribution_chart_placeholder = st.empty()

    # Update layout to make timeline more visible
    st.markdown("---")  # Add a divider

    # Create a dedicated section for the timeline
    st.subheader("Attention Status Over Time")
    timeline_chart_container = st.container()

    # Function to transfer data from video processor to session state
    def transfer_data():
        if (webrtc_ctx.video_processor is not None and
                hasattr(webrtc_ctx.video_processor, 'results_queue') and
                len(webrtc_ctx.video_processor.results_queue) > 0):

            # Get the data from the video processor queue
            new_data = webrtc_ctx.video_processor.results_queue

            # Add to session state
            st.session_state.attention_data.extend(new_data)

            # Clear the queue
            webrtc_ctx.video_processor.results_queue = []

            # Limit the data history
            if len(st.session_state.attention_data) > DATA_HISTORY_LENGTH:
                st.session_state.attention_data = st.session_state.attention_data[-DATA_HISTORY_LENGTH:]

            return True
        return False

    # Update charts only periodically to improve performance
    current_time = time.time()
    chart_update_needed = current_time - st.session_state.last_chart_update >= 2 or refresh

    # Transfer data if the webrtc context is playing
    if webrtc_ctx.state.playing:
        new_data_added = transfer_data()

        # Only update charts if needed and we have data
        if chart_update_needed and st.session_state.attention_data:
            st.session_state.last_chart_update = current_time

            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.attention_data)

            # Calculate statistics
            stats = compute_attention_statistics(df)

            # Display statistics
            stats_html = f"""
            <div style="padding: 10px; background-color: #f0f2f6; border-radius: 10px;">
                <h4>Session Summary</h4>
                <p>Duration: {int(stats['duration_seconds'] // 60)} min {int(stats['duration_seconds'] % 60)} sec</p>
                <p>Attention Ratio: {stats['attention_ratio']:.2%}</p>
                <p>Max Attention Streak: {stats['max_attention_streak']} frames</p>
                <p>Max Distraction Streak: {stats['max_distraction_streak']} frames</p>
                <p>Attention Transitions: {stats['attention_transitions']}</p>
                <p>Current Status: {stats['current_streak_type']} ({stats['current_streak']} frames)</p>
            </div>
            """
            stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

            # Create and display charts
            attention_chart = create_attention_time_chart(df)
            if attention_chart:
                attention_chart_placeholder.altair_chart(attention_chart, use_container_width=True)

            distribution_chart = create_attention_distribution_chart(df)
            if distribution_chart:
                distribution_chart_placeholder.altair_chart(distribution_chart, use_container_width=True)

            # Create and display the status timeline
            distraction_chart = create_distraction_chart(df)
            if distraction_chart:
                timeline_chart_container.altair_chart(distraction_chart, use_container_width=True)

    # Display initial info if no data is available
    if not st.session_state.attention_data:
        stats_placeholder.info("No data collected yet. Start the camera and begin tracking to see statistics.")
        attention_chart_placeholder.info("Attention data will appear here once collected.")
        distribution_chart_placeholder.info("Attention distribution will appear here once collected.")
        timeline_chart_container.info("Attention timeline will appear here once collected.")


if __name__ == "__main__":
    main()
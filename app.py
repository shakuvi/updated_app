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
from datetime import datetime, timedelta
import altair as alt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import sqlite3
import hashlib
import uuid
from typing import Optional, List, Dict, Any
import threading
import queue
import plotly.express as px
import plotly.graph_objects as go

# ───────── CONFIG ─────────
MODEL_DIR = 'run_mlp'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_geom_mlp.pth')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
AXIS_LEN = 50  # mm for drawing head‐pose axes
SIGMOID_TH = 0.5  # threshold on sigmoid(logit)
SMOOTHING_ALPHA = 0.8  # EMA smoothing factor
DATA_HISTORY_LENGTH = 1000  # Maximum number of data points to keep in memory
DATABASE_PATH = 'attention_tracking.db'

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

# RTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ────────────────────────────────────────────────────────────
# DATABASE MANAGEMENT
# ────────────────────────────────────────────────────────────

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           username
                           TEXT
                           UNIQUE
                           NOT
                           NULL,
                           password_hash
                           TEXT
                           NOT
                           NULL,
                           role
                           TEXT
                           NOT
                           NULL
                           CHECK (
                           role
                           IN
                       (
                           'lecturer',
                           'student'
                       )),
                           full_name TEXT NOT NULL,
                           email TEXT,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           )
                       ''')

        # Lecture sessions table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS lecture_sessions
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           session_id
                           TEXT
                           UNIQUE
                           NOT
                           NULL,
                           lecturer_id
                           INTEGER
                           NOT
                           NULL,
                           title
                           TEXT
                           NOT
                           NULL,
                           description
                           TEXT,
                           start_time
                           TIMESTAMP,
                           end_time
                           TIMESTAMP,
                           status
                           TEXT
                           DEFAULT
                           'planned'
                           CHECK (
                           status
                           IN
                       (
                           'planned',
                           'active',
                           'ended'
                       )),
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           FOREIGN KEY
                       (
                           lecturer_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        # Session participants table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS session_participants
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           session_id
                           TEXT
                           NOT
                           NULL,
                           student_id
                           INTEGER
                           NOT
                           NULL,
                           joined_at
                           TIMESTAMP,
                           left_at
                           TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           session_id
                       ) REFERENCES lecture_sessions
                       (
                           session_id
                       ),
                           FOREIGN KEY
                       (
                           student_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        # Attention data table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS attention_data
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           session_id
                           TEXT
                           NOT
                           NULL,
                           student_id
                           INTEGER
                           NOT
                           NULL,
                           timestamp
                           TIMESTAMP
                           NOT
                           NULL,
                           yaw
                           REAL,
                           pitch
                           REAL,
                           left_h
                           REAL,
                           left_v
                           REAL,
                           right_h
                           REAL,
                           right_v
                           REAL,
                           prob
                           REAL,
                           smoothed_prob
                           REAL,
                           attention_status
                           TEXT,
                           FOREIGN
                           KEY
                       (
                           session_id
                       ) REFERENCES lecture_sessions
                       (
                           session_id
                       ),
                           FOREIGN KEY
                       (
                           student_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        conn.commit()
        conn.close()

        # Create default admin user if not exists
        self.create_default_users()

    def create_default_users(self):
        """Create default lecturer and student for testing"""
        try:
            self.create_user("admin", "admin123", "lecturer", "Administrator", "admin@example.com")
            self.create_user("student1", "student123", "student", "John Doe", "john@example.com")
        except Exception:
            pass  # Users might already exist

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, role: str, full_name: str, email: str = None) -> bool:
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            password_hash = self.hash_password(password)
            cursor.execute('''
                           INSERT INTO users (username, password_hash, role, full_name, email)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (username, password_hash, role, full_name, email))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = self.hash_password(password)
        cursor.execute('''
                       SELECT id, username, role, full_name, email
                       FROM users
                       WHERE username = ?
                         AND password_hash = ?
                       ''', (username, password_hash))

        user = cursor.fetchone()
        conn.close()

        if user:
            return {
                'id': user[0],
                'username': user[1],
                'role': user[2],
                'full_name': user[3],
                'email': user[4]
            }
        return None

    def create_session(self, lecturer_id: int, title: str, description: str = "") -> str:
        """Create a new lecture session"""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO lecture_sessions (session_id, lecturer_id, title, description)
                       VALUES (?, ?, ?, ?)
                       ''', (session_id, lecturer_id, title, description))

        conn.commit()
        conn.close()
        return session_id

    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information by session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT ls.*, u.full_name as lecturer_name
                       FROM lecture_sessions ls
                                JOIN users u ON ls.lecturer_id = u.id
                       WHERE ls.session_id = ?
                       ''', (session_id,))

        session = cursor.fetchone()
        conn.close()

        if session:
            return {
                'id': session[0],
                'session_id': session[1],
                'lecturer_id': session[2],
                'title': session[3],
                'description': session[4],
                'start_time': session[5],
                'end_time': session[6],
                'status': session[7],
                'created_at': session[8],
                'lecturer_name': session[9]
            }
        return None

    def start_session(self, session_id: str) -> bool:
        """Start a lecture session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE lecture_sessions
                       SET status     = 'active',
                           start_time = CURRENT_TIMESTAMP
                       WHERE session_id = ?
                       ''', (session_id,))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def end_session(self, session_id: str) -> bool:
        """End a lecture session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE lecture_sessions
                       SET status   = 'ended',
                           end_time = CURRENT_TIMESTAMP
                       WHERE session_id = ?
                       ''', (session_id,))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def join_session(self, session_id: str, student_id: int) -> bool:
        """Student joins a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already joined
        cursor.execute('''
                       SELECT id
                       FROM session_participants
                       WHERE session_id = ?
                         AND student_id = ?
                         AND left_at IS NULL
                       ''', (session_id, student_id))

        if cursor.fetchone():
            conn.close()
            return True  # Already joined

        cursor.execute('''
                       INSERT INTO session_participants (session_id, student_id, joined_at)
                       VALUES (?, ?, CURRENT_TIMESTAMP)
                       ''', (session_id, student_id))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def leave_session(self, session_id: str, student_id: int) -> bool:
        """Student leaves a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE session_participants
                       SET left_at = CURRENT_TIMESTAMP
                       WHERE session_id = ?
                         AND student_id = ?
                         AND left_at IS NULL
                       ''', (session_id, student_id))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def get_session_participants(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all participants in a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT u.id, u.full_name, u.username, sp.joined_at, sp.left_at
                       FROM session_participants sp
                                JOIN users u ON sp.student_id = u.id
                       WHERE sp.session_id = ?
                       ORDER BY sp.joined_at DESC
                       ''', (session_id,))

        participants = []
        for row in cursor.fetchall():
            participants.append({
                'id': row[0],
                'full_name': row[1],
                'username': row[2],
                'joined_at': row[3],
                'left_at': row[4],
                'is_active': row[4] is None
            })

        conn.close()
        return participants

    def save_attention_data(self, session_id: str, student_id: int, data: Dict[str, Any]):
        """Save attention tracking data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO attention_data (session_id, student_id, timestamp, yaw, pitch,
                                                   left_h, left_v, right_h, right_v, prob, smoothed_prob,
                                                   attention_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           session_id, student_id, data['timestamp'], data.get('yaw'), data.get('pitch'),
                           data.get('left_h'), data.get('left_v'), data.get('right_h'), data.get('right_v'),
                           data.get('prob'), data.get('smoothed_prob'), data.get('attention')
                       ))

        conn.commit()
        conn.close()

    def get_attention_data(self, session_id: str, student_id: int = None) -> pd.DataFrame:
        """Get attention data for a session"""
        conn = sqlite3.connect(self.db_path)

        if student_id:
            query = '''
                    SELECT * \
                    FROM attention_data
                    WHERE session_id = ? \
                      AND student_id = ?
                    ORDER BY timestamp \
                    '''
            df = pd.read_sql_query(query, conn, params=(session_id, student_id))
        else:
            query = '''
                    SELECT ad.*, u.full_name, u.username
                    FROM attention_data ad
                             JOIN users u ON ad.student_id = u.id
                    WHERE ad.session_id = ?
                    ORDER BY ad.timestamp \
                    '''
            df = pd.read_sql_query(query, conn, params=(session_id,))

        conn.close()

        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_lecturer_sessions(self, lecturer_id: int) -> List[Dict[str, Any]]:
        """Get all sessions created by a lecturer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT *
                       FROM lecture_sessions
                       WHERE lecturer_id = ?
                       ORDER BY created_at DESC
                       ''', (lecturer_id,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'session_id': row[1],
                'lecturer_id': row[2],
                'title': row[3],
                'description': row[4],
                'start_time': row[5],
                'end_time': row[6],
                'status': row[7],
                'created_at': row[8]
            })

        conn.close()
        return sessions

    def get_student_sessions(self, student_id: int) -> List[Dict[str, Any]]:
        """Get all sessions a student has participated in"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT DISTINCT ls.*, u.full_name as lecturer_name
                       FROM lecture_sessions ls
                                JOIN session_participants sp ON ls.session_id = sp.session_id
                                JOIN users u ON ls.lecturer_id = u.id
                       WHERE sp.student_id = ?
                       ORDER BY ls.created_at DESC
                       ''', (student_id,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'session_id': row[1],
                'lecturer_id': row[2],
                'title': row[3],
                'description': row[4],
                'start_time': row[5],
                'end_time': row[6],
                'status': row[7],
                'created_at': row[8],
                'lecturer_name': row[9]
            })

        conn.close()
        return sessions

    def get_all_students(self) -> List[Dict[str, Any]]:
        """Get all students in the system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT id, username, full_name, email
                       FROM users
                       WHERE role = 'student'
                       ORDER BY full_name
                       ''')

        students = []
        for row in cursor.fetchall():
            students.append({
                'id': row[0],
                'username': row[1],
                'full_name': row[2],
                'email': row[3]
            })

        conn.close()
        return students


# ────────────────────────────────────────────────────────────
# ML MODEL
# ────────────────────────────────────────────────────────────

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
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────

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


def create_attention_time_chart(df, title="Attention Level Over Time"):
    """Create attention timeline chart using Plotly"""
    if df.empty:
        return None

    # Convert timestamp to relative time (minutes from start)
    df = df.copy()
    start_time = df['timestamp'].min()
    df['minutes_elapsed'] = (df['timestamp'] - start_time).dt.total_seconds() / 60

    colors = ['red' if status == 'NOT LOOKING' else 'green' for status in df['attention_status']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['minutes_elapsed'],
        y=df['smoothed_prob'],
        mode='lines+markers',
        name='Attention Score',
        line=dict(color='blue', width=2),
        marker=dict(
            color=colors,
            size=4,
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>Time:</b> %{x:.1f} min<br><b>Attention:</b> %{y:.2f}<br><extra></extra>'
    ))

    fig.add_hline(y=SIGMOID_TH, line_dash="dash", line_color="gray",
                  annotation_text="Threshold")

    fig.update_layout(
        title=title,
        xaxis_title="Time (minutes)",
        yaxis_title="Attention Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=True
    )

    return fig


def create_attention_distribution_chart(df, title="Attention Distribution"):
    """Create attention distribution pie chart"""
    if df.empty:
        return None

    attention_counts = df['attention_status'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=attention_counts.index,
        values=attention_counts.values,
        marker_colors=['green' if x == 'LOOKING' else 'red' for x in attention_counts.index],
        textinfo='label+percent',
        textfont_size=12
    )])

    fig.update_layout(
        title=title,
        height=300
    )

    return fig


def create_realtime_gauge(current_attention, title="Current Attention"):
    """Create a gauge chart for real-time attention"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_attention * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': SIGMOID_TH * 100
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


def create_student_comparison_chart(df):
    """Create chart comparing all students' attention levels"""
    if df.empty:
        return None

    # Calculate attention ratio for each student
    student_stats = df.groupby(['student_id', 'full_name']).agg({
        'attention_status': lambda x: (x == 'LOOKING').mean(),
        'smoothed_prob': 'mean'
    }).reset_index()

    student_stats.columns = ['student_id', 'full_name', 'attention_ratio', 'avg_attention_score']
    student_stats = student_stats.sort_values('attention_ratio', ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=student_stats['full_name'],
        y=student_stats['attention_ratio'] * 100,
        name='Attention Percentage',
        marker_color=['green' if x > 0.5 else 'red' for x in student_stats['attention_ratio']],
        text=[f"{x:.1f}%" for x in student_stats['attention_ratio'] * 100],
        textposition='auto'
    ))

    fig.update_layout(
        title='Student Attention Comparison',
        xaxis_title='Student',
        yaxis_title='Attention Percentage (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )

    return fig


def compute_attention_statistics(df):
    """Compute comprehensive attention statistics"""
    if df.empty:
        return {}

    total_records = len(df)
    looking_records = len(df[df['attention_status'] == 'LOOKING'])
    attention_ratio = looking_records / total_records if total_records > 0 else 0

    # Calculate streaks
    df_copy = df.copy()
    attention_changes = df_copy['attention_status'].ne(df_copy['attention_status'].shift()).cumsum()
    df_copy['change_group'] = attention_changes

    grouped = df_copy.groupby(['attention_status', 'change_group']).size()
    attention_streaks = pd.DataFrame({'streak_length': grouped}).reset_index()

    max_attention_streak = 0
    looking_streaks = attention_streaks[attention_streaks['attention_status'] == 'LOOKING']
    if not looking_streaks.empty:
        max_attention_streak = looking_streaks['streak_length'].max()

    max_distraction_streak = 0
    not_looking_streaks = attention_streaks[attention_streaks['attention_status'] == 'NOT LOOKING']
    if not not_looking_streaks.empty:
        max_distraction_streak = not_looking_streaks['streak_length'].max()

    current_streak_type = df['attention_status'].iloc[-1] if not df.empty else None
    current_streak = df.iloc[::-1]['attention_status'].eq(current_streak_type).cumsum().iloc[0]

    attention_transitions = sum(df['attention_status'].ne(df['attention_status'].shift()).fillna(0))

    if len(df) >= 2:
        duration_seconds = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    else:
        duration_seconds = 0

    avg_attention_score = df['smoothed_prob'].mean()

    return {
        'attention_ratio': attention_ratio,
        'max_attention_streak': max_attention_streak,
        'max_distraction_streak': max_distraction_streak,
        'current_streak_type': current_streak_type,
        'current_streak': current_streak,
        'attention_transitions': attention_transitions,
        'duration_seconds': duration_seconds,
        'avg_attention_score': avg_attention_score
    }


# ────────────────────────────────────────────────────────────
# VIDEO PROCESSORS
# ────────────────────────────────────────────────────────────

class LecturerVideoProcessor(VideoProcessorBase):
    """Simple video processor for lecturer without ML model"""

    def __init__(self):
        pass

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Simple overlay for lecturer
        cv2.putText(img, "LECTURER CAMERA", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img, "Session Active", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return frame.from_ndarray(img, format="bgr24")


class StudentAttentionVideoProcessor(VideoProcessorBase):
    """Advanced video processor for students with ML model"""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.smoothed_prob = None
        self.results_queue = []

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            self.clf = GazeClassifier().to(self.device)
            self.clf.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.clf.eval()
            self.model_loaded = True
        else:
            self.model_loaded = False

    def recv(self, frame):
        if not self.model_loaded:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "Model not loaded!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame.from_ndarray(img, format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Head pose
            yaw, pitch, rvec, tvec, cam = compute_head_pose(lm, w, h)
            if rvec is not None:
                draw_axes(img, rvec, tvec, cam)

            # Iris
            l_h, l_v, r_h, r_v, lx, ly, rx, ry = compute_iris_ratios(lm, w, h)
            for xs, ys in ((lx, ly), (rx, ry)):
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                pad = 6
                cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                              (255, 255, 0), thickness=3)
                for x, y in zip(xs.astype(int), ys.astype(int)):
                    cv2.circle(img, (x, y), 4, (0, 255, 255), -1)

            # Build & scale features
            if yaw is not None and pitch is not None:
                raw = np.array([[yaw, pitch, l_h, l_v, r_h, r_v]], dtype=np.float32)
                scaled = self.scaler.transform(raw)
                feat = torch.from_numpy(scaled).to(self.device)

                # Classify
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

                # Enhanced overlay with progress bar
                cv2.putText(img, f"{status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
                cv2.putText(img, f"Score: {self.smoothed_prob:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Progress bar for attention score
                bar_width = 200
                bar_height = 20
                bar_x, bar_y = 10, 80

                # Background bar
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                # Fill bar based on attention score
                fill_width = int(bar_width * self.smoothed_prob)
                bar_color = (0, 255, 0) if self.smoothed_prob > SIGMOID_TH else (0, 0, 255)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

                # Threshold line
                threshold_x = int(bar_x + bar_width * SIGMOID_TH)
                cv2.line(img, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (255, 255, 255), 2)

                # Record data
                if prob is not None and self.smoothed_prob is not None:
                    current_time = datetime.now()
                    self.results_queue.append({
                        'timestamp': current_time,
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


# ────────────────────────────────────────────────────────────
# STREAMLIT APPLICATION
# ────────────────────────────────────────────────────────────

def initialize_session_state():
    """Initialize session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'attention_data' not in st.session_state:
        st.session_state.attention_data = []
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'selected_session_analytics' not in st.session_state:
        st.session_state.selected_session_analytics = None


def login_page():
    """Login page for both lecturers and students"""
    st.title("🎓 AI-Driven Attention Analytics for Online Lectures")
    st.subheader("Login")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username and password:
                    user = st.session_state.db_manager.authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")

        # Registration section
        st.subheader("New User Registration")
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_full_name = st.text_input("Full Name", key="reg_full_name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_role = st.selectbox("Role", ["student", "lecturer"], key="reg_role")
            register = st.form_submit_button("Register")

            if register:
                if reg_username and reg_password and reg_full_name:
                    success = st.session_state.db_manager.create_user(
                        reg_username, reg_password, reg_role, reg_full_name, reg_email
                    )
                    if success:
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Please fill in all required fields")


def lecturer_dashboard():
    """Main dashboard for lecturers"""
    st.title("👨‍🏫 Lecturer Dashboard")
    st.write(f"Welcome, {st.session_state.user['full_name']}")

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", [
            "My Sessions",
            "Create Session",
            "Active Session",
            "Manage Students",
            "Session Analytics"
        ])

        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.current_session = None
            st.rerun()

    if page == "My Sessions":
        show_lecturer_sessions()
    elif page == "Create Session":
        create_new_session()
    elif page == "Active Session":
        manage_active_session()
    elif page == "Manage Students":
        manage_students()
    elif page == "Session Analytics":
        session_analytics()


def show_lecturer_sessions():
    """Display lecturer's sessions"""
    st.subheader("My Lecture Sessions")

    sessions = st.session_state.db_manager.get_lecturer_sessions(st.session_state.user['id'])

    if not sessions:
        st.info("No sessions created yet. Create your first session!")
        return

    for session in sessions:
        with st.expander(f"📚 {session['title']} - {session['status'].upper()}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Session ID:** `{session['session_id']}`")
                st.write(f"**Description:** {session['description'] or 'No description'}")
                st.write(f"**Status:** {session['status']}")
                st.write(f"**Created:** {session['created_at']}")

            with col2:
                if session['start_time']:
                    st.write(f"**Started:** {session['start_time']}")
                if session['end_time']:
                    st.write(f"**Ended:** {session['end_time']}")

                # Action buttons
                if session['status'] == 'planned':
                    if st.button(f"Start Session", key=f"start_{session['id']}"):
                        st.session_state.db_manager.start_session(session['session_id'])
                        st.session_state.current_session = session['session_id']
                        st.success("Session started!")
                        st.rerun()

                elif session['status'] == 'active':
                    if st.button(f"Go to Active Session", key=f"goto_{session['id']}"):
                        st.session_state.current_session = session['session_id']
                        st.rerun()

                if session['status'] == 'ended':
                    if st.button(f"View Analytics", key=f"analytics_{session['id']}"):
                        st.session_state.selected_session_analytics = session['session_id']

            # Show participants
            participants = st.session_state.db_manager.get_session_participants(session['session_id'])
            if participants:
                st.write("**Participants:**")
                for p in participants:
                    status = "🟢 Active" if p['is_active'] else "🔴 Left"
                    st.write(f"- {p['full_name']} ({p['username']}) - {status}")


def create_new_session():
    
    st.subheader("Create New Session")

    # -------- FORM --------
    with st.form("create_session_form"):
        title = st.text_input("Session Title", placeholder="e.g., Introduction to Machine Learning")
        description = st.text_area("Description (Optional)",
                                   placeholder="Brief description of the lecture…")
        submitted = st.form_submit_button("Create Session")

    # -------- AFTER THE FORM SUBMITS --------
    if submitted:
        if title:
            session_id = st.session_state.db_manager.create_session(
                st.session_state.user['id'], title, description
            )
            # Remember the new session so the button outside the form can use it
            st.session_state.new_session_id = session_id

            st.success("Session created successfully!")
            st.info(f"**Session ID:** `{session_id}`")
            st.info("📋 **Share this Session ID with your students so they can join.**")
        else:
            st.error("Please enter a session title")

    # -------- OPTIONAL “START NOW” BUTTON --------
    if st.session_state.get("new_session_id"):
        if st.button("Start Session Now"):
            st.session_state.db_manager.start_session(st.session_state.new_session_id)
            st.session_state.current_session = st.session_state.new_session_id
            st.success("Session started!")
            # Clean up
            del st.session_state.new_session_id
            st.experimental_rerun()



def manage_active_session():
    """Manage the currently active session"""
    if not st.session_state.current_session:
        st.info("No active session. Please start a session first.")
        return

    session_info = st.session_state.db_manager.get_session_by_id(st.session_state.current_session)
    if not session_info:
        st.error("Session not found")
        return

    st.subheader(f"Active Session: {session_info['title']}")
    st.info(f"**Session ID:** `{st.session_state.current_session}`")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Lecturer Camera")
        # Lecturer's camera feed WITHOUT model
        webrtc_ctx = webrtc_streamer(
            key="lecturer-camera",
            video_processor_factory=LecturerVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.subheader("Session Controls")

        if st.button("🛑 End Session", type="primary"):
            st.session_state.db_manager.end_session(st.session_state.current_session)
            st.session_state.current_session = None
            st.success("Session ended!")
            st.rerun()

        # Show participants
        st.subheader("👥 Active Participants")
        participants = st.session_state.db_manager.get_session_participants(st.session_state.current_session)
        active_participants = [p for p in participants if p['is_active']]

        if active_participants:
            for p in active_participants:
                st.write(f"🟢 {p['full_name']} ({p['username']})")
        else:
            st.write("No active participants")

        # Refresh button
        if st.button("🔄 Refresh Participants"):
            st.rerun()

    # Real-time attention monitoring
    st.subheader("📊 Real-time Attention Monitoring")

    # Get recent attention data for all participants
    df = st.session_state.db_manager.get_attention_data(st.session_state.current_session)

    if not df.empty:
        # Show overall statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_attention = df['smoothed_prob'].mean()
            st.metric("📈 Average Attention", f"{avg_attention:.2%}")

        with col2:
            active_students = df['student_id'].nunique()
            st.metric("👨‍🎓 Active Students", active_students)

        with col3:
            looking_ratio = len(df[df['attention_status'] == 'LOOKING']) / len(df)
            st.metric("👀 Looking Ratio", f"{looking_ratio:.2%}")

        with col4:
            total_datapoints = len(df)
            st.metric("📊 Data Points", total_datapoints)

        # Student comparison chart
        comparison_chart = create_student_comparison_chart(df)
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)

        # Individual student attention in expandable sections
        st.subheader("👤 Individual Student Attention")

        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id]
            if not student_data.empty:
                student_name = student_data['full_name'].iloc[0]
                recent_attention = student_data.tail(50)
                stats = compute_attention_statistics(recent_attention)

                with st.expander(f"📊 {student_name} - {stats['attention_ratio']:.1%} Attention"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric("Current Attention", f"{stats['attention_ratio']:.1%}")
                        st.metric("Avg Score", f"{stats['avg_attention_score']:.2f}")
                        st.metric("Current Streak", f"{stats['current_streak']} ({stats['current_streak_type']})")

                    with col2:
                        # Individual student chart
                        chart = create_attention_time_chart(recent_attention.tail(100), f"{student_name}'s Attention")
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No attention data available yet. Students need to join and start their cameras.")

    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()


def manage_students():
    """Manage students in the system"""
    st.subheader("👥 Manage Students")

    tab1, tab2 = st.tabs(["View Students", "Add Student"])

    with tab1:
        students = st.session_state.db_manager.get_all_students()
        if students:
            df = pd.DataFrame(students)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No students registered yet.")

    with tab2:
        st.subheader("Add New Student")
        with st.form("add_student_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")

            submit = st.form_submit_button("Add Student")

            if submit:
                if username and password and full_name:
                    success = st.session_state.db_manager.create_user(
                        username, password, "student", full_name, email
                    )
                    if success:
                        st.success("Student added successfully!")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Please fill in all required fields")


def session_analytics():
    """Show detailed analytics for completed sessions"""
    st.subheader("📊 Session Analytics")

    sessions = st.session_state.db_manager.get_lecturer_sessions(st.session_state.user['id'])
    ended_sessions = [s for s in sessions if s['status'] == 'ended']

    if not ended_sessions:
        st.info("No completed sessions available for analytics.")
        return

    # Session selector
    session_options = {f"{s['title']} ({s['session_id'][:8]}...)": s['session_id'] for s in ended_sessions}
    selected_session_name = st.selectbox("Select Session", list(session_options.keys()))
    selected_session_id = session_options[selected_session_name]

    # Get session data
    df = st.session_state.db_manager.get_attention_data(selected_session_id)

    if df.empty:
        st.warning("No attention data available for this session.")
        return

    # Overall session statistics
    st.subheader("📈 Overall Session Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_participants = df['student_id'].nunique()
        st.metric("👥 Total Participants", total_participants)

    with col2:
        avg_attention = df['smoothed_prob'].mean()
        st.metric("📊 Average Attention", f"{avg_attention:.2%}")

    with col3:
        session_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
        st.metric("⏱️ Session Duration", f"{session_duration:.1f} min")

    with col4:
        looking_ratio = len(df[df['attention_status'] == 'LOOKING']) / len(df)
        st.metric("👀 Overall Looking Ratio", f"{looking_ratio:.2%}")

    # Charts section
    st.subheader("📊 Session Overview Charts")
    col1, col2 = st.columns(2)

    with col1:
        chart = create_attention_time_chart(df, "Session Attention Timeline")
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    with col2:
        chart = create_attention_distribution_chart(df, "Session Attention Distribution")
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    # Student comparison chart
    comparison_chart = create_student_comparison_chart(df)
    if comparison_chart:
        st.plotly_chart(comparison_chart, use_container_width=True)

    # Detailed individual student performance
    st.subheader("👤 Individual Student Performance Analysis")

    # Create summary table first
    student_summary = []
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        student_name = student_data['full_name'].iloc[0]
        stats = compute_attention_statistics(student_data)

        student_summary.append({
            'Student': student_name,
            'Attention Ratio': f"{stats['attention_ratio']:.2%}",
            'Avg Score': f"{stats['avg_attention_score']:.3f}",
            'Max Attention Streak': stats['max_attention_streak'],
            'Max Distraction Streak': stats['max_distraction_streak'],
            'Transitions': stats['attention_transitions'],
            'Duration (min)': f"{stats['duration_seconds'] / 60:.1f}"
        })

    # Display summary table
    summary_df = pd.DataFrame(student_summary)
    st.dataframe(summary_df, use_container_width=True)

    # Detailed individual analysis
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        student_name = student_data['full_name'].iloc[0]
        stats = compute_attention_statistics(student_data)

        with st.expander(f"📊 Detailed Analysis: {student_name}"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("🎯 Attention Ratio", f"{stats['attention_ratio']:.2%}")
                st.metric("📊 Average Score", f"{stats['avg_attention_score']:.3f}")
                st.metric("🔥 Max Attention Streak", f"{stats['max_attention_streak']} frames")
                st.metric("😴 Max Distraction Streak", f"{stats['max_distraction_streak']} frames")
                st.metric("🔄 Attention Transitions", stats['attention_transitions'])
                st.metric("⏱️ Active Duration", f"{stats['duration_seconds']:.0f} seconds")

            with col2:
                # Individual timeline chart
                chart = create_attention_time_chart(student_data, f"{student_name}'s Attention Timeline")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                # Distribution chart
                dist_chart = create_attention_distribution_chart(student_data,
                                                                 f"{student_name}'s Attention Distribution")
                if dist_chart:
                    st.plotly_chart(dist_chart, use_container_width=True)


def student_dashboard():
    """Main dashboard for students"""
    st.title("🎓 Student Dashboard")
    st.write(f"Welcome, {st.session_state.user['full_name']}")

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", [
            "Join Session",
            "My Current Session",
            "Session History"
        ])

        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.current_session = None
            st.rerun()

    if page == "Join Session":
        join_session_page()
    elif page == "My Current Session":
        current_session_page()
    elif page == "Session History":
        student_session_history()


def join_session_page():
    """Page for students to join a session"""
    st.subheader("🚪 Join Lecture Session")

    with st.form("join_session_form"):
        session_id = st.text_input("Session ID", placeholder="Enter the session ID provided by your lecturer")
        submit = st.form_submit_button("Join Session")

        if submit:
            if session_id:
                session_info = st.session_state.db_manager.get_session_by_id(session_id)
                if session_info:
                    if session_info['status'] == 'active':
                        success = st.session_state.db_manager.join_session(session_id, st.session_state.user['id'])
                        if success:
                            st.session_state.current_session = session_id
                            st.success(f"✅ Joined session: {session_info['title']}")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("Failed to join session")
                    else:
                        st.error(f"Session is not active (Status: {session_info['status']})")
                else:
                    st.error("Session not found")
            else:
                st.error("Please enter a session ID")


def current_session_page():
    """Current active session for student with enhanced real-time analytics"""
    if not st.session_state.current_session:
        st.info("You are not currently in any session. Join a session to start tracking.")
        return

    session_info = st.session_state.db_manager.get_session_by_id(st.session_state.current_session)
    if not session_info:
        st.error("Session not found")
        st.session_state.current_session = None
        return

    if session_info['status'] != 'active':
        st.warning("This session has ended.")
        st.session_state.current_session = None
        return

    st.subheader(f"📚 Current Session: {session_info['title']}")
    st.write(f"👨‍🏫 Lecturer: {session_info['lecturer_name']}")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Your Camera Feed")
        # Student's camera feed WITH attention tracking model
        webrtc_ctx = webrtc_streamer(
            key="student-camera",
            video_processor_factory=StudentAttentionVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        # Save attention data to database
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            if hasattr(webrtc_ctx.video_processor, 'results_queue') and webrtc_ctx.video_processor.results_queue:
                # Save data to database
                for data in webrtc_ctx.video_processor.results_queue:
                    st.session_state.db_manager.save_attention_data(
                        st.session_state.current_session,
                        st.session_state.user['id'],
                        data
                    )
                # Clear the queue
                webrtc_ctx.video_processor.results_queue = []

    with col2:
        st.subheader("🎛️ Session Info")

        if st.button("🚪 Leave Session", type="primary"):
            st.session_state.db_manager.leave_session(st.session_state.current_session, st.session_state.user['id'])
            st.session_state.current_session = None
            st.success("Left session!")
            st.rerun()

        # Show session participants
        participants = st.session_state.db_manager.get_session_participants(st.session_state.current_session)
        active_participants = [p for p in participants if p['is_active']]
        st.write(f"👥 **Participants:** {len(active_participants)}")

    # Real-time personal analytics
    st.subheader("📊 Your Real-time Analytics")

    # Get personal attention data
    df = st.session_state.db_manager.get_attention_data(
        st.session_state.current_session,
        st.session_state.user['id']
    )

    if not df.empty:
        recent_data = df.tail(100)  # Last 100 data points
        stats = compute_attention_statistics(recent_data)

        # Current attention metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 Your Attention", f"{stats['attention_ratio']:.1%}")

        with col2:
            current_score = recent_data['smoothed_prob'].iloc[-1] if not recent_data.empty else 0
            st.metric("📊 Current Score", f"{current_score:.3f}")

        with col3:
            st.metric("🔥 Current Streak", f"{stats['current_streak']} ({stats['current_streak_type']})")

        with col4:
            st.metric("⏱️ Active Time", f"{stats['duration_seconds'] / 60:.1f} min")

        # Real-time gauge
        if not recent_data.empty:
            current_attention = recent_data['smoothed_prob'].iloc[-1]
            gauge_chart = create_realtime_gauge(current_attention, "Your Current Attention Level")
            st.plotly_chart(gauge_chart, use_container_width=True)

        # Personal attention timeline
        col1, col2 = st.columns(2)

        with col1:
            chart = create_attention_time_chart(recent_data, "Your Attention Timeline")
            if chart:
                st.plotly_chart(chart, use_container_width=True)

        with col2:
            dist_chart = create_attention_distribution_chart(recent_data, "Your Attention Distribution")
            if dist_chart:
                st.plotly_chart(dist_chart, use_container_width=True)

        # Detailed statistics
        with st.expander("📈 Detailed Statistics"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**🎯 Overall Attention Ratio:** {stats['attention_ratio']:.2%}")
                st.write(f"**📊 Average Attention Score:** {stats['avg_attention_score']:.3f}")
                st.write(f"**🔥 Max Attention Streak:** {stats['max_attention_streak']} frames")
                st.write(f"**😴 Max Distraction Streak:** {stats['max_distraction_streak']} frames")

            with col2:
                st.write(f"**🔄 Attention Transitions:** {stats['attention_transitions']}")
                st.write(f"**⏱️ Total Active Duration:** {stats['duration_seconds']:.0f} seconds")
                st.write(f"**📊 Data Points Collected:** {len(recent_data)}")

                # Performance feedback
                if stats['attention_ratio'] >= 0.8:
                    st.success("🌟 Excellent attention! Keep it up!")
                elif stats['attention_ratio'] >= 0.6:
                    st.info("👍 Good attention level!")
                elif stats['attention_ratio'] >= 0.4:
                    st.warning("⚠️ Try to focus more on the lecture.")
                else:
                    st.error("❗ Low attention detected. Please focus!")
    else:
        st.info("📊 No attention data recorded yet. Make sure your camera is active and your face is visible.")

    # Auto-refresh every 3 seconds for real-time updates
    time.sleep(3)
    st.rerun()


def student_session_history():
    """Show student's session history and personal analytics"""
    st.subheader("📚 My Session History")

    sessions = st.session_state.db_manager.get_student_sessions(st.session_state.user['id'])

    if not sessions:
        st.info("You haven't participated in any sessions yet.")
        return

    # Summary statistics across all sessions
    st.subheader("📊 Your Overall Performance")

    all_session_data = []
    for session in sessions:
        df = st.session_state.db_manager.get_attention_data(
            session['session_id'],
            st.session_state.user['id']
        )
        if not df.empty:
            stats = compute_attention_statistics(df)
            all_session_data.append({
                'session_title': session['title'],
                'attention_ratio': stats['attention_ratio'],
                'avg_score': stats['avg_attention_score'],
                'duration_minutes': stats['duration_seconds'] / 60,
                'session_date': session['created_at']
            })

    if all_session_data:
        summary_df = pd.DataFrame(all_session_data)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_attention = summary_df['attention_ratio'].mean()
            st.metric("📈 Average Attention", f"{avg_attention:.2%}")

        with col2:
            best_session = summary_df.loc[summary_df['attention_ratio'].idxmax()]
            st.metric("🏆 Best Session", f"{best_session['attention_ratio']:.2%}")

        with col3:
            total_time = summary_df['duration_minutes'].sum()
            st.metric("⏱️ Total Study Time", f"{total_time:.1f} min")

        with col4:
            total_sessions = len(summary_df)
            st.metric("📚 Sessions Attended", total_sessions)

        # Performance trend chart
        if len(summary_df) > 1:
            summary_df['session_date'] = pd.to_datetime(summary_df['session_date'])
            summary_df = summary_df.sort_values('session_date')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=summary_df['session_date'],
                y=summary_df['attention_ratio'] * 100,
                mode='lines+markers',
                name='Attention Percentage',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title='Your Attention Performance Trend',
                xaxis_title='Session Date',
                yaxis_title='Attention Percentage (%)',
                yaxis=dict(range=[0, 100]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # Individual session details
    st.subheader("📋 Session Details")

    for session in sessions:
        with st.expander(f"📚 {session['title']} - {session['status'].upper()}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**👨‍🏫 Lecturer:** {session['lecturer_name']}")
                st.write(f"**📅 Date:** {session['created_at']}")
                st.write(f"**🔄 Status:** {session['status']}")

            with col2:
                if session['start_time']:
                    st.write(f"**🕐 Started:** {session['start_time']}")
                if session['end_time']:
                    st.write(f"**🕑 Ended:** {session['end_time']}")

            # Show personal attention data for this session
            df = st.session_state.db_manager.get_attention_data(
                session['session_id'],
                st.session_state.user['id']
            )

            if not df.empty:
                stats = compute_attention_statistics(df)

                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Attention Ratio", f"{stats['attention_ratio']:.2%}")
                with col2:
                    st.metric("📊 Average Score", f"{stats['avg_attention_score']:.3f}")
                with col3:
                    st.metric("⏱️ Duration", f"{stats['duration_seconds'] / 60:.1f} min")

                # Detailed stats in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**🔥 Max Attention Streak:** {stats['max_attention_streak']} frames")
                    st.write(f"**😴 Max Distraction Streak:** {stats['max_distraction_streak']} frames")
                    st.write(f"**🔄 Attention Transitions:** {stats['attention_transitions']}")

                with col2:
                    # Performance grade
                    if stats['attention_ratio'] >= 0.9:
                        grade = "A+ 🌟"
                        color = "green"
                    elif stats['attention_ratio'] >= 0.8:
                        grade = "A 🌟"
                        color = "green"
                    elif stats['attention_ratio'] >= 0.7:
                        grade = "B+ 👍"
                        color = "blue"
                    elif stats['attention_ratio'] >= 0.6:
                        grade = "B 👍"
                        color = "blue"
                    elif stats['attention_ratio'] >= 0.5:
                        grade = "C ⚠️"
                        color = "orange"
                    else:
                        grade = "D ❗"
                        color = "red"

                    st.markdown(f"**Performance Grade:** :{color}[{grade}]")

                # Charts for this session
                col1, col2 = st.columns(2)
                with col1:
                    chart = create_attention_time_chart(df, f"Attention Timeline")
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                with col2:
                    chart = create_attention_distribution_chart(df, f"Attention Distribution")
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No attention data recorded for this session.")


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Lecture Attention Tracking System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }

    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }

    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }

    .stExpander > div:first-child {
        background-color: #f8f9fa;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .session-id {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 5px 10px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Check if user is logged in
    if not st.session_state.user:
        login_page()
    else:
        # Add header with user info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 🎓 Lecture Attention Tracking System")
        with col2:
            st.write(f"👤 **{st.session_state.user['full_name']}**")
        with col3:
            st.write(f"🎭 **{st.session_state.user['role'].title()}**")

        st.divider()

        # Route based on user role
        if st.session_state.user['role'] == 'lecturer':
            lecturer_dashboard()
        elif st.session_state.user['role'] == 'student':
            student_dashboard()
        else:
            st.error("Invalid user role")
            st.session_state.user = None
            st.rerun()


if __name__ == "__main__":
    main()
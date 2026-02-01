import os
import re
import csv
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ───────────── CONFIG ─────────────
SRC_ROOT = r'..\Columbia Gaze Data Set'
OUT_CSV = 'geom_features.csv'

VERT_THRESH = 10
HORIZ_THRESH = 10

PATT = re.compile(
    r'\d+_\d+m_([+-]?\d+)P_([+-]?\d+)V_([+-]?\d+)H',
    re.IGNORECASE
)

PNP_IDS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64
)
# ────────────────────────────────────


def compute_head_pose(landmarks, w, h):
    pts2d = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in PNP_IDS],
        dtype=np.float64
    )

    cam = np.array(
        [[w, 0, w / 2],
         [0, w, h / 2],
         [0, 0, 1]],
        dtype=np.float64
    )

    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D,
        pts2d,
        cam,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    return yaw, pitch


def compute_iris_ratios(landmarks, w, h, iris_ids):
    ids = sorted(iris_ids)
    half = len(ids) // 2

    left_ids = ids[:half]
    right_ids = ids[half:]

    def ratios(idxs):
        xs = np.array([landmarks[i].x * w for i in idxs])
        ys = np.array([landmarks[i].y * h for i in idxs])

        cx, cy = xs.mean(), ys.mean()
        xL, xR = xs.min(), xs.max()
        yT, yB = ys.min(), ys.max()

        h_ratio = (cx - xL) / (xR - xL + 1e-6)
        v_ratio = (cy - yT) / (yB - yT + 1e-6)

        return float(h_ratio), float(v_ratio)

    l_h, l_v = ratios(left_ids)
    r_h, r_v = ratios(right_ids)

    return l_h, l_v, r_h, r_v


class _CSVWriter:
    def __init__(self, path):
        self.path = path
        self.fp = None
        self.writer = None

    def __enter__(self):
        self.fp = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.fp)
        self.writer.writerow([
            'subject', 'filename',
            'yaw', 'pitch',
            'l_h_ratio', 'l_v_ratio',
            'r_h_ratio', 'r_v_ratio',
            'label'
        ])
        return self.writer

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fp:
                self.fp.close()
        finally:
            self.fp = None
            self.writer = None
        return False


def _iter_subject_dirs(SRC_ROOT):
    for subj in sorted(os.listdir(SRC_ROOT)):
        subj_dir = os.path.join(SRC_ROOT, subj)
        if os.path.isdir(subj_dir):
            yield subj, subj_dir


def _iter_image_files(subj_dir):
    for fn in os.listdir(subj_dir):
        low = fn.lower()
        if low.endswith(('.jpg', 'jpeg', 'png')):
            yield fn


def _label_from_filename(fn):
    m = PATT.match(fn)
    if not m:
        return None
    gv = int(m.group(2))
    gh = int(m.group(3))
    label = 1 if abs(gv) <= VERT_THRESH and abs(gh) <= HORIZ_THRESH else 0
    return label


def _build_iris_ids(mp_face):
    iris_id_set = set()
    for edge in mp_face.FACEMESH_IRISES:
        iris_id_set.update(edge)
    return sorted(iris_id_set)


def main():
    mp_face = mp.solutions.face_mesh

    with _CSVWriter(OUT_CSV) as writer:
        with mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:

            iris_ids = _build_iris_ids(mp_face)

            for subj, subj_dir in _iter_subject_dirs(SRC_ROOT):
                for fn in tqdm(list(_iter_image_files(subj_dir)), desc=f"Subject {subj}"):
                    label = _label_from_filename(fn)
                    if label is None:
                        continue

                    img_path = os.path.join(subj_dir, fn)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]

                    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if not res.multi_face_landmarks:
                        continue

                    lm = res.multi_face_landmarks[0].landmark

                    yaw, pitch = compute_head_pose(lm, w, h)
                    if yaw is None:
                        continue

                    l_h, l_v, r_h, r_v = compute_iris_ratios(lm, w, h, iris_ids)

                    writer.writerow([
                        subj, fn,
                        f"{yaw:.3f}", f"{pitch:.3f}",
                        f"{l_h:.3f}", f"{l_v:.3f}",
                        f"{r_h:.3f}", f"{r_v:.3f}",
                        label
                    ])

    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print(f"✅ Wrote {len(df)} rows to {OUT_CSV}")
    print(df.head())


if __name__ == '__main__':
    main()

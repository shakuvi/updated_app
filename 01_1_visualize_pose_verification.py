import os, cv2, random, numpy as np, mediapipe as mp
from tqdm import tqdm

# ───────── CONFIG ─────────
# Path to your Columbia Gaze raw folders (0001,0002,…)
SRC_ROOT    = r'..\Columbia Gaze Data Set'
# Where to save the annotated images
OUT_DIR     = 'pose_iris_verification_full'
# How many random images to process
NUM_SAMPLES = 20
# PnP generic 3D face model points
PNP_IDS     = [1,152,33,263,61,291]
MODEL_3D    = np.array([
    ( 0.0,   0.0,    0.0),
    ( 0.0, -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    ( 43.3,  32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    ( 28.9, -28.9,  -24.1)
], dtype=np.float64)
AXIS_LEN    = 50   # mm, length of each PnP axis
# ────────────────────────────

def draw_axes(img, rvec, tvec, cam_mat):
    """Draw red/green/blue 3D axes at the nose tip."""
    axes_3d = np.float32([
        [AXIS_LEN, 0, 0],
        [0, AXIS_LEN, 0],
        [0, 0, AXIS_LEN],
    ])
    imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam_mat, None)
    origin, _ = cv2.projectPoints(np.zeros((1,3)), rvec, tvec, cam_mat, None)
    o = tuple(origin.ravel().astype(int))
    colors = [(0,0,255),(0,255,0),(255,0,0)]  # x=red, y=green, z=blue
    for pt, col in zip(imgpts.reshape(-1,2).astype(int), colors):
        cv2.line(img, o, tuple(pt), col, 2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Gather all (subject,filename) pairs
    all_imgs = []
    for subj in os.listdir(SRC_ROOT):
        d = os.path.join(SRC_ROOT, subj)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg','jpeg','png')):
                all_imgs.append((subj, fn))

    sample = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))

    # Initialize FaceMesh with iris refinement
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True  # <— needed for iris points
    ) as face_mesh:

        for subj, fn in tqdm(sample, desc="Visualizing"):
            src = os.path.join(SRC_ROOT, subj, fn)
            img = cv2.imread(src)
            if img is None:
                continue

            h, w = img.shape[:2]
            cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)

            # Run FaceMesh
            res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark

            # ── HEAD-POSE (PnP) ──────────────────────────
            pts2d = np.array([(lm[i].x*w, lm[i].y*h) for i in PNP_IDS], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_3D, pts2d, cam, np.zeros((4,1)),
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                draw_axes(img, rvec, tvec, cam)

            # ── IRIS (rectangle + small circles) ────────
            # collect all iris landmark indices dynamically
            iris_ids = set()
            for edge in mp_face.FACEMESH_IRISES:
                iris_ids.update(edge)
            iris_ids = sorted(iris_ids)

            # compute bounding box of those iris points
            xs = []; ys = []
            for idx in iris_ids:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                xs.append(x); ys.append(y)
                # small filled circle per landmark
                cv2.circle(img, (x,y), 4, (0,255,255), -1)  # bright yellow

            if xs and ys:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                pad = 6
                x1, y1 = max(x1-pad,0), max(y1-pad,0)
                x2, y2 = min(x2+pad,w), min(y2+pad,h)
                # bold teal rectangle
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), thickness=4)

            # ── SAVE ─────────────────────────────────────
            od = os.path.join(OUT_DIR, subj)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, fn), img)

    print("✅ All done! Check out the annotated frames in", OUT_DIR)

if __name__ == '__main__':
    main()

import cv2
import joblib
import numpy as np
import mediapipe as mp

from collections import deque
from threading import Thread
from queue import Queue, Empty

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

HAND_MODEL_PATH = "models/hand_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"

MODEL_PATH = "models/best_model.joblib"
METADATA_PATH = "models/metadata.joblib"

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

PREDICTION_HISTORY_SIZE = 7
MIN_MAJORITY_COUNT = 5

# Number of stable majority confirmations required
# before changing the committed sign.
COMMIT_CONFIRMATIONS = 2


# ============================================================
# SIGN HISTORY
# ============================================================

MAX_SIGN_HISTORY = 8


# ============================================================
# HUMAN-READABLE SIGN NAMES
# ============================================================

DISPLAY_NAMES = {
    "friend": "Friend",
    "good morning": "Good Morning",
    "happybirthday": "Happy Birthday",
    "hello": "Hello",
    "howareyou": "How Are You",
    "iloveyou": "I Love You",
    "ineedhelp": "I Need Help",
    "mistake": "Mistake",
    "no": "No",
    "ok": "OK",
    "pain": "Pain",
    "please": "Please",
    "sorry": "Sorry",
    "understand": "Understand",
    "where": "Where",
    "yes": "Yes",
}


def display_name(label):
    return DISPLAY_NAMES.get(
        label,
        label.replace("_", " ").title()
    )


# ============================================================
# UI COLORS
# ============================================================

WHITE = (245, 245, 245)
LIGHT_GRAY = (205, 205, 205)
GRAY = (145, 145, 145)

DARK_PANEL = (18, 18, 18)
DARK_PANEL_2 = (28, 28, 28)

CYAN = (255, 210, 60)
GREEN = (80, 220, 130)
BLUE = (220, 150, 70)
YELLOW = (40, 220, 255)
RED = (80, 80, 235)

HAND_COLOR = (80, 220, 130)
HAND_CONNECTION_COLOR = (60, 180, 100)

POSE_COLOR = (230, 170, 70)
POSE_CONNECTION_COLOR = (180, 120, 50)

OUTLINE_COLOR = (15, 15, 15)


# ============================================================
# SPEECH ENGINE
# ============================================================

class SpeechEngine:

    def __init__(self):

        self.queue = Queue()
        self.running = True

        self.thread = Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    def _worker(self):

        pythoncom = None
        speaker = None

        try:

            import pythoncom
            import win32com.client

            # Initialize COM for this thread.
            pythoncom.CoInitialize()

            # Direct Windows SAPI voice.
            speaker = win32com.client.Dispatch(
                "SAPI.SpVoice"
            )

            # Speech rate:
            # -10 = slow
            # 0   = normal
            # 10  = fast
            speaker.Rate = -1

            # Volume: 0-100
            speaker.Volume = 100

            print(
                "Windows SAPI speech engine ready."
            )

            while self.running:

                try:

                    text = self.queue.get(
                        timeout=0.2
                    )

                except Empty:

                    continue

                if text is None:
                    break

                try:

                    print(
                        f"Speaking: {text}"
                    )

                    # 0 = synchronous speech.
                    #
                    # The worker waits until the
                    # current word/sentence finishes,
                    # then processes the next sign.
                    speaker.Speak(
                        str(text),
                        0
                    )

                except Exception as error:

                    print(
                        f"SAPI speech error: {error}"
                    )

        except Exception as error:

            print()
            print("=" * 70)
            print("WINDOWS SPEECH ENGINE ERROR")
            print("=" * 70)
            print(error)
            print("=" * 70)
            print()

        finally:

            if pythoncom is not None:

                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def speak(self, text):

        if not self.running:
            return

        if not text:
            return

        # Add the newest sign to the speech queue.
        self.queue.put(
            str(text)
        )

    def stop(self):

        self.running = False

        try:
            self.queue.put_nowait(
                None
            )
        except Exception:
            pass

        if self.thread.is_alive():

            self.thread.join(
                timeout=2.0
            )


# ============================================================
# CREATE HAND DETECTOR
# ============================================================

def create_hand_detector():

    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# CREATE POSE DETECTOR
# ============================================================

def create_pose_detector():

    base_options = python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.PoseLandmarker.create_from_options(
        options
    )


# ============================================================
# NORMALIZE HAND
# MUST MATCH extract_landmarks.py
# ============================================================

def normalize_hand(
    landmarks,
    handedness
):

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    scale_x = (
        middle_mcp.x
        - wrist.x
    )

    scale_y = (
        middle_mcp.y
        - wrist.y
    )

    scale_z = (
        middle_mcp.z
        - wrist.z
    )

    scale = (
        scale_x ** 2
        + scale_y ** 2
        + scale_z ** 2
    ) ** 0.5

    if scale < 1e-6:
        scale = 1.0

    normalized = []

    for landmark in landmarks:

        x = (
            landmark.x
            - wrist.x
        ) / scale

        y = (
            landmark.y
            - wrist.y
        ) / scale

        z = (
            landmark.z
            - wrist.z
        ) / scale

        # Same left-hand canonicalization
        # used during dataset extraction.
        if handedness == "Left":

            x = -x
            z = -z

        normalized.extend([
            x,
            y,
            z
        ])

    return normalized


# ============================================================
# NORMALIZE POSE
# MUST MATCH extract_landmarks.py
# ============================================================

def normalize_pose(landmarks):

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center_x = (
        left_shoulder.x
        + right_shoulder.x
    ) / 2

    center_y = (
        left_shoulder.y
        + right_shoulder.y
    ) / 2

    shoulder_x = (
        left_shoulder.x
        - right_shoulder.x
    )

    shoulder_y = (
        left_shoulder.y
        - right_shoulder.y
    )

    shoulder_distance = (
        shoulder_x ** 2
        + shoulder_y ** 2
    ) ** 0.5

    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    normalized = []

    for landmark in landmarks:

        normalized.extend([
            (
                landmark.x
                - center_x
            ) / shoulder_distance,

            (
                landmark.y
                - center_y
            ) / shoulder_distance,

            landmark.z
            / shoulder_distance
        ])

    return normalized


# ============================================================
# EMPTY FEATURES
# ============================================================

def empty_hand():
    return [0.0] * 63


def empty_pose():
    return [0.0] * 99


# ============================================================
# HAND CENTER
# ============================================================

def get_hand_center(hand):

    x = sum(
        landmark.x
        for landmark in hand
    ) / len(hand)

    y = sum(
        landmark.y
        for landmark in hand
    ) / len(hand)

    return x, y


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    frame,
    hand_detector,
    pose_detector
):

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    hand_result = hand_detector.detect(
        mp_image
    )

    pose_result = pose_detector.detect(
        mp_image
    )

    hands = []

    for index, hand in enumerate(
        hand_result.hand_landmarks
    ):

        handedness = (
            hand_result
            .handedness[index][0]
            .category_name
        )

        center_x, center_y = (
            get_hand_center(hand)
        )

        features = normalize_hand(
            hand,
            handedness
        )

        hands.append({
            "center_x": center_x,
            "center_y": center_y,
            "handedness": handedness,
            "features": features
        })

    # SAME ordering as training
    hands.sort(
        key=lambda hand: (
            hand["center_x"],
            hand["center_y"]
        )
    )

    hand_1 = empty_hand()
    hand_2 = empty_hand()

    if len(hands) >= 1:
        hand_1 = hands[0]["features"]

    if len(hands) >= 2:
        hand_2 = hands[1]["features"]

    pose_features = empty_pose()

    if pose_result.pose_landmarks:

        pose_features = normalize_pose(
            pose_result.pose_landmarks[0]
        )

    features = (
        hand_1
        + hand_2
        + pose_features
    )

    if len(features) != 225:

        raise RuntimeError(
            f"Feature count mismatch: "
            f"{len(features)} instead of 225"
        )

    return (
        np.array(
            features,
            dtype=np.float32
        ),
        hand_result,
        pose_result,
        hands
    )


# ============================================================
# PREDICTION STABILIZER
# ============================================================

class PredictionStabilizer:

    def __init__(self):

        # Consecutive identical predictions required
        # before accepting a new sign.
        self.required_frames = 10

        self.candidate = None
        self.candidate_frames = 0

        self.committed = None

    def update(self, prediction):

        # First prediction.
        if self.candidate is None:

            self.candidate = prediction
            self.candidate_frames = 1

            return self.committed, False

        # Same prediction continues.
        if prediction == self.candidate:

            self.candidate_frames += 1

        # New prediction starts from zero.
        else:

            self.candidate = prediction
            self.candidate_frames = 1

        # Not stable yet.
        if self.candidate_frames < self.required_frames:

            return self.committed, False

        # Same as currently committed sign.
        if self.candidate == self.committed:

            return self.committed, False

        # New sign is now committed.
        self.committed = self.candidate

        return self.committed, True

    def reset(self):

        self.candidate = None
        self.candidate_frames = 0
        self.committed = None


# ============================================================
# DRAW HELPERS
# ============================================================

def point_from_landmark(
    landmark,
    width,
    height,
    mirrored=True
):

    x = landmark.x

    if mirrored:
        x = 1.0 - x

    y = landmark.y

    return (
        int(x * width),
        int(y * height)
    )


def draw_thick_line(
    image,
    point1,
    point2,
    color,
    thickness=2
):

    # Dark outline first.
    cv2.line(
        image,
        point1,
        point2,
        OUTLINE_COLOR,
        thickness + 3,
        cv2.LINE_AA
    )

    cv2.line(
        image,
        point1,
        point2,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_hand_skeleton(
    frame,
    hand
):

    height, width = frame.shape[:2]

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    points = []

    for landmark in hand:

        points.append(
            point_from_landmark(
                landmark,
                width,
                height,
                mirrored=True
            )
        )

    # Skeleton bones.
    for start, end in connections:

        draw_thick_line(
            frame,
            points[start],
            points[end],
            HAND_CONNECTION_COLOR,
            3
        )

    # Landmark nodes.
    for index, point in enumerate(points):

        radius = 5

        if index in [4, 8, 12, 16, 20]:
            radius = 7

        cv2.circle(
            frame,
            point,
            radius + 2,
            OUTLINE_COLOR,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            point,
            radius,
            HAND_COLOR,
            -1,
            cv2.LINE_AA
        )

    # Wrist highlight.
    cv2.circle(
        frame,
        points[0],
        9,
        OUTLINE_COLOR,
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        points[0],
        6,
        YELLOW,
        -1,
        cv2.LINE_AA
    )


def draw_hand_box(
    frame,
    hand
):

    height, width = frame.shape[:2]

    points = [
        point_from_landmark(
            landmark,
            width,
            height,
            mirrored=True
        )
        for landmark in hand
    ]

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    if not xs or not ys:
        return

    padding = 18

    x1 = max(
        0,
        min(xs) - padding
    )

    y1 = max(
        0,
        min(ys) - padding
    )

    x2 = min(
        width - 1,
        max(xs) + padding
    )

    y2 = min(
        height - 1,
        max(ys) + padding
    )

    # Rounded-style corner brackets.
    corner = 18

    color = HAND_COLOR
    thickness = 2

    # Top-left.
    cv2.line(
        frame,
        (x1, y1),
        (x1 + corner, y1),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + corner),
        color,
        thickness,
        cv2.LINE_AA
    )

    # Top-right.
    cv2.line(
        frame,
        (x2 - corner, y1),
        (x2, y1),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + corner),
        color,
        thickness,
        cv2.LINE_AA
    )

    # Bottom-left.
    cv2.line(
        frame,
        (x1, y2 - corner),
        (x1, y2),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1 + corner, y2),
        color,
        thickness,
        cv2.LINE_AA
    )

    # Bottom-right.
    cv2.line(
        frame,
        (x2 - corner, y2),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x2, y2 - corner),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW POSE SKELETON
# ============================================================

def draw_pose_skeleton(
    frame,
    pose_landmarks
):

    if not pose_landmarks:
        return

    height, width = frame.shape[:2]

    # Upper-body connections.
    connections = [
        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28),
    ]

    points = []

    for landmark in pose_landmarks:

        points.append(
            point_from_landmark(
                landmark,
                width,
                height,
                mirrored=True
            )
        )

    # Draw only if the landmark is reasonably visible.
    for start, end in connections:

        if (
            start >= len(points)
            or end >= len(points)
        ):
            continue

        draw_thick_line(
            frame,
            points[start],
            points[end],
            POSE_CONNECTION_COLOR,
            3
        )

    important_points = [
        11, 12,
        13, 14,
        15, 16,
        23, 24,
        25, 26,
        27, 28
    ]

    for index in important_points:

        if index >= len(points):
            continue

        point = points[index]

        cv2.circle(
            frame,
            point,
            6,
            OUTLINE_COLOR,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            point,
            4,
            POSE_COLOR,
            -1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW TRANSPARENT RECTANGLE
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    color=DARK_PANEL,
    alpha=0.88
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# TEXT HELPER
# ============================================================

def draw_text(
    frame,
    text,
    position,
    size=0.6,
    color=WHITE,
    thickness=1
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("HANDLINK")
print("=" * 70)

print()
print("Loading model...")

model = joblib.load(
    MODEL_PATH
)

metadata = joblib.load(
    METADATA_PATH
)

id_to_label = metadata["id_to_label"]

print(
    f"Model: {metadata['best_model']}"
)

print(
    f"Features: {metadata['feature_count']}"
)

if metadata["feature_count"] != 225:

    raise RuntimeError(
        "Model does not expect 225 features."
    )

if not hasattr(model, "predict"):

    raise RuntimeError(
        "Model does not support predict()."
    )

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

print()
print("Loading MediaPipe...")

hand_detector = create_hand_detector()
pose_detector = create_pose_detector()

print("MediaPipe loaded.")


# ============================================================
# SPEECH
# ============================================================

speech = SpeechEngine()


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

if not camera.isOpened():

    hand_detector.close()
    pose_detector.close()
    speech.stop()

    raise RuntimeError(
        "Could not open webcam."
    )


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


# ============================================================
# STATE
# ============================================================

stabilizer = PredictionStabilizer()

sign_history = deque(
    maxlen=MAX_SIGN_HISTORY
)

current_sign = None

last_hand_count = 0


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "HandLink"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    FRAME_WIDTH,
    FRAME_HEIGHT
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read webcam frame."
        )

        break

    # --------------------------------------------------------
    # ORIGINAL FRAME GOES INTO MODEL
    # --------------------------------------------------------

    (
        features,
        hand_result,
        pose_result,
        hands
    ) = extract_features(
        frame,
        hand_detector,
        pose_detector
    )

    hand_count = len(
        hand_result.hand_landmarks
    )

    last_hand_count = hand_count


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    if hand_count > 0:

        prediction_id = model.predict(
            features.reshape(1, -1)
        )[0]

        raw_prediction = id_to_label[
            int(prediction_id)
        ]

        committed_sign, changed = (
            stabilizer.update(
                raw_prediction
            )
        )

        # ----------------------------------------------------
        # NEW SIGN COMMITTED
        # ----------------------------------------------------

        if changed:

            current_sign = committed_sign

            readable_sign = display_name(
                committed_sign
            )

            # Add to history.
            sign_history.appendleft(
                readable_sign
            )

            # Speak the new sign.
            speech.speak(
                readable_sign
            )

            print(
                f"Committed sign: "
                f"{readable_sign}"
            )

    else:

        # IMPORTANT:
        #
        # We DO NOT erase the committed sign.
        #
        # If the hand briefly disappears,
        # the last sign remains on screen.
        pass


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    display_frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # TRACKING OVERLAY
    # ========================================================

    for hand in (
        hand_result.hand_landmarks
    ):

        draw_hand_skeleton(
            display_frame,
            hand
        )

        draw_hand_box(
            display_frame,
            hand
        )


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    if pose_result.pose_landmarks:

        draw_pose_skeleton(
            display_frame,
            pose_result.pose_landmarks[0]
        )


    # ========================================================
    # TOP HEADER
    # ========================================================

    header_height = 92

    draw_panel(
        display_frame,
        0,
        0,
        display_frame.shape[1],
        header_height,
        DARK_PANEL,
        0.94
    )

    # Accent line.
    cv2.rectangle(
        display_frame,
        (0, header_height - 3),
        (
            display_frame.shape[1],
            header_height
        ),
        CYAN,
        -1
    )

    # Logo.
    draw_text(
        display_frame,
        "HANDLINK",
        (28, 38),
        0.9,
        WHITE,
        2
    )

    # Status indicator.
    if hand_count > 0:

        status_text = "LIVE"
        status_color = GREEN

    else:

        status_text = "READY"
        status_color = GRAY

    cv2.circle(
        display_frame,
        (255, 31),
        7,
        status_color,
        -1,
        cv2.LINE_AA
    )

    draw_text(
        display_frame,
        status_text,
        (270, 37),
        0.55,
        status_color,
        1
    )

    draw_text(
        display_frame,
        f"{hand_count} hand"
        + ("s" if hand_count != 1 else "")
        + " detected",
        (28, 70),
        0.52,
        LIGHT_GRAY,
        1
    )


    # ========================================================
    # CURRENT SIGN PANEL
    # ========================================================

    panel_width = 430

    panel_x1 = (
        display_frame.shape[1]
        - panel_width
        - 22
    )

    panel_y1 = 112

    panel_x2 = (
        display_frame.shape[1]
        - 22
    )

    panel_y2 = 275

    draw_panel(
        display_frame,
        panel_x1,
        panel_y1,
        panel_x2,
        panel_y2,
        DARK_PANEL,
        0.92
    )

    draw_text(
        display_frame,
        "CURRENT SIGN",
        (
            panel_x1 + 22,
            panel_y1 + 32
        ),
        0.5,
        CYAN,
        1
    )

    if current_sign is not None:

        sign_text = display_name(
            current_sign
        )

        draw_text(
            display_frame,
            sign_text,
            (
                panel_x1 + 22,
                panel_y1 + 88
            ),
            1.05,
            WHITE,
            2
        )

        draw_text(
            display_frame,
            "Sign committed",
            (
                panel_x1 + 22,
                panel_y1 + 125
            ),
            0.48,
            GREEN,
            1
        )

    else:

        draw_text(
            display_frame,
            "Waiting...",
            (
                panel_x1 + 22,
                panel_y1 + 88
            ),
            0.9,
            LIGHT_GRAY,
            1
        )


    # ========================================================
    # SIGN HISTORY PANEL
    # ========================================================

    history_x1 = panel_x1

    history_y1 = 295

    history_x2 = panel_x2

    history_y2 = (
        history_y1
        + 300
    )

    draw_panel(
        display_frame,
        history_x1,
        history_y1,
        history_x2,
        history_y2,
        DARK_PANEL,
        0.92
    )

    draw_text(
        display_frame,
        "SIGN HISTORY",
        (
            history_x1 + 22,
            history_y1 + 32
        ),
        0.5,
        CYAN,
        1
    )

    if len(sign_history) == 0:

        draw_text(
            display_frame,
            "No signs committed yet",
            (
                history_x1 + 22,
                history_y1 + 70
            ),
            0.5,
            GRAY,
            1
        )

    else:

        for index, sign in enumerate(
            sign_history
        ):

            y = (
                history_y1
                + 68
                + index * 29
            )

            if index == 0:

                number_color = CYAN
                text_color = WHITE

            else:

                number_color = GRAY
                text_color = LIGHT_GRAY

            draw_text(
                display_frame,
                f"{index + 1:02d}",
                (
                    history_x1 + 22,
                    y
                ),
                0.46,
                number_color,
                1
            )

            draw_text(
                display_frame,
                sign,
                (
                    history_x1 + 65,
                    y
                ),
                0.5,
                text_color,
                1
            )


    # ========================================================
    # BOTTOM STATUS BAR
    # ========================================================

    bottom_height = 42

    bottom_y1 = (
        display_frame.shape[0]
        - bottom_height
    )

    draw_panel(
        display_frame,
        0,
        bottom_y1,
        display_frame.shape[1],
        display_frame.shape[0],
        DARK_PANEL,
        0.94
    )

    draw_text(
        display_frame,
        "Q  Quit",
        (25, bottom_y1 + 27),
        0.45,
        LIGHT_GRAY,
        1
    )

    draw_text(
        display_frame,
        "R  Reset History",
        (110, bottom_y1 + 27),
        0.45,
        LIGHT_GRAY,
        1
    )

    draw_text(
        display_frame,
        "LIVE SIGN LANGUAGE TRANSLATION",
        (
            display_frame.shape[1] - 310,
            bottom_y1 + 27
        ),
        0.4,
        GRAY,
        1
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Quit.
    if key == ord("q"):

        break

    # Reset.
    if key == ord("r"):

        stabilizer.reset()

        sign_history.clear()

        current_sign = None

        print(
            "Sign history reset."
        )


# ============================================================
# CLEANUP
# ============================================================

camera.release()

hand_detector.close()
pose_detector.close()

speech.stop()

cv2.destroyAllWindows()

print()
print("=" * 70)
print("HANDLINK CLOSED")
print("=" * 70)
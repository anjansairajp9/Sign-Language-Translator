import os
import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DATASET_DIR = "dataset"
OUTPUT_FILE = "data/landmarks_v2.csv"

HAND_MODEL_PATH = "models/hand_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"

GESTURES = [
    "friend",
    "good morning",
    "happybirthday",
    "hello",
    "howareyou",
    "iloveyou",
    "ineedhelp",
    "mistake",
    "no",
    "ok",
    "pain",
    "please",
    "sorry",
    "understand",
    "where",
    "yes"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


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


def normalize_hand(landmarks, handedness):
    wrist = landmarks[0]

    scale = max(
        (
            (landmark.x - wrist.x) ** 2
            + (landmark.y - wrist.y) ** 2
            + (landmark.z - wrist.z) ** 2
        ) ** 0.5
        for landmark in landmarks
    )

    if scale < 1e-6:
        scale = 1.0

    normalized = []

    for landmark in landmarks:
        x = (
            landmark.x - wrist.x
        ) / scale

        y = (
            landmark.y - wrist.y
        ) / scale

        z = (
            landmark.z - wrist.z
        ) / scale

        if handedness == "Left":
            x = -x
            z = -z

        normalized.extend([
            x,
            y,
            z
        ])

    return normalized


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


def empty_hand():
    return [0.0] * 63


def empty_pose():
    return [0.0] * 99


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


def process_image(
    image_path,
    hand_detector,
    pose_detector
):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(
        image,
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

        center_x, center_y = get_hand_center(
            hand
        )

        features = normalize_hand(
            hand,
            handedness
        )

        hands.append({
            "center_x": center_x,
            "center_y": center_y,
            "features": features
        })

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

    if (
        hand_1 == empty_hand()
        and hand_2 == empty_hand()
        and pose_features == empty_pose()
    ):
        return None

    return (
        hand_1
        + hand_2
        + pose_features
    )


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

hand_detector = create_hand_detector()
pose_detector = create_pose_detector()

rows = []

total_processed = 0
total_skipped = 0

print("=" * 60)
print("HANDLINK V2 NORMALIZED LANDMARK EXTRACTION")
print("=" * 60)

for gesture in GESTURES:

    gesture_folder = os.path.join(
        DATASET_DIR,
        gesture
    )

    if not os.path.isdir(
        gesture_folder
    ):
        print(
            f"\nWARNING: Folder not found: "
            f"{gesture_folder}"
        )
        continue

    image_files = [
        file
        for file in os.listdir(
            gesture_folder
        )
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    processed = 0
    skipped = 0

    print()
    print(f"Processing: {gesture}")
    print(
        f"Images found: "
        f"{len(image_files)}"
    )

    for image_file in sorted(
        image_files
    ):

        image_path = os.path.join(
            gesture_folder,
            image_file
        )

        features = process_image(
            image_path,
            hand_detector,
            pose_detector
        )

        if features is None:
            skipped += 1
            total_skipped += 1
            continue

        row = {
            "label": gesture
        }

        for index, value in enumerate(
            features
        ):
            row[
                f"feature_{index}"
            ] = value

        rows.append(row)

        processed += 1
        total_processed += 1

    print(
        f"Images processed: "
        f"{processed}"
    )

    print(
        f"Images skipped: "
        f"{skipped}"
    )


hand_detector.close()
pose_detector.close()

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Images processed : "
    f"{total_processed}"
)

print(
    f"Images skipped   : "
    f"{total_skipped}"
)

print(
    f"Dataset shape    : "
    f"{df.shape}"
)

print(
    f"Features         : "
    f"{df.shape[1] - 1}"
)

print(
    f"Saved to         : "
    f"{OUTPUT_FILE}"
)

print("=" * 60)

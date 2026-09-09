"""PUPPET EXHIBITION BUILD: lower-load tracking and visitor handover.

The camera stays visually clean while only a few important hand joints are
marked with white dots. MediaPipe provides wave detection, visible hand
recognition, and audience-presence detection. The exhibition view combines
layered memory windows, a CRT/particle scan of the real articulated puppet,
clear cinematic text, physical energy, response number, and a bottle that
visibly empties during the thirty-second water break.

Behaviour arc:
    waves 1-6:  180 degrees gradually falls to 60 degrees
    wave 7:     a sudden double 180-degree second-wind burst
    waves 8-9:  controlled random, exhausted responses
    wave 10:    0 degrees; the puppet refuses, but the visitor session remains
    water break: after ten separate visitor sessions, rest for 30 seconds

Companion Arduino sketch: PUPPET_BUILD_5_D9.ino
"""

from collections import deque
from functools import lru_cache
import json
import math
import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import serial
from serial.tools import list_ports

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


BUILD_VERSION = "5.3"
BUILD_SIGNATURE = "15 FPS / 5S THANK YOU / VISITOR CHANGE"
ARDUINO_FIRMWARE_ID = "PUPPET_BUILD_4"


# ============================================================
# Device settings
# ============================================================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
CAMERA_INDEX = 0
WINDOW_NAME = "PUPPET"

START_FULLSCREEN = True
SHOW_SHORTCUTS_AT_START = False
SHOW_HAND_LANDMARKS = True

# MediaPipe still processes the camera frame at webcam resolution. The live
# image is then cropped to 16:9 and enlarged BEFORE text and UI are drawn, so
# exhibition typography is never a 640x480 image stretched to fullscreen.
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
TARGET_FPS = 15.0


# ============================================================
# Ten-response cycle
# ============================================================

TOTAL_INTERACTIONS = 10
REST_DURATION_SECONDS = 30.0
VISITORS_PER_WATER_BREAK = 10
THANK_YOU_DURATION_SECONDS = 5.0
VISITOR_COUNTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "PUPPET_VISITOR_COUNTER.json",
)

# The first six responses form a clearly visible shrinking sequence. Response
# 7 uses the Arduino's dedicated burst motion: 180 -> 40 -> 180 -> rest.
# Responses 8-9 are chosen from exhausted angle ranges.
FIXED_RESPONSE_ANGLES = {
    1: 180,
    2: 160,
    3: 140,
    4: 115,
    5: 85,
    6: 60,
    7: 180,
    10: 0,
}

RANDOM_RESPONSE_ANGLES = {
    8: (35, 70, 105, 135),
    9: (15, 40, 65, 90),
}

# Delay stays readable but does not dominate the work. The sudden response at
# interaction 7 is immediate, reinforcing its unexpected "second wind".
RESPONSE_DELAYS = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.35,
    5: 0.70,
    6: 1.10,
    7: 0.0,
    8: 1.35,
    9: 1.85,
    10: 0.0,
}

PUPPET_LINES = {
    1: "I'M READY.",
    2: "I CAN DO THIS ALL DAY.",
    3: "AGAIN. I'M STILL HERE.",
    4: "MY ARM FEELS HEAVIER.",
    5: "I'M GETTING TIRED.",
    6: "I NEED A MOMENT.",
    7: "I'M FINE. WATCH THIS.",
    8: "THAT WASN'T QUITE RIGHT.",
    9: "MY BODY IS NOT LISTENING.",
    10: "NO. I'M DONE.",
}

# Physical energy is intentionally non-linear: repeated effort drains the
# body faster as the round continues. The second-wind burst does not restore
# energy; it spends what little remains.
PHYSICAL_ENERGY_LEVELS = {
    0: 100,
    1: 88,
    2: 75,
    3: 62,
    4: 48,
    5: 34,
    6: 22,
    7: 13,
    8: 7,
    9: 3,
    10: 0,
}


# ============================================================
# Audience-presence and autonomous activity
# ============================================================

# A short grace period smooths momentary segmentation errors. Once a round has
# started, four continuous seconds without a person resets it.
PERSON_DETECTION_GRACE_SECONDS = 1.0
NO_PERSON_RESTART_SECONDS = 4.0
POSE_VISIBILITY_THRESHOLD = 0.45

# A session stores only a compact colour histogram from the visitor's torso.
# It is not face recognition and no source crop is saved. A deliberately long
# confirmation time prevents one turn, arm crossing, or tracking glitch from
# being mistaken for a new visitor.
VISITOR_CHANGE_DISTANCE = 0.52
VISITOR_CHANGE_CONFIRM_SECONDS = 1.10
VISITOR_SIGNATURE_CHECK_INTERVAL = 0.25
VISITOR_SIGNATURE_WARMUP_SECONDS = 1.25

# In the middle and late stages, the puppet may move without being asked. These
# actions do not consume stamina or increase the interaction count. Limiting
# them to one per round reduces unnecessary servo heat during a long exhibition.
RANDOM_ACTIVITY_START_COUNT = 4
RANDOM_ACTIVITY_END_COUNT = 9
RANDOM_ACTIVITY_MIN_WAIT = 6.0
RANDOM_ACTIVITY_MAX_WAIT = 10.0
MAX_RANDOM_ACTIVITIES_PER_ROUND = 1
RANDOM_ACTIVITY_ANGLES = (30, 45, 65, 90, 120)


# ============================================================
# Wave detection
# ============================================================

MAX_HANDS = 4
WAVE_WINDOW_SECONDS = 1.8
MIN_WAVE_SPAN = 0.04
MIN_DIRECTION_STEP = 0.010
MIN_DIRECTION_CHANGES = 1
MIN_WAVE_FRAMES = 4
COMMAND_COOLDOWN_SECONDS = 1.30
TRACK_TIMEOUT_SECONDS = 0.65
TRACK_MATCH_DISTANCE = 0.25


# ============================================================
# Colour exhibition interface with monochrome tracking marks
# ============================================================

PANEL_BACKGROUND = (8, 8, 8)
TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (172, 172, 172)
TEXT_FAINT = (104, 104, 104)
WINDOW_LINE = (156, 156, 156)
MEMORY_FRAME_LIMIT = 5

# Only a small set of important joints is displayed as clean white dots. No
# MediaPipe rainbow colours, black outline, skeleton lines, or trails are used.
HAND_FOREGROUND_COLOUR = (255, 255, 255)
HAND_POINT_RADIUS = 4
# Six points only: wrist plus five fingertips. This keeps recognition readable
# without turning the participant's hand into a technical skeleton.
DISPLAY_HAND_LANDMARK_INDICES = (0, 4, 8, 12, 16, 20)
WAVE_RECOGNIZED_DISPLAY_SECONDS = 1.0
SECOND_WIND_DISPLAY_SECONDS = 3.0
# The build id belongs in the terminal. It appears on-screen only when the
# operator intentionally opens the shortcut/debug layer with H.
BUILD_LABEL_DISPLAY_SECONDS = 0.0


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def find_arduino_port(preferred_port):
    ports = list(list_ports.comports())
    available_names = [item.device for item in ports]

    if preferred_port in available_names:
        return preferred_port

    keywords = (
        "arduino",
        "mega",
        "ch340",
        "wch",
        "usb serial",
        "serial device",
    )
    likely_ports = []

    for item in ports:
        description = f"{item.description} {item.manufacturer or ''}".lower()
        if any(word in description for word in keywords):
            likely_ports.append(item.device)

    if len(likely_ports) == 1:
        print(f"{preferred_port} was not found. Using {likely_ports[0]} instead.")
        return likely_ports[0]

    available_text = ", ".join(available_names) or "none"
    raise RuntimeError(
        f"Arduino was not found. Available ports: {available_text}"
    )


def connect_arduino():
    port = find_arduino_port(SERIAL_PORT)
    print(f"Connecting to Arduino on {port}...")
    arduino = serial.Serial(
        port=port,
        baudrate=BAUD_RATE,
        timeout=0.05,
        write_timeout=1,
    )
    time.sleep(2.4)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    verify_arduino_firmware(arduino)
    print(f"Arduino connected and verified: {port} / {ARDUINO_FIRMWARE_ID}")
    return arduino


def verify_arduino_firmware(arduino):
    """Refuse to run silently against the old W-only Arduino sketch."""
    arduino.write(b"V\n")
    arduino.flush()
    deadline = time.monotonic() + 2.0
    replies = []

    while time.monotonic() < deadline:
        if arduino.in_waiting <= 0:
            time.sleep(0.03)
            continue

        reply = (
            arduino.readline()
            .decode("utf-8", errors="replace")
            .strip()
        )
        if not reply:
            continue
        replies.append(reply)
        if reply == f"FIRMWARE:{ARDUINO_FIRMWARE_ID}":
            return

    received = ", ".join(replies) if replies else "no reply"
    raise RuntimeError(
        "ARDUINO CODE IS OUTDATED OR ON THE WRONG PORT. "
        "Upload PUPPET_BUILD_5_D9.ino, keep the servo signal on D9, "
        f"then restart. Arduino replied: {received}"
    )


def open_camera(index):
    print(f"Opening camera {index}...")
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera {index}. Try CAMERA_INDEX 0, 1 or 2."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Camera opened / target {TARGET_FPS:.0f} FPS")
    return camera


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def get_hand_position(hand_landmarks):
    important_points = [0, 5, 9, 13, 17]
    x_values = [hand_landmarks.landmark[index].x for index in important_points]
    y_values = [hand_landmarks.landmark[index].y for index in important_points]
    return (
        sum(x_values) / len(x_values),
        sum(y_values) / len(y_values),
    )


def update_track_history(track, current_time, position):
    track["history"].append((current_time, position[0]))

    while (
        track["history"]
        and current_time - track["history"][0][0] > WAVE_WINDOW_SECONDS
    ):
        track["history"].popleft()


def detect_wave(history):
    if len(history) < MIN_WAVE_FRAMES:
        return False, 0.0, 0

    positions = [position for _, position in history]
    wave_span = max(positions) - min(positions)

    if wave_span < MIN_WAVE_SPAN:
        return False, wave_span, 0

    directions = []
    anchor = positions[0]

    for position in positions[1:]:
        movement = position - anchor
        if abs(movement) < MIN_DIRECTION_STEP:
            continue

        direction = 1 if movement > 0 else -1
        if not directions or direction != directions[-1]:
            directions.append(direction)
        anchor = position

    direction_changes = max(0, len(directions) - 1)
    return direction_changes >= MIN_DIRECTION_CHANGES, wave_span, direction_changes


def match_hands_to_tracks(hand_positions, tracks, current_time, next_track_id):
    assignments = []
    used_track_ids = set()

    for position in hand_positions:
        best_track_id = None
        best_distance = TRACK_MATCH_DISTANCE

        for track_id, track in tracks.items():
            if track_id in used_track_ids:
                continue

            current_distance = distance(position, track["position"])
            if current_distance < best_distance:
                best_distance = current_distance
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = next_track_id
            next_track_id += 1
            tracks[best_track_id] = {
                "position": position,
                "last_seen": current_time,
                "history": deque(),
            }

        track = tracks[best_track_id]
        track["position"] = position
        track["last_seen"] = current_time
        used_track_ids.add(best_track_id)
        assignments.append(best_track_id)

    expired_ids = [
        track_id
        for track_id, track in tracks.items()
        if current_time - track["last_seen"] > TRACK_TIMEOUT_SECONDS
    ]
    for track_id in expired_ids:
        del tracks[track_id]

    return assignments, next_track_id


def clear_track_histories(tracks):
    for track in tracks.values():
        track["history"].clear()


def send_angle_command(arduino, angle, source="participant"):
    safe_angle = int(np.clip(round(angle), 0, 180))
    command = f"A{safe_angle}\n".encode("ascii")

    try:
        arduino.write(command)
        arduino.flush()
        print(f"A{safe_angle} sent ({source})")
        return True
    except serial.SerialException as error:
        print(f"Serial send failed: {error}")
        return False


def send_burst_command(arduino, angle=180):
    """Request the Arduino's one-off double-hit second-wind motion."""
    safe_angle = int(np.clip(round(angle), 0, 180))
    command = f"B{safe_angle}\n".encode("ascii")

    try:
        arduino.write(command)
        arduino.flush()
        print(f"B{safe_angle} sent (second-wind burst)")
        return True
    except serial.SerialException as error:
        print(f"Serial burst send failed: {error}")
        return False


def send_reset_command(arduino, source="round reset"):
    """Immediately cancel any physical motion and return the arm to 0 degrees."""
    try:
        arduino.write(b"R\n")
        arduino.flush()
        print(f"R sent ({source})")
        return True
    except serial.SerialException as error:
        print(f"Serial reset failed: {error}")
        return False


def pose_confirms_person(pose_landmarks):
    """Use stable torso joints for presence instead of background segmentation."""
    if pose_landmarks is None:
        return False

    landmark_indices = (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    )
    visible_count = sum(
        pose_landmarks.landmark[index].visibility >= POSE_VISIBILITY_THRESHOLD
        for index in landmark_indices
    )
    return visible_count >= 2


def extract_visitor_signature(frame, pose_landmarks):
    """Return a disposable torso-colour signature for visitor handover.

    The signature is a small normalised histogram, not a photograph or face
    embedding. It is kept only in memory for the current visitor session.
    """
    if pose_landmarks is None:
        return None

    indices = (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    )
    landmarks = [pose_landmarks.landmark[index] for index in indices]
    if sum(point.visibility >= POSE_VISIBILITY_THRESHOLD for point in landmarks) < 3:
        return None

    height, width = frame.shape[:2]
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    span_x = max(24.0, max(xs) - min(xs))
    span_y = max(36.0, max(ys) - min(ys))
    left = max(0, int(min(xs) - span_x * 0.12))
    right = min(width, int(max(xs) + span_x * 0.12))
    top = max(0, int(min(ys) + span_y * 0.08))
    bottom = min(height, int(max(ys) - span_y * 0.06))
    if right - left < 24 or bottom - top < 32:
        return None

    torso = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.astype(np.float32)


def compare_visitor_signatures(reference, current):
    if reference is None or current is None:
        return 0.0
    return float(
        cv2.compareHist(
            reference.astype(np.float32),
            current.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def choose_response_angle(interaction_count, rng):
    """Return the fixed or controlled-random angle for this response."""
    if interaction_count in FIXED_RESPONSE_ANGLES:
        return FIXED_RESPONSE_ANGLES[interaction_count]
    return rng.choice(RANDOM_RESPONSE_ANGLES[interaction_count])


def get_response_delay(interaction_count):
    return RESPONSE_DELAYS[interaction_count]


def get_stamina(interaction_count):
    return PHYSICAL_ENERGY_LEVELS.get(
        int(np.clip(interaction_count, 0, TOTAL_INTERACTIONS)),
        0,
    )


def get_effort_label(energy):
    if energy >= 75:
        return "RESTED"
    if energy >= 50:
        return "STEADY"
    if energy >= 25:
        return "STRAINED"
    if energy > 0:
        return "EXHAUSTED"
    return "DEPLETED"


def load_next_visitor_number():
    """Continue the five-digit visitor archive number across app restarts."""
    try:
        with open(VISITOR_COUNTER_PATH, "r", encoding="utf-8") as counter_file:
            data = json.load(counter_file)
        return max(1, int(data.get("next_visitor_number", 1)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1


def save_next_visitor_number(next_number):
    """Atomically persist the next N.00001-style archive identifier."""
    temporary_path = VISITOR_COUNTER_PATH + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as counter_file:
            json.dump(
                {"next_visitor_number": int(next_number)},
                counter_file,
                indent=2,
            )
        os.replace(temporary_path, VISITOR_COUNTER_PATH)
    except OSError as error:
        print(f"Visitor counter could not be saved: {error}")


def schedule_random_activity(current_time, interaction_count, activity_count, rng):
    if not (
        RANDOM_ACTIVITY_START_COUNT
        <= interaction_count
        <= RANDOM_ACTIVITY_END_COUNT
    ):
        return None
    if activity_count >= MAX_RANDOM_ACTIVITIES_PER_ROUND:
        return None
    return current_time + rng.uniform(
        RANDOM_ACTIVITY_MIN_WAIT,
        RANDOM_ACTIVITY_MAX_WAIT,
    )


@lru_cache(maxsize=96)
def get_ui_font(pixel_size, bold=False):
    """Load a real monospaced TrueType font for crisp fullscreen text."""
    if ImageFont is None:
        return None

    if bold:
        candidates = (
            r"C:\Windows\Fonts\consolab.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
        )
    else:
        candidates = (
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\lucon.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, pixel_size)
    return ImageFont.load_default()


def draw_text(
    frame,
    text,
    y,
    colour=TEXT_PRIMARY,
    scale=0.58,
    centred=False,
    thickness=1,
    x_override=None,
    bold=False,
):
    # Pillow draws TrueType directly at the final 1080p resolution. This fixes
    # the blurred OpenCV text seen in BUILD 4.x. A fallback remains available
    # in case Pillow is missing on an operator machine.
    pixel_size = max(13, int(round(40 * scale)))
    font = get_ui_font(pixel_size, bold)
    if font is not None and Image is not None:
        stroke_width = 1 if pixel_size < 36 else 2
        bbox = font.getbbox(text, stroke_width=stroke_width)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        pad = stroke_width + 4
        if x_override is not None:
            x = int(x_override)
        elif centred:
            x = max(0, (frame.shape[1] - text_width) // 2)
        else:
            x = 14
        top = int(y - text_height)

        patch = Image.new(
            "RGBA",
            (text_width + pad * 2, text_height + pad * 2),
            (0, 0, 0, 0),
        )
        drawer = ImageDraw.Draw(patch)
        rgb_colour = (int(colour[2]), int(colour[1]), int(colour[0]), 255)
        drawer.text(
            (pad - bbox[0], pad - bbox[1]),
            text,
            font=font,
            fill=rgb_colour,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 210),
        )
        rgba = np.asarray(patch)
        left = max(0, x - pad)
        upper = max(0, top - pad)
        right = min(frame.shape[1], left + rgba.shape[1])
        lower = min(frame.shape[0], upper + rgba.shape[0])
        if right > left and lower > upper:
            cut = rgba[: lower - upper, : right - left]
            alpha = cut[:, :, 3:4].astype(np.float32) / 255.0
            rgb = cut[:, :, :3][:, :, ::-1].astype(np.float32)
            roi = frame[upper:lower, left:right].astype(np.float32)
            frame[upper:lower, left:right] = np.clip(
                rgb * alpha + roi * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
        return

    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(text, font_cv, scale, thickness)
    if x_override is not None:
        x = x_override
    elif centred:
        x = max(0, (frame.shape[1] - text_width) // 2)
    else:
        x = 14
    cv2.putText(frame, text, (x, y), font_cv, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font_cv, scale, colour, thickness, cv2.LINE_AA)


def draw_pixel_text(
    frame,
    text,
    y,
    colour=TEXT_PRIMARY,
    block_scale=7,
    centred=False,
    x_override=None,
    centre_x_override=None,
    max_width=None,
):
    """Draw portable bitmap type without relying on an installed pixel font."""
    if Image is None or ImageDraw is None or ImageFont is None:
        draw_text(
            frame,
            text,
            y,
            colour,
            max(0.8, block_scale / 4.5),
            centred=centred,
            x_override=x_override,
            bold=True,
        )
        return

    font = ImageFont.load_default()
    scratch = Image.new("L", (1, 1), 0)
    drawer = ImageDraw.Draw(scratch)
    bbox = drawer.textbbox((0, 0), text, font=font, stroke_width=0)
    source_width = max(1, bbox[2] - bbox[0])
    source_height = max(1, bbox[3] - bbox[1])
    scale = max(2, int(block_scale))
    if max_width is not None:
        scale = max(2, min(scale, int(max_width) // source_width))

    glyph = Image.new("L", (source_width + 2, source_height + 2), 0)
    glyph_draw = ImageDraw.Draw(glyph)
    glyph_draw.text(
        (1 - bbox[0], 1 - bbox[1]),
        text,
        font=font,
        fill=255,
        stroke_width=0,
    )
    bitmap = np.asarray(glyph)
    bitmap = np.where(bitmap >= 128, 255, 0).astype(np.uint8)
    bitmap = cv2.resize(
        bitmap,
        (bitmap.shape[1] * scale, bitmap.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    text_width = bitmap.shape[1]
    text_height = bitmap.shape[0]
    if centre_x_override is not None:
        left = int(centre_x_override - text_width / 2)
    elif x_override is not None:
        left = int(x_override)
    elif centred:
        left = (frame.shape[1] - text_width) // 2
    else:
        left = 14
    top = int(y - text_height)

    left = max(0, min(frame.shape[1] - 1, left))
    top = max(0, min(frame.shape[0] - 1, top))
    right = min(frame.shape[1], left + text_width)
    bottom = min(frame.shape[0], top + text_height)
    if right <= left or bottom <= top:
        return

    mask = bitmap[: bottom - top, : right - left]
    # A one-cell shadow keeps the white bitmap legible over the live camera.
    shadow_offset = max(2, scale // 2)
    shadow_left = min(frame.shape[1], left + shadow_offset)
    shadow_top = min(frame.shape[0], top + shadow_offset)
    shadow_right = min(frame.shape[1], shadow_left + mask.shape[1])
    shadow_bottom = min(frame.shape[0], shadow_top + mask.shape[0])
    if shadow_right > shadow_left and shadow_bottom > shadow_top:
        shadow_mask = mask[: shadow_bottom - shadow_top, : shadow_right - shadow_left]
        shadow_roi = frame[shadow_top:shadow_bottom, shadow_left:shadow_right]
        shadow_roi[shadow_mask > 0] = (
            shadow_roi[shadow_mask > 0].astype(np.float32) * 0.18
        ).astype(np.uint8)

    roi = frame[top:bottom, left:right]
    roi[mask > 0] = colour


def draw_rounded_rectangle(frame, left, top, right, bottom, colour, radius):
    """Draw a filled rounded rectangle using only OpenCV primitives."""
    radius = int(max(1, min(radius, (right - left) // 2, (bottom - top) // 2)))
    cv2.rectangle(frame, (left + radius, top), (right - radius, bottom), colour, -1)
    cv2.rectangle(frame, (left, top + radius), (right, bottom - radius), colour, -1)
    cv2.circle(frame, (left + radius, top + radius), radius, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (right - radius, top + radius), radius, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (left + radius, bottom - radius), radius, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (right - radius, bottom - radius), radius, colour, -1, cv2.LINE_AA)


def blend_rounded_panel(frame, left, top, right, bottom, opacity=0.42, radius=18):
    overlay = frame.copy()
    draw_rounded_rectangle(
        overlay,
        left,
        top,
        right,
        bottom,
        PANEL_BACKGROUND,
        radius,
    )
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)


def prepare_display_frame(frame):
    """Centre-crop the webcam to 16:9, then enlarge before drawing the UI."""
    source_height, source_width = frame.shape[:2]
    target_ratio = DISPLAY_WIDTH / DISPLAY_HEIGHT
    source_ratio = source_width / max(1, source_height)

    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = int(round(crop_height * target_ratio))
        crop_x = max(0, (source_width - crop_width) // 2)
        crop_y = 0
    else:
        crop_width = source_width
        crop_height = int(round(crop_width / target_ratio))
        crop_x = 0
        crop_y = max(0, (source_height - crop_height) // 2)

    cropped = frame[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    display = cv2.resize(
        cropped,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    transform = (
        source_width,
        source_height,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
    )
    return display, transform


def apply_camera_style(frame):
    """Keep the live camera in colour, with restrained CRT texture."""
    styled = cv2.convertScaleAbs(frame, alpha=1.035, beta=-4)

    # A small noise field is enlarged instead of allocating full-HD random
    # noise every frame. The grain remains subtle enough for the camera image
    # and large type to stay legible.
    noise_small = np.random.default_rng().normal(0.0, 3.2, (135, 240))
    noise = cv2.resize(
        noise_small.astype(np.float32),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    styled = np.clip(
        styled.astype(np.float32) + noise[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    styled[1::4, :, :] = (
        styled[1::4, :, :].astype(np.float32) * 0.94
    ).astype(np.uint8)
    return styled


def map_landmark_to_display(landmark, transform):
    source_width, source_height, crop_x, crop_y, crop_width, crop_height = transform
    source_x = landmark.x * source_width
    source_y = landmark.y * source_height
    display_x = int(round((source_x - crop_x) / max(1, crop_width) * DISPLAY_WIDTH))
    display_y = int(round((source_y - crop_y) / max(1, crop_height) * DISPLAY_HEIGHT))
    if not (0 <= display_x < DISPLAY_WIDTH and 0 <= display_y < DISPLAY_HEIGHT):
        return None
    return display_x, display_y


def draw_collective_memory_windows(frame, memory_frames, current_time):
    """Layer recent participant echoes as sparse translucent archive windows."""
    height, width = frame.shape[:2]
    layouts = (
        (0.10, 0.16, 0.26, 0.42),
        (0.17, 0.28, 0.24, 0.43),
        (0.29, 0.18, 0.27, 0.46),
        (0.43, 0.30, 0.31, 0.46),
        (0.57, 0.20, 0.25, 0.40),
    )
    stored = list(memory_frames)[-len(layouts) :]
    visible_count = max(3, len(stored))

    for index in range(visible_count):
        left_n, top_n, width_n, height_n = layouts[index]
        drift = int(3.0 * math.sin(current_time * 0.35 + index * 1.7))
        left = int(width * left_n) + drift
        top = int(height * top_n) - drift
        window_width = int(width * width_n)
        window_height = int(height * height_n)
        right = min(width - 1, left + window_width)
        bottom = min(height - 1, top + window_height)

        visitor_id = None
        if index < len(stored):
            visitor_id, memory = stored[index]
            memory = cv2.resize(
                memory,
                (right - left, bottom - top),
                interpolation=cv2.INTER_AREA,
            )
            # Memory remains recognisably coloured, but is less saturated than
            # the live camera so the present body still reads as the foreground.
            memory = cv2.convertScaleAbs(memory, alpha=1.08, beta=-10)
            memory_hsv = cv2.cvtColor(memory, cv2.COLOR_BGR2HSV)
            memory_hsv[:, :, 1] = np.clip(
                memory_hsv[:, :, 1].astype(np.float32) * 0.72,
                0,
                255,
            ).astype(np.uint8)
            memory = cv2.cvtColor(memory_hsv, cv2.COLOR_HSV2BGR)
            memory[1::4, :, :] = (
                memory[1::4, :, :].astype(np.float32) * 0.82
            ).astype(np.uint8)
            opacity = 0.085 + index * 0.020
            roi = frame[top:bottom, left:right]
            cv2.addWeighted(memory, opacity, roi, 1.0 - opacity, 0, roi)

        # The frames are deliberately thin, incomplete and slightly unstable:
        # they read as accumulated memory rather than application windows.
        line_colour = tuple(int(value * (0.44 + 0.07 * index)) for value in WINDOW_LINE)
        cv2.rectangle(frame, (left, top), (right, bottom), line_colour, 1, cv2.LINE_AA)
        cv2.line(frame, (left, top + 29), (right, top + 29), line_colour, 1, cv2.LINE_AA)
        cv2.circle(frame, (left + 13, top + 14), 2, line_colour, -1, cv2.LINE_AA)
        cv2.line(frame, (right - 25, top + 14), (right - 10, top + 14), line_colour, 1, cv2.LINE_AA)
        if visitor_id is not None:
            # One faint five-digit archive line per visitor. The identifier is
            # intentionally quieter than the live instructions and puppet text.
            draw_text(
                frame,
                visitor_id,
                top + 24,
                tuple(min(135, value + 24) for value in line_colour),
                max(0.34, width / 5650.0),
                x_override=left + 31,
            )

    if stored:
        draw_text(
            frame,
            f"COLLECTIVE MEMORY  {len(stored):02d}",
            int(height * 0.13),
            TEXT_FAINT,
            max(0.34, width / 5600.0),
            x_override=int(width * 0.10),
        )


def limb_polygon(start, end, start_width, end_width):
    """Return a tapered quadrilateral around a limb centre line."""
    x1, y1 = start
    x2, y2 = end
    length = max(1.0, math.hypot(x2 - x1, y2 - y1))
    normal_x = -(y2 - y1) / length
    normal_y = (x2 - x1) / length
    return np.array(
        [
            (x1 + normal_x * start_width, y1 + normal_y * start_width),
            (x2 + normal_x * end_width, y2 + normal_y * end_width),
            (x2 - normal_x * end_width, y2 - normal_y * end_width),
            (x1 - normal_x * start_width, y1 - normal_y * start_width),
        ],
        dtype=np.int32,
    )


def build_articulated_puppet_masks(frame_shape, centre_x, top, height):
    """Build the silhouette of Jialin's flat plywood puppet, part by part."""
    canvas_height, canvas_width = frame_shape[:2]
    part_masks = []
    joints = []

    def new_part(points=None):
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        if points is not None:
            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255, cv2.LINE_AA)
        part_masks.append(mask)
        return mask

    unit = float(height)
    cx = float(centre_x)
    y0 = float(top)

    # Rounded, slightly irregular laser-cut head.
    head = np.array(
        [
            (cx - 0.052 * unit, y0 + 0.020 * unit),
            (cx - 0.072 * unit, y0 + 0.055 * unit),
            (cx - 0.069 * unit, y0 + 0.108 * unit),
            (cx - 0.046 * unit, y0 + 0.148 * unit),
            (cx, y0 + 0.162 * unit),
            (cx + 0.052 * unit, y0 + 0.143 * unit),
            (cx + 0.071 * unit, y0 + 0.094 * unit),
            (cx + 0.064 * unit, y0 + 0.047 * unit),
            (cx + 0.032 * unit, y0 + 0.018 * unit),
        ],
        dtype=np.int32,
    )
    new_part(head)

    torso = np.array(
        [
            (cx - 0.028 * unit, y0 + 0.176 * unit),
            (cx - 0.092 * unit, y0 + 0.205 * unit),
            (cx - 0.105 * unit, y0 + 0.285 * unit),
            (cx - 0.082 * unit, y0 + 0.402 * unit),
            (cx - 0.067 * unit, y0 + 0.485 * unit),
            (cx + 0.067 * unit, y0 + 0.485 * unit),
            (cx + 0.082 * unit, y0 + 0.402 * unit),
            (cx + 0.105 * unit, y0 + 0.285 * unit),
            (cx + 0.092 * unit, y0 + 0.205 * unit),
            (cx + 0.028 * unit, y0 + 0.176 * unit),
        ],
        dtype=np.int32,
    )
    new_part(torso)

    # Long arms and enlarged hands are distinctive features of the real work.
    for side in (-1, 1):
        shoulder = (cx + side * 0.091 * unit, y0 + 0.225 * unit)
        elbow = (cx + side * 0.165 * unit, y0 + 0.370 * unit)
        wrist = (cx + side * 0.214 * unit, y0 + 0.505 * unit)
        new_part(limb_polygon(shoulder, elbow, 0.033 * unit, 0.027 * unit))
        new_part(limb_polygon(elbow, wrist, 0.030 * unit, 0.025 * unit))

        hand_cx = cx + side * 0.228 * unit
        hand_top = y0 + 0.485 * unit
        mirrored = np.array(
            [
                (side * -0.030, 0.000),
                (side * -0.045, 0.050),
                (side * -0.043, 0.122),
                (side * -0.027, 0.158),
                (side * -0.012, 0.108),
                (side * -0.002, 0.161),
                (side * 0.012, 0.153),
                (side * 0.012, 0.103),
                (side * 0.027, 0.143),
                (side * 0.040, 0.128),
                (side * 0.026, 0.058),
                (side * 0.030, 0.000),
            ],
            dtype=np.float32,
        )
        hand = np.column_stack(
            (hand_cx + mirrored[:, 0] * unit, hand_top + mirrored[:, 1] * unit)
        ).astype(np.int32)
        new_part(hand)
        joints.extend((shoulder, elbow))

    for side in (-1, 1):
        hip = (cx + side * 0.052 * unit, y0 + 0.492 * unit)
        knee = (cx + side * 0.058 * unit, y0 + 0.705 * unit)
        ankle = (cx + side * 0.058 * unit, y0 + 0.920 * unit)
        new_part(limb_polygon(hip, knee, 0.038 * unit, 0.034 * unit))
        new_part(limb_polygon(knee, ankle, 0.033 * unit, 0.026 * unit))

        foot = np.array(
            [
                (ankle[0] - 0.027 * unit, ankle[1] - 0.004 * unit),
                (ankle[0] + 0.025 * unit, ankle[1] - 0.004 * unit),
                (ankle[0] + side * 0.030 * unit, y0 + 0.972 * unit),
                (ankle[0] + side * 0.054 * unit, y0 + 0.992 * unit),
                (ankle[0] - side * 0.016 * unit, y0 + 0.995 * unit),
                (ankle[0] - side * 0.027 * unit, y0 + 0.962 * unit),
            ],
            dtype=np.int32,
        )
        new_part(foot)
        joints.extend((hip, knee))

    joints.append((cx, y0 + 0.171 * unit))
    full_mask = np.maximum.reduce(part_masks)
    joints = [(int(round(x)), int(round(y))) for x, y in joints]
    return full_mask, part_masks, joints


def draw_puppet_energy_scan(frame, centre_x, top, height, energy, current_time):
    """CRT/particle energy scan shaped exactly like the physical puppet."""
    full_mask, part_masks, joints = build_articulated_puppet_masks(
        frame.shape,
        centre_x,
        top,
        height,
    )
    y_min = max(0, top)
    y_max = min(frame.shape[0], top + height + 4)
    x_min = max(0, centre_x - int(height * 0.32))
    x_max = min(frame.shape[1], centre_x + int(height * 0.32))

    outline = np.zeros_like(full_mask)
    kernel = np.ones((3, 3), np.uint8)
    for part_mask in part_masks:
        dilated = cv2.dilate(part_mask, kernel, iterations=1)
        eroded = cv2.erode(part_mask, kernel, iterations=1)
        outline = cv2.max(outline, cv2.subtract(dilated, eroded))
    outline_overlay = frame.copy()
    outline_overlay[outline > 0] = (205, 205, 205)
    cv2.addWeighted(outline_overlay, 0.55, frame, 0.45, 0, frame)

    local_mask = full_mask[y_min:y_max, x_min:x_max] > 0
    local_height, local_width = local_mask.shape
    if local_height and local_width:
        fill_top = top + int(height * (1.0 - energy / 100.0))
        rows = np.arange(y_min, y_max)[:, None]
        remaining = local_mask & (rows >= fill_top)
        depleted = local_mask & ~remaining

        seed = int(current_time * 4.0) + int(energy) * 193 + 701
        rng = np.random.default_rng(seed)
        random_field = rng.random((local_height, local_width))
        particles = (remaining & (random_field < 0.67)) | (
            depleted & (random_field < 0.105)
        )
        particle_layer = frame[y_min:y_max, x_min:x_max]
        particle_layer[particles] = (238, 238, 238)

        # Irregular denser islands echo the torn silicone skin patches while
        # the exposed jointed outline still reads as plywood.
        patch = np.zeros((local_height, local_width), dtype=np.uint8)
        patch_shapes = (
            np.array(((0.45, 0.22), (0.58, 0.26), (0.55, 0.43), (0.43, 0.39))),
            np.array(((0.30, 0.50), (0.39, 0.48), (0.40, 0.68), (0.31, 0.65))),
            np.array(((0.55, 0.69), (0.64, 0.70), (0.61, 0.87), (0.54, 0.84))),
        )
        for shape in patch_shapes:
            points = np.column_stack(
                (shape[:, 0] * local_width, shape[:, 1] * local_height)
            ).astype(np.int32)
            cv2.fillPoly(patch, [points], 255, cv2.LINE_AA)
        skin = (patch > 0) & local_mask & (random_field < 0.31)
        particle_layer[skin] = (255, 255, 255)

    # Horizontal scan slices are clipped to each wooden body part.
    for scan_y in range(top + 2, top + height, 6):
        if not (0 <= scan_y < full_mask.shape[0]):
            continue
        row_x = np.where(full_mask[scan_y] > 0)[0]
        if len(row_x):
            cv2.line(
                frame,
                (int(row_x.min()), scan_y),
                (int(row_x.max()), scan_y),
                (132, 132, 132),
                1,
                cv2.LINE_AA,
            )

    scan_y = top + int((current_time * 42.0) % max(1, height))
    cv2.line(
        frame,
        (centre_x - int(height * 0.29), scan_y),
        (centre_x + int(height * 0.29), scan_y),
        (224, 224, 224),
        1,
        cv2.LINE_AA,
    )

    joint_radius = max(3, int(height * 0.010))
    for joint in joints:
        cv2.circle(frame, joint, joint_radius + 2, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(frame, joint, joint_radius, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(frame, joint, 1, (245, 245, 245), -1, cv2.LINE_AA)
    return joints


def draw_stamina_panel(
    frame,
    interaction_count,
    visitor_count,
    puppet_state,
    current_time,
):
    """Large archive frame with physical energy and articulated puppet scan."""
    height, width = frame.shape[:2]
    stamina = get_stamina(interaction_count)
    figure_height = int(height * 0.52)
    figure_top = int(height * 0.060)
    figure_centre_x = int(width * 0.888)
    info_left = int(width * 0.615)

    # A translucent black field isolates the pale scan from a colourful live
    # background. The square frame belongs to the archive-window language and
    # deliberately avoids a rounded dashboard/card appearance.
    panel_left = int(width * 0.588)
    panel_top = int(height * 0.022)
    panel_right = width - int(width * 0.016)
    panel_bottom = int(height * 0.620)
    panel_overlay = frame.copy()
    cv2.rectangle(
        panel_overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(panel_overlay, 0.47, frame, 0.53, 0, frame)
    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (148, 148, 148),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (panel_left, panel_top + 28),
        (panel_right, panel_top + 28),
        (92, 92, 92),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (panel_left + 14, panel_top + 14),
        2,
        (170, 170, 170),
        -1,
        cv2.LINE_AA,
    )

    draw_text(
        frame,
        f"BODY ENERGY  {stamina:02d}%",
        int(height * 0.105),
        TEXT_PRIMARY,
        max(0.80, width / 2400.0),
        x_override=info_left,
        bold=True,
    )
    cv2.line(
        frame,
        (info_left, int(height * 0.132)),
        (figure_centre_x - int(width * 0.100), int(height * 0.132)),
        TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )
    draw_text(
        frame,
        f"{interaction_count:02d} / {TOTAL_INTERACTIONS:02d}",
        int(height * 0.205),
        TEXT_SECONDARY,
        max(0.76, width / 2500.0),
        x_override=info_left,
    )

    tick_y = int(height * 0.235)
    tick_gap = max(20, int(width * 0.016))
    for index in range(TOTAL_INTERACTIONS):
        tick_x = info_left + index * tick_gap
        completed = index < interaction_count
        colour = TEXT_PRIMARY if completed else TEXT_FAINT
        tick_height = 38 if completed else 25
        cv2.line(
            frame,
            (tick_x, tick_y),
            (tick_x, tick_y + tick_height),
            colour,
            4 if completed else 2,
            cv2.LINE_AA,
        )

    draw_text(
        frame,
        get_effort_label(stamina),
        int(height * 0.335),
        TEXT_FAINT,
        max(0.50, width / 3900.0),
        x_override=info_left,
    )
    draw_text(
        frame,
        f"VISITORS  {visitor_count:02d} / {VISITORS_PER_WATER_BREAK:02d}",
        int(height * 0.410),
        TEXT_SECONDARY,
        max(0.58, width / 3300.0),
        x_override=info_left,
    )
    draw_puppet_energy_scan(
        frame,
        figure_centre_x,
        figure_top,
        figure_height,
        stamina,
        current_time,
    )


def draw_gesture_status(frame, hand_count, wave_recognized):
    """One small archive marker; no rounded app-style status pill."""
    if hand_count <= 0 and not wave_recognized:
        return

    status = "WAVE RECOGNIZED" if wave_recognized else "HAND DETECTED"
    if not wave_recognized and hand_count > 1:
        status = f"{hand_count} HANDS DETECTED"

    left = max(22, int(frame.shape[1] * 0.025))
    top = max(22, int(frame.shape[0] * 0.025))
    cv2.circle(
        frame,
        (left + 5, top + 8),
        5 if wave_recognized else 3,
        HAND_FOREGROUND_COLOUR,
        -1,
        cv2.LINE_AA,
    )
    draw_text(
        frame,
        status,
        top + 17,
        TEXT_PRIMARY if wave_recognized else TEXT_SECONDARY,
        max(0.38, frame.shape[1] / 4800.0),
        x_override=left + 19,
    )


def draw_hand_landmarks_monochrome(frame, hand_landmarks, transform):
    """Draw only selected live hand joints as pure white points."""
    points = []
    for index in DISPLAY_HAND_LANDMARK_INDICES:
        landmark = hand_landmarks.landmark[index]
        mapped = map_landmark_to_display(landmark, transform)
        if mapped is None:
            continue
        x, y = mapped
        points.append((x, y))
        cv2.circle(
            frame,
            (x, y),
            max(HAND_POINT_RADIUS, int(frame.shape[0] * 0.006)),
            HAND_FOREGROUND_COLOUR,
            -1,
            cv2.LINE_AA,
        )
    return points


def draw_build_label(frame, current_time, startup_time):
    """Operator-only build label; the exhibition view stays clean."""
    height, width = frame.shape[:2]
    label = f"BUILD {BUILD_VERSION}  |  {BUILD_SIGNATURE}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.34, width / 2200.0)
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    left = max(8, (width - text_width) // 2 - 12)
    top = max(8, int(height * 0.012))
    right = min(width - 8, left + text_width + 24)
    bottom = top + text_height + baseline + 14
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BACKGROUND, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), TEXT_PRIMARY, 1)
    cv2.putText(
        frame,
        label,
        (left + 12, top + text_height + 6),
        font,
        scale,
        TEXT_PRIMARY,
        thickness,
        cv2.LINE_AA,
    )


def draw_water_bottle(frame, centre_x, top, bottle_height, fill_fraction):
    """Draw a thin monochrome bottle whose water slowly disappears."""
    bottle_width = max(72, int(bottle_height * 0.34))
    neck_width = int(bottle_width * 0.38)
    cap_height = max(8, int(bottle_height * 0.06))
    shoulder_y = top + int(bottle_height * 0.20)
    bottom = top + bottle_height
    body_left = centre_x - bottle_width // 2
    body_right = centre_x + bottle_width // 2
    neck_left = centre_x - neck_width // 2
    neck_right = centre_x + neck_width // 2

    bottle_points = np.array(
        [
            (neck_left, top + cap_height),
            (neck_left, shoulder_y - 8),
            (body_left, shoulder_y + 14),
            (body_left, bottom - 12),
            (body_left + 12, bottom),
            (body_right - 12, bottom),
            (body_right, bottom - 12),
            (body_right, shoulder_y + 14),
            (neck_right, shoulder_y - 8),
            (neck_right, top + cap_height),
        ],
        dtype=np.int32,
    )

    cv2.rectangle(
        frame,
        (neck_left - 4, top),
        (neck_right + 4, top + cap_height),
        TEXT_SECONDARY,
        1,
    )

    bottle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(bottle_mask, [bottle_points], 255, cv2.LINE_AA)
    inner_mask = cv2.erode(bottle_mask, np.ones((7, 7), np.uint8), iterations=1)
    liquid_top = bottom - int((bottom - shoulder_y) * np.clip(fill_fraction, 0.0, 1.0))
    liquid_mask = inner_mask.copy()
    liquid_mask[:liquid_top, :] = 0
    liquid_overlay = frame.copy()
    liquid_overlay[liquid_mask > 0] = (232, 232, 232)
    cv2.addWeighted(liquid_overlay, 0.66, frame, 0.34, 0, frame)

    cv2.polylines(
        frame,
        [bottle_points],
        True,
        TEXT_SECONDARY,
        2,
        cv2.LINE_AA,
    )

    if fill_fraction > 0.02:
        cv2.line(
            frame,
            (body_left + 7, liquid_top),
            (body_right - 7, liquid_top),
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )


def draw_water_break(frame, remaining_seconds):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), PANEL_BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.64, frame, 0.36, 0, frame)

    remaining_seconds = max(0.0, remaining_seconds)
    progress = 1.0 - remaining_seconds / REST_DURATION_SECONDS
    bottle_remaining = 1.0 - progress
    countdown = int(math.ceil(remaining_seconds))

    draw_text(
        frame,
        "WATER BREAK",
        int(height * 0.17),
        TEXT_PRIMARY,
        max(0.68, width / 1450.0),
        centred=True,
        thickness=1,
    )
    draw_text(
        frame,
        "the puppet is recovering",
        int(height * 0.22),
        TEXT_SECONDARY,
        max(0.34, width / 3200.0),
        centred=True,
    )
    draw_text(
        frame,
        f"new round in {countdown:02d}s",
        int(height * 0.86),
        TEXT_SECONDARY,
        max(0.38, width / 2800.0),
        centred=True,
    )

    bottle_height = max(170, int(height * 0.42))
    draw_water_bottle(
        frame,
        width // 2,
        int(height * 0.29),
        bottle_height,
        bottle_remaining,
    )
    draw_text(
        frame,
        f"{int(round(bottle_remaining * 100)):02d}% water",
        int(height * 0.76),
        TEXT_PRIMARY,
        max(0.42, width / 2500.0),
        centred=True,
    )


def draw_audience_interface(
    frame,
    speech,
    instruction,
    interaction_count,
    visitor_count,
    puppet_state,
    show_shortcuts,
    current_time,
):
    height, width = frame.shape[:2]
    left = max(46, int(width * 0.034))

    # The audience instruction is a primary exhibition element. Before the
    # first wave it explains how to wake the work; after colour returns it
    # keeps the next action unambiguous without a small app-style tooltip.
    prompt_centre_x = int(width * 0.372)
    if interaction_count <= 0:
        prompt = "WAVE YOUR HANDS"
        prompt_detail = "MOVE YOUR HANDS SIDE TO SIDE"
    elif interaction_count >= TOTAL_INTERACTIONS:
        prompt = "THANK YOU"
        prompt_detail = "STEP AWAY FOR THE NEXT PERSON"
    else:
        prompt = "KEEP WAVING"
        prompt_detail = "THE PUPPET IS STILL RESPONDING"

    pulse = 0.90 + 0.10 * math.sin(current_time * 2.2)
    prompt_value = int(235 + 20 * pulse)
    prompt_colour = (prompt_value, prompt_value, prompt_value)
    line_left = int(width * 0.172)
    line_right = int(width * 0.573)
    cv2.line(
        frame,
        (line_left, int(height * 0.305)),
        (line_right, int(height * 0.305)),
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    draw_pixel_text(
        frame,
        prompt,
        int(height * 0.405),
        prompt_colour,
        block_scale=max(8, int(width / 215.0)),
        centre_x_override=prompt_centre_x,
        max_width=int(width * 0.58),
    )
    draw_text(
        frame,
        prompt_detail,
        int(height * 0.462),
        TEXT_PRIMARY,
        max(0.55, width / 3500.0),
        centred=False,
        x_override=prompt_centre_x - int(len(prompt_detail) * width * 0.00345),
    )
    cv2.line(
        frame,
        (int(width * 0.224), int(height * 0.490)),
        (int(width * 0.521), int(height * 0.490)),
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # A dark lower field gives the puppet's inner voice enough visual weight
    # to be read from across an exhibition space.
    lower_overlay = frame.copy()
    cv2.rectangle(
        lower_overlay,
        (0, int(height * 0.775)),
        (width, height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(lower_overlay, 0.74, frame, 0.26, 0, frame)

    draw_text(
        frame,
        "PUPPET THOUGHT",
        int(height * 0.850),
        TEXT_SECONDARY,
        max(0.50, width / 3850.0),
        x_override=left,
    )
    draw_pixel_text(
        frame,
        speech,
        int(height * 0.952),
        TEXT_PRIMARY,
        block_scale=max(6, int(width / 285.0)),
        x_override=left,
        max_width=int(width * 0.92),
    )

    if show_shortcuts:
        draw_text(
            frame,
            "N NEXT   T TEST   B BURST   C RESTART   F FULLSCREEN   Q QUIT",
            height - 7,
            (125, 125, 125),
            0.30,
            centred=True,
        )

    draw_stamina_panel(
        frame,
        interaction_count,
        visitor_count,
        puppet_state,
        current_time,
    )


def draw_absence_reset(frame, absence_remaining):
    """A thin left-edge four-second reset indicator, matching the concept art."""
    height, width = frame.shape[:2]
    x = max(34, int(width * 0.024))
    top = int(height * 0.075)
    bottom = int(height * 0.77)
    cv2.rectangle(frame, (x, top), (x + 8, bottom), (88, 88, 88), 1, cv2.LINE_AA)

    if absence_remaining is None:
        return

    remaining = float(np.clip(absence_remaining, 0.0, NO_PERSON_RESTART_SECONDS))
    progress = 1.0 - remaining / NO_PERSON_RESTART_SECONDS
    fill_top = int(top + (bottom - top) * progress)
    cv2.rectangle(frame, (x + 2, fill_top), (x + 6, bottom - 2), TEXT_PRIMARY, -1)
    draw_text(
        frame,
        f"RESET IN {remaining:0.1f}s",
        int(top + (bottom - top) * 0.42),
        TEXT_SECONDARY,
        max(0.42, width / 4600.0),
        x_override=x + 28,
    )


def draw_starburst(frame, centre, strength, base_radius=34):
    if strength <= 0.01:
        return
    x, y = centre
    outer = int(base_radius * (0.7 + strength * 1.35))
    inner = max(3, int(base_radius * 0.12))
    colour_value = int(150 + 105 * strength)
    colour = (colour_value, colour_value, colour_value)
    cv2.line(frame, (x - outer, y), (x + outer, y), colour, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - outer), (x, y + outer), colour, 1, cv2.LINE_AA)
    diagonal = int(outer * 0.56)
    cv2.line(
        frame,
        (x - diagonal, y - diagonal),
        (x + diagonal, y + diagonal),
        colour,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (x - diagonal, y + diagonal),
        (x + diagonal, y - diagonal),
        colour,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (x, y), inner, (255, 255, 255), -1, cv2.LINE_AA)


def draw_second_wind_effect(frame, current_time, display_until, hand_points=None):
    """Two star pulses connect the screen to the physical double burst."""
    if current_time >= display_until:
        return

    height, width = frame.shape[:2]
    age = SECOND_WIND_DISPLAY_SECONDS - (display_until - current_time)
    first_flash = math.exp(-((age - 0.06) / 0.13) ** 2)
    second_flash = math.exp(-((age - 0.95) / 0.15) ** 2)
    flash = max(first_flash, second_flash)
    if flash > 0.01:
        white = np.full_like(frame, 255)
        alpha = 0.08 * flash
        cv2.addWeighted(white, alpha, frame, 1.0 - alpha, 0, frame)

        for point in hand_points or ():
            draw_starburst(frame, point, flash, base_radius=30)
        draw_starburst(
            frame,
            (width - int(width * 0.075), int(height * 0.19)),
            flash,
            base_radius=48,
        )

    if age > 2.15:
        return

    label = "SECOND WIND"
    baseline_y = int(height * 0.43)
    line_width = int(width * 0.30)
    cv2.line(
        frame,
        ((width - line_width) // 2, baseline_y + 28),
        ((width + line_width) // 2, baseline_y + 28),
        TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )
    draw_text(
        frame,
        label,
        baseline_y,
        TEXT_PRIMARY,
        max(1.15, width / 1700.0),
        centred=True,
    )


def main():
    arduino = None
    camera = None
    tracks = {}
    # These image echoes intentionally survive a four-second round reset. The
    # next participant therefore enters an interface already inhabited by
    # traces of previous bodies: the collective memory is not theirs alone.
    memory_frames = deque(maxlen=MEMORY_FRAME_LIMIT)
    next_track_id = 1
    rng = random.Random()

    interaction_count = 0
    visitor_count = 0
    current_session_registered = False
    current_visitor_id = None
    next_visitor_number = load_next_visitor_number()
    system_mode = "ACTIVE"
    puppet_state = "READY"
    current_line = "I'M READY. SHOW ME YOUR MOVEMENT."
    current_response_angle = None

    last_wave_time = -COMMAND_COOLDOWN_SECONDS
    last_person_seen = -math.inf
    no_person_since = None
    pending_response_time = None
    pending_response_angle = None
    pending_response_count = None
    response_display_until = 0.0
    special_message = None
    special_message_until = 0.0
    rest_started_at = None
    rest_ends_at = None
    random_activity_count = 0
    next_random_activity_time = None
    manual_wave_requested = False
    wave_recognized_until = 0.0
    second_wind_display_until = 0.0
    thank_you_started_at = None
    visitor_signature = None
    visitor_signature_started_at = None
    person_mismatch_since = None
    last_signature_check_time = -math.inf
    show_shortcuts = SHOW_SHORTCUTS_AT_START
    is_fullscreen = START_FULLSCREEN
    startup_time = time.monotonic()
    next_frame_deadline = startup_time

    def restart_round(current_time, reason):
        nonlocal interaction_count
        nonlocal current_session_registered
        nonlocal current_visitor_id
        nonlocal system_mode
        nonlocal puppet_state
        nonlocal current_line
        nonlocal current_response_angle
        nonlocal last_wave_time
        nonlocal no_person_since
        nonlocal pending_response_time
        nonlocal pending_response_angle
        nonlocal pending_response_count
        nonlocal response_display_until
        nonlocal special_message
        nonlocal special_message_until
        nonlocal rest_started_at
        nonlocal rest_ends_at
        nonlocal random_activity_count
        nonlocal next_random_activity_time
        nonlocal next_track_id
        nonlocal wave_recognized_until
        nonlocal second_wind_display_until
        nonlocal thank_you_started_at
        nonlocal visitor_signature
        nonlocal visitor_signature_started_at
        nonlocal person_mismatch_since
        nonlocal last_signature_check_time

        if arduino is not None:
            send_reset_command(arduino, reason)

        interaction_count = 0
        current_session_registered = False
        current_visitor_id = None
        system_mode = "ACTIVE"
        puppet_state = "READY"
        current_line = "I'M READY. SHOW ME YOUR MOVEMENT."
        current_response_angle = None
        last_wave_time = current_time
        no_person_since = None
        pending_response_time = None
        pending_response_angle = None
        pending_response_count = None
        response_display_until = 0.0
        special_message = "NEW ROUND. I'M READY."
        special_message_until = current_time + 2.2
        rest_started_at = None
        rest_ends_at = None
        random_activity_count = 0
        next_random_activity_time = None
        wave_recognized_until = 0.0
        second_wind_display_until = 0.0
        thank_you_started_at = None
        visitor_signature = None
        visitor_signature_started_at = None
        person_mismatch_since = None
        last_signature_check_time = -math.inf
        tracks.clear()
        next_track_id = 1
        print(f"Round restarted ({reason})")

    def start_water_break(current_time, reason):
        nonlocal system_mode
        nonlocal puppet_state
        nonlocal current_line
        nonlocal pending_response_time
        nonlocal pending_response_angle
        nonlocal pending_response_count
        nonlocal rest_started_at
        nonlocal rest_ends_at
        nonlocal next_random_activity_time

        if arduino is not None:
            send_reset_command(arduino, reason)
        system_mode = "RESTING"
        puppet_state = "WATER BREAK"
        current_line = "TEN VISITORS. I NEED WATER."
        pending_response_time = None
        pending_response_angle = None
        pending_response_count = None
        rest_started_at = current_time
        rest_ends_at = current_time + REST_DURATION_SECONDS
        next_random_activity_time = None
        print("Ten visitor sessions completed: 30-second water break started")

    try:
        arduino = connect_arduino()
        camera = open_camera(CAMERA_INDEX)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if is_fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        print("\nSystem running")
        print("T = manual 180-degree servo test")
        print("B = manual 180 -> 30 -> 180 second-wind test")
        print("N = manually advance one audience response")
        print("C = restart the current round")
        print("F = fullscreen")
        print("H = show/hide shortcuts")
        print("Q = quit\n")

        print(f"RUNNING BUILD {BUILD_VERSION}: {BUILD_SIGNATURE}")
        print("If this exact line is not visible, a different Python file is running.")

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=0,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        ) as hands, mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        ) as pose:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Failed to read camera frame")
                    continue

                current_time = time.monotonic()
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                hand_results = hands.process(rgb_frame)
                pose_results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                detected_hands = hand_results.multi_hand_landmarks or []

                # A torso or a hand confirms presence. Unlike background
                # segmentation, an empty room cannot hold the round open.
                person_detected_now = (
                    pose_confirms_person(pose_results.pose_landmarks)
                    or bool(detected_hands)
                )
                if person_detected_now:
                    last_person_seen = current_time
                person_present = (
                    current_time - last_person_seen
                    <= PERSON_DETECTION_GRACE_SECONDS
                )

                # Once a visitor has started, compare only a disposable torso
                # colour histogram. A sustained large change opens a fresh
                # round immediately, even when the handover happens without a
                # four-second empty gap.
                current_signature = None
                if (
                    system_mode == "ACTIVE"
                    and current_session_registered
                    and interaction_count < TOTAL_INTERACTIONS
                    and current_time - last_signature_check_time
                    >= VISITOR_SIGNATURE_CHECK_INTERVAL
                ):
                    last_signature_check_time = current_time
                    current_signature = extract_visitor_signature(
                        frame,
                        pose_results.pose_landmarks,
                    )
                    if visitor_signature is None and current_signature is not None:
                        visitor_signature = current_signature
                        visitor_signature_started_at = current_time
                        person_mismatch_since = None
                    elif (
                        current_signature is not None
                        and visitor_signature_started_at is not None
                        and current_time - visitor_signature_started_at
                        >= VISITOR_SIGNATURE_WARMUP_SECONDS
                    ):
                        signature_distance = compare_visitor_signatures(
                            visitor_signature,
                            current_signature,
                        )
                        if signature_distance >= VISITOR_CHANGE_DISTANCE:
                            if person_mismatch_since is None:
                                person_mismatch_since = current_time
                            elif (
                                current_time - person_mismatch_since
                                >= VISITOR_CHANGE_CONFIRM_SECONDS
                            ):
                                print(
                                    "Different visitor appearance detected "
                                    f"(distance {signature_distance:.2f})"
                                )
                                restart_round(
                                    current_time,
                                    "different visitor detected",
                                )
                        else:
                            person_mismatch_since = None
                            # Follow slow illumination and pose changes while
                            # keeping the original visitor as the reference.
                            visitor_signature = cv2.normalize(
                                visitor_signature * 0.96
                                + current_signature * 0.04,
                                None,
                                alpha=1.0,
                                norm_type=cv2.NORM_L1,
                            )

                hand_positions = [get_hand_position(hand) for hand in detected_hands]
                assignments, next_track_id = match_hands_to_tracks(
                    hand_positions,
                    tracks,
                    current_time,
                    next_track_id,
                )

                wave_found = False
                for track_id in assignments:
                    track = tracks[track_id]
                    update_track_history(track, current_time, track["position"])
                    waved, _, _ = detect_wave(track["history"])
                    wave_found = wave_found or waved

                # Keep the physical state visible by reading the Arduino's
                # non-blocking state-machine messages.
                while arduino.in_waiting > 0:
                    reply = (
                        arduino.readline()
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    if not reply:
                        continue
                    print(f"Arduino: {reply}")
                    if reply.startswith(("WAVE_START", "BURST_START")):
                        puppet_state = "MOVING"
                    elif reply == "WAVE_DONE":
                        puppet_state = (
                            "WATER BREAK"
                            if system_mode == "RESTING"
                            else "READY"
                        )
                    elif reply == "BUSY":
                        puppet_state = "MOVING"
                    elif reply.startswith("READY:") and system_mode == "ACTIVE":
                        puppet_state = "READY"
                    elif reply == "RESET_DONE":
                        puppet_state = "READY"

                if system_mode == "RESTING":
                    if current_time >= rest_ends_at:
                        visitor_count = 0
                        restart_round(current_time, "water break complete")
                else:
                    # A completed visitor always receives a full five-second
                    # THANK YOU screen. Afterwards the work returns to the
                    # monochrome start page, except that visitor 10 enters the
                    # scheduled water break first.
                    if (
                        interaction_count >= TOTAL_INTERACTIONS
                        and thank_you_started_at is not None
                        and current_time - thank_you_started_at
                        >= THANK_YOU_DURATION_SECONDS
                    ):
                        if visitor_count >= VISITORS_PER_WATER_BREAK:
                            start_water_break(
                                current_time,
                                "ten visitors completed",
                            )
                        else:
                            restart_round(
                                current_time,
                                "thank-you screen complete",
                            )

                    # Restart an abandoned partial round after a continuous
                    # absence. The 30-second water break always gets its full
                    # duration and is handled above.
                    if (
                        0 < interaction_count < TOTAL_INTERACTIONS
                        and not person_present
                    ):
                        if no_person_since is None:
                            no_person_since = current_time
                        elif (
                            current_time - no_person_since
                            >= NO_PERSON_RESTART_SECONDS
                        ):
                            restart_round(current_time, "visitor left")
                    else:
                        no_person_since = None

                can_accept_wave = (
                    system_mode == "ACTIVE"
                    and interaction_count < TOTAL_INTERACTIONS
                    and pending_response_time is None
                    and puppet_state == "READY"
                    and current_time - last_wave_time
                    >= COMMAND_COOLDOWN_SECONDS
                )

                if (wave_found or manual_wave_requested) and can_accept_wave:
                    manual_wave_requested = False
                    wave_recognized_until = (
                        current_time + WAVE_RECOGNIZED_DISPLAY_SECONDS
                    )
                    if interaction_count == 0 and not current_session_registered:
                        visitor_count = min(
                            VISITORS_PER_WATER_BREAK,
                            visitor_count + 1,
                        )
                        current_session_registered = True
                        current_visitor_id = f"N.{next_visitor_number:05d}"
                        next_visitor_number += 1
                        save_next_visitor_number(next_visitor_number)
                        visitor_signature = extract_visitor_signature(
                            frame,
                            pose_results.pose_landmarks,
                        )
                        visitor_signature_started_at = current_time
                        person_mismatch_since = None
                        last_signature_check_time = current_time

                        # A visitor contributes one representative archive
                        # image, not one image per wave. The data window and
                        # five-digit identifier therefore stay one-to-one.
                        memory_capture, _ = prepare_display_frame(frame)
                        memory_capture = apply_camera_style(memory_capture)
                        memory_frames.append(
                            (
                                current_visitor_id,
                                cv2.resize(
                                    memory_capture,
                                    (960, 540),
                                    interpolation=cv2.INTER_AREA,
                                ),
                            )
                        )
                        print(
                            f"Registered visitor {current_visitor_id} "
                            f"({visitor_count}/{VISITORS_PER_WATER_BREAK})"
                        )
                    interaction_count += 1
                    current_line = PUPPET_LINES[interaction_count]
                    response_angle = choose_response_angle(
                        interaction_count,
                        rng,
                    )
                    current_response_angle = response_angle
                    response_delay = get_response_delay(interaction_count)
                    last_wave_time = current_time

                    if interaction_count == TOTAL_INTERACTIONS:
                        thank_you_started_at = current_time
                        if send_angle_command(arduino, 0, "strike"):
                            puppet_state = "ON STRIKE"
                        response_display_until = current_time + 2.4
                        next_random_activity_time = None
                        print(
                            "Visitor completed ten responses; water break "
                            "waits for ten separate visitor sessions"
                        )
                    elif interaction_count == 7:
                        if send_burst_command(arduino, 180):
                            puppet_state = "SECOND WIND"
                            response_display_until = (
                                current_time + SECOND_WIND_DISPLAY_SECONDS
                            )
                            second_wind_display_until = (
                                current_time + SECOND_WIND_DISPLAY_SECONDS
                            )
                    elif response_delay <= 0.0:
                        if send_angle_command(
                            arduino,
                            response_angle,
                            f"response {interaction_count}",
                        ):
                            puppet_state = "MOVING"
                            response_display_until = current_time + 1.15
                    else:
                        pending_response_time = current_time + response_delay
                        pending_response_angle = response_angle
                        pending_response_count = interaction_count
                        puppet_state = "WAITING"
                        print(
                            f"Response {interaction_count} scheduled in "
                            f"{response_delay:.2f}s at {response_angle} degrees"
                        )

                    next_random_activity_time = schedule_random_activity(
                        current_time,
                        interaction_count,
                        random_activity_count,
                        rng,
                    )
                    clear_track_histories(tracks)

                if (
                    system_mode == "ACTIVE"
                    and pending_response_time is not None
                    and current_time >= pending_response_time
                    and puppet_state in ("READY", "WAITING")
                ):
                    if send_angle_command(
                        arduino,
                        pending_response_angle,
                        f"delayed response {pending_response_count}",
                    ):
                        puppet_state = "MOVING"
                        last_wave_time = current_time
                        response_display_until = current_time + 1.15
                        clear_track_histories(tracks)
                    pending_response_time = None
                    pending_response_angle = None
                    pending_response_count = None

                # Autonomous movement is intentionally sparse and only occurs
                # when a visitor is present and the servo is free.
                if (
                    system_mode == "ACTIVE"
                    and next_random_activity_time is not None
                    and current_time >= next_random_activity_time
                    and person_present
                    and pending_response_time is None
                    and puppet_state == "READY"
                    and current_time - last_wave_time
                    >= COMMAND_COOLDOWN_SECONDS
                ):
                    autonomous_angle = rng.choice(RANDOM_ACTIVITY_ANGLES)
                    if send_angle_command(
                        arduino,
                        autonomous_angle,
                        "autonomous activity",
                    ):
                        puppet_state = "MOVING ON ITS OWN"
                        special_message = "THAT MOVEMENT WAS MINE."
                        special_message_until = current_time + 2.4
                        last_wave_time = current_time
                        random_activity_count += 1

                    next_random_activity_time = schedule_random_activity(
                        current_time,
                        interaction_count,
                        random_activity_count,
                        rng,
                    )

                display, display_transform = prepare_display_frame(frame)
                display = apply_camera_style(display)
                draw_collective_memory_windows(
                    display,
                    memory_frames,
                    current_time,
                )

                # Dormant installation state: the entire camera and archive
                # field remain monochrome. The first accepted wave raises the
                # interaction count and immediately restores the camera colour.
                if system_mode == "ACTIVE" and interaction_count == 0:
                    monochrome = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                    display = cv2.cvtColor(monochrome, cv2.COLOR_GRAY2BGR)

                hand_point_positions = []
                if SHOW_HAND_LANDMARKS:
                    for hand in detected_hands:
                        hand_point_positions.extend(
                            draw_hand_landmarks_monochrome(
                                display,
                                hand,
                                display_transform,
                            )
                        )

                if system_mode == "RESTING":
                    draw_water_break(display, rest_ends_at - current_time)
                else:
                    if current_time < special_message_until and special_message:
                        speech = special_message
                        instruction = "the puppet moved without your wave"
                    elif pending_response_time is not None:
                        remaining = max(0.0, pending_response_time - current_time)
                        speech = current_line
                        instruction = f"responding in {remaining:.1f}s"
                    elif current_time < response_display_until:
                        speech = current_line
                        instruction = (
                            f"response {interaction_count} / {TOTAL_INTERACTIONS}"
                        )
                    elif no_person_since is not None:
                        remaining = max(
                            0.0,
                            NO_PERSON_RESTART_SECONDS
                            - (current_time - no_person_since),
                        )
                        speech = "IS ANYONE THERE?"
                        instruction = f"resetting in {remaining:.1f}s"
                    elif detected_hands:
                        speech = current_line
                        instruction = "keep waving — move side to side"
                    else:
                        speech = current_line
                        instruction = "wave your hand from side to side"

                    draw_audience_interface(
                        display,
                        speech,
                        instruction,
                        interaction_count,
                        visitor_count,
                        puppet_state,
                        show_shortcuts,
                        current_time,
                    )

                    absence_remaining = None
                    if no_person_since is not None:
                        absence_remaining = max(
                            0.0,
                            NO_PERSON_RESTART_SECONDS
                            - (current_time - no_person_since),
                        )
                    draw_absence_reset(display, absence_remaining)

                    draw_gesture_status(
                        display,
                        len(detected_hands),
                        current_time < wave_recognized_until,
                    )
                    draw_second_wind_effect(
                        display,
                        current_time,
                        second_wind_display_until,
                        hand_point_positions,
                    )

                if show_shortcuts:
                    draw_build_label(display, current_time, startup_time)

                cv2.imshow(WINDOW_NAME, display)
                # The explicit limiter matters because many webcams ignore the
                # requested capture FPS. It also reduces MediaPipe and 1080p UI
                # work instead of merely dropping frames after processing.
                frame_interval = 1.0 / max(1.0, TARGET_FPS)
                next_frame_deadline = max(
                    next_frame_deadline + frame_interval,
                    time.monotonic(),
                )
                wait_milliseconds = max(
                    1,
                    int(round((next_frame_deadline - time.monotonic()) * 1000)),
                )
                key = cv2.waitKey(wait_milliseconds) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key in (ord("t"), ord("T")):
                    if system_mode == "ACTIVE" and puppet_state == "READY":
                        if send_angle_command(arduino, 180, "manual test"):
                            puppet_state = "MOVING"
                            special_message = "MANUAL TEST."
                            special_message_until = current_time + 1.2

                if key in (ord("b"), ord("B")):
                    if system_mode == "ACTIVE" and puppet_state == "READY":
                        if send_burst_command(arduino, 180):
                            puppet_state = "SECOND WIND"
                            current_line = "I'M FINE. WATCH THIS."
                            current_response_angle = 180
                            response_display_until = (
                                current_time + SECOND_WIND_DISPLAY_SECONDS
                            )
                            second_wind_display_until = (
                                current_time + SECOND_WIND_DISPLAY_SECONDS
                            )
                            special_message = "MANUAL SECOND-WIND TEST."
                            special_message_until = (
                                current_time + SECOND_WIND_DISPLAY_SECONDS
                            )

                if key in (ord("n"), ord("N")):
                    if system_mode == "ACTIVE":
                        manual_wave_requested = True
                        print("Manual next response requested")

                if key in (ord("c"), ord("C")):
                    restart_round(current_time, "manual restart")

                if key in (ord("h"), ord("H")):
                    show_shortcuts = not show_shortcuts

                if key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                    )

    except serial.SerialException as error:
        print(f"Arduino connection failed: {error}")
        print("Close Arduino Serial Monitor and check the COM port.")
    except RuntimeError as error:
        print(f"System error: {error}")
    except KeyboardInterrupt:
        print("Program stopped")
    except Exception as error:
        print(f"Unexpected error: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if arduino is not None and arduino.is_open:
            arduino.close()
        print("Program closed")


if __name__ == "__main__":
    main()

from collections import deque

import cv2
import numpy as np
import pytesseract
import time
import math


def mark_gas_section():
    x, y = 1030, 470
    w, h = 170, 110
    return x, y, w, h


def mark_rpm_section():
    x, y = 660, 260
    w, h = 340, 280
    return x, y, w, h


def mark_digital_section():
    x, y = 415, 545
    w, h = 80, 45
    return x, y, w, h


def mark_beacon_section():
    x, y = 580, 545
    w, h = 100, 55
    return x, y, w, h


def mark_lights_section():
    x, y = 200, 615
    w, h = 100, 45
    return x, y, w, h


def set_important_sections(image):
    result = image.copy()

    section_functions = [
        mark_gas_section,
        mark_rpm_section,
        mark_digital_section,
        mark_beacon_section,
        mark_lights_section
    ]

    for section_func in section_functions:
        x, y, w, h = section_func()
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return result


def detect_low_lights(image):
    x, y, w, h = mark_lights_section()
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)

    threshold = 500
    if green_pixels > threshold:
        return True
    else:
        return False


def detect_high_lights(image):
    x, y, w, h = mark_lights_section()
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(mask)

    threshold = 500
    if blue_pixels > threshold:
        return True
    else:
        return False


def detect_display_info(image):
    x, y, w, h = mark_digital_section()
    roi = image[y:y+h, x:x+w]

    basewidth = 200
    wpercent = (basewidth/float(roi.shape[1]))
    hsize = int((float(roi.shape[0])*float(wpercent)))
    roi = cv2.resize(roi, (basewidth, hsize), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'
    text = pytesseract.image_to_string(thresh, config=config)
    print("Detected text:", text)
    cv2.imshow("Thresh", thresh)


def detect_blinkers_from_beacon(image):
    x, y, w, h = mark_beacon_section()
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mid = w // 2
    left_mask = mask[:, :mid]
    right_mask = mask[:, mid:]

    left_on = cv2.countNonZero(left_mask) > 500
    right_on = cv2.countNonZero(right_mask) > 500

    if left_on and right_on:
        return "Balizas encendidas"
    elif left_on:
        return "Luz de giro izquierda encendida"
    elif right_on:
        return "Luz de giro derecha encendida"
    else:
        return ""

# TODO: mejorar, es muy sensible
rpm_history = deque(maxlen=20)
def detect_rpm_change(frame, tolerance=3):
    x, y, w, h = mark_rpm_section()
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    blurred = cv2.GaussianBlur(red_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=10)

    rpm_angle = None
    rpm_state = ""
    if lines is not None:
        longest = max(lines, key=lambda l: np.linalg.norm([l[0][2]-l[0][0], l[0][3]-l[0][1]]))
        x1, y1, x2, y2 = longest[0]
        cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        rpm_angle = round(angle / 2) * 2
        rpm_history.append(rpm_angle)
        if len(rpm_history) >= 18:
            diff = rpm_history[-1] - rpm_history[0]
            if abs(diff) <= 10:
                rpm_state = "Estable"
            elif diff > 0:
                rpm_state = "Subiendo"
            else:
                rpm_state = "Bajando"

    return rpm_state, rpm_angle


def show_board_info(frame, low_lights, high_lights, blinkers, rpm_state):
    low_lights_str = "Luces bajas: %s" % ("ON" if low_lights else "OFF")
    high_lights_str = "Luces altas: %s" % ("ON" if high_lights else "OFF")
    rpm_str = "RPM: %s" % rpm_state

    cv2.putText(
        frame, low_lights_str, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        frame, high_lights_str, (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        frame, blinkers, (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        frame, rpm_str, (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )


video = cv2.VideoCapture('./camioneta.mov')
frame_count = 0
prev_time = time.time()
last_rpm_angle = None

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = set_important_sections(frame)
    low_lights = detect_low_lights(frame)
    high_lights = detect_high_lights(frame)
    blinker_state = detect_blinkers_from_beacon(frame)
    rpm_state, last_rpm_angle = detect_rpm_change(frame, last_rpm_angle)
    show_board_info(frame, low_lights, high_lights, blinker_state, rpm_state)

    current_time = time.time()
    if current_time - prev_time >= 1:
        try:
            detect_display_info(frame)
            prev_time = current_time
        except Exception as e:
            print(f"Error processing frame: {e}")

    cv2.imshow('frame', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

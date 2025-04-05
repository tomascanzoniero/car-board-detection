import cv2
import numpy as np
import pytesseract
import time


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


def show_board_info(frame, low_lights, high_lights):
    low_lights_str = "Luces bajas: %s" % ("ON" if low_lights else "OFF")
    high_lights_str = "Luces altas: %s" % ("ON" if high_lights else "OFF")

    cv2.putText(
        frame, low_lights_str, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        frame, high_lights_str, (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )


video = cv2.VideoCapture('./camioneta.mov')
frame_count = 0
prev_time = time.time()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = set_important_sections(frame)
    low_lights = detect_low_lights(frame)
    high_lights = detect_high_lights(frame)
    show_board_info(frame, low_lights, high_lights)

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

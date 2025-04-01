import cv2


def mark_gas_section():
    x, y = 1030, 470
    w, h = 170, 110
    return x, y, w, h


def mark_rpm_section():
    x, y = 660, 260
    w, h = 340, 280
    return x, y, w, h


def mark_digital_section():
    x, y = 380, 505
    w, h = 120, 95
    return x, y, w, h


def mark_beacon_section():
    x, y = 580, 540
    w, h = 100, 50
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


video = cv2.VideoCapture('./camioneta.mov')
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    try:
        frame = set_important_sections(frame)
    except Exception as e:
        print(f"Error processing frame: {e}")
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

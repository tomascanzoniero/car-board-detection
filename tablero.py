from collections import deque

import cv2
import numpy as np
import pytesseract
import time
import math


class DashboardSection:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_coordinates(self):
        return self.x, self.y, self.w, self.h

    def get_roi(self, image):
        x, y, w, h = self.get_coordinates()
        return image[y:y+h, x:x+w]


class DashboardMonitor:
    def __init__(self):
        self.sections = {
            'gas': DashboardSection(1030, 470, 170, 110),
            'rpm': DashboardSection(660, 260, 340, 280),
            'digital': DashboardSection(415, 545, 80, 45),
            'beacon': DashboardSection(580, 545, 100, 55),
            'lights': DashboardSection(200, 615, 100, 45)
        }

        # Historial de valores para suavizado
        self.rpm_history = deque(maxlen=20)
        self.filtered_rpm_history = deque(maxlen=3)
        self.gas_history = deque(maxlen=20)
        self.filtered_gas_history = deque(maxlen=10)

        # Calibraciones
        self.rpm_calibration = [
            (-30, 0),    # 0%
            (4, 7.5),    # 7.5%
            (50, 20)     # 20%
        ]

        self.gas_calibration = [
            (-85, 75),  # (75%)
            (-45, 25),  # (25%)
            (38, 0),    # (0%)
        ]

    def mark_sections(self, image):
        result = image.copy()
        for section in self.sections.values():
            x, y, w, h = section.get_coordinates()
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return result

    def detect_lights(self, image, color_range):
        lower_bound, upper_bound = color_range
        roi = self.sections['lights'].get_roi(image)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        pixel_count = cv2.countNonZero(mask)
        return pixel_count > 500

    def detect_low_lights(self, image):
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        return self.detect_lights(image, (lower_green, upper_green))

    def detect_high_lights(self, image):
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        return self.detect_lights(image, (lower_blue, upper_blue))

    def detect_display_info(self, image):
        roi = self.sections['digital'].get_roi(image)

        basewidth = 200
        wpercent = (basewidth / float(roi.shape[1]))
        hsize = int((float(roi.shape[0]) * float(wpercent)))
        roi = cv2.resize(roi, (basewidth, hsize), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(thresh, config=config)
        return text

    def detect_blinkers_from_beacon(self, image):
        roi = self.sections['beacon'].get_roi(image)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        mid = roi.shape[1] // 2
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

    def angle_to_percentage(self, angle, calibration_points):
        if angle <= calibration_points[0][0]:
            return calibration_points[0][1]
        elif angle >= calibration_points[-1][0]:
            return calibration_points[-1][1]

        for i in range(len(calibration_points) - 1):
            point1 = calibration_points[i]
            point2 = calibration_points[i + 1]

            if point1[0] <= angle <= point2[0]:
                angle1, percent1 = point1
                angle2, percent2 = point2

                percentage = percent1 + (angle - angle1) * (percent2 - percent1) / (angle2 - angle1)
                return round(percentage, 1)

        return None

    def rpm_angle_to_percentage(self, angle):
        return self.angle_to_percentage(angle, self.rpm_calibration)

    def gas_angle_to_percentage(self, angle):
        return self.angle_to_percentage(angle, self.gas_calibration)

    def detect_dial_position(self, frame, section_name, angle_range, color_range):
        roi = self.sections[section_name].get_roi(frame)
        center_x, center_y = roi.shape[1] // 2, roi.shape[0] // 2

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower1, upper1, lower2, upper2 = color_range

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=15,
            minLineLength=30, maxLineGap=10
        )

        valid_lines = []
        angle_min, angle_max = angle_range

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                if angle_min <= angle <= angle_max:
                    dist_from_center = min(
                        math.sqrt((x1 - center_x)**2 + (y1 - center_y)**2),
                        math.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
                    )

                    if dist_from_center < min(roi.shape[1], roi.shape[0]) // 3:
                        valid_lines.append((line[0], angle, np.linalg.norm([x2-x1, y2-y1])))

            if valid_lines:
                valid_lines.sort(key=lambda x: x[2], reverse=True)
                (x1, y1, x2, y2), angle, _ = valid_lines[0]
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0) if section_name == 'rpm' else (0, 0, 255), 2)
                return angle

        return None

    def detect_rpm_change(self, frame):
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])

        rpm_angle = self.detect_dial_position(
            frame, 'rpm', (-60, 60),
            (red_lower1, red_upper1, red_lower2, red_upper2)
        )

        rpm_state = ""
        rpm_percentage = None

        if rpm_angle is not None:
            rounded_angle = round(rpm_angle / 2) * 2
            self.rpm_history.append(rounded_angle)

            recent_values = list(self.rpm_history)[-min(len(self.rpm_history), 5):]
            smoothed_angle = sum(recent_values) / len(recent_values)

            self.filtered_rpm_history.append(smoothed_angle)

            display_angle = sum(self.filtered_rpm_history) / len(self.filtered_rpm_history)
            rpm_percentage = self.rpm_angle_to_percentage(display_angle)

            if len(self.rpm_history) >= 10:
                recent_trend = self.rpm_history[-1] - self.rpm_history[-10]
                if abs(recent_trend) <= 5:
                    rpm_state = "Estable"
                elif recent_trend > 0:
                    rpm_state = "Subiendo"
                else:
                    rpm_state = "Bajando"

        return rpm_state, rpm_angle, rpm_percentage

    def detect_gas_level(self, frame):
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 50, 50])
        red_upper2 = np.array([180, 255, 255])

        gas_angle = self.detect_dial_position(
            frame, 'gas', (-135, 45),
            (red_lower1, red_upper1, red_lower2, red_upper2)
        )

        gas_percentage = None

        if gas_angle is not None:
            rounded_angle = round(gas_angle / 5) * 5
            self.gas_history.append(rounded_angle)

            recent_values = list(self.gas_history)[-min(len(self.gas_history), 5):]
            smoothed_angle = sum(recent_values) / len(recent_values)

            self.filtered_gas_history.append(smoothed_angle)

            display_angle = sum(self.filtered_gas_history) / len(self.filtered_gas_history)
            gas_percentage = self.gas_angle_to_percentage(display_angle)

        return gas_angle, gas_percentage

    def show_board_info(
        self, frame, low_lights, high_lights, blinkers,
        rpm_percentage, ocr_text, gas_percentage
    ):
        info_lines = [
            f"Luces bajas: {'ON' if low_lights else 'OFF'}",
            f"Luces altas: {'ON' if high_lights else 'OFF'}",
            blinkers,
            f"RPM: {rpm_percentage}",
            f"Combustible: {gas_percentage}%",
            f"OCR: {ocr_text}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame, line, (20, 40 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )


def main():
    monitor = DashboardMonitor()
    video = cv2.VideoCapture('./camioneta.mov')

    prev_time = time.time()
    prev_time_ocr = time.time()

    rpm_percentage = None
    ocr_text = ""
    gas_percentage = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # frame = monitor.mark_sections(frame)
        low_lights = monitor.detect_low_lights(frame)
        high_lights = monitor.detect_high_lights(frame)
        blinker_state = monitor.detect_blinkers_from_beacon(frame)

        current_time = time.time()
        if current_time - prev_time >= 0.3:
            try:
                prev_time = current_time
                _, _, new_rpm_percentage = monitor.detect_rpm_change(frame)
                if new_rpm_percentage is not None:
                    rpm_percentage = new_rpm_percentage

                _, new_gas_percentage = monitor.detect_gas_level(frame)
                if new_gas_percentage is not None:
                    gas_percentage = new_gas_percentage
            except Exception as e:
                print(f"Error processing frame: {e}")

        current_time_ocr = time.time()
        if current_time_ocr - prev_time_ocr >= 1:
            try:
                prev_time_ocr = current_time_ocr
                ocr_text = monitor.detect_display_info(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")

        monitor.show_board_info(
            frame, low_lights, high_lights, blinker_state,
            rpm_percentage, ocr_text, gas_percentage
        )

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
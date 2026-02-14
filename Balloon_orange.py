import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(0)

AREA_THRESHOLD = 3000
ALIGNMENT_THRESHOLD = 30
BALLOON_DIAMETER_M = 0.25  # real balloon diameter in metres

COLOR_RANGES = {
    "RED": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "ORANGE": [
        (np.array([10, 120, 120]), np.array([20, 255, 255]))
    ],
    "YELLOW": [
        (np.array([20, 120, 120]), np.array([35, 255, 255]))
    ],
    "GREEN": [
        (np.array([35, 100, 50]), np.array([85, 255, 255]))
    ],
    "BLUE": [
        (np.array([100, 150, 50]), np.array([140, 255, 255]))
    ]
}

# ---- Balloon shape check ----
def is_balloon(cnt, min_circularity=0.7):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity > min_circularity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_center = (w // 2, h // 2)
    cv2.circle(frame, frame_center, 5, (255, 255, 0), -1)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    best_contour = None
    best_mask = None
    detected_color = "UNKNOWN"
    max_area = 0

    # Detect balloon by color + shape
    for color_name, ranges in COLOR_RANGES.items():
        color_mask = None

        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            color_mask = temp_mask if color_mask is None else cv2.bitwise_or(color_mask, temp_mask)

        color_mask = cv2.erode(color_mask, None, iterations=2)
        color_mask = cv2.dilate(color_mask, None, iterations=2)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if (
                area > AREA_THRESHOLD
                and is_balloon(cnt)          # <-- KEY FILTER
                and area > max_area
            ):
                max_area = area
                best_contour = cnt
                best_mask = color_mask
                detected_color = color_name

    if best_contour is not None:
        balloon_area_px = cv2.contourArea(best_contour)

        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        balloon_center = (int(x), int(y))

        cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, balloon_center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, balloon_center, 5, (0, 0, 255), -1)

        dx_px = balloon_center[0] - frame_center[0]
        dy_px = frame_center[1] - balloon_center[1]

        balloon_diameter_px = 2 * radius
        meters_per_pixel = BALLOON_DIAMETER_M / balloon_diameter_px

        dx_m = dx_px * meters_per_pixel
        dy_m = dy_px * meters_per_pixel
        distance_m = np.sqrt(dx_m**2 + dy_m**2)

        balloon_area_m2 = balloon_area_px * (meters_per_pixel ** 2)

        cv2.line(frame, frame_center, balloon_center, (255, 255, 0), 2)

        instructions = []
        aligned = False

        if abs(dx_px) > ALIGNMENT_THRESHOLD:
            instructions.append(f"MOVE {'RIGHT' if dx_px > 0 else 'LEFT'} {abs(dx_m):.2f} m")

        if abs(dy_px) > ALIGNMENT_THRESHOLD:
            instructions.append(f"MOVE {'UP' if dy_px > 0 else 'DOWN'} {abs(dy_m):.2f} m")

        if not instructions:
            instructions.append("ALIGNED - READY TO POP")
            aligned = True

        instruction_text = " | ".join(instructions)

        cv2.putText(frame, "BALLOON DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Color: {detected_color}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"Offset: {distance_m:.2f} m", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Area: {int(balloon_area_px)} px", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Area: {balloon_area_m2:.4f} m^2", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, instruction_text, (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if aligned:
            cv2.putText(frame, "POP",
                        (w // 2 - 60, h // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            print("POP COMMAND TRIGGERED")

        cv2.imshow("Mask", best_mask)

    else:
        cv2.putText(frame, "Searching...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Underwater Balloon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
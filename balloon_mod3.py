import cv2
import numpy as np

cap = cv2.VideoCapture(0)

AREA_THRESHOLD = 3000
ALIGNMENT_THRESHOLD = 30
BALLOON_DIAMETER_M = 0.25

# ✅ Balanced HSV Ranges (No overlap, not too strict)
COLOR_RANGES = {
    "RED": [
        (np.array([0, 100, 70]), np.array([8, 255, 255])),
        (np.array([170, 100, 70]), np.array([180, 255, 255]))
    ],
    "ORANGE": [
        (np.array([8, 100, 80]), np.array([20, 255, 255]))
    ],
    "YELLOW": [
        (np.array([23, 100, 80]), np.array([35, 255, 255]))
    ],
    "GREEN": [
        (np.array([35, 80, 60]), np.array([85, 255, 255]))
    ],
    "BLUE": [
        (np.array([95, 100, 60]), np.array([140, 255, 255]))
    ]
}

# ---------- WHITE BALANCE ----------
def white_balance(img):
    result = img.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)

# ---------- CLAHE ----------
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ---------- Circularity ----------
def is_balloon(cnt, min_circularity=0.65):
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

    # ---- PREPROCESSING ----
    wb = white_balance(frame)
    enhanced = apply_clahe(wb)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    best_contour = None
    detected_color = "UNKNOWN"
    max_area = 0
    best_mask = None

    for color_name, ranges in COLOR_RANGES.items():

        mask = None
        for lower, upper in ranges:
            temp = cv2.inRange(hsv, lower, upper)
            mask = temp if mask is None else cv2.bitwise_or(mask, temp)

        # Noise removal
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < AREA_THRESHOLD:
                continue

            if not is_balloon(cnt):
                continue

            # Edge validation
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            edge_pixels = cv2.bitwise_and(edges, edges, mask=contour_mask)
            edge_count = cv2.countNonZero(edge_pixels)

            if edge_count < 40:
                continue

            if area > max_area:
                max_area = area
                best_contour = cnt
                detected_color = color_name
                best_mask = mask

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

        cv2.putText(frame, instruction_text, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if aligned:
            cv2.putText(frame, "POP",
                        (w // 2 - 60, h // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            print("POP COMMAND TRIGGERED")

        cv2.imshow("Mask", best_mask)

    else:
        cv2.putText(frame, "Searching...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    cv2.imshow("Underwater Balloon Detection - Final", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

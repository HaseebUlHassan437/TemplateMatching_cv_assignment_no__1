import cv2
import numpy as np


IMAGE_PATH = "C:/Users/hasee/Desktop/MSCS/3rd semester/computer vision/Assignments/Assignment1/input.png"
TEMPLATE_PATH = "C:/Users/hasee/Desktop/MSCS/3rd semester/computer vision/Assignments/Assignment1/template.png"
MODE = "single"          # "single" or "multi"
THRESHOLD = 0.8          # only used in multi mode
OVERLAP_THRESH = 0.3     # only used in multi mode (NMS)
OUTPUT_PATH = "result.jpg"
DOT_RADIUS = 6


def draw_corner_dots(image, top_left, bottom_right, radius=6, color=(0, 0, 255), thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for (cx, cy) in corners:
        cv2.circle(image, (cx, cy), radius, color, thickness)
    return image


def single_match(image_gray, template_gray, method=cv2.TM_CCOEFF_NORMED):
    h, w = template_gray.shape[:2]
    result = cv2.matchTemplate(image_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        top_left, score = min_loc, min_val
    else:
        top_left, score = max_loc, max_val

    bottom_right = (top_left[0] + w, top_left[1] + h)
    return [(top_left, bottom_right, score)]


def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    x1 = np.array([b[0][0] for b in boxes], dtype=float)
    y1 = np.array([b[0][1] for b in boxes], dtype=float)
    x2 = np.array([b[1][0] for b in boxes], dtype=float)
    y2 = np.array([b[1][1] for b in boxes], dtype=float)
    scores = np.array([b[2] for b in boxes], dtype=float)

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        remaining = np.where(iou <= overlap_thresh)[0]
        order = order[remaining + 1]

    return [boxes[i] for i in keep]


def multi_match(image_gray, template_gray, threshold=0.8, method=cv2.TM_CCOEFF_NORMED, overlap_thresh=0.3):
    h, w = template_gray.shape[:2]
    result = cv2.matchTemplate(image_gray, template_gray, method)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        locations = np.where(result <= (1 - threshold))
    else:
        locations = np.where(result >= threshold)

    boxes = []
    for (y, x) in zip(*locations):
        top_left = (int(x), int(y))
        bottom_right = (int(x) + w, int(y) + h)
        score = float(result[y, x])
        boxes.append((top_left, bottom_right, score))

    return non_max_suppression(boxes, overlap_thresh=overlap_thresh)





image = cv2.imread(IMAGE_PATH)
template = cv2.imread(TEMPLATE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
if template is None:
    raise FileNotFoundError(f"Could not read template: {TEMPLATE_PATH}")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

if MODE == "single":
    boxes = single_match(image_gray, template_gray)
else:
    boxes = multi_match(image_gray, template_gray, threshold=THRESHOLD, overlap_thresh=OVERLAP_THRESH)

print(f"Found {len(boxes)} match(es).")
output = image.copy()
for i, (top_left, bottom_right, score) in enumerate(boxes):
    print(f"  Match {i + 1}: top_left={top_left}, bottom_right={bottom_right}, score={score:.3f}")
    draw_corner_dots(output, top_left, bottom_right, radius=DOT_RADIUS)

cv2.imwrite(OUTPUT_PATH, output)
print(f"Saved annotated image to: {OUTPUT_PATH}")
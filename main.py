import mediapipe as mp
import cv2
import numpy as np
import math

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Anatomical bone colors (distinct for each finger)
COLORS = {
    'thumb': (180, 130, 200),
    'index': (130, 180, 130),
    'middle': (180, 160, 130),
    'ring': (130, 160, 200),
    'pinky': (200, 150, 150),
    'carpal': (200, 190, 180),
    'highlight': (0, 255, 255),  # Yellow highlight
}

# Bone definitions with anatomical names
FINGER_BONES = {
    'thumb': [
        ('metacarpal', 1, 2, '1st Metacarpal'),
        ('prox_phalanx', 2, 3, 'Thumb Proximal Phalanx'),
        ('dist_phalanx', 3, 4, 'Thumb Distal Phalanx'),
    ],
    'index': [
        ('metacarpal', 0, 5, '2nd Metacarpal'),
        ('prox_phalanx', 5, 6, 'Index Proximal Phalanx'),
        ('mid_phalanx', 6, 7, 'Index Middle Phalanx'),
        ('dist_phalanx', 7, 8, 'Index Distal Phalanx'),
    ],
    'middle': [
        ('metacarpal', 0, 9, '3rd Metacarpal'),
        ('prox_phalanx', 9, 10, 'Middle Proximal Phalanx'),
        ('mid_phalanx', 10, 11, 'Middle Middle Phalanx'),
        ('dist_phalanx', 11, 12, 'Middle Distal Phalanx'),
    ],
    'ring': [
        ('metacarpal', 0, 13, '4th Metacarpal'),
        ('prox_phalanx', 13, 14, 'Ring Proximal Phalanx'),
        ('mid_phalanx', 14, 15, 'Ring Middle Phalanx'),
        ('dist_phalanx', 15, 16, 'Ring Distal Phalanx'),
    ],
    'pinky': [
        ('metacarpal', 0, 17, '5th Metacarpal'),
        ('prox_phalanx', 17, 18, 'Pinky Proximal Phalanx'),
        ('mid_phalanx', 18, 19, 'Pinky Middle Phalanx'),
        ('dist_phalanx', 19, 20, 'Pinky Distal Phalanx'),
    ],
}


def get_point(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def get_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def rotate_point(point, center, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (
        center[0] + dx * cos_a - dy * sin_a,
        center[1] + dx * sin_a + dy * cos_a
    )


def point_to_line_distance(point, line_start, line_end):
    """Calculate the shortest distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    if line_len_sq == 0:
        return get_distance(point, line_start)
    
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    
    return get_distance(point, (closest_x, closest_y))


def find_touched_bone(pointer_tip, target_points, touch_threshold=35):
    """
    Find which bone the pointer finger is touching.
    Returns (finger_name, bone_type, start_idx, end_idx, bone_name) or None.
    """
    closest_bone = None
    closest_distance = touch_threshold
    
    for finger_name, bones in FINGER_BONES.items():
        for bone_type, start_idx, end_idx, bone_name in bones:
            p1 = target_points[start_idx]
            p2 = target_points[end_idx]
            
            distance = point_to_line_distance(pointer_tip, p1, p2)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_bone = (finger_name, bone_type, start_idx, end_idx, bone_name)
    
    return closest_bone


def draw_phalanx(img, p1, p2, color, is_distal=False, highlight=False):
    """Draw a phalanx bone (finger bone)."""
    if highlight:
        color = COLORS['highlight']
    
    angle = get_angle(p1, p2)
    length = get_distance(p1, p2)
    
    if length < 10:
        return
    
    base_width = length * 0.28
    shaft_width = length * 0.18
    head_width = length * 0.24
    
    if is_distal:
        shape = [
            (-base_width * 0.15, -base_width * 0.5),
            (0, -base_width * 0.35),
            (base_width * 0.15, -base_width * 0.5),
            (length * 0.15, -shaft_width * 0.5),
            (length * 0.4, -shaft_width * 0.45),
            (length * 0.7, -shaft_width * 0.5),
            (length * 0.85, -shaft_width * 0.35),
            (length, 0),
            (length * 0.85, shaft_width * 0.35),
            (length * 0.7, shaft_width * 0.5),
            (length * 0.4, shaft_width * 0.45),
            (length * 0.15, shaft_width * 0.5),
            (base_width * 0.15, base_width * 0.5),
            (0, base_width * 0.35),
            (-base_width * 0.15, base_width * 0.5),
        ]
    else:
        shape = [
            (-base_width * 0.1, -base_width * 0.55),
            (0, -base_width * 0.4),
            (base_width * 0.1, -base_width * 0.55),
            (length * 0.2, -shaft_width * 0.5),
            (length * 0.5, -shaft_width * 0.45),
            (length * 0.8, -shaft_width * 0.5),
            (length * 0.9, -head_width * 0.45),
            (length * 0.95, -head_width * 0.3),
            (length, -head_width * 0.15),
            (length + head_width * 0.1, 0),
            (length, head_width * 0.15),
            (length * 0.95, head_width * 0.3),
            (length * 0.9, head_width * 0.45),
            (length * 0.8, shaft_width * 0.5),
            (length * 0.5, shaft_width * 0.45),
            (length * 0.2, shaft_width * 0.5),
            (base_width * 0.1, base_width * 0.55),
            (0, base_width * 0.4),
            (-base_width * 0.1, base_width * 0.55),
        ]
    
    bone_points = []
    for px, py in shape:
        rotated = rotate_point((p1[0] + px, p1[1] + py), p1, angle)
        bone_points.append([int(rotated[0]), int(rotated[1])])
    
    bone_points = np.array(bone_points, np.int32)
    
    darker = tuple(max(0, c - 50) for c in color)
    lighter = tuple(min(255, c + 40) for c in color)
    
    cv2.fillPoly(img, [bone_points], color)
    cv2.polylines(img, [bone_points], True, darker, 2 if not highlight else 3)
    cv2.polylines(img, [bone_points[:len(bone_points)//2]], False, lighter, 1)


def draw_metacarpal(img, p1, p2, color, highlight=False):
    """Draw a metacarpal bone (palm bone)."""
    if highlight:
        color = COLORS['highlight']
    
    angle = get_angle(p1, p2)
    length = get_distance(p1, p2)
    
    if length < 10:
        return
    
    base_width = length * 0.22
    shaft_width = length * 0.12
    neck_width = length * 0.10
    head_width = length * 0.20
    
    shape = [
        (-base_width * 0.15, -base_width * 0.6),
        (0, -base_width * 0.55),
        (length * 0.08, -base_width * 0.5),
        (length * 0.15, -shaft_width * 0.5),
        (length * 0.3, -shaft_width * 0.45),
        (length * 0.5, -shaft_width * 0.4),
        (length * 0.7, -shaft_width * 0.45),
        (length * 0.82, -neck_width * 0.5),
        (length * 0.88, -neck_width * 0.55),
        (length * 0.92, -head_width * 0.6),
        (length * 0.96, -head_width * 0.5),
        (length, -head_width * 0.3),
        (length + head_width * 0.15, 0),
        (length, head_width * 0.3),
        (length * 0.96, head_width * 0.5),
        (length * 0.92, head_width * 0.6),
        (length * 0.88, neck_width * 0.55),
        (length * 0.82, neck_width * 0.5),
        (length * 0.7, shaft_width * 0.45),
        (length * 0.5, shaft_width * 0.4),
        (length * 0.3, shaft_width * 0.45),
        (length * 0.15, shaft_width * 0.5),
        (length * 0.08, base_width * 0.5),
        (0, base_width * 0.55),
        (-base_width * 0.15, base_width * 0.6),
    ]
    
    bone_points = []
    for px, py in shape:
        rotated = rotate_point((p1[0] + px, p1[1] + py), p1, angle)
        bone_points.append([int(rotated[0]), int(rotated[1])])
    
    bone_points = np.array(bone_points, np.int32)
    
    darker = tuple(max(0, c - 50) for c in color)
    lighter = tuple(min(255, c + 40) for c in color)
    
    cv2.fillPoly(img, [bone_points], color)
    cv2.polylines(img, [bone_points], True, darker, 2 if not highlight else 3)
    cv2.polylines(img, [bone_points[:len(bone_points)//2]], False, lighter, 1)


def draw_carpal_bones(img, points):
    """Draw the 8 carpal bones of the wrist."""
    wrist = points[0]
    index_base = points[5]
    pinky_base = points[17]
    
    palm_width = get_distance(index_base, pinky_base)
    
    carpal_center_x = int(wrist[0] * 0.6 + (index_base[0] + pinky_base[0]) / 2 * 0.4)
    carpal_center_y = int(wrist[1] * 0.6 + (index_base[1] + pinky_base[1]) / 2 * 0.4)
    
    hand_angle = get_angle(wrist, ((index_base[0] + pinky_base[0]) // 2, 
                                    (index_base[1] + pinky_base[1]) // 2))
    
    carpal_info = [
        {'name': 'scaphoid', 'offset': (-0.25, -0.12), 'size': (0.14, 0.11), 'angle': 20},
        {'name': 'lunate', 'offset': (-0.08, -0.15), 'size': (0.11, 0.09), 'angle': 0},
        {'name': 'triquetrum', 'offset': (0.08, -0.12), 'size': (0.10, 0.09), 'angle': -10},
        {'name': 'pisiform', 'offset': (0.18, -0.08), 'size': (0.06, 0.06), 'angle': 0},
        {'name': 'trapezium', 'offset': (-0.28, 0.08), 'size': (0.12, 0.10), 'angle': 35},
        {'name': 'trapezoid', 'offset': (-0.14, 0.10), 'size': (0.09, 0.08), 'angle': 15},
        {'name': 'capitate', 'offset': (0.0, 0.12), 'size': (0.11, 0.12), 'angle': 0},
        {'name': 'hamate', 'offset': (0.14, 0.10), 'size': (0.11, 0.10), 'angle': -15},
    ]
    
    color = COLORS['carpal']
    darker = tuple(max(0, c - 40) for c in color)
    lighter = tuple(min(255, c + 30) for c in color)
    
    for carpal in carpal_info:
        ox, oy = carpal['offset']
        rotated_offset = rotate_point((ox * palm_width, oy * palm_width), (0, 0), hand_angle)
        
        cx = int(carpal_center_x + rotated_offset[0])
        cy = int(carpal_center_y + rotated_offset[1])
        
        w = int(carpal['size'][0] * palm_width)
        h = int(carpal['size'][1] * palm_width)
        
        bone_angle = math.radians(carpal['angle']) + hand_angle
        
        num_points = 6
        bone_points = []
        for i in range(num_points):
            angle = bone_angle + (2 * math.pi * i / num_points)
            radius_variation = 0.85 + 0.3 * ((i % 2) * 0.5 + (i % 3) * 0.2)
            px = cx + int(w * radius_variation * math.cos(angle))
            py = cy + int(h * radius_variation * math.sin(angle))
            bone_points.append([px, py])
        
        bone_points = np.array(bone_points, np.int32)
        
        cv2.fillPoly(img, [bone_points], color)
        cv2.polylines(img, [bone_points], True, darker, 1)
        cv2.circle(img, (cx - w//4, cy - h//4), max(2, w//6), lighter, -1)


def draw_hand_anatomy(img, landmarks, w, h, highlighted_bone=None):
    """Draw anatomically accurate hand bones with optional highlighting."""
    points = {}
    for i in range(21):
        points[i] = get_point(landmarks.landmark[i], w, h)
    
    draw_carpal_bones(img, points)
    
    for finger_name, bones in FINGER_BONES.items():
        color = COLORS[finger_name]
        
        for bone_type, start_idx, end_idx, bone_name in bones:
            p1 = points[start_idx]
            p2 = points[end_idx]
            
            is_highlighted = (highlighted_bone is not None and 
                              highlighted_bone[2] == start_idx and 
                              highlighted_bone[3] == end_idx)
            
            if bone_type == 'metacarpal':
                draw_metacarpal(img, p1, p2, color, highlight=is_highlighted)
            elif bone_type == 'dist_phalanx':
                draw_phalanx(img, p1, p2, color, is_distal=True, highlight=is_highlighted)
            else:
                draw_phalanx(img, p1, p2, color, is_distal=False, highlight=is_highlighted)
    
    return img


def draw_pointer_indicator(img, tip_point):
    """Draw a small indicator at the pointer fingertip."""
    cv2.circle(img, tip_point, 10, (0, 0, 255), 2)
    cv2.circle(img, tip_point, 4, (0, 0, 255), -1)


def draw_bone_label(img, bone_name, w):
    """Draw the bone name label on screen."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(bone_name, font, font_scale, thickness)
    
    x = (w - text_width) // 2
    y = 60
    
    padding = 15
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + padding), 
                  (0, 0, 0), -1)
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + padding), 
                  COLORS['highlight'], 2)
    
    cv2.putText(img, bone_name, (x, y), font, font_scale, COLORS['highlight'], thickness)


def main():
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, img = cap.read()
            if not success:
                continue
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            touched_bone = None
            pointer_tip = None
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                
                # Get index fingertip (landmark 8) from both hands
                tip1 = get_point(hand1.landmark[8], w, h)
                tip2 = get_point(hand2.landmark[8], w, h)
                
                # Get points for both hands
                points1 = {i: get_point(hand1.landmark[i], w, h) for i in range(21)}
                points2 = {i: get_point(hand2.landmark[i], w, h) for i in range(21)}
                
                # Check if hand1's finger is touching hand2's bones
                touched_by_hand1 = find_touched_bone(tip1, points2)
                # Check if hand2's finger is touching hand1's bones
                touched_by_hand2 = find_touched_bone(tip2, points1)
                
                # Draw both hands
                if touched_by_hand1:
                    touched_bone = touched_by_hand1
                    pointer_tip = tip1
                    draw_hand_anatomy(img, hand1, w, h, highlighted_bone=None)
                    draw_hand_anatomy(img, hand2, w, h, highlighted_bone=touched_bone)
                elif touched_by_hand2:
                    touched_bone = touched_by_hand2
                    pointer_tip = tip2
                    draw_hand_anatomy(img, hand1, w, h, highlighted_bone=touched_bone)
                    draw_hand_anatomy(img, hand2, w, h, highlighted_bone=None)
                else:
                    draw_hand_anatomy(img, hand1, w, h)
                    draw_hand_anatomy(img, hand2, w, h)
                
                # Draw pointer indicator
                if pointer_tip:
                    draw_pointer_indicator(img, pointer_tip)
                
                # Display bone name
                if touched_bone:
                    draw_bone_label(img, touched_bone[4], w)
            
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                # Just one hand - draw it normally
                draw_hand_anatomy(img, results.multi_hand_landmarks[0], w, h)

            # Instructions
            cv2.putText(img, "Touch one hand's index finger to the other hand to reveal scientific names for ", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Hand Anatomy", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
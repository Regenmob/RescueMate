
# import cv2
# import numpy as np
# import openvino as ov
# import time
# import os

# # ✅ 클래스 라벨 정의
# class_labels = ["safety", "unsafety"]

# # ✅ 모델 로딩 및 입력 설정
# core = ov.Core()
# model = core.read_model("/home/sunghyuck/12/son/otx-workspace-DETECTION/outputs/export/openvino.xml")
# model.reshape({model.inputs[0].any_name: [1, 3, 736, 992]})
# compiled_model = core.compile_model(model=model, device_name="GPU")
# input_layer = compiled_model.input(0)
# _, C, H, W = input_layer.shape

# # ✅ NMS 함수 (겹치는 박스 제거)
# def apply_nms(boxes, iou_threshold=0.5):
#     if not boxes:
#         return []

#     boxes_array = np.array(boxes)
#     coords = boxes_array[:, :4].tolist()
#     scores = boxes_array[:, 4].tolist()
#     indices = cv2.dnn.NMSBoxes(coords, scores, 0.4, iou_threshold)
#     return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

# # ✅ 시각화 함수
# def draw_boxes(image, boxes, threshold=0.5):
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     for box in boxes:
#         if len(box) < 6:
#             continue
#         x1, y1, x2, y2, conf, class_id = box
#         if conf >= threshold:
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#             class_id = int(class_id)
#             label = f"{class_labels[class_id]} {int(conf * 100)}%"

#             cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0) if class_labels[class_id] == "safety" else (0, 0, 255), 2)
#             cv2.putText(rgb, label, (x1, max(20, y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# # ✅ 웹캠 실행
# cap = cv2.VideoCapture(0)

# # 상태 변수 초기화
# time_condition_start = None
# save_photo_timer = None
# show_danger_text = False
# danger_text_timer = None
# danger_identified = False

# # 저장 폴더 확인 및 생성
# save_folder = "danger"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if danger_identified:
#         cv2.imshow("Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("y"):
#             break
#         continue

#     resized = cv2.resize(frame, (W, H))
#     input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

#     result = compiled_model([input_tensor])
#     boxes = result[compiled_model.output("boxes")].squeeze()
#     labels = result[compiled_model.output("labels")].squeeze()

#     max_safety_confidence = 0
#     max_unsafety_confidence = 0
#     found_detection = False
#     for i in range(len(boxes)):
#         if np.all(boxes[i] == 0):
#             continue
#         x1, y1, x2, y2, conf = boxes[i]
#         class_id = int(labels[i])
#         label = class_labels[class_id]
#         if label == "safety":
#             max_safety_confidence = max(max_safety_confidence, conf)
#             found_detection = True
#         elif label == "unsafety":
#             max_unsafety_confidence = max(max_unsafety_confidence, conf)
#             found_detection = True

#     combined = []
#     for i in range(len(boxes)):
#         if np.all(boxes[i] == 0):
#             continue
#         x1, y1, x2, y2, conf = boxes[i]
#         class_id = int(labels[i])
#         label = class_labels[class_id]
#         if label == "safety" or label == "unsafety":
#             combined.append([x1, y1, x2, y2, conf, class_id])

#     combined = apply_nms(combined)

#     current_time = time.time()
#     condition_met = (max_safety_confidence < 0.5) or (max_unsafety_confidence > 0.5)

#     if condition_met:
#         if time_condition_start is None:
#             time_condition_start = current_time
#             save_photo_timer = current_time
#             show_danger_text = False
#             danger_text_timer = None
#         elif current_time - time_condition_start >= 3:
#             if save_photo_timer is None or current_time - save_photo_timer >= 15:
#                 timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#                 filename = os.path.join(save_folder, f"capture_{timestamp}.jpg")
#                 cv2.imwrite(filename, frame)
#                 save_photo_timer = current_time

#             if not show_danger_text and current_time - time_condition_start >= 3:
#                 show_danger_text = True
#                 danger_text_timer = current_time
#     else:
#         time_condition_start = None
#         save_photo_timer = None
#         # show_danger_text = False # 이 줄을 제거합니다.
#         danger_text_timer = None

#     output_img = draw_boxes(frame, combined, threshold=0.5)

#     if show_danger_text:
#         text = "Danger"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 2
#         font_thickness = 3
#         text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#         text_x = (output_img.shape[1] - text_size[0]) // 2
#         text_y = (output_img.shape[0] + text_size[1]) // 2
#         cv2.putText(output_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

#     cv2.imshow("Detection", output_img)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
#     elif key == ord("y"):
#         danger_identified = True
#         time_condition_start = None
#         save_photo_timer = None
#         show_danger_text = False
#         danger_text_timer = None

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import openvino as ov
import time
import os

# ✅ 클래스 라벨 정의
class_labels = ["safety", "unsafety"]

# ✅ 모델 로딩 및 입력 설정
core = ov.Core()
model = core.read_model("/home/sunghyuck/12/son/otx-workspace-DETECTION/outputs/export/openvino.xml")
model.reshape({model.inputs[0].any_name: [1, 3, 736, 992]})
compiled_model = core.compile_model(model=model, device_name="GPU")
input_layer = compiled_model.input(0)
_, C, H, W = input_layer.shape

# ✅ NMS 함수 (겹치는 박스 제거)
def apply_nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes_array = np.array(boxes)
    coords = boxes_array[:, :4].tolist()
    scores = boxes_array[:, 4].tolist()
    indices = cv2.dnn.NMSBoxes(coords, scores, 0.4, iou_threshold)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

# ✅ 시각화 함수
def draw_boxes(image, boxes, threshold=0.5):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        if len(box) < 6:
            continue
        x1, y1, x2, y2, conf, class_id = box
        if conf >= threshold:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
            label = f"{class_labels[class_id]} {int(conf * 100)}%"

            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0) if class_labels[class_id] == "safety" else (0, 0, 255), 2)
            cv2.putText(rgb, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ✅ 웹캠 실행
cap = cv2.VideoCapture(0)

# 상태 변수 초기화
time_condition_start = None
save_photo_timer = None
show_danger_text = False
danger_text_timer = None
danger_identified = False

# 저장 폴더 확인 및 생성
save_folder = "danger"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if danger_identified:
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("y"):
            break
        continue

    resized = cv2.resize(frame, (W, H))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

    result = compiled_model([input_tensor])
    boxes = result[compiled_model.output("boxes")].squeeze()
    labels = result[compiled_model.output("labels")].squeeze()

    max_safety_confidence = 0
    max_unsafety_confidence = 0
    found_detection = False
    for i in range(len(boxes)):
        if np.all(boxes[i] == 0):
            continue
        x1, y1, x2, y2, conf = boxes[i]
        class_id = int(labels[i])
        label = class_labels[class_id]
        if label == "safety":
            max_safety_confidence = max(max_safety_confidence, conf)
            found_detection = True
        elif label == "unsafety":
            max_unsafety_confidence = max(max_unsafety_confidence, conf)
            found_detection = True

    combined = []
    for i in range(len(boxes)):
        if np.all(boxes[i] == 0):
            continue
        x1, y1, x2, y2, conf = boxes[i]
        class_id = int(labels[i])
        label = class_labels[class_id]
        if label == "safety" or label == "unsafety":
            combined.append([x1, y1, x2, y2, conf, class_id])

    combined = apply_nms(combined)

    current_time = time.time()
    condition_met = (max_safety_confidence < 0.5) or (max_unsafety_confidence > 0.5)

    if not show_danger_text: # show_danger_text가 False일 때만 조건 확인
        if condition_met:
            if time_condition_start is None:
                time_condition_start = current_time
                save_photo_timer = current_time
                danger_text_timer = None
            elif current_time - time_condition_start >= 3:
                if save_photo_timer is None or current_time - save_photo_timer >= 15:
                    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    filename = os.path.join(save_folder, f"capture_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    save_photo_timer = current_time

                if not show_danger_text and current_time - time_condition_start >= 3:
                    show_danger_text = True
                    danger_text_timer = current_time
        else:
            time_condition_start = None
            save_photo_timer = None
            danger_text_timer = None

    output_img = draw_boxes(frame, combined, threshold=0.5)

    if show_danger_text:
        text = "Danger"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (output_img.shape[1] - text_size[0]) // 2
        text_y = (output_img.shape[0] + text_size[1]) // 2
        cv2.putText(output_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    cv2.imshow("Detection", output_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("y"):
        danger_identified = True
        time_condition_start = None
        save_photo_timer = None
        show_danger_text = False
        danger_text_timer = None

cap.release()
cv2.destroyAllWindows()
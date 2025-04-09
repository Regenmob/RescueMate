import cv2
import numpy as np
import openvino as ov
import time

# 모델 파일 이름과 모델 입력 해상도 설정 (모델이 학습된 해상도)
MODEL_NAME = "/home/sunghyuck/project/RescueMate/JJH/otx-workspace-DETECTION/outputs/deploy/model/model.xml"
MODEL_WIDTH = 992    # 모델 입력 너비 (W)
MODEL_HEIGHT = 736   # 모델 입력 높이 (H)

# OpenVINO Core 초기화 및 모델 로드/컴파일
core = ov.Core()
print("사용 가능한 디바이스:", core.available_devices)
# GPU 사용 시 변경 요망
DEVICE = "GPU"

model = core.read_model(MODEL_NAME)
compiled_model = core.compile_model(model=model, device_name=DEVICE)

# 모델의 입력 및 출력 노드 가져오기
input_layer = compiled_model.input(0)
output_box = compiled_model.output("boxes")   # 박스 좌표 및 score 출력 노드
output_label = compiled_model.output("labels")  # 레이블 출력 노드

# 클래스 목록: 인덱스 0은 Safety, 1은 Unsafety 등.
CLASSES = ["Safety", "Unsafety"]

danger_start_time = None
is_danger = False

# 정보 창 크기 설정
INFO_COUNT_WIDTH = 150
INFO_COUNT_HEIGHT = 100
INFO_STATUS_WIDTH = 200
INFO_STATUS_HEIGHT = 100


def process_results(frame: np.ndarray, boxes: np.ndarray, labels: np.ndarray, thresh: float = 0.3) -> tuple[list, int, int, int]:
    """
    모델 추론 결과로부터 detection 결과를 처리합니다.
    동일한 객체에 대해 여러 클래스가 검출된 경우, 가장 높은 score를 가진 클래스만 유지합니다.

    Args:
        frame (np.ndarray): 원본 이미지 프레임.
        boxes (np.ndarray): 모델의 박스 출력 텐서, shape: [1, 1, N, 5] (N: 검출 개수)
        labels (np.ndarray): 모델의 레이블 출력 텐서, shape: [1, 1, N, ?] (여기서는 label 정보)
        thresh (float): score 임계값. 이 값 이상인 검출만 사용합니다.

    Returns:
        tuple[list, int, int, int]: 각 검출 결과는 (label, score, box) 튜플로 구성된 리스트와,
                                     Safety와 Unsafety 객체 수, 그리고 총 객체 수를 반환합니다.
    """
    frame_height, frame_width = frame.shape[:2]
    ratio_x = frame_width / MODEL_WIDTH
    ratio_y = frame_height / MODEL_HEIGHT

    boxes_array = boxes.squeeze()
    labels_array = labels.squeeze()

    # 바운딩 박스별로 검출 결과 저장
    detections_by_box = {}

    safety_count = 0
    unsafety_count = 0

    for box_vals, label_vals in zip(boxes_array, labels_array):
        xmin, ymin, xmax, ymax, score = box_vals
        if score >= thresh:
            # 모델 출력은 정규화 좌표이므로 실제 픽셀 좌표로 변환
            box_pixels = tuple(
                map(
                    int,
                    (
                        xmin * ratio_x,
                        ymin * ratio_y,
                        (xmax - xmin) * ratio_x,
                        (ymax - ymin) * ratio_y,
                    ),
                )
            )
            label = int(label_vals)  # label 값을 정수로 변환

            # 튜플 형태의 박스 좌표를 키로 사용
            if box_pixels not in detections_by_box:
                detections_by_box[box_pixels] = []
            detections_by_box[box_pixels].append((label, float(score)))

    final_detections = []
    for box, class_scores in detections_by_box.items():
        # 각 바운딩 박스에 대해 가장 높은 score를 가진 클래스 찾기
        best_label = -1
        best_score = -1.0
        for label, score in class_scores:
            if score > best_score:
                best_score = score
                best_label = label
        if best_label != -1:
            final_detections.append((best_label, best_score, box))
            if best_label == 0:
                safety_count += 1
            elif best_label == 1:
                unsafety_count += 1

    total_count = safety_count + unsafety_count
    return final_detections, safety_count, unsafety_count, total_count


def draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    검출 결과에 따라 원본 이미지에 박스와 레이블 텍스트를 그립니다.

    Args:
        frame (np.ndarray): 원본 이미지 프레임.
        detections (list): (label, score, box) 튜플로 구성된 검출 결과 리스트.

    Returns:
        np.ndarray: 박스 및 텍스트가 그려진 이미지.
    """
    colors = {
        0: (10, 240, 30),  # Safety: 초록 계열
        1: (10, 30, 240)   # Unsafety: 붉은 계열
    }

    for label, score, box in detections:
        color = colors.get(label, (100, 100, 100))
        x1, y1, box_width, box_height = box
        x2 = x1 + box_width
        y2 = y1 + box_height

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

        cv2.putText(
            frame,
            f"{CLASSES[label]} {score:.2f}",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame


def display_info(safety_count: int, unsafety_count: int) -> None:
    """
    검출된 Safety 및 Unsafety 객체 수를 별도의 창에 표시합니다.

    Args:
        safety_count (int): 검출된 Safety 객체 수.
        unsafety_count (int): 검출된 Unsafety 객체 수.
    """
    info_window_name = "Detection Count"
    info_image = np.zeros((INFO_COUNT_HEIGHT, INFO_COUNT_WIDTH, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.4
    font_thickness = 1

    cv2.putText(
        info_image,
        f"Safety: {safety_count}",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        info_image,
        f"Unsafety: {unsafety_count}",
        (5, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    cv2.imshow(info_window_name, info_image)


def display_status_info(safety_count: int, unsafety_count: int, total_count: int) -> None:
    """
    Safety 및 Unsafety 비율에 따라 위험 또는 정상 상태를 별도의 창에 표시합니다.

    Args:
        safety_count (int): 검출된 Safety 객체 수.
        unsafety_count (int): 검출된 Unsafety 객체 수.
        total_count (int): 총 검출된 객체 수.
    """
    global danger_start_time, is_danger

    status_window_name = "Detection Status"
    status_image = np.zeros((INFO_STATUS_HEIGHT, INFO_STATUS_WIDTH, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.6
    font_thickness = 1
    status_text = "Normal"
    status_color = (0, 255, 0)  # 녹색

    if total_count > 0:
        safety_percentage = (safety_count / total_count) * 100
        unsafety_percentage = (unsafety_count / total_count) * 100

        if safety_percentage < 30 or unsafety_percentage > 50:
            if not is_danger:
                is_danger = True
                danger_start_time = time.time()
            elif danger_start_time is not None and (time.time() - danger_start_time) >= 10:
                status_text = "Danger"
                status_color = (0, 0, 255)  # 빨간색
        else:
            is_danger = False
            danger_start_time = None
            status_text = "Normal"
            status_color = (0, 255, 0)
    else:
        status_text = "Normal" # No objects detected, consider as normal

    cv2.putText(
        status_image,
        status_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        status_color,
        font_thickness,
        cv2.LINE_AA,
    )

    cv2.imshow(status_window_name, status_image)


def run_object_detection(source: int = 0) -> None:
    """
    웹캠 등에서 영상을 입력받아 객체 검출을 실행하고, 결과를 화면에 출력합니다.

    Args:
        source (int): 웹캠 소스 번호 (기본값: 0).
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("이미지 캡쳐에 실패했습니다.")
            break

        resized_img = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA)
        input_img = np.expand_dims(resized_img.transpose(2, 0, 1), axis=0)

        results = compiled_model([input_img])
        boxes = results[output_box]
        labels = results[output_label]
        detections, safety_count, unsafety_count, total_count = process_results(frame, boxes, labels)

        frame = draw_boxes(frame, detections)

        cv2.imshow("results", frame)
        # 카메라 창 위치 및 크기 가져오기
        results_rect = cv2.getWindowImageRect("results")
        results_x = results_rect[0]
        results_y = results_rect[1]
        results_width = results_rect[2]
        results_height = results_rect[3]

        # "Detection Count" 창 위치 설정 (카메라 창 바로 오른쪽)
        count_x = results_x + results_width + 10  # 약간의 간격 추가
        count_y = results_y
        cv2.imshow("Detection Count", np.zeros((INFO_COUNT_HEIGHT, INFO_COUNT_WIDTH, 3), dtype=np.uint8)) # 먼저 창 생성
        cv2.moveWindow("Detection Count", count_x, count_y)
        display_info(safety_count, unsafety_count)

        # "Detection Status" 창 위치 설정 ("Detection Count" 창 바로 아래)
        status_x = results_x + results_width + 10  # 동일한 x 위치
        status_y = results_y + INFO_COUNT_HEIGHT + 100 # "Detection Count" 창 아래 간격 추가
        cv2.imshow("Detection Status", np.zeros((INFO_STATUS_HEIGHT, INFO_STATUS_WIDTH, 3), dtype=np.uint8)) # 먼저 창 생성
        cv2.moveWindow("Detection Status", status_x, status_y)
        display_status_info(safety_count, unsafety_count, total_count)

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow("Detection Count")
    cv2.destroyWindow("Detection Status")


if __name__ == "__main__":
    run_object_detection()
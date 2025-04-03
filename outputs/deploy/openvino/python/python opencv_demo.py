import cv2
import time

# 모델 파일 경로 및 설정 파일 경로 (사용자 환경에 맞게 수정해야 합니다.)
model_path = "home/sunghyuck/12/son/otx-workspace-DETECTION/outputs/export/openvino.xml"  # OpenVINO 모델 파일 (.xml)
config_path = "home/sunghyuck/12/son/otx-workspace-DETECTION/outputs/export/openvino.bin"  # OpenVINO 설정 파일 (.bin)

# 클래스 이름 (OTX 학습 시 사용한 클래스 순서와 동일해야 합니다.)
classes = ["unsafety", "safety"]  # 예시: "unsafety"가 0번, "safety"가 1번 클래스인 경우

# 신뢰도 임계값 (40% = 0.4)
confidence_threshold = 0.4

# 카메라 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 모델 로드
net = cv2.dnn.readNet(model_path, config_path)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    height, width = frame.shape[:2]

    # blob 생성 (모델 입력 형식에 맞춰 조정해야 합니다.)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (300, 300), swapRB=True, crop=False)
    net.setInput(blob)

    # 추론 실행
    detections = net.forward()

    # 탐지 결과 처리
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            label = classes[class_id]

            # 안전 또는 위험 클래스인 경우에만 처리
            if label == "safety" or label == "unsafety":
                box = detections[0, 0, i, 3:7] * [width, height, width, height]
                (startX, startY, endX, endY) = box.astype("int")

                # 바운딩 박스 및 레이블 그리기
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0) if label == "safety" else (0, 0, 255), 2)
                text = f"{label}: {confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if label == "safety" else (0, 0, 255), 2)

    # 화면에 프레임 표시
    cv2.imshow("Object Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()
import cv2
import time
import re
from gpiozero import MotionSensor, LED
from picamera2 import Picamera2
from ultralytics import YOLO
import easyocr

# 1. 하드웨어 핀 설정 (PIR: GPIO 4, LED: GPIO 17)
pir = MotionSensor(4)
led = LED(17)

print("1/2. 차량 검출용 YOLOv8n 모델 로드 중...")
vehicle_model = YOLO("yolov8n.pt", task="detect")

print("2/2. 초경량 번호판 글자 인식(EasyOCR) 엔진 로드 중...")
reader = easyocr.Reader(['ko', 'en'], gpu=False)

print("Picamera2 로딩 및 스트림 개방 중...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

time.sleep(1.5)
print("✅ 주차 관제 시스템 준비 완료! (PIR-GPIO4 / LED-GPIO17 연동 완료)")

# 차량 클래스 ID (car, bus, truck)
VEHICLE_CLASSES = [2, 5, 7]

# 💡 허가된 차량 번호판 목록 (테스트용 숫자 패턴 등록)
AUTHORIZED_PLATES = ["233", "1181", "123가4567"]

def stop_system():
    print("\n🔴 주차 관제 시스템이 안전하게 종료됩니다.")
    led.off()
    picam2.stop()
    cv2.destroyAllWindows()
    exit()

# 메인 루프 실행
try:
    while True:
        # [STAGE 1] PIR 센서 움직임 감지 대기
        if pir.motion_detected:
            print("\n🚨 [PIR 트리거] 차량 진입 구역 움직임 감지! 스캔을 시작합니다.")
            
            scan_start = time.time()
            while pir.motion_detected and (time.time() - scan_start < 5):
                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # [STAGE 2] 차량 검출
                v_results = vehicle_model.predict(frame_bgr, conf=0.4, verbose=False)
                v_boxes = v_results[0].boxes
                
                for v_box in v_boxes:
                    v_class = int(v_box.cls[0])
                    
                    if v_class in VEHICLE_CLASSES:
                        v_xyxy = list(map(int, v_box.xyxy[0].tolist()))
                        
                        # 차량 영역 이미지 크롭
                        crop_vehicle = frame_bgr[v_xyxy[1]:v_xyxy[3], v_xyxy[0]:v_xyxy[2]]
                        if crop_vehicle.size == 0:
                            continue
                            
                        cv2.imwrite("current_vehicle.jpg", crop_vehicle)
                        print(f"🚘 [차량 발견] 차량 포착 -> 번호판 글자 분석 스캔 시작")
                        
                        # [STAGE 3] 차량 이미지 내부에서 EasyOCR로 글자 및 위치 추출
                        ocr_results = reader.readtext(crop_vehicle)
                        annotated_img = crop_vehicle.copy()
                        detected_any_plate = False
                        
                        for (bbox, text, prob) in ocr_results:
                            clean_text = re.sub(f"[^가-힣0-9a-zA-Z]", "", text)
                            
                            # 숫자가 포함된 3글자 이상의 패턴을 번호판으로 인식
                            if len(clean_text) >= 3 and any(char.isdigit() for char in clean_text):
                                detected_any_plate = True
                                print(f"🎫 [번호판 문자 추출 성공]: {clean_text} (신뢰도: {prob:.2f})")
                                
                                # 이미지에 번호판 박스 및 텍스트 마킹
                                pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                                pt2 = (int(bbox[2][0]), int(bbox[2][1]))
                                cv2.rectangle(annotated_img, pt1, pt2, (0, 255, 0), 2)
                                cv2.putText(annotated_img, clean_text, (pt1[0], pt1[1] - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                # plate_logs.txt 파일 기록
                                with open("plate_logs.txt", "a", encoding="utf-8") as f:
                                    log_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                    f.write(f"[{log_time}] 감지된 번호판: {clean_text} (정확도: {prob:.2f})\n")
                                print("📝 [Plate 로그 기록 완료] -> plate_logs.txt 적재")
                                
                                # [STAGE 4] 등록 차량 승인 여부 대조 및 빵판 LED 제어
                                is_authorized = False
                                for auth_plate in AUTHORIZED_PLATES:
                                    if auth_plate in clean_text:
                                        is_authorized = True
                                        break
                                
                                if is_authorized:
                                    print(f"🔓 [출입 허가] 등록 차량 승인: {clean_text}")
                                    print("💡 [HARDWARE] 빵판의 LED(GPIO 17)를 3초간 점등합니다.")
                                    led.on()
                                    time.sleep(3)
                                    led.off()
                                else:
                                    print(f"❌ [출입 통제] 미등록 외부 차량: {clean_text}")
                                    
                                break 
                        
                        if detected_any_plate:
                            cv2.imwrite("detected_result.jpg", annotated_img)
                            print("💾 [결과 이미지 저장 완료] -> detected_result.jpg")
                        
                        break 
                
                time.sleep(0.3)
                
            print("🟢 [상태 전환] 대기 상태로 복귀합니다.\n")
            
        time.sleep(0.5)

except KeyboardInterrupt:
    stop_system()

import cv2

# 동영상 파일에서 프레임 불러오기
cap = cv2.VideoCapture('movingobj01.mp4')

# 첫 번째 프레임을 읽고 배경으로 설정
ret, base_frame = cap.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    cap.release()
    exit()

# 시퀀스 사진의 기본 베이스 이미지 생성 (초기 프레임)
sequence_image = base_frame.copy()

# 배경 프레임을 그레이스케일로 변환
gray_base = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)

while True:
    # 현재 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 프레임이 더 이상 없으면 종료
    
    # 현재 프레임을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 배경 프레임과 현재 프레임 간의 차이 계산
    diff_frame = cv2.absdiff(gray_base, gray_frame)
    
    # 차이 값에 대한 임계처리 (thresholding)
    _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
    
    # 차이 영역을 반전하여 배경 영역을 만듦
    inv_thresh_frame = cv2.bitwise_not(thresh_frame)
    
    # 배경 영역과 현재 프레임의 AND 연산으로 움직임이 없는 부분 복원
    static_area = cv2.bitwise_and(sequence_image, sequence_image, mask=inv_thresh_frame)
    
    # 차이 있는 부분(움직임 영역)과 현재 프레임을 AND 연산하여 움직임 포착
    moving_area = cv2.bitwise_and(frame, frame, mask=thresh_frame)
    
    # 배경 이미지에 움직임이 있는 영역을 합성
    sequence_image = cv2.add(static_area, moving_area)
    
    # 실시간 모션샷 이미지 출력
    cv2.imshow("Motion Shot", sequence_image)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 결과 이미지를 파일로 저장
cv2.imwrite('motion_shot_output.jpg', sequence_image)

# 자원 해제
cap.release()
cv2.destroyAllWindows()

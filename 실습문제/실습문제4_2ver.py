import cv2
import numpy as np

def create_motion_shot(video_path, output_path):
    """
    동영상 파일에서 움직임이 포착된 모션샷 이미지를 생성하고 저장합니다.

    Args:
        video_path (str): 입력 동영상 파일 경로
        output_path (str): 출력 모션샷 이미지 파일 경로
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    # 첫 번째 프레임을 배경으로 설정
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 번째 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    # 모션샷 이미지 초기화
    motion_shot = np.zeros_like(prev_frame)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # 흑백 이미지 변환
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 움직임 감지
        diff = cv2.absdiff(gray_prev, gray_curr)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 움직임 영역을 모션샷 이미지에 합성
        motion_shot = cv2.addWeighted(motion_shot, 0.95, thresh, 0.05, 0)

        # 현재 프레임을 이전 프레임으로 설정
        prev_frame = current_frame

    # 모션샷 이미지 저장
    cv2.imwrite(output_path, motion_shot)

    cap.release()
    cv2.destroyAllWindows()

# 사용 예시
video_path = 'movingobj01.mp4'
output_path = 'motion_shot.jpg'
create_motion_shot(video_path, output_path)

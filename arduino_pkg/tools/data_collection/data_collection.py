from datetime import datetime 
import keyboard
import cv2
import serial
import os
import time

########################################################################

# 사진 저장 경로
DATA_PATH= os.path.dirname(os.path.realpath(__file__)) + '/img' 

# 카메라 번호
CAMERA_NUM = 0

# Arduino 장치 주소
SERIAL_PORT = "/dev/ttyACM0"

# 조향 단계
MAX_STEERING = 10

# 조향 간격
INTERVAL = 1

########################################################################

# Data Collector Class 추출본
class Data_Collect:
    def __init__(self, path, cam_num, max_steering=7, image_width=640, image_height=480, keyboard_sensing_period=0.1):
        self.data_collection_path = os.path.join(path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.left_speed = 0
        self.right_speed = 0
        self.steering = 0
        self.max_steering = max_steering
        self.cap = cv2.VideoCapture(cam_num)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
        self.frame_num = 0
        self.keyboard_sensing_period = keyboard_sensing_period
        self.exit = False
        self.last_input_time = time.time()

    def process(self):
        """
        한 번의 키보드 입력을 처리하고 상태를 반환.
        :return: {'exit': True/False} - 프로세스 종료 여부 포함.
        """
        current_time = time.time()
        if current_time - self.last_input_time < self.keyboard_sensing_period:
            return {'exit': self.exit}
        if keyboard.is_pressed('w'):
            self.left_speed = min(self.left_speed + 10, 250)
            self.right_speed = min(self.right_speed + 10, 250)
        elif keyboard.is_pressed('s'):
            self.left_speed = max(self.left_speed - 10, -250)
            self.right_speed = max(self.right_speed - 10, -250)
        elif keyboard.is_pressed('a'):
            self.steering = max(self.steering - INTERVAL, -self.max_steering)
        elif keyboard.is_pressed('d'):
            self.steering = min(self.steering + INTERVAL, self.max_steering)
        elif keyboard.is_pressed('r'):
            self.steering = 0
            self.left_speed = 0
            self.right_speed = 0
        elif keyboard.is_pressed('c'):
            print('Saving frame...')
            if not os.path.exists(self.data_collection_path):
                os.makedirs(self.data_collection_path)
            file_name = f'{self.data_collection_path}/{self.frame_num}_steer:{self.steering}_left_speed:{self.left_speed}_right_speed:{self.right_speed}.png'
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(file_name, frame)
                self.frame_num += 1
        elif keyboard.is_pressed('f'):
            self.left_speed = 0
            self.right_speed = 0
            self.steering = 0
            print("You pressed 'f'. Exiting...")
            self.exit = True
        self.last_input_time = current_time
        print(f'Steering: {self.steering}, Left Speed: {self.left_speed}, Right Speed: {self.right_speed}')
        return {'exit': self.exit}

    def get_control_values(self):
        """현재 속도 및 조향 값을 반환."""
        return {'steering': self.steering, 'left_speed': self.left_speed, 'right_speed': self.right_speed}

    def cleanup(self):
        """자원 정리"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print('Program finished')


# Main
def main():
    print(DATA_PATH)

    # 데이터 수집 객체 초기화
    data_collector = Data_Collect(path=DATA_PATH, cam_num=CAMERA_NUM, max_steering=MAX_STEERING)
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
    time.sleep(1)
    try:
        # 숨겨진 코드 프로세스 시작
        while True:
            # 한 번의 키보드 입력 처리
            result = data_collector.process()

            # 프로세스 종료 플래그 확인
            if result["exit"]:
                steering = 0
                left_speed = 0
                right_speed = 0
                message = f"s{steering}l{left_speed}r{right_speed}\n"
                ser.write(message.encode())
                break

            # 현재 제어 값 가져오기
            control_values = data_collector.get_control_values()

            # 시리얼 송신
            message = f"s{control_values['steering']}l{control_values['left_speed']}r{control_values['right_speed']}\n"
            ser.write(message.encode())

            # 디버깅용 출력
            print(f"Sent: {message.strip()}")

    except KeyboardInterrupt:
        steering = 0
        left_speed = 0
        right_speed = 0
        message = f"s{steering}l{left_speed}r{right_speed}\n"
        ser.write(message.encode())
        print("Program interrupted.")
        
    finally:
        ser.close()
        data_collector.cleanup()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()

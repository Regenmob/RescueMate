import subprocess
import re
import time

def monitor_demo_output(demo_command):
    """
    데모 스크립트를 실행하고 안전/위험 백분율 출력을 감시합니다.

    Args:
        demo_command (list): 데모 스크립트를 실행하는 명령어 (예: ["python", "demo.py", "-i", "0"]).
    """
    try:
        process = subprocess.Popen(demo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"데모 스크립트가 PID: {process.pid}로 시작되었습니다.")

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(f"[데모 출력]: {line.strip()}")

            # 대소문자 구분 없이 "safe" 또는 "unsafe" 뒤에 오는 백분율 값을 찾습니다.
            match = re.search(r"(?i)(safe|unsafe)\s*\((\d+)%\)", line)
            if match:
                label = match.group(1).lower()
                percentage = int(match.group(2))
                if percentage >= 50:
                    print(f"**[{label.capitalize()}] 값 {percentage}%가 50% 이상입니다.**")

            time.sleep(0.1)  # CPU 사용률을 낮추기 위한 작은 지연

        stdout, stderr = process.communicate()
        if stderr:
            print(f"[데모 에러]: {stderr}")

    except FileNotFoundError:
        print(f"에러: 데모 스크립트 '{demo_command[1]}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    demo_script_path = "demo.py"  # demo.py 파일이 현재 디렉토리에 있다고 가정합니다.
    camera_input = "0"  # 카메라 입력을 위한 값입니다. 필요에 따라 변경하세요.

    # demo.py를 실행하는 명령어
    demo_command = ["python", demo_script_path, "-i", camera_input]

    print(f"감시할 데모 스크립트: {' '.join(demo_command)}")
    monitor_demo_output(demo_command)
    print("감시가 종료되었습니다.")
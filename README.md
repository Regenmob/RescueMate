# RescueMate
RescueMate is an AI-powered program that detects falls and sends instant alerts to ensure safety in emergencies.

# 프로젝트 소개
딥뷰를 참고하여 만든 실시간 사람탐지 카메라

# 개발 기간
25.03.31 ~ 25.04.10

# 개발환경

ubuntu 22.04
openvino == 2023.3.0
openvino.model.api == 0.1.9.1
otx==1.6.5
numpy>=1.21.0,<=1.23.5 # np.bool was removed in 1.24.0 which was used in openvino runtime
opencv
python3.10



clone code
● (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)
```
https://github.com/Regenmob/RescueMate.git
```

Prerequite
● (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Steps to build
● (프로젝트를 실행을 위해 빌드 절차 기술)
```
cd ~/xxxx
source .venv/bin/activate

make
make install
```

Steps to run
● (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)
```
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

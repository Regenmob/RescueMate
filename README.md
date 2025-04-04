# RescueMate
RescueMate is an AI-powered program that detects falls and sends instant alerts to ensure safety in emergencies.

# 프로젝트 소개
딥뷰를 참고하여 CVAT툴을 이용한 실시간 사람탐지 카메라

# 개발 기간
25.03.31 ~ 25.04.10

# 실행해 필요한 requirement.txt

```
openvino == 2023.3.0
openvino.model.api == 0.1.9.1
otx==1.6.5
numpy>=1.21.0,<=1.23.5 # np.bool was removed in 1.24.0 which was used in openvino runtime
```


clone code
● (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)
```
https://github.com/Regenmob/RescueMate.git
```


#개발 환경 셋팅


1. 학습 진행 할 폴더생성

```
mkdir <학습폴더 이름>
```


2.학습 폴더 내 가상 작업환경 설정 및 OTX 다운


```
python -m venv .venv
source .otx/bin/activate
pip install -U pip
pip install otx[full]==1.6.5
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip uninstall mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
```

   

3. 작업할 툴 CVAT 사용

```
URL 주소 : https://www.cvat.ai/
```

CVAT 사용자 메뉴얼

```
URL 주소 : https://docs.cvat.ai/docs/manual/
```

cvat으로 생성한 폴더에 annotation폴더 밖에 없는대 해당 위치에 image폴더 생성 후 표폰 사진을 image 폴더에 넣는다
그러면 otx에서 트레이닝에 사용되는 폴더 완성

4. CVAT에서 생성한 파일을 이용한 OTX사용 Example

1) Build task

Example)

* Find template what we want to train
```
otx find --template --task DETECTION # find template what we to train
```

* Build task to create model
```
otx build Object_Detection_YOLO_X --train-data-roots  ../datumaro/export-coco/
otx build Object_Detection_YOLO_X --val-data-roots  ../datumaro/export-coco/
```

2)Train

Example)

* Start training with parameters
```
otx train params --learning_parameters.num_iters 8 --learning_parameters.batch_size 1
otx train --enable-hpo
```
3) Evaluation

Example)

* Evaluate model on the specific dataset
```
otx eval --test-data-roots ../../datumaro/export-coco
```

4) Export & Optimize

Example)

* Exports a trained model to the OpenVINO to run it on Intel hardware
```
otx export --load-weights ./outputs/latest_trained_model/models/weights.pth --output ./outputs/export
```

5) Deploy

Example)

* Creates openvino.zip with a demo application and an exported model
```
otx deploy --load-weights ./outputs/export/openvino.xml --output ./outputs/deploy
```



Steps to build
● (프로젝트를 실행을 위해 빌드 절차 기술)

Example)

* 학습된 openvino압출파일 풀기
```
cd <트레이닝 폴더> / otx-wokrspace-DETECTION / output / deploy
unzip openvino.zip
cd openvino / python
```




Steps to run
● (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

*실행 전 requirement.txt 로 파일 다운
```
pip install requirement.txt
```


* 카메라 실행
```
python demo.py -i xxx -m yyy -d zzz
```

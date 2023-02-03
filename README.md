# SCA-SignMatching (SIFT, VIT)

이 프로젝트는 SIFT와 VIT를 이용해 상가간판 매칭 기술을 구현한 프로젝트입니다.

## Introduction
본 모듈은 한 쌍의 로드뷰 파노라마 이미지와 SCA-ObjectDetection 모듈의 결과를 입력으로 받아 동일한 간판끼리 매칭하는 기능을 담당합니다.

간판 매칭에는 SIFT and VIT feature matching을 이용하였으며 본 모듈의 시각화 결과는 아래 이미지와 같습니다.

![pair_400-400_vit](docs/result/pair_400-400_vit.jpg)

## Pipeline
![Structure](docs/images/Structure.jpg)

## Project Architecture

```shell
${WORKSPACE}/SCA-SignMatching/
├─data
│  ├─result
│  │  ├─sift_best_pair    # SIFT 매칭 결과 텍스트 파일
│  │  ├─visualization     # 매칭된 결과 이미지
│  │  └─vit_best_pair     # VIT 매칭 결과 텍스트 파일
│  └─sample
│      ├─db               # 예시 DB 이미지
│      └─query            # 예시 Query 이미지
├─docker
├─docs
│  ├─images
│  └─result
├─models                  # Feature 추출 모델
│  ├─sift_vlad
│  │  └─utils
│  └─vit
│      └─utils
├─scripts                 # 데이터셋 다운로드 스크립트
└─utils                   # 모듈 실행을 위한 유틸리티
```

## Installation

본 프로젝트는 Docker 컨테이너 상에서 실행하기를 권장합니다.

설치 방법은 아래 링크를 참조하세요.

[HowToInstall.md](docs/HowToInstall.md)

## Inference

추론 방법은 아래 링크를 참조하세요.

[HowToInfer.md](docs/HowToInfer.md)

## Evaluation

### Best F1-score
| Recall | Precision | F1-score |
|--------|-----------|----------|
| 0.82   | 0.80      | 0.81     |

평가 결과 및 평가 방법은 아래 링크를 참조하세요.

[HowToEvaluate.md](docs/HowToEvaluate.md)

## How to run RESTful API Server
### Start API Server
* 아래 명령어를 순차적으로 수행하면 docker container 내에서 background로 실행됩니다.
```shell
# in Host
git clone -b SIFT_VIT https://${PERSONAL_TOKEN}@github.com/sogang-mm/SCA-SignMatching.git SCA-SIFT_VIT
cd SCA-SIFT_VIT/
docker-compose up -d --build
docker attach SIFT_VIT_main
# in docker container
source ~/.bashrc
sh scripts/entrypoint.sh 
sh scripts/server_start.sh
```
### Shutdown API Server
* 아래 쉘스크립트 명령어를 수행하면 background로 실행 중인 API 서버가 종료됩니다.
```shell
sh scripts/server_shutdown.sh
```

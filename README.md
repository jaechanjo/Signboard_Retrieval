# SCA-SignMatching (LoFTR)
본 프로젝트는 LoFTR feature matching을 이용해 상가간판 매칭기술을 구현한 프로젝트이다.
## Pipeline
![img.png](docs/images/pipeline.png)
## Project Architecture
```shell
.
├── README.md
├── data                        # data 디렉토리
├── docker                      # docker container 구축을 위해 필요한 설정파일
├── docker-compose.yml          # docker-compose 환경 설정 파일
├── docs                        # Readme 작성에 필요한 파일
├── eval.py                     # LoFTR SignMatching 성능 평가를 위한 모듈
├── main.py                     # LoFTR SignMatching 추론 모듈
├── requirements.txt            
├── scripts                     
│   └── download.sh             # 성능평가 데이터셋 다운로드 스크립트
└── utils
    ├── __init__.py
    └── common.py               # 모듈 실행을 위한 유틸리티

```

## Performance
### Method 별 성능
|Method| Recall   | Precision | F1-score |
|------|----------|-----------|----------|
|SIFT| 0.42     | 0.55      | 0.48     |
|VIT| 0.82     | **0.80**  | **0.81**     |
|SIFT+VIT| 0.69     | 0.74      | 0.71     |
|SuperPoint + Superglue| 0.82     | 0.76      | 0.79     |
|LoFTR| **0.85** | 0.77      | **0.81** |
### LoFTR 파라미터별 성능

| k   | Match threshold | recall   | precision | F1-score |
|-----|-----------------|----------|-----------|----------|
| 1   | 0.4             | **0.88** | 0.69      | 0.77     |
| 1   | 0.5             | 0.86     | 0.73      | 0.79     |
| 2   | 0.4             | 0.85     | 0.77      | **0.81** |
| 2   | 0.5             | 0.81     | 0.78      | 0.79     |
| 3   | 0.4             | 0.81     | 0.8       | 0.80     |
| 3   | 0.5             | 0.76     | 0.82      | 0.79     |
| 4   | 0.4             | 0.77     | 0.82      | 0.79     |
| 4   | 0.5             | 0.72     | 0.85      | 0.78     |
| 5   | 0.4             | 0.73     | 0.85      | 0.79     |
| 5   | 0.5             | 0.67     | **0.87**  | 0.76     |

## Installation
```shell
$ cd ${WORKSPACE}
$ git clone -b LoFTR --single-branch https:github.com/gyusik19/SCA-SignMatching
$ cd SCA-SignMatching
$ vim docker-compose.yml
  # volume 경로, ports 수정
$ docker-compose up -d --build
$ docker attach ${CONTAINER NAME}
$ pip install -r requirements.txt
```
## Inference
Data preparation

- 파노라마 이미지와 간판 detector 모듈의 결과인 json 파일을 같은 경로에 위치시킨다.
- 이미지와 json 파일은 같은 이름을 가지도록 한다.

```shell
data/sample/db_dir/db_img.jpg
data/sample/db_dir/db_img.json
data/sample/query_dir/query_img.jpg
data/sample/query_dir/query_img.json
```
```shell
python main.py \
--db_path 'data/sample/1/19.jpg' \
--query_path 'data/sample/1/21.jpg' \
--match_threshold 0.4 \
--k 2 \
--visualize \
--output_dir 'data/result'
```
visualize 결과
![img.png](docs/images/img.png)

## Evaluation
```shell
# 평가용 데이터 다운로드
sh scripts/download.sh
python eval.py \
--db_path 'data/gt/db' \
--query_path 'data/gt/query' \
--match_threshold 0.4 \
--k 2
```
다음과 같은 결과를 확인할 수 있다.

```shell
num GT :  704
num TP, num FP 600 184
recall :  0.852
precision :  0.765
eval time :  51.310 sec
```

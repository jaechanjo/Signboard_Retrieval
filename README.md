# SCA-SignMatching (Superpoint + Superglue)
본 프로젝트는 Superglue feature matching을 이용해 상가간판 매칭기술을 구현한 프로젝트이다.
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
├── models                      # Superglue 관련 weight 및 모듈
├── eval.py                     # Superglue SignMatching 성능 평가를 위한 모듈
├── main.py                     # Superglue SignMatching 추론 모듈
├── requirements.txt            
├── scripts      
          ├── download.sh              
│   └── download_weights.sh             # 성능평가 데이터셋 다운로드 스크립트
└── utils
    ├── __init__.py
    └── common.py               # 모듈 실행을 위한 유틸리티

```

## Performance
### Method 별 성능
|Method| Recall   | Precision | F1-score |
|------|----------|-----------|----------|
|SIFT| 0.37     | 0.49      | 0.42     |
|VIT| 0.78     | **0.79**  | 0.78     |
|SIFT+VIT| 0.66     | 0.73      | 0.69     |
|SuperPoint + Superglue| 0.82     | 0.76      | 0.79     |
|LoFTR| **0.85** | 0.77      | **0.81** |
### Superglue 파라미터별 성능

| k   | Match threshold | recall   | precision | F1-score |
|-----|-----------------|----------|-----------|----------|
| 1   | 0.4             | **0.85** | 0.68      | 0.76     |
| 1   | 0.5             | 0.82     | 0.73      | 0.77     |
| 2   | 0.4             | 0.82     | 0.76      | 0.78     |
| 2   | 0.5             | 0.81     | 0.78      | **0.79** |
| 3   | 0.4             | 0.78     | 0.78      | 0.78     |
| 3   | 0.5             | 0.74     | 0.81      | 0.78     |
| 4   | 0.4             | 0.74     | 0.80      | 0.77     |
| 4   | 0.5             | 0.70     | 0.83      | 0.76     |
| 5   | 0.4             | 0.69     | 0.82      | 0.75     |
| 5   | 0.5             | 0.66     | **0.84**  | 0.74     |

## Installation
```shell
$ cd ${WORKSPACE}
$ git clone -b superglue --single-branch https:github.com/gyusik19/SCA-SignMatching
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
sh scripts/download_weights.sh
python eval.py \
--db_path 'data/gt/db' \
--query_path 'data/gt/query' \
--match_threshold 0.4 \
--k 2
```
다음과 같은 결과를 확인할 수 있다.

```shell
num GT :  704
num TP, num FP 410 68
recall :  0.582
precision :  0.858
F1-score : 0.6937394247038918
eval time :  14.831 sec
```
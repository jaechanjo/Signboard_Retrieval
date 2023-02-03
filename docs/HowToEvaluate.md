## Evaluation

### Performance
#### Method 별 성능
|Method| Recall   | Precision | F1-score |
|------|----------|-----------|----------|
|SIFT| 0.42     | 0.55      | 0.48     |
|VIT| 0.82     | **0.80**  | **0.81**     |
|SIFT+VIT| 0.69     | 0.74      | 0.71     |
|SuperPoint + Superglue| 0.82     | 0.76      | 0.79     |
|LoFTR| **0.85** | 0.77      | **0.81** |


#### LoFTR 파라미터별 성능

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

#### 메모리 및 추론시간

resize_float = 0.8 (1600 * 800 px 크기 이미지)

| Inference time | Memory  |
|----------------|---------|
| 2.114s         | 11.75GB |

#### Evaluation 방법
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
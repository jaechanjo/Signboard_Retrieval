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


#### Superglue 파라미터별 성능

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

#### 메모리 및 추론시간

| Inference time | Memory |
|----------------|--------|
| 1.7s           | 2.53GB |

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
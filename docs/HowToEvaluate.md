## Evaluation

### 1. 파노라마 데이터셋
<table style="width:100%">
  <tr>
    <th>Dataset details</th>
    <th>Count</th>
  </tr>
  <tr>
    <td><b>total pairs</b></td>
    <td><b>97</b></td>
  </tr>
  <tr>
    <td>query cropped signs</td>
    <td>1,301</td>
  </tr>
  <tr>
    <td>db cropped signs</td>
    <td>1,229</td>
  </tr>
  <tr>
    <td>Matched pairs</td>
    <td>704</td>
  </tr>
  <tr>
    <td>Unmatched pairs</td>
    <td>597</td>
  </tr>
</table>

> 데이터셋 다운로드
  ```shell
     sh scripts/download.sh
  ```        
### 2. Method 별 성능
|Method| Recall   | Precision | F1-score |
|------|----------|-----------|----------|
|SIFT| 0.42     | 0.55      | 0.48     |
|VIT| 0.82     | **0.80**  | **0.81**     |
|SIFT+VIT| 0.69     | 0.74      | 0.71     |
|SuperPoint + Superglue| 0.82     | 0.76      | 0.79     |
|LoFTR| **0.85** | 0.77      | **0.81** |

### 3. SIFT, VIT 파라미터별 성능 평가

 <table style="width:100%">
  <tr>
    <th>Match threshold</th>
    <th>Recall@1</th>
    <th>Precsion@1</th>
  </tr>
  <tr>
    <td><b>1/4</b></td>
    <td><b>0.82</b></td>
    <td><b>0.80</b></td>
  </tr>
  <tr>
    <td>1/2</td>
    <td>0.69</td>
    <td>0.87</td>
  </tr>
  <tr>
    <td>3/4</td>
    <td>0.58</td>
    <td>0.92</td>
  </tr>
</table>
   
 <table style="width:100%">
  <tr>
    <th>Match threshold_1/4</th>
    <th>Recall@1</th>
    <th>Precsion@1</th>
  </tr>
  <tr>
    <td>SIFT</td>
    <td>0.42</td>
    <td>0.55</td>
  </tr>
  <tr>
    <td><b>VIT</b></td>
    <td><b>0.82</b></td>
    <td><b>0.80</b></td>
  </tr>
  <tr>
    <td>SIFT+VIT</td>
    <td>0.69</td>
    <td>0.74</td>
  </tr>
</table>

### 3. 추론 시간 (sec)

<table style="width:100%">
 <tr>
   <th>Model</th>
   <th>Crop</th>
   <th><b>Feature Extraction</b></th>
   <th>Merge</th>
   <th>Total</th>
 </tr>
 <tr>
   <td>SIFT</td>
   <td rowspan=3>0.051</td>
   <td>0.655</td>
   <td rowspan=3>0.006</td>
   <td rowspan=3><b>4.460</b></td>
  </tr>
  <tr>
    <td>VIT</td>
    <td><b>4.403</b></td>
  </tr>
</table>

처음 실행할 때, VIT 사전학습 가중치를 불러오기 때문에 시간이 더 걸릴 수 있습니다.
  > 가중치 이름 : swin_large_patch4_window12_384_22k.pth
  > 
  > 가중치 용량 : 886MB

### 4. 메모리 (MB)

- 파노라마 이미지 해상도 2000 * 1000 px

| 실행 전  | 실행 후  | Memory |
|----------|----------|--------|
| 319.95MB | 322.02MB | **2.07MB** |

전, 후 메모리는 로컬 환경에 따라 달라질 수 있습니다.

### 5. Evaluation 방법

추가 설정 파라미터는 다음 링크를 참조하세요. ([params.py](/config/params.py))

```shell
# 평가를 위해 예시 데이터셋 다운로드
sh scripts/download.sh

# evaluation
python3 eval.py\
--query_dir ./data/gt/query/\  # Query 파노라마 이미지 폴더 경로
--db_dir ./data/gt/db/\        # DB 파노라마 이미지 폴더 
```

```shell
# Best result
macro_mAP@1: 0.91
micro_mAP@1: 0.92
recall@1: 0.82
precision@1: 0.80
```

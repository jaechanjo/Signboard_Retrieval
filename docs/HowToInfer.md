## Inference

### Data preparation



```shell
data/sample/db_dir/db_img.jpg
data/sample/query_dir/query_img.jpg
```

### Inference 예시
```python
from main import SuperGlue

db_path = 'data/sample/1/19.jpg'
query_path = 'data/sample/1/21.jpg'
db_json = 'data/sample/1/19.json'
query_json = 'data/sample/1/21.json'
with open(db_json, 'r') as f:
    db_det = json.load(f)           # db 이미지 SignBoard Detection 결과
with open(query_json, 'r') as f:    # query 이미지 SignBoard Detection 결과
    query_det = json.load(f)

model = SuperGlue()
result = model.inference(db_path, query_path, db_det, query_det)

```

### 결과

결과의 ```db_box_index```는 db 이미지의 SCA-ObjectDetection 모듈 추론 결과에서의 index를 의미합니다.

결과의 ```query_box_index```는 query 이미지의 SCA-ObjectDetection 모듈 추론 결과에서의 index를 의미합니다.
```python
result = {
    'db_image': 'data/sample/1/19.jpg', 
    'query_image': 'data/sample/1/21.jpg', 
    'matches': [
        {
            'db_box_index': 6,          # db 이미지 SignBoard Detection 의 6번 결과
            'query_box_index': 5        # query 이미지 SignBoard Detection 의 5번 결과
        }, 
        {
            'db_box_index': 2, 
            'query_box_index': 6
         }, 
        {
            'db_box_index': 4, 
            'query_box_index': 9
        }
    ]
}
```

### Parameter setting
SuperGlue parameter 은 ```cfg/config/params.py``` 에서 조정할 수 있으며
용도에 맞게 ```match_threshold```, ``` k``` 값을 수정하면 됩니다.

parameter 값에 따른 성능은 [HowToEvaluate.md](https://github.com/sogang-mm/SCA-SignMatching/tree/superglue/docs/HowToEvaluate.md)
를 참고하시면 됩니다.

```python
superglue_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1000
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.4,          # 용도에 맞게 수정
            }
        }
k = 2                                            # 용도에 맞게 수정
```
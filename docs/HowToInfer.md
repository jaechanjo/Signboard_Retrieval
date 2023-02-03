## Inference

### Data preparation



```shell
data/sample/db_dir/db_img.jpg
data/sample/query_dir/query_img.jpg
```

### Inference 예시
```python
from main import LoFTR

db_path = 'data/sample/1/19.jpg'
query_path = 'data/sample/1/21.jpg'
db_json = 'data/sample/1/19.json'
query_json = 'data/sample/1/21.json'
with open(db_json, 'r') as f:
    db_det = json.load(f)
with open(query_json, 'r') as f:
    query_det = json.load(f)

# load model
model = LoFTR()
# inference
result = model.inference(db_path, query_path, db_det, query_det)
print(result)
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
용도에 맞게 ```'match_coarse': {'thr'}```, ``` k``` 값을 수정하면 됩니다.

```resize_float``` 변수를 작게 수정하면 메모리를 줄일 수 있으나 성능이 감소할 수 있습니다.

parameter 값에 따른 성능은 [HowToEvaluate.md](https://github.com/sogang-mm/SCA-SignMatching/tree/LoFTR/docs/HowToEvaluate.md)
를 참고하시면 됩니다.

```python
LoFTR_config = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        'attention': 'linear',
        'temp_bug_fix': False,
    },
    'match_coarse': {
        'thr': 0.4,                 # mathing point 의 confidence
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': True,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
    },
    'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear'},
}

k = 2                               # Bounding Box에 k개 이상의 matching point시 같은 간판으로 간주함.

resize_float = 0.8                                  # 용도에 맞게 수정
```
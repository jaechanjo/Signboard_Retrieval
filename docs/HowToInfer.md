## Inference

모든 파일은 `${WORKSPACE}/SCA-SignMatching/` 폴더에서 실행합니다.


추가 설정 파라미터는 다음 링크를 참조하세요. ([params.py](/config/params.py))

### 1. Data preparation

  - 파노라마 이미지와 간판 detector 모듈의 결과인 json 파일을 같은 경로에 위치시킨다.
  - 이미지와 json 파일은 같은 이름을 가지도록 한다.
        
    ```shell
    data/sample/db/db_img.jpg
    data/sample/db/db_img.json
    data/sample/query/query_img.jpg
    data/sample/query/query_img.json
    ```

### 2. Run python

```shell
python3 main.py\
--query_path ./data/sample/query/400@230124.jpg\        # query 파노라마 이미지 경로
--db_path ./data/sample/db/400@190124.jpg\              # db 파노라마 이미지 경로
--query_json_path ./data/sample/query/400@230124.json\  # query 간판 탐지 json 경로
--db_json_path ./data/sample/db/400@190124.json\        # db 간판 탐지 json 경로
-vis/ --visualize                                       # 결과 매칭 이미지를 저장할지 여부
```

### 3. Import class

```python
from main import SIFT_VIT

query_path = './data/sample/query/400@230124.jpg'
db_path = './data/sample/db/400@190124.jpg'
query_json_path = './data/sample/query/400@230124.json'
db_json_path = './data/sample/db/400@190124.json'

#load json
with open(query_json_path, 'r') as qj:
    query_det = json.load(qj)
with open(db_json_path, 'r') as dbj:
    db_det = json.load(dbj)

#load model
model = SIFT_VIT()

#inference
result_dict, result_json = model.inference(opt.query_path, opt.db_path, query_det, db_det)
```
```shell
#sample result
result_dict = {'400-400': {'0': ['23'], '1': ['18'], '2': [], '3': ['28'], '4': ['26'], '5': ['26'],
                '6': ['15'], '7': [], '8': ['22'], '9': ['6'], '10': [], '11': ['8'], '12': ['26'],
                '13': ['7'], '14': ['9'], '15': ['25'], '16': ['11'], '17': ['12'], '18': ['13'],
                '19': [], '20': ['16'], '21': [], '22': [], '23': ['27'], '24': ['19'], '25': ['14'],
                '26': ['29'], '27': ['31'], '28': [], '29': ['30'], '30': ['32']}}

# '400-400' : "(query_id)-(db_id)"
# '0', '1', '2' ... : cropped 간판 인덱스
# [] : 매칭이 안된 경우


result_json = {'db_image': './data/sample/db/400@190124.jpg', 
               'query_image': './data/sample/query/400@230124.jpg', 
               'matches': [
                           {'db_box_index': 23, 'query_box_index': 0}, 
                           {'db_box_index': 18, 'query_box_index': 1}, 
                           {'db_box_index': 28, 'query_box_index': 3}, 
                           {'db_box_index': 26, 'query_box_index': 4},
                            ... ,
                           {'db_box_index': 32, 'query_box_index': 30}
                          ]
              }

# 'db_image' is db 이미지 경로
# 'query_image' is query 이미지 경로
# 'matches' is 매칭된 간판 쌍 리스트
```

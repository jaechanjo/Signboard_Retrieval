## Inference

All commands should be executed within the `Signboard_Retrieval/` subfolder

For additional hyperparameters, see the following file. [params.py](./config/params.py)

### 1. Run Python

```shell
python3 main.py\
--query_path ./data/sample/query/400@230124.jpg\  # query panorama image path
--db_path ./data/sample/db/400@190124.jpg\        # db panorama image path
-vis/ --visualize       # Whether to save the resulting image
```

### 2. Import Class

```python
from main import SIFT_VIT

query_path = './data/sample/query/400@230124.jpg'
db_path = './data/sample/db/400@190124.jpg'

#load model
model = SIFT_VIT()

#inference
result_dict, result_json = model.inference(opt.query_path, opt.db_path)
```

```shell
#sample result

result_dict = {'400-400': {'0': ['23'], '1': ['18'], '2': [], '3': ['28'], '4': ['26'], '5': ['26'],
                '6': ['15'], '7': [], '8': ['22'], '9': ['6'], '10': [], '11': ['8'], '12': ['26'],
                '13': ['7'], '14': ['9'], '15': ['25'], '16': ['11'], '17': ['12'], '18': ['13'],
                '19': [], '20': ['16'], '21': [], '22': [], '23': ['27'], '24': ['19'], '25': ['14'],
                '26': ['29'], '27': ['31'], '28': [], '29': ['30'], '30': ['32']}}

# '400-400' is "(query_id)-(db_id)"
# '0', '1', '2' ... is cropped sign index
# [] means unmatched pairs

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

# 'db_image' is db_image_path
# 'query_image' is query_image_path
# 'matches' is list of matched box index dictionary
```
For more detailed instructions, see the instructions in the [main_guide.ipynb](./main_guide.ipynb)

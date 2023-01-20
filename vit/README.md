# 간판 특징 추출 엔진

### Dataset

파노라마 뷰 이미지 

### Learning based feature extractor s**ystem**

![Image_Retrieval_System](./imgs/image_retrieval_system.png)

```bash
#1. Extract Image Feature
python feature_extractor.py

#2. Get retrieval ranking
python rank.py
```

### Argument

--result_path=ViT로 inference한 간판 매칭 결과 저장 경로

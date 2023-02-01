## Setup

### 1. Docker compose

```shell
cd ${WORKSPACE}  # directory for git clone
git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/jaechanjo/Signboard_Retrieval.git
cd Signboard_Retrieval
docker-compose up -d  # build docker container
```

### 2. Packages

```shell
cat requirements.txt | while read PACKAGE; do pip install "$PACKAGE"; done  # ignore error of install version 
```

### 3. File Tree

```shell
${WORKSPACE}/Signboard_Retrieval/
├─data
│  ├─result
│  │  ├─sift_best_pair    # sift matching result txt
│  │  ├─visualization     # image of boxes and lines connected
│  │  └─vit_best_pair     # vit matching result txt
│  └─sample
│      ├─db               # sample db
│      └─query            # sample query
├─docker
├─docs
│  ├─images
│  └─result
├─models                  # Matching Models
│  ├─sift_vlad
│  │  └─utils
│  ├─vit
│  │    └─utils
│  └─ocr                  # try& error
│      ├─ ...
│      └─utils
├─scripts                 # script for downloading sample dataset
└─utils                   # utility for module
```

### 4. Data preparation

  - Place the panoramic image and the json file that is the result of the signboard detector module in the same path.
  - The image and json file must have the same name.
        
    ```shell
    data/sample/db/db_img.jpg
    data/sample/db/db_img.json
    data/sample/query/query_img.jpg
    data/sample/query/query_img.json
    ```

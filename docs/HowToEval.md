## Evaluation

### 1. Panorama datasets
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

> datasets download
  ```shell
     # (optional) it can be used for evaluation
     sh scripts/download.sh
  ```        
  
### 2. Performace evaluation
- According to the match threshold, recall and precision of top1
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
   
- When the best match threshold is 1/4, depending on the model
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

### 3. Time (sec)

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

Please understand that it will take some time because the weight file is downloaded at first.
  > weight : swin_large_patch4_window12_384_22k.pth (886MB)

### 4. Memory (MB)

- resolution of panorama image is 2000 * 1000 px

| Before   | After    | Memory |
|----------|----------|--------|
| 319.95MB | 322.02MB | **2.07MB** |

The before and after memory may vary depending on the local environment.

### 5. Validation on labeled dataset

```shell
# download eval dataset
sh scripts/download.sh

# evaluation
python3 eval.py\
--query_dir ./data/gt/query/\  # query panorama image directory
--db_dir ./data/gt/db/\        # db panorama image directory
```

For more detailed instructions, see the instructions in the [eval_guide.ipynb](./eval_guide.ipynb)

```shell
# Best result
macro_mAP@1: 0.91
micro_mAP@1: 0.92
recall@1: 0.82
precision@1: 0.80
```

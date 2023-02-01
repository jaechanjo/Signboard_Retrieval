result_path='./data/result/' # directory of saving results

topk=1                       # the number of matching candidates

match_weight='1/4'           # threshold of whether matched or not

method='vit'                 # module, ['vit', 'sift', 'vit_sift', 'sift_vit']

algo='max'                   # matching algorithm, ['max', 'erase']

batch_size=64                # the batch size extracting vit feature

num_workers=0                # dataset load multi-processing
def merge_topk_val(result_path, q_panorama_id, db_panorama_id, topk, match_weight=1/4, method='vit', algo='max'):
    
    '''
    topk: int, the number of matching candidates
    match_weight: float, threshold of whether matched or not
    result_path: str, folder path saving results, './data/result/'
    method: str, what you want to use, ['vit', 'sift', 'vit_sift', 'sift_vit']
    algo: str, matching algorithm, ['max', 'erase']
        - 'max' : matching pairs for maximizing score
        - 'erase' : matching pairs removing top1 prediction sequentially.
    '''
    
    #merge
    merge_dict = {}
        
    for m in method.split('_'):
        try:
            with open(f"{result_path}/eval/{m}_best_pair/pair_{q_panorama_id}-{db_panorama_id}_{m}.txt", "r") as f:
                for line in f.readlines():
                    data = line.strip().split(',')
                    img1 = data[0].split('.')[0]
                    img2 = data[1].split('.')[0]
                    score = float(data[2]) #cosine_similarity

                    if img1+'-'+img2 not in merge_dict:
                        merge_dict[img1+'-'+img2] = score
                    else:
                        merge_dict[img1+'-'+img2] += score
        except:
            print(f"method {m} error")
    
    #change form
    result_dict = {}
    
    for key in merge_dict:
        imgs = key.split('-')
        q = imgs[0]
        db = imgs[1]
        del imgs
        score = merge_dict[key]
        
        if q not in result_dict: #query emege at first
            result_dict[q] = [(db, score)]
        else: #query emege repeatedly
            db_list = result_dict[q]
            db_list.append((db, score))
            result_dict[q] = db_list
    
    #Matching Algorithm
    if algo == 'max':
    
        result_topk = {}

        for q in result_dict.keys():

            #use match score sorted
            topk_list = sorted(result_dict[q], key=lambda x: -x[1])[:topk] #(db, score)
            result_topk[q] = topk_list
            
    else: #algo == 'erase'
        result_topk = {}
        erase_list = []

        for q in result_dict.keys():

            #erase top1 matching
            erased_list = [(db, score) for (db, score) in result_dict[q] if db not in erase_list]

            if len(erased_list) != 0: #not empty
                #use match score sorted
                topk_list = sorted(erased_list, key=lambda x: -x[1])[:topk] #(db, score)
                result_topk[q] = topk_list

                #add erased top1 matched db, not score
                erase_list.append(topk_list[0][0])
            else: #empty -> score maximization
                topk_list = sorted(result_dict[q], key=lambda x: -x[1])[:topk] #(db, score)
                result_topk[q] = topk_list
    
    
    ############ evaluation ################
    #ex) query_name : '0_100_False'/ db_name : '2_100_False'
    ########################################
        
    #matched pair score
    matched_score = []
    #unmatched pair score
    unmatched_score = []

    #panorama mAP : if there is in topk, +1/ if not 0 -> ap per panorama
    pap = 0
    #crop mAP : ap per 1 sign crop
    ap = 0

    for q_name in list(result_topk.keys()):
        #query_label
        q_lbl = int(q_name.split('_')[1])

        for rank, (db_name, score) in enumerate(result_topk[q_name], start=1): #only db_name
            #db_label
            db_lbl = int(db_name.split('_')[1])

            if (q_lbl == db_lbl) & (q_lbl != 100): #('False' in q_name) # Count #100 not matched / + except change

                #if matched, record score
                matched_score.append(score)

                #panorama mAP
                pap += 1
                #crop mAP
                qap = 1/rank
                ap += qap
            else: #unmatched

                unmatched_score.append(score)

    #crop mAP
    matched_cnt = len([q_name for q_name in result_topk.keys() if ('100' not in q_name)]) #& ('True' not in q_name)]) ##Divide #except unmatched '100' / + except change

    #panorama mAP
    try:
        pap/=matched_cnt
        print(f"macro AP@{topk} is {pap}") # check map per panorama

    except:
        if matched_cnt == 0:
            print(f"there is no gt, pass!")
        else:
            print("stop! - what happen now?")
            
    return result_topk, pap, (ap, matched_cnt), (matched_score, unmatched_score)
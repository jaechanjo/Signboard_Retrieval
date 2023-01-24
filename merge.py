#merge sift & vit
def merge_topk(result_path, panorama_id, topk, match_weight, method='vit', algo='max'):
    
    '''
    result_path: str, folder path saving results, './match_score/'
    method: str, what you want to use, ['vit', 'sift', 'vit_sift', 'sift_vit']
    algo: str, which you want to use for matching algorithm, ['max', 'erase']
        - 'max' : matching pairs for maximizing score
        - 'erase' : matching pairs removing top1 prediction sequentially.
    '''
    
    #merge
    merge_dict = {}
        
    for m in method.split('_'):
        try:
            with open(f"{result_path}/{m}_best_pair/pair_{panorama_id}_{m}.txt", "r") as f:
                for line in f.readlines():
                    data = line.strip().split(',')
                    img1 = data[0].split('.')[0]
                    img2 = data[1].split('.')[0]
                    score = float(data[2]) #cosine_similarity

                    if img1+'-'+img2 not in merge_dict: #3가지 방법에서 특정 q_db pair가 중복된다면, 점수를 합산!
                        merge_dict[img1+'-'+img2] = score
                    else:
                        merge_dict[img1+'-'+img2] += score
        except:
            raise Exception(f"method {m} error")
    
    #change form
    result_dict = {}
    
    for key in merge_dict:
        imgs = key.split('-')
        q = imgs[0]
        db = imgs[1]
        del imgs
        score = merge_dict[key]
        
        #특정 query에 대해, 모든 ref를 list에 모아주기
        if q not in result_dict: #이 query가 처음 나왔을 때
            result_dict[q] = [(db, score)]
        else: #query가 중복될 경우
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

    #match or not
    match_thres_dict = {'vit' : [0.34818036103100775, 0.18382692764634148], 'sift' : [0.24703212600804292, 0.14514607468375135], \
                        'vit_sift' : [0.5255381287510346, 0.28420164151355065], 'sift_vit' : [0.5255381287510346, 0.28420164151355065]}        
    
    # magic line for distinguish 'matched' from 'unmatched'
    match_thres = match_weight*match_thres_dict[method][0] + (1-match_weight)*match_thres_dict[method][1]

    # final result dict
    final_result_dict = {}
    
    final_dict = {}

    for q in result_topk.keys():
            
        thres_list = []

        for rank, (db, score) in enumerate(result_topk[q], start=1): #only db_name

            if score > match_thres: # positve : predict top 1 db

                thres_list.append(db)    
            else: # negative : predict empty list []
                pass
            
        final_dict[q] = thres_list
    
    final_result_dict[panorama_id] = final_dict
    
    #done
    print(f"\nMerge...\nresult_dict\nDone.")
    
    return final_result_dict

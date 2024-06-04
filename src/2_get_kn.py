'''
Với mỗi relation e.g. "P30", tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
- Input: một tập tin .rlt.jsonl e.g. "TREx-all-P30.rlt.jsonl" được tạo ra bởi "/src/1_analyze_mlm.py"
- Output: bốn tập tin .json lưu trữ vị trí (l, i) của các nơ-ron tri thức tại đường dẫn "../results/kn/"
'''

import jsonlines, json
import numpy as np
from collections import Counter
import os
import time


threshold_ratio = 0.2
mode_ratio_bag = 0.7
mode_ratio_rel = 0.1
kn_dir = '/content/drive/MyDrive/Thesis/code_research/results/kn/'
rlts_dir = '/content/drive/MyDrive/Thesis/code_research/results/'


def re_filter(metric_triplets):
    '''
    (list) -> (list)

    Description:
    metric_triplets: [[2, 3071, 0.003070291830226779], [3, 1935, 0.0019019835162907839],...]
    
    - Lọc điểm phân bổ của các nơ-ron với prompt tương ứng sử dụng threshold t (xem thêm ở mục 4.1 main paper). Input và output có CÙNG size.
    '''

    # find a max attribution score
    metric_max = -999
    for i in range(len(metric_triplets)):
        metric_max = max(metric_max, metric_triplets[i][2])
    
    # filter with threshold t (t = 0.2*max from main paper)
    metric_triplets = [triplet for triplet in metric_triplets if triplet[2] >= metric_max * threshold_ratio]
    return metric_triplets


def pos_list2str(pos_list):
    '''
    (list) -> (str)

    Description:
    pos_list: e.g. [2, 3071]
    
    - Chuyển đổi list đầu vào thành chuỗi (string) với các phần tử được ngăn cách bởi '@'.

    Example:
    >>> pos_list2str([2, 3071])
    '2@3071'
    '''

    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    '''
    (str) -> (list)

    Description:
    pos_str: e.g. '2@3071'
    
    - Chuyển đổi chuỗi 'l@i' (l: chỉ số layer, i: chỉ số nơ-ron ở layer tương ứng) đầu vào thành list số nguyên tương ứng.

    Example:
    >>> pos_str2list('2@3071')
    [2, 3071]
    '''

    return [int(pos) for pos in pos_str.split('@')]


def parse_kn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
    '''
    (collections.Counter, int, int, int) -> (list)

    Description:
    pos_cnt: bộ đếm vị trí (l, i) e.g. Counter({'2@3071': 2, '1@3071': 1, ...})
    tot_num: số lượng prompt trong template prompt đang xét hoặc số lượng template prompt trong relation
    mode_ratio: thresh hold p% e.g. 0.7
    
    - Lọc các nơ-ron tri thức được chia sẻ bởi p% số lượng prompt thuộc template prompt tương ứng sử dụng kết quả của bộ đếm "Counter.update()" từ thư viện "collections".
    '''

    # define number of prompt for filtering based on p%
    mode_threshold = tot_num * mode_ratio
    mode_threshold = max(mode_threshold, min_threshold)
    
    # filter with threshold p%
    kn_bag = []
    for pos_str, cnt in pos_cnt.items():
        if cnt >= mode_threshold:
            kn_bag.append(pos_str2list(pos_str))
    return kn_bag # kn_bag [[2, 3071], [1, 3071], ...]


def analysis_file(filename, metric='ig_gold'):
    '''
    (str, str) -> (int, list, list)

    Description:
    filename: tên tập tin (.rlt.jsonl) chứa điểm phân bổ được tạo từ /src/1_analyze_mlm.py e.g. TREx-all-P30.rlt.jsonl
    metric: tên phương pháp có điểm phân bổ cần lọc (baseline, ig_gold...) thuộc "res_dict" e.g. 'ig_gold', 'base'

    - Tiến hành lọc các nơ-ron tri thức ở tập tin dữ liệu đầu vào với (STEP 3) threshold t (thực hiện trên mỗi prompt) cùng với (STEP 4.1) threshold p% (thực hiện trên các prompt của cùng một template prompt) và (STEP 4.2) threshold p'% (thực hiện tương đối với số lượng template prompt trong relation).
    '''

    # Take relation name and print out
    rel = filename.split('.')[0].split('-')[-1] # e.g. P30
    print(f'===========> parsing important position in {rel}-{metric}..., mode_ratio_bag={mode_ratio_bag}')

    # Read data file
    rlts_bag_list = []
    with open(os.path.join(rlts_dir, filename), 'r') as fr:
        for rlts_bag in jsonlines.Reader(fr): # read .rlt.jsonl file line by line (list by list)
            rlts_bag_list.append(rlts_bag)

    ave_kn_num = 0
    kn_bag_list = []

    # Get imp pos by bag_ig
    # loop considers each template prompt (tuple) 
    for bag_idx, rlts_bag in enumerate(rlts_bag_list): # refer to /Drive/code_research/printed/rlts_bag_list.txt
        pos_cnt_bag = Counter()
        # loop considers each triple (prompt)
        for rlt in rlts_bag:
            res_dict = rlt[1] # rlt is [tokens_info, res_dict] from /src/1_analyze_mlm.py
            # (STEP 3) filter with threshold t
            metric_triplets = re_filter(res_dict[metric])
            # count knowledge neuron by indices l, i
            for metric_triplet in metric_triplets: # metric_triplet [l, i, W(i)l]
                pos_cnt_bag.update([pos_list2str(metric_triplet[:2])]) # e.g. pos_cnt_bag.update(['2@3071']) - specific neuron (l, i) is counted minimum 0 and maximum 1 for each prompt

        # (STEP 4.1) filter with threshold p% (for the specified template prompt)
        kn_bag = parse_kn(pos_cnt_bag, len(rlts_bag), mode_ratio_bag, 3) # kn_bag [[2, 3071], [1, 3071], ...]
        ave_kn_num += len(kn_bag) # number of kn for each template prompt
        kn_bag_list.append(kn_bag) # kn_bag_list [[template[prompt]], [template[prompt]]] - size similar to .rlt.jsonl input file

    ave_kn_num /= len(rlts_bag_list) # denominator is number of template prompt in the relation

    # Get imp pos by rel_ig
    pos_cnt_rel = Counter()
    for kn_bag in kn_bag_list:
        for kn in kn_bag:
            pos_cnt_rel.update([pos_list2str(kn)])
    kn_rel = parse_kn(pos_cnt_rel, len(kn_bag_list), mode_ratio_rel) # (STEP 4.2) filter with threshold p'% # kn_rel [[2, 3071], [1, 3071], ...] 

    return ave_kn_num, kn_bag_list, kn_rel


def stat(data, pos_type, rel):
    '''
    (list, str, str) -> print

    Description:
    data: e.g. template prompt scope - [[template[prompt]], [template[prompt]]], relation scope - [[2, 3071], [1, 3071], ...]
    pos_type: e.g. 'kn_bag', 'kn_rel'
    rel: e.g. 'P30'

    - Tính toán và in ra màn hình số lượng nơ-ron tri thức trong phạm vi template prompt hoặc relation.   

    Example (kết quả in ra màn hình):
    P30's kn_bag has on average 4.313846153846153 imp pos. 
    P30's kn_rel has 11 imp pos.
    '''

    if pos_type == 'kn_rel':
        print(f'{rel}\'s {pos_type} has {len(data)} imp pos. ')
        return
    
    ave_len = 0
    for kn_bag in data:
        ave_len += len(kn_bag)
    ave_len /= len(data)
    print(f'{rel}\'s {pos_type} has on average {ave_len} imp pos. ')

# create "../results/kn/"" directory
if not os.path.exists(kn_dir):
    os.makedirs(kn_dir)
# MAIN process
for filename in os.listdir(rlts_dir): # loop lists all items in "../results/"
    if filename.endswith('.rlt.jsonl'): # just considers .rlt.jsonl
        # record running time
        tic = time.perf_counter()
        # ig_gold
        # 1. filter
        threshold_ratio = 0.2
        mode_ratio_bag = 0.7
        for max_it in range(6):
            ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename)
            if ave_kn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_kn_num > 5:
                mode_ratio_bag += 0.05
            if ave_kn_num >= 2 and ave_kn_num <= 5:
                break
        rel = filename.split('.')[0].split('-')[-1] # e.g. 'P30'
        # 2. print out statistic
        # stat(kn_bag_list, 'kn_bag', rel)
        # stat(kn_rel, 'kn_rel', rel)
        # 3. write file
        with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'w') as fw:
            json.dump(kn_bag_list, fw, indent=2)
        with open(os.path.join(kn_dir, f'kn_rel-{rel}.json'), 'w') as fw:
            json.dump(kn_rel, fw, indent=2)

        # baseline
        # 1. filter
        threshold_ratio = 0.5
        mode_ratio_bag = 0.7
        for max_it in range(6):
            ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename, 'base')
            if ave_kn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_kn_num > 5:
                mode_ratio_bag += 0.05
            if ave_kn_num >= 2 and ave_kn_num <= 5:
                break
        rel = filename.split('.')[0].split('-')[-1]
        # 2. print out statistic
        # stat(kn_bag_list, 'kn_bag', rel)
        # stat(kn_rel, 'kn_rel', rel)
        # 3. write file
        with open(os.path.join(kn_dir, f'base_kn_bag-{rel}.json'), 'w') as fw:
            json.dump(kn_bag_list, fw, indent=2)
        with open(os.path.join(kn_dir, f'base_kn_rel-{rel}.json'), 'w') as fw:
            json.dump(kn_rel, fw, indent=2)

        # record running time
        toc = time.perf_counter()
        print(f"***** {filename} KNs filtered. Costing time: {toc - tic:0.4f} seconds *****")
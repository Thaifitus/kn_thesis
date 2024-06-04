'''
Chương trình thống kê nơ-ron tri thức dựa trên kết quả từ "/src/2_get_kn.py". Tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
- Input: các tập tin .json chứa nơ-ron tri thức tương ứng với từng quan hệ từ "/src/2_get_kn.py" ở đường dẫn "/results/kn".
- Output: một (hai gồm baseline và IG) tập tin .pdf trực quan sự phân bố các nơ-ron tri thức và các thông tin thống kê được in ra màn hình.
'''

import json
import numpy as np
import os
from collections import Counter
from matplotlib import pyplot as plt
import random

kn_dir = '../results/'
fig_dir = '../results/figs/'

# =========== stat kn_bag ig ==============

# y_points = []
tot_bag_num = 0
tot_kneurons = 0
kn_bag_counter = Counter()
for filename in os.listdir(kn_dir): # count number of kn in each "kn_bag-{rel}.json" file
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f) # read all of "kn_bag-{rel}.json" file with data format [[template[kn]], [template[kn]]]
        for kn_bag in kn_bag_list: # loop considers each bag (template prompt)
            for kn in kn_bag: # loop considers each kn
                kn_bag_counter.update([kn[0]]) # count a kn's layer
                # y_points.append(kn[0])
        tot_bag_num += len(kn_bag_list) # number of bag
for k, v in kn_bag_counter.items():
    tot_kneurons += kn_bag_counter[k]
for k, v in kn_bag_counter.items():
    kn_bag_counter[k] /= tot_kneurons # calculate percentage of each layer's kn (used for visualization)
# average # Kneurons
print('average ig_kn', tot_kneurons / tot_bag_num) # average kn of each relational fact (based on all relations)

# =========== stat kn_bag base ==============
# (process is similar to the above "stat kn_bag ig" process)

tot_bag_num = 0
tot_kneurons = 0
base_kn_bag_counter = Counter()
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                base_kn_bag_counter.update([kn[0]])
        tot_bag_num += len(kn_bag_list)
for k, v in base_kn_bag_counter.items():
    tot_kneurons += base_kn_bag_counter[k]
for k, v in base_kn_bag_counter.items():
    base_kn_bag_counter[k] /= tot_kneurons
# average # Kneurons
print('average base_kn', tot_kneurons / tot_bag_num)

# =========== plot knowledge neuron distribution ===========

if not os.path.exists(fig_dir): # create '../results/figs/'
    os.makedirs(fig_dir)

# IG METHOD
plt.figure(figsize=(8, 3))

x = np.array([i + 1 for i in range(12)])
y = np.array([kn_bag_counter[i] for i in range(12)])
plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i + 1 for i in range(12)], labels=[i + 1 for i in range(12)], fontsize=20)
plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=18)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=18)
plt.ylim(-y.max() - 0.03, y.max() + 0.03)
plt.xlim(0.3, 12.7)
bottom = -y
y = y * 2
plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, 'kneurons_distribution.pdf'), dpi=100)
plt.close()

# BASELINE
plt.figure(figsize=(8, 3))

x = np.array([i + 1 for i in range(12)])
y = np.array([base_kn_bag_counter[i] for i in range(12)])
plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i + 1 for i in range(12)], labels=[i + 1 for i in range(12)], fontsize=20)
plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=18)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=18)
plt.ylim(-y.max() - 0.03, y.max() + 0.03)
plt.xlim(0.3, 12.7)
bottom = -y
y = y * 2
plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, 'kneurons_distribution-base.pdf'), dpi=100)
plt.close()


# ========================================================================================
#                       knowledge neuron intersection analysis
# ========================================================================================

def cal_intersec(kn_bag_1, kn_bag_2):
    '''
    (list, list) -> (int)

    Description:
    kn_bag_1, kn_bag_2: nơ-ron tri thức với định dạng [template[kn]] (xem chi tiết ở /results/kn/kn_bag-{rel}.json)

    - Tìm số lượng những nơ-ron tri thức xuất hiện ở cả hai kn bag.
    '''

    kn_bag_1 = set(['@'.join(map(str, kn)) for kn in kn_bag_1]) # e.g. kn_bag_1 {'9@1944', '8@1944','l@i'...}
    kn_bag_2 = set(['@'.join(map(str, kn)) for kn in kn_bag_2])
    return len(kn_bag_1.intersection(kn_bag_2))
    
# (this "ig kn" process and the below "base kn" process is similar. Authors duplicated the code then changed directory and printed messages)
# ====== load ig kn =======

kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir): # loop considers each "kn_bag-{rel}.json" file
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f) # read all of "kn_bag-{rel}.json" file with data format [[template[kn]], [template[kn]]]
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list
    # kn_bag_list_per_rel: {'P30': [[template[kn]], [template[kn]]], ...,
    #                       'P24': [[template[kn]], [template[kn]]], ...}

# ig inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    # print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list) # number of bag (template prompt)
    for i in range(0, len_kn_bag_list): # considers one bag with the others
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i] # [template[kn]]
            kn_bag_2 = kn_bag_list[j] # [template[kn]]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean() # average number of kn shared by two bag (template prompt) in the same relation
print(f'ig kn has on average {inner_ave_intersec} inner kn interseciton')

# ig inter
# this process is similar to the above "ig inner" process excepts (1) number of kn_bag_2 is 100 (compared in 100 times) (2) kn_bag_2 is random choice
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    # print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(0, 100):
            kn_bag_1 = kn_bag_list[i]
            other_rel = random.choice([x for x in kn_bag_list_per_rel.keys() if x != rel])
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inter_ave_intersec.append(num_intersec)
inter_ave_intersec = np.array(inter_ave_intersec).mean() # average number of kn shared by two bag (template prompt) in two relations
print(f'ig kn has on average {inter_ave_intersec} inter kn interseciton')


# ====== load base kn =======
kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list

# base inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    # print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i]
            kn_bag_2 = kn_bag_list[j]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'base kn has on average {inner_ave_intersec} inner kn interseciton')

# base inter
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    # print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(0, 100):
            kn_bag_1 = kn_bag_list[i]
            other_rel = random.choice([x for x in kn_bag_list_per_rel.keys() if x != rel])
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inter_ave_intersec.append(num_intersec)
inter_ave_intersec = np.array(inter_ave_intersec).mean()
print(f'base kn has on average {inter_ave_intersec} inter kn interseciton')

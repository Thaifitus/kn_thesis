"""
BERT MLM runner
"""

'''
Chương trình sử dụng hàm eval_modification() thực hiện khảo sát ở mục 4.5 (suppressing, amplifying) thuộc bài báo "Knowledge Neurons in Pretrained Transformers". Tập tin đầu vào và đầu ra của chương trình bao gồm:
- Input: các tập tin "base_kn_bag-{rel}.json" hoặc "kn_bag-{rel}.json" là kết quả từ tập tin "2_get_kn.py"
- Output: tập tin "3_modify_activation.args.json" chứa cấu hình của chương trình; tập tin kết quả "{prefix}modify_activation_rlt.json" với prefix là "base_" hoặc "" chứa sự chênh lệch xác xuất dự đoán từ [MASK] là ground truth (gold label) tương ứng
'''

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import time

import transformers
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    '''
    (list,int, object) -> (dict, dict)

    Description:
    example (a triple): ['Mount Markham is a part of the continent of [MASK].', 'Antarctica', 'P30(continent)']
    max_seq_length: 128
    tokenizer: BertTokenizer.from_pretrained
    - Với các biến "ori_tokens", "segment_ids", "input_ids' tham khảo /Drive/Thesis/code_research/printed/Example2Feature_Output.txt cho giá trị cụ thể được in ra.

    Step 1: 
    - Sử dụng phương thức tokenize của BERT để chuyển chuỗi câu của example[0] thành list tokens
    - Sau khi có được list tokens, thực hiện kiểm tra lại độ dài của list, nếu vượt quá max_seq_length - 2 thì sẽ giảm bớt

    Step 2:
    - Tạo list tokens bằng cách thêm những kí tự đặc biệt ([CLS] vào đầu câu, [SEP] vào cuối câu) trên ori_tokens
    - Tạo base_tokens có độ dài bằng (len(ori_token) + 2) chứa tokens [UNK]
    - Tạo segment_ids có độ dài bằng với độ dài list tokens chứa phần tử 0

    Step 3:
    - Thực hiện chuyển list tokens, base_tokens thành dạng list các số nguyên tương ứng với phương thức convert_tokens_to_ids
    - Tạo list input_mask chứa phần tử 1 có độ dài bằng với input_ids (==tokens)

    Step 4:
    - Thêm padding cho các giá trị input_ids, baseline_ids, segment_ids, input_mask
    Step 5: 
    - Khởi tạo 2 dict features và tokens_info
    - dict features chứa các key và value của input_ids, baseline_ids, segment_ids, input_mask
    - dict tokens_info chứa các key và value của tokens, example[2], example[1], pred_label=None

    Example:
    >>> eval_features, tokens_info = example2feature(['Mount Markham is a part of the continent of [MASK].', 'Antarctica', 'P30(continent)'], 128, tokenizer)
    >>> print(eval_features)
    {'input_ids': [101, 3572, 2392, 2522, 1110, 170, 1226, 1104, 1103, 10995, 1104, 103, 
                  119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0],
    'input_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0],
    'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'baseline_ids': [100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    >>> print(tokens_info)
    {'tokens': ['[CLS]', 'Mount', 'Mark', '##ham', 'is', 'a', 'part', 
                 'of', 'the', 'continent', 'of', '[MASK]', '.', '[SEP]'],
    'relation': 'P30(continent)',
    'gold_obj': 'Antarctica', 
    'pred_obj': None}
    '''

    """Convert an example into input features"""   
    # Step 1
    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]
    
    # Step 2
    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)
    
    # Step 3
    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)
    
    # Step 4
    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    # Step 5
    features = {
        'input_ids': input_ids, # ids of tokens (CLS-SEP added)
        'input_mask': input_mask, # list of 1
        'segment_ids': segment_ids, # list of 0 (len==len(input_mask))
        'baseline_ids': baseline_ids, # ids of [UNK] tokens
    }
    tokens_info = {
        "tokens":tokens,
        "relation":example[2],
        "gold_obj":example[1],
        "pred_obj": None
    }

    return features, tokens_info


def main():
    '''
    Function's procedures:
    parse arguments ⭢ set device ⭢ set random seeds ⭢ save program args ⭢ init tokenizer ⭢ Load pre-trained BERT ⭢ (data parallel) ⭢ prepare eval set (generate data_all_allbags.json) ⭢ eval_modification()
    '''

    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--tmp_data_path",
                        default=None,
                        type=str,
                        help="Temporary input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--kn_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")
    parser.add_argument("--output_prefix",
                        default='',
                        type=str,
                        help="The output prefix to indentify each running of experiment. ")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--debug",
                        type=int,
                        default=-1,
                        help="How many examples to debug. -1 denotes no debugging")

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    # os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, '3_modify_activation.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load pre-trained BERT
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    if os.path.exists(args.tmp_data_path): # read data_all_allbags.json
        with open(args.tmp_data_path, 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else: # create data_all_allbags.json
        with open(args.data_path, 'r') as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(args.tmp_data_path, 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)


    def eval_modification(prefix=''):
        '''
        (str) -> tập tin {prefix}modify_activation_rlt.json

        Description:
        prefix: loại phương pháp xác định nơ-ron tri thức e.g. '' hoặc 'base_'.

        Mỗi relation có 2 file kn_bag (base và ig):
        - Xử lý với tập tin kn_bag của mỗi relation với "prefix" tương ứng.
        - Với mỗi kn_bag, loop cho mỗi prompt thuộc template tương ứng: ori_gold_prob ⭢ int_gold_prob (remove) ⭢ int_gold_prob (enhance). (xem hình minh họa tại /Drive/thesis/code_research/printed/eval_modification.jpg)
        - Tính giá trị trung bình và giá trị ratio.
        '''

        rlt_dict = {}
        for filename in os.listdir(args.kn_dir):
            if not filename.startswith(f'{prefix}kn_bag-'): # "base_kn_bag-{rel}.json" or "kn_bag-{rel}.json"
                continue
            relation = filename.split('.')[0].split('-')[-1]
            save_key = filename.split('.')[0]
            print(f'calculating {prefix}relation {relation} ...')
            rlt_dict[save_key] = {
                'own:ori_prob': [],
                'rm_own:ave_delta': [],
                'rm_own:ave_delta_ratio': None,
                'eh_own:ave_delta': [],
                'eh_own:ave_delta_ratio': None,
                'oth:ori_prob': [],
                'rm_oth:ave_delta': [],
                'rm_oth:ave_delta_ratio': None,
                'eh_oth:ave_delta': [],
                'eh_oth:ave_delta_ratio': None
            }
            with open(os.path.join(args.kn_dir, filename), 'r') as fr: # read file "base_kn_bag-{rel}.json" or "kn_bag-{rel}.json"
                kn_bag_list = json.load(fr)
            # record running time
            tic = time.perf_counter()

            # loop considers each template prompt
            for bag_idx, kn_bag in enumerate(kn_bag_list):
                if (bag_idx + 1) % 100 == 0:
                    print(f'calculating {prefix}bag {bag_idx} ...')
                # =============== eval own bag: remove & enhance ================
                eval_bag = eval_bag_list_perrel[relation][bag_idx]
                for eval_example in eval_bag: # eval_example is a list (a triple); eval_bag is a list (template prompts)
                    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                    # convert features to long type tensors
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    # baseline_ids = baseline_ids.to(device) # shape==(1, 128)
                    input_ids = input_ids.to(device) # shape==(1, 128)
                    input_mask = input_mask.to(device) # shape==(1, 128)
                    segment_ids = segment_ids.to(device) # shape==(1, 128)

                    # record [MASK]'s position
                    tgt_pos = tokens_info['tokens'].index('[MASK]')
                    # record [MASK]'s gold label
                    gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])

                    # original probability
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab) - corresponding to model's output 1 in "1_analyze_mlm.py" (config 0)
                    ori_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar

                    # remove
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='remove')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['own:ori_prob'].append(ori_gold_prob.item())
                    rlt_dict[save_key]['rm_own:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                    # enhance
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='enhance')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['eh_own:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                # =============== eval another bag: remove & enhance ================
                # oth_relations = list(eval_bag_list_perrel.keys())
                # oth_relations = [rel for rel in oth_relations if rel != relation]
                # oth_relation = random.choice(oth_relations)
                # oth_idx = random.randint(0, len(eval_bag_list_perrel[oth_relation]) - 1)
                # eval_bag = eval_bag_list_perrel[oth_relation][oth_idx]
                # for eval_example in eval_bag:
                #     eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                #     # convert features to long type tensors
                #     baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                #     baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                #     input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                #     input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                #     segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                #     baseline_ids = baseline_ids.to(device)
                #     input_ids = input_ids.to(device)
                #     input_mask = input_mask.to(device)
                #     segment_ids = segment_ids.to(device)

                #     # record [MASK]'s position
                #     tgt_pos = tokens_info['tokens'].index('[MASK]')
                #     # record [MASK]'s gold label
                #     gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])

                #     _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                #     ori_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar

                #     # remove
                #     _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='remove')  # (1, n_vocab)
                #     int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                #     rlt_dict[save_key]['oth:ori_prob'].append(ori_gold_prob.item())
                #     rlt_dict[save_key]['rm_oth:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                #     # enhance
                #     _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='enhance')  # (1, n_vocab)
                #     int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                #     rlt_dict[save_key]['eh_oth:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

            # record running time
            toc = time.perf_counter()
            print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")
            logger.info(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")

            # calculate mean and ratio values for the relation
            for k, v in rlt_dict[save_key].items():
                if rlt_dict[save_key][k] is not None and len(rlt_dict[save_key][k]) > 0: # all data is not None excepts '{}_ratio'
                    rlt_dict[save_key][k] = np.array(rlt_dict[save_key][k]).mean()
            rlt_dict[save_key]['rm_own:ave_delta_ratio'] = rlt_dict[save_key]['rm_own:ave_delta'] / rlt_dict[save_key]['own:ori_prob']
            rlt_dict[save_key]['eh_own:ave_delta_ratio'] = rlt_dict[save_key]['eh_own:ave_delta'] / rlt_dict[save_key]['own:ori_prob']
            # rlt_dict[save_key]['rm_oth:ave_delta_ratio'] = rlt_dict[save_key]['rm_oth:ave_delta'] / rlt_dict[save_key]['oth:ori_prob']
            # rlt_dict[save_key]['eh_oth:ave_delta_ratio'] = rlt_dict[save_key]['eh_oth:ave_delta'] / rlt_dict[save_key]['oth:ori_prob']
            # print(save_key, '==============>', rlt_dict[save_key])

        # write to file for all relations
        with open(os.path.join(args.output_dir, f'{prefix}modify_activation_rlt.json'), 'w') as fw:
            json.dump(rlt_dict, fw, indent=2)

    eval_modification('')
    # eval_modification('base_')

if __name__ == "__main__":
    main()
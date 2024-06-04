"""
BERT MLM runner
"""

'''
Chương trình thực hiện xóa tri thức về quan hệ được chỉ định trong mô hình đồng thời đánh giá độ đo Perplexity, độ chính xác trước và sau quá trình loại bỏ tri thức. Tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
- Input: tập tin "data_all_allbags.json" chứa các prompt dùng cho quá trình đánh giá hiệu suất; tập tin "kn_bag-{rel}.json" chứa các nơ-ron tri thức cần xóa.
- Output: tập tin "7_erase_knowledge.args.json" lưu trữ cấu hình của chương trình; các giá trị về độ chính xác, độ đo Perplexity được in ra màn hình.
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
from collections import Counter

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


def main():
    '''
    Function's procedures:
    parse arguments ⭢ set device ⭢ set random seeds ⭢ save program args ⭢ init tokenizer ⭢ (Load pre-trained BERT) ⭢ prepare eval set (generate data_all_allbags.json) ⭢ erase()
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
    parser.add_argument("--pt_relation",
                        type=str,
                        default=None,
                        help="Relation to calculate on clusters")

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
    json.dump(args.__dict__, open(os.path.join(args.output_dir, '7_erase_knowledge.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load pre-trained BERT
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

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
    # print(type(eval_bag_list_perrel)) # <class 'dict'>, same as json file


    def erase(rel):
        '''
        (str) -> (print)

        Description:
        rel: tên quan hệ trong tập dữ liệu chứa tri thức cần xóa e.g. 'P30'

        - Hàm thực hiện xóa tri thức về quan hệ "rel" trong mô hình đồng thời đánh giá độ đo Perplexity, độ chính xác trước và sau quá trình loại bỏ tri thức. Quá trình thực hiện gồm các bước sau:
        1. Tìm kiếm các nơ-ron tri thức phổ biến nhất của quan hệ ở tập tin "kn_bag-{rel}.json".
        2. Tính toán độ đo Perplexity (ppl), độ chính xác (acc) của mô hình ở quan hệ cần xóa và các quan hệ còn lại trước khi loại bỏ tri thức.
        3. Tiến hành loại bỏ tri thức.
        4. Tính toán độ đo Perplexity (ppl), độ chính xác (acc) của mô hình ở quan hệ cần xóa và các quan hệ còn lại sau khi loại bỏ tri thức.
        5. In ra màn hình các kết quả về độ đo Perplexity (ppl) và độ chính xác (acc).
        '''

        print(f'evaluating {rel}...')
        with open(os.path.join(args.kn_dir, f'kn_bag-{rel}.json'), 'r') as fr: # read file e.g. "kn_bag-P30.json"
            kn_bag_list = json.load(fr)

        # ======================== calculate kn_rel =================================
        kn_rel = []
        kn_counter = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_counter.update([pos_list2str(kn)])
        most_common_kn = kn_counter.most_common(20)
        print(most_common_kn)
        kn_rel = [pos_str2list(kn_str[0]) for kn_str in most_common_kn] # e.g kn_rel == [[2, 3071], [1, 3071], ...]

        # ======================== load model =================================
        model = BertForMaskedLM.from_pretrained(args.bert_model)
        model.to(device)
        model.eval()

        # record running time
        tic = time.perf_counter()
        # ========================== eval self =====================
        correct = 0
        total = 0
        log_ppl_list = []

        # loop considers each template prompt
        for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[rel]):
            # print(f'evaluating ori {bag_idx} / {len(eval_bag_list_perrel[rel])}')
            # loop considers each triple
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

                # ppl
                gold_prob = F.softmax(logits, dim=-1)[0][gold_label] # tensor.size() == 1
                log_ppl = np.log(1.0 / gold_prob.item())
                log_ppl_list.append(log_ppl)
                # predicted token
                ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                # compare predicted token to ground truth
                total += 1
                if ori_pred_label == eval_example[1]:
                    correct += 1
        ppl = np.exp(np.array(log_ppl_list).mean())
        acc = correct / total
        # ========================== eval other =====================
        o_correct = 0
        o_total = 0
        o_log_ppl_list = []
        for o_rel, eval_bag_list in eval_bag_list_perrel.items():
            if o_rel == rel:
                continue
            # print(f'evaluating for another relation {o_rel}')
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                # if bag_idx % 100 != 0:
                #     continue
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

                    # ppl
                    gold_prob = F.softmax(logits, dim=-1)[0][gold_label]
                    o_log_ppl = np.log(1.0 / gold_prob.item())
                    o_log_ppl_list.append(o_log_ppl)
                    # predicted token
                    ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                    ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                    # compare predicted token to grouth truth
                    o_total += 1
                    if ori_pred_label == eval_example[1]:
                        o_correct += 1
        o_ppl = np.exp(np.array(o_log_ppl_list).mean())
        o_acc = o_correct / o_total

        # ============================================== erase knowledge begin ===========================================================
        print(f'\nerasing -- kn_num: {len(kn_rel)}')
        # unk_emb = model.bert.embeddings.word_embeddings.weight[100]
        with torch.no_grad():
            for layer, pos in kn_rel:
                # model.bert.encoder.layer[layer].output.dense.weight[:, pos] = unk_emb
                # model.bert.encoder.layer[layer].intermediate.dense.weight[pos, :] = torch.zeros(768) # weight.size() = (3072, 768); cannot erase in intermediate (BertIntermediate) layer because of GELU function
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] = torch.zeros(768) # weight.size() == (768, 3072)
        # ============================================== erase knowledge end =============================================================

        # ========================== eval self =====================
        new_correct = 0
        new_total = 0
        new_log_ppl_list = []

        # loop considers each template prompt
        for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[rel]):
            # print(f'evaluating erased {bag_idx} / {len(eval_bag_list_perrel[rel])}')
            # loop considers each triple
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

                # ppl
                gold_prob = F.softmax(logits, dim=-1)[0][gold_label]
                new_log_ppl = np.log(1.0 / gold_prob.item())
                new_log_ppl_list.append(new_log_ppl)
                # predicted token
                ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                # compare predicted token to ground truth
                new_total += 1
                if ori_pred_label == eval_example[1]:
                    new_correct += 1
        new_ppl = np.exp(np.array(new_log_ppl_list).mean())
        new_acc = new_correct / new_total

        # ========================== eval other =====================
        o_new_correct = 0
        o_new_total = 0
        o_new_log_ppl_list = []
        for o_rel, eval_bag_list in eval_bag_list_perrel.items():
            if o_rel == rel:
                continue
            # print(f'evaluating for another relation {o_rel}')
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                # if bag_idx % 100 != 0:
                #     continue
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

                    # ppl
                    gold_prob = F.softmax(logits, dim=-1)[0][gold_label]
                    o_new_log_ppl = np.log(1.0 / gold_prob.item())
                    o_new_log_ppl_list.append(o_new_log_ppl)
                    # predicted token
                    ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                    ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                    # compare predicted token to ground truth
                    o_new_total += 1
                    if ori_pred_label == eval_example[1]:
                        o_new_correct += 1
        o_new_ppl = np.exp(np.array(o_new_log_ppl_list).mean())
        o_new_acc = o_new_correct / o_new_total

        # record running time
        toc = time.perf_counter()
        print(f"\n***** {rel} erased. Costing time: {toc - tic:0.4f} seconds *****")

        print(f'======================================== {rel} ===========================================')
        print(f'original accuracy: {acc:.4}')
        print(f'erased accuracy: {new_acc:.4}')
        erased_ratio = (acc - new_acc) / acc
        print(f'erased ratio: {erased_ratio:.4}')
        print(f'# Kneurons: {len(kn_rel)}')

        print(f'original ppl: {ppl:.4}')
        print(f'erased ppl: {new_ppl:.4}')
        erased_ratio = (ppl - new_ppl) / ppl
        print(f'ppl increasing ratio: {erased_ratio:.4}')

        print(f'(for other) original accuracy: {o_acc:.4}')
        print(f'(for other) erased accuracy: {o_new_acc:.4}')
        o_erased_ratio = (o_acc - o_new_acc) / o_acc
        print(f'(for other) erased ratio: {o_erased_ratio:.4}')

        print(f'(for other) original ppl: {o_ppl:.4}')
        print(f'(for other) erased ppl: {o_new_ppl:.4}')
        o_erased_ratio = (o_ppl - o_new_ppl) / o_ppl
        print(f'(for other) ppl increasing ratio: {o_erased_ratio:.4}')

    # erase('P19')
    # erase('P27')
    # erase('P106')
    # erase('P937')
    erase(args.pt_relation)


if __name__ == "__main__":
    main()
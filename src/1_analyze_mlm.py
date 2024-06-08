"""
BERT MLM runner
"""

'''
Chương trình xử lý một relation với mỗi lần thực thi. Tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
- Input: tập tin "/data/PARAREL/data_all_allbags.json" (hoặc tập tin "/data/PARAREL/data_all.json" để tạo ra data_all_allbags)
- Output: một tập tin chứa các config của chương trình "/results/{}.args.json"; một tập tin chứa điểm phân bổ "/results/{}.rlt.jsonl" (tập tin dữ liệu "/data/PARAREL/data_all_allbags.json")
'''

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
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


def scaled_input(emb, batch_size, num_batch):
    '''
    (tensor, int, int) -> (tensor, tensor)

    Description:
    emb là một tensor có kích thước (1, ffn_size). Là vector embedding, thường là đầu ra từ một lớp ẩn trong mạng nơ-ron nhân tạo.
    emb: tensor([[-0.0792, -0.0529, -0.1438,  ..., -0.0602, -0.0076, -0.0139]],
       grad_fn=<SliceBackward0>)
    batch_size: 20
    num_batch: 1
    
    - Khởi tạo một tensor "baseline" có cùng kích thước với emb và được gán giá trị 0 (torch.zeros_like(emb)). Tensor này được sử dụng làm điểm tham chiếu.

    - Tính toán tensor "step" bằng cách lấy hiệu của emb và baseline chia cho tổng số điểm (num_points = batch_size * num_batch). Tensor step này biểu thị sự thay đổi kích thước cần thiết cho mỗi điểm trên đường thẳng được tạo ra từ baseline đến emb.

    - Sử dụng list để tạo ra một danh sách chứa các tensor đại diện cho các điểm trên đường thẳng được xác định bởi baseline và step.
    - Mỗi phần tử trong danh sách được tính bằng cách cộng baseline với step nhân với một chỉ số i (for i in range(num_points)). Về cơ bản, vòng lặp này tạo ra num_points tensor, mỗi tensor cách nhau một khoảng bằng step trên đường thẳng đi từ baseline đến emb.
    - Sử dụng torch.cat để nối các tensor trong danh sách thành một tensor duy nhất có kích thước (num_points, ffn_size). Tensor này chứa tất cả các điểm được tạo ra trên đường thẳng.

    - Với các biến "baseline", "step", "res" tham khảo /Drive/Thesis/code_research/printed/Scaled_input_output.txt cho giá trị cụ thể được in ra.

    Example:
    >>> scaled_weights, weights_step = scaled_input(tensor([[-0.0792, -0.0529, -0.1438,  ..., -0.0602, -0.0076, -0.0139]],
                                                    grad_fn=<SliceBackward0>), 
                                                    20, 
                                                    1)

    >>> print(scaled_weights)
    tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0040, -0.0026, -0.0072,  ..., -0.0030, -0.0004, -0.0007],
        [-0.0079, -0.0053, -0.0144,  ..., -0.0060, -0.0008, -0.0014],
        ...,
        [-0.0674, -0.0449, -0.1223,  ..., -0.0512, -0.0065, -0.0118],
        [-0.0713, -0.0476, -0.1295,  ..., -0.0542, -0.0068, -0.0125],
        [-0.0753, -0.0502, -0.1366,  ..., -0.0572, -0.0072, -0.0132]],
       grad_fn=<CatBackward0>)

    >>> print(weights_step)
    tensor([-0.0040, -0.0026, -0.0072,  ..., -0.0030, -0.0004, -0.0007],
       grad_fn=<SelectBackward0>)
    '''

    # base --> emd (m = 20 num_points)
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size) 
    
    num_points = batch_size * num_batch 
    step = (emb - baseline) / num_points  # (1, ffn_size), "step" is intergrated gradient step

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    '''
    (list) -> (list)

    Description:
    ig_list: ma trận chứa điểm phân bổ của các nơ-ron.
    ig_list: [[-1.1967224963882472e-05, -1.0288573321304284e-05, ...,  -5.880503977095941e-06],
              [2.6398491172585636e-05,   2.284685685083332e-08 , ..., -1.531957241240889e-05],
             ...]
    ig_list."shape" == (12, 3072)

    - Hàm duyệt qua numpy.ndarray "ig" (được chuyển đổi từ "ig_list"), với những giá trị lớn hơn hoặc bằng 10% giá trị ig.max thì sẽ lưu lại vị trí (i, j) và giá trị tương ứng thành triplet có dạng (i, j, ig[i][j]).

    Example:
    >>> print(ig)
    [[-1.1967224963882472e-05, -1.0288573321304284e-05, ...,  -5.880503977095941e-06],
    [2.6398491172585636e-05,   2.284685685083332e-08 , ..., -1.531957241240889e-05],
    ...]
    >>> print(ig_triplet)
    [[2, 3071, 0.003070289269089699], [3, 1935, 0.0019019896863028407], ..., 
    [11, 2772, 0.002441480988636613], [11, 2824, 0.0021156712900847197], 
    [11, 2958, 0.0018888862105086446], [11, 3052, 0.0021194075234234333]]
    '''

    ig_triplet = []
    ig = np.array(ig_list) # print(ig.shape) - (12, 3072)

    # this filter is not threshold t (refer to section 4.1 in main paper for threshold t)
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def main():
    '''
    Function's procedures:
    parse arguments ⭢ set device ⭢ set random seeds ⭢ save program args ⭢ init tokenizer ⭢ Load pre-trained BERT ⭢ (data parallel) ⭢ prepare eval set (generate data_all_allbags.json) ⭢ loop checks relation (only a input relation is executed) ⭢ loop considers template prompts (tuple)
        ⭢ loop considers a prompt (triple)
            ⭢ generate input feature with example2feature() ⭢ model's first output (loop for 12 layer and decode) ⭢ generate input (for ig method) with scaled_input() ⭢ apply method (model's second output) ⭢ get result ⭢ write file.
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
    parser.add_argument("--output_prefix",
                        default=None,
                        type=str,
                        required=True,
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
    parser.add_argument("--pt_relation",
                        type=str,
                        default=None,
                        help="Relation to calculate on clusters")

    # parameters about integrated grad
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--get_ig_pred",
                        action='store_true',
                        help="Whether to get integrated gradient at the predicted label.")
    parser.add_argument("--get_ig_gold",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label.")
    parser.add_argument("--get_base",
                        action='store_true',
                        help="Whether to get base values. ")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

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
        # device = torch.device("cuda")
        # n_gpu = 2
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, '1_analyze_mlm' + '.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    # print(f"\n\n{type(model)}") # <class 'custom_bert.BertForMaskedLM'>
    # print(model)

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
    # print(type(eval_bag_list_perrel)) # <class 'dict'>, same as json file

    # evaluate args.debug bags for each relation
    for relation, eval_bag_list in eval_bag_list_perrel.items():
        if args.pt_relation is not None and relation != args.pt_relation: # reason for using data_all_allbags.json instead of data_all.json
            continue
        # print(f"TEST: {args.pt_relation}") # TEST: P30 # only one time because dataset has only one P30
        # record running time
        tic = time.perf_counter()
        
        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + relation + '.rlt' + '.jsonl'), 'w') as fw:
            # print(f"TEST: {list(enumerate(eval_bag_list))}") # refer to /Drive/Thesis/code_research/printed/eval_bag_list.txt
            # enumerate(eval_bag_list) contains tuples. Each tuple contains a number and a list (template prompts), this list contains lists (triples)
            
            # loop considers each template prompt
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                res_dict_bag = [] # list for final result (write to file)
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

                    # record various results
                    res_dict = {
                        'pred': [],
                        'ig_pred': [],
                        'ig_gold': [],
                        'base': []
                    }

                    # original pred prob
                    if args.get_pred: # NOT HERE with current setting
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    for tgt_layer in range(model.bert.config.num_hidden_layers):
                        '''
                        This loop considers hidden layer's output of each transformer block (by using "tgt_layer"). The "ffn_weights" are different and ig method is applied to each "ffn_weights" (logits, pred_label is the same for all loops).
                        '''

                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                        # print(f"ffn_weights size: {ffn_weights.size()}") # ffn_weights size: torch.Size([1, 3072])
                        # print(f"logits size: {logits.size()}") # logits size: torch.Size([1, 28996])
                        
                        pred_label = int(torch.argmax(logits[0, :]))  # scalar (logits = x @ W.T + b)
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label) # a predicted token
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        # interpolate input for integrated grad
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                        scaled_weights.requires_grad_(True)

                        # integrated grad at the pred label for each layer
                        if args.get_ig_pred: # NOT HERE with current setting
                            ig_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                                # calculate sum of gradients
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_pred = grad if ig_pred is None else torch.add(ig_pred, grad)  # (ffn_size)
                            # integral approximation 
                            ig_pred = ig_pred * weights_step  # (ffn_size)
                            res_dict['ig_pred'].append(ig_pred.tolist())

                        # integrated grad at the gold label for each layer
                        if args.get_ig_gold:
                            ig_gold = None
                            for batch_idx in range(args.num_batch): # args.num_batch==1
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] # (num_points, ffn_size) - the same as scaled_weights
                                '''
                                >>> temp_rand = torch.rand(20, 3072)
                                >>> temp = temp_rand[0 * 20:(0+1) * 20]
                                >>> temp - temp_0  # tensor 0
                                '''
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                                # calculate sum of gradients
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
                            # integral approximation
                            ig_gold = ig_gold * weights_step  # (ffn_size)
                            res_dict['ig_gold'].append(ig_gold.tolist())

                        # base ffn_weights for each layer # END of tgt_layer loop
                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    # first filter (not threshold t), res_dict['ig_gold'('base')]."shape" == (12, 3072)
                    if args.get_ig_gold:
                        res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
                    if args.get_base:
                        res_dict['base'] = convert_to_triplet_ig(res_dict['base'])

                    res_dict_bag.append([tokens_info, res_dict])

                fw.write(res_dict_bag)

        # record running time
        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()
'''
Chương trình đếm và lưu trữ số lượng template prompt và số lượng prompt của mỗi relation tại tập tin .txt. Tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
Input: tập tin "data_all_allbags.json".
Output: một tập tin chứa .txt chứa tên các relation, số lượng template và số lượng prompt tương ứng tại "/results/".
'''

import json, jsonlines
import os

total_prompt = 0
total_template = 0

def count_prompt(data_directory):
	'''
	(str) -> text (.txt) file in "/results/"

	Description:
	data_directory: e.g. '../data/PARAREL/data_all_allbags.json'
	
	- Hàm nhận vào một đường dẫn "data_directory" chứa tập tin dữ liệu (data_all_allbags) để thực hiện đếm và ghi nhận số lượng template prompt cũng như prompt của các relation.
	'''

	global total_prompt, total_template

	if os.path.exists(data_directory): # read file if data file exists
		with open(data_directory, 'r') as f:
			eval_bag_list_perrel = json.load(f)
	else: # stop the program if data file does not exist
	 	print("Data does not exist")
	 	return

	counts = {}
	with jsonlines.open('../results/count_prompt.txt', 'w') as fw:
		# loop considers each relation
		for relation, eval_bag_list in eval_bag_list_perrel.items():
			# Initialize record
			counts[relation] = {
				'num of template (bag)': 0,
				'num of prompt': 0
			}
			counts[relation]['num of template (bag)'] += len(eval_bag_list)
			total_template += len(eval_bag_list)

			# loop considers each template prompt
			for bag_idx, eval_bag in enumerate(eval_bag_list):
				counts[relation]['num of prompt'] += len(eval_bag)
				total_prompt += len(eval_bag)
			
			# Write file for each relation
			fw.write(f'{relation} : {counts[relation]}')

		# Write total number of template and prompt to file
		fw.write(f'Number of templates: {total_template} - Number of prompts: {total_prompt}')

if __name__ == "__main__":
	data_directory = '../data/PARAREL/data_all_allbags.json'
	count_prompt(data_directory)
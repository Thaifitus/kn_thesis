'''
Chương trình tiến hành đếm và lưu trữ số lượng template prompt, số lượng prompt của mỗi relation. Tập tin đầu vào và đầu ra tương ứng của chương trình bao gồm:
Input: tập tin "data_all_allbags.json"
Output: một tập tin csv chứa tên các relation, số lượng template và số lượng prompt tương ứng
'''

import json, jsonlines
import os
import csv

total_prompt = 0
total_template = 0

def count_prompt(data_directory, out_dir):
	'''
	(str, str) -> csv file

	Description:
	data_directory: đường dẫn chứa tập tin dữ liệu e.g. '../data/PARAREL/data_all_allbags.json'
	out_dir: đường dẫn lưu trữ kết quả đếm e.g. './count_prompt.csv'

	- Hàm nhận vào một đường dẫn "data_directory" chứa tập tin dữ liệu (data_all_allbags) để thực hiện đếm và ghi nhận số lượng template prompt cũng như prompt của các relation và lưu trữ ở "out_dir".
	'''

	global total_prompt, total_template

	# Read data
	if os.path.exists(data_directory): # read if data file exists
		with open(data_directory, 'r') as f:
			eval_bag_list_perrel = json.load(f)
	else: # stop the program if data file does not exist
	 	print("Data does not exist")
	 	return

	# Counting
	count_record = []
	for relation, eval_bag_list in eval_bag_list_perrel.items(): # loop considers each relation
		# Initialize record
		rel_dict = {'relation': '',
					'num of template (bag)': 0,
					'num of prompt': 0}
		# Count template prompt
		rel_dict['num of template (bag)'] += len(eval_bag_list)
		total_template += len(eval_bag_list)
		# Count prompt
		for bag_idx, eval_bag in enumerate(eval_bag_list):
			rel_dict['num of prompt'] += len(eval_bag)
			total_prompt += len(eval_bag)
		# Take relation with name e.g. P101(field of work)
		rel_with_name = eval_bag_list_perrel[relation][0][0][2]
		rel_dict['relation'] = rel_with_name

		count_record.append(rel_dict)

	# Write file for all relations
	with open(out_dir, 'w') as fw:
		# creating a csv dict writer object
		writer = csv.DictWriter(fw, fieldnames=count_record[0].keys())

		# writing headers (field names)
		writer.writeheader()
	
		# writing data rows
		writer.writerows(count_record)
		
		# Write total number of template and prompt to file
		fw.write(f'\nNumber of templates: {total_template} - Number of prompts: {total_prompt}')

if __name__ == "__main__":
	data_directory = '../data/PARAREL/data_all_allbags.json'
	out_dir = './count_prompt.csv'
	count_prompt(data_directory, out_dir)
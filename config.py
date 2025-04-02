import numpy as np


# core_gym_env.py & DQN.py
embedding_len = 768
action_set = [1, 2, 3]

# llama3.1  config
api_key = ""
base_url=""

sem_types = ["database", "game", "image", "video", "network"]

eval_samples_list = {"HeapSort":["heap_sort"], "RGB2YCbCr":["rgb2ycbcr"]}
eval_func_types = {"rgb2ycbcr":["image","network"], "heap_sort":["sort", "image"]}

train_samples_list = {}
train_func_types = {}

template1 = 'Analyze the actual functionality of this code, and output the possible functional categories of this code along with their confidence values. Output strictly in the format: {"Type1":score1, "Type2":score2,...}. The sum of all confidence values must be 1. Do not output anything outside the {"Type1":score1, "Type2":score2,...} dictionary. [Finally, only output the {} dictionary, no description;] [It must be ensured that the sum of all confidence values is 1, and the confidence values cannot all be 0]\n'
template2 = 'The functional categories only include:["database", "game", "image", "video", "network", "'
template_head = template1 + template2
template_body = '"]\nCode are as follow:\n'
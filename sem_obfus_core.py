import os
import torch
import subprocess
from unixcoder import UniXcoder
from openai import OpenAI
import json
import time
import re
from config import *

obfus_type = ""
record_file_name = "record.txt"
insert_record_file_name = "insert_record.txt"
model_path = os.path.join('model', 'unixcoder-base-nine')
original_bin_name = "bin.exe"
original_bin_symboal_name = "bin_symbol.exe"
insert_bin_name = "bin_insert.exe"
insert_bin_symboal_name = "bin_insert_symbol.exe"

RED = '\033[31m'
YELLOW = '\033[33m'
RESET = '\033[0m'

COLOR_List = ['\033[32m', '\033[34m', '\033[35m', '\033[36m', '\033[97m']


def set_obfus_type(ob_type):
    global obfus_type
    obfus_type = ob_type
    

# ----------------------- code bert embedding -----------------------------
def initial_model(gpu_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    
    if device.type == 'cuda':
        gpu_index = torch.cuda.current_device()
        print(f"\tmodel load on CUDA:{gpu_index}")
    elif device.type == 'cpu':
        print(f"\tmodel load on CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(model_path)
    model.to(device)
    return model, device

def get_bin_embedding(model, device, text):
    tokens_ids = model.tokenize([text], max_length=1023, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, max_func_embedding = model(source_ids)
    semantic_vector = max_func_embedding.cpu().detach().numpy() if torch.is_tensor(max_func_embedding) else max_func_embedding
    if semantic_vector.ndim > 1:
        semantic_vector = semantic_vector[0]
    return semantic_vector


# ---------------------- IDA parser ------------------------------
# build before obfus
def run_build_sh(work_dir):
    try:
        result = subprocess.run(["build.sh"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
    except subprocess.CalledProcessError as e:
        print("run build.sh failed")
    return

def run_get_bincode_sh(work_dir, funcname=None):
    if funcname:
        try:
            result = subprocess.run(["get_bin_code.sh", funcname],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("run get_bin_code.sh failed:{e}")
    else:
        try:
            result = subprocess.run(["get_bin_code.sh"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("run get_bin_code.sh failed:{e}")
    return

def get_bin_code(work_dir, bin_path, symboal_bin_path, funcname):
    if os.path.exists(bin_path) and os.path.exists(symboal_bin_path):
        run_get_bincode_sh(work_dir, funcname)
        run_get_bincode_sh(work_dir)
        bin_code = ""
        func_code_path = os.path.join(work_dir, "func_code.log")
        if os.path.exists(func_code_path):
            with open(func_code_path, "r") as fp:
                bin_code = fp.read()
            os.remove(func_code_path)
            return bin_code
    return None

# build after obfus
def run_build_insert_sh(work_dir):
    rm_list = ["bin_insert.exe", "bin_insert.o", "bin_insert_symboal.exe"]
    for rm_item in rm_list:
        if os.path.exists(os.path.join(work_dir, rm_item)):
            os.remove(os.path.join(work_dir, rm_item))
    try:
        result = subprocess.run(["build_insert.sh"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
    except subprocess.CalledProcessError as e:
        print("run build_insert.sh failed")
        return False
    return True

def run_get_insert_bincode_sh(work_dir, funcname=None):
    if funcname:
        try:
            result = subprocess.run(["get_insert_bin_code.sh", funcname],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
        except subprocess.CalledProcessError as e:
            print("run get_insert_bin_code.sh failed")
    else:
        try:
            result = subprocess.run(["get_insert_bin_code.sh"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
        except subprocess.CalledProcessError as e:
            print("run get_insert_bin_code.sh failed")
    return

def get_insert_bin_code(work_dir, bin_path, symboal_bin_path, funcname):
    if os.path.exists(bin_path) and os.path.exists(symboal_bin_path):
        run_get_insert_bincode_sh(work_dir, funcname)
        run_get_insert_bincode_sh(work_dir)
        bin_code = ""
        func_code_path = os.path.join(work_dir, "func_code.log")
        if os.path.exists(func_code_path):
            with open(func_code_path, "r") as fp:
                bin_code = fp.read()
            os.remove(func_code_path)
            return bin_code
    return None


# ------------------------ LLVM Pass support --------------------------
def run_record_pass(work_dir, funcname):
    try:
        result = subprocess.run(["./seobfus", "bin_src.ll", "bin.ll", funcname, "-RecordSe"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
    except subprocess.CalledProcessError as e:
        print("run record pass failed")
    return

def run_insert_pass(work_dir, funcname):
    try:
        result = subprocess.run(["./seobfus", "bin.ll", "bin_insert.ll", funcname, "-InsertSe"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work_dir)
    except subprocess.CalledProcessError as e:
        print("run insert pass failed:")
        print(e)
        return False
    return True

# ------------------------- env support ----------------------------
def get_record_list(work_dir):
    record_list = []
    record_path = os.path.join(work_dir, record_file_name)
    #if not os.path.exists(record_path):
        #run_record_pass(work_dir, funcname)
    
    try:
        with open(record_path, 'r') as fp:
            lines = fp.readlines()
    except Exception as e:
        print(f"[workdir: {work_dir}]error open record file")
        return record_list, 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            record_list.append(stripped_line)
    return record_list, len(record_list)
        



def build_insert_record(work_dir, insert_list):
    insert_f_path = os.path.join(work_dir, insert_record_file_name)
    content = '\n'.join([item for item in insert_list if item])
    with open(insert_f_path, 'w') as fp:
        fp.write(content)
    return

def send2onlineLLVM(content):
    client = OpenAI(api_key=api_key,base_url=base_url)
    ret_content = ""
    try:
        response = client.chat.completions.create(
            model='meta-llama/Meta-Llama-3.1-80B-Instruct',
            messages=[
                {'role': 'user', 'content': content}
            ],
            temperature=0,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                ret_content += chunk.choices[0].delta.content
    except Exception as e:
        return None
    return ret_content


def parse_response(recv_data):
    if not recv_data:
        return None, 0
    # parser returned json
    start_index = recv_data.find('{')
    end_index = recv_data.rfind('}') + 1

    if start_index != -1 and end_index != 0:
        json_content = recv_data[start_index:end_index]
        try:
            json_data = json.loads(json_content)
            return json_data, 2
        except Exception as e:
            return None, 1
    return None, 1
    
def loop_get_json_data(send_data):
    recv_data = send2onlineLLVM(send_data)
    json_data, retType = parse_response(recv_data)
    if retType == 0 or retType == 1:
        return None, retType
    
    total_score = 0.00
    for key, value in json_data.items():
        total_score += value
    total_score = round(total_score, 1)
    if total_score < 0.5:
        return json_data, 3
    return json_data, 2

def get_reward(work_dir, bin_code, funcname, progress_str, main_color):
    score = 0.00
    global obfus_type
    try:
        sample_type = train_func_types[funcname]
    except Exception as e:
        print("not find sample type")
        return score, None
    send_data = template_head + sample_type + template_body + bin_code
    json_data, retType = loop_get_json_data(send_data)
    attempt_index = 0
    while retType != 2 and attempt_index < 10:
        if retType == 0:
            print(f"{main_color}[{work_dir}: {progress_str}]:{RESET} {RED}LLM rate limit, resending...{RESET}")
            time.sleep(10)
        elif retType == 1 or retType == 3:
            print(f"{main_color}[{work_dir}: {progress_str}]:{RESET} {RED}json parser error/nvalid score:[{json_data}], resending...{RESET}")
            time.sleep(1)
        json_data, retType = loop_get_json_data(send_data)
        attempt_index += 1
    if json_data:
        for key, value in json_data.items():
            if key == obfus_type:
                score = score + value
            elif key == sample_type:
                score = score - value
            else:
                score = score - 0.5*value
        reward = round(score, 2)
        print(f"{main_color}[{work_dir}: {progress_str}]{RESET}: {json_data} --> {YELLOW}[reward:{reward}]{RESET}")
        return reward, json_data
    return score, None

def add_insert_point(bin_code, insert_sem):
    lines = bin_code.splitlines()
    modified_lines = []
    flag = True
    for line in lines:
        if flag and insert_sem in line:
            match = re.match(r'^(\s*)', line)
            if match:
                head_white = match.group(1)
            else:
                head_white = ''
            new_line = head_white + '<INSERTION_POINT>'
            modified_lines.append(new_line)
            flag = False
        else:
            modified_lines.append(line)
    modified_code = '\n'.join(modified_lines)
    #print(modified_code)
    return modified_code
def delete_insert_point(bin_code):
    lines = bin_code.splitlines()
    modified_lines = []
    for line in lines:
        if '<INSERTION_POINT>' in line:
            continue
        modified_lines.append(line)
    modified_code = '\n'.join(modified_lines)
    return modified_code

def get_state(model_info, work_dir, insert_list, funcname, id):
    main_color = COLOR_List[id]
    progress_str = "Initial"
    insert_sem = insert_list[-1]
    insert_sem = insert_sem.split('|', 3)[3]
    state = None
    build_insert_record(work_dir, insert_list)
    if not run_insert_pass(work_dir, funcname):
        print(f"[{work_dir}]: run_insert_pass fail")
        return None, None, None
    if run_build_insert_sh(work_dir):
        bin_path = os.path.join(work_dir, insert_bin_name)
        bin_symboal = os.path.join(work_dir, insert_bin_symboal_name)
        bin_code = get_insert_bin_code(work_dir, bin_path, bin_symboal, funcname)
        if bin_code:
            bin_code = add_insert_point(bin_code, insert_sem)
            state = get_bin_embedding(model_info[0], model_info[1], bin_code)
            #print(f"[{work_dir}]: get state successfully")
            #print(f"[{work_dir} state]: {state}]\n")
            if state is None:
                print(f"[{work_dir}:{progress_str}]: get state error")
        else:
            print(f"[{work_dir}:{progress_str}]: Bin_code None\n")
    else:
        print(f"[{work_dir}:{progress_str}]: run_build_insert_sh failed\n")
    return state

def apply_obfu(model_info, work_dir, insert_list, funcname, progress_str, id):
    main_color = COLOR_List[id]
    insert_sem = insert_list[-1]
    if insert_sem != '':
        insert_sem = '<INSERTION_POINT>'
    state = None
    reward = None
    build_insert_record(work_dir, insert_list)
    if not run_insert_pass(work_dir, funcname):
        print(f"[{work_dir}]: run_insert_pass fail")
        return None, None, None
    if run_build_insert_sh(work_dir):
        bin_path = os.path.join(work_dir, insert_bin_name)
        bin_symboal = os.path.join(work_dir, insert_bin_symboal_name)
        bin_code = get_insert_bin_code(work_dir, bin_path, bin_symboal, funcname)
        if bin_code:
            if insert_sem != '':
                bin_code = add_insert_point(bin_code, insert_sem)
            state = get_bin_embedding(model_info[0], model_info[1], bin_code)
            bin_code = delete_insert_point(bin_code)
            #print(f"[{work_dir}]: get state successfully")
            #print(f"[{work_dir} state]: {state}]\n")
            if state is not None:
                reward, json_data = get_reward(work_dir, bin_code, funcname, progress_str, main_color)
            else:
                print(f"[{work_dir}:{progress_str}]: get state error")
        else:
            print(f"[{work_dir}:{progress_str}]: Bin_code None\n")
    else:
        print(f"[{work_dir}:{progress_str}]: run_build_insert_sh failed\n")

    return state, reward, json_data

def apply_eval_obfu(model_info, work_dir, insert_list, funcname, progress_str, id):
    main_color = COLOR_List[id]
    insert_sem = insert_list[-1]
    if insert_sem != '':
        insert_sem = '<INSERTION_POINT>'
    state = None

    build_insert_record(work_dir, insert_list)
    if not run_insert_pass(work_dir, funcname):
        print(f"[{work_dir}]: run_insert_pass fail")
        return None, None
    if run_build_insert_sh(work_dir):
        bin_path = os.path.join(work_dir, insert_bin_name)
        bin_symboal = os.path.join(work_dir, insert_bin_symboal_name)
        bin_code = get_insert_bin_code(work_dir, bin_path, bin_symboal, funcname)
        if bin_code:
            if insert_sem != '':
                bin_code = add_insert_point(bin_code, insert_sem)
            state = get_bin_embedding(model_info[0], model_info[1], bin_code)
            bin_code = delete_insert_point(bin_code)
            print(f"{main_color}[{work_dir}: {progress_str}]:{RESET} {YELLOW} Finished{RESET}")
            if state is None:
                print(f"[{work_dir}:{progress_str}]: get state error")
        else:
            print(f"[{work_dir}:{progress_str}]: Bin_code None\n")
    else:
        print(f"[{work_dir}:{progress_str}]: run_build_insert_sh failed\n")

    return state, bin_code
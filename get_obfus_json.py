import torch
import faiss
import os
import json
import shutil
import networkx as nx
import math
import subprocess
import re
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path
from unixcoder import UniXcoder
from graph2vec import Graph2Vec
from joblib import Parallel, delayed
from tqdm import tqdm
from networkx.drawing.nx_pydot import read_dot, write_dot
from scipy.spatial.distance import cosine

BranchNode_Feature = 1
LoopNode_Feature = 2
OtherNode_Feature = 3
semantic_dimension = 768
cfg_dimension = 128
#  ---------- some score weights ----------
weight_alpha = 0.5    # struct
weight_beta =  20   # semantic
weight_mixed = 0.6  # struct


def get_embedding(text):
    tokens_ids = model.tokenize([text],max_length=1023,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, max_func_embedding = model(source_ids)
    return tokens_embeddings, max_func_embedding    

def build_classify_index(class_name):
    index_file_path = f'index/{class_name}.index'
    metadata_file = f'index/{class_name}.json'
    input_json_file = f'datasets/json/{class_name}.json'
    code_dir = f"datasets/code/{class_name}"
    metadata = []
    input_json = []
    with open(input_json_file, 'r', encoding='utf-8') as fp:
        input_json = json.load(fp)
    
    process_num = 0
    json_len = len(input_json)
    index = faiss.IndexIDMap2(faiss.IndexFlatL2())
    

    # Initial Faiss index
    dimension = 768
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    
    for json_item in input_json:
        try:
            json_data = {
                "If-Condition-Str": json_item["If-Condition-Str"],
                "Loop-Condition-Str": json_item["Loop-Condition-Str"],
                "Normal-Str": json_item["Normal-Str"]
            }
        except Exception as e:
            process_num = process_num + 1
            print(f"error read json item: [{process_num}/{json_len}]")
            continue
        data_all_content = json.dumps(json_data)
        if not data_all_content.isascii():
            process_num = process_num + 1
            print(f"error read not ascii: [{process_num}/{json_len}]")
            continue
        code_name = json_item["func_file_path"].split('\\')[-1]
        code_path = f"{code_dir}/{code_name}"
        #code_path = os.path.join(code_dir, os.path.basename(json_item["func_file_path"]))
        #print(code_path)
        try:
            with open(code_path, 'r', encoding='utf-8') as fp:
                code_content = fp.read()
        except Exception as e:
            process_num = process_num + 1
            print(f"error read: {code_path} [{process_num}/{json_len}]")
            continue
        # add index vector
        metadata.append(json_data)
        _, embedding = get_embedding(code_content)
        embedding_np = embedding.cpu().detach().numpy()
        index.add(embedding_np)
        process_num = process_num + 1
        print(f"[{class_name}]insert content: [{process_num}/{json_len}]")

    # save index
    faiss.write_index(index, index_file)
    print(f"[{class_name}] faiss vector database saved to {index_file}")
    with open(metadata_file, 'w',encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=4)
    print(f"[{class_name}] metadata file saved to {metadata_file}")

# ---------------- Following are build cfg json datasets ------------------------------
def execute_joern_parse(id, input_path, output_path):
    env = os.environ.copy()
    env["JAVA_HOME"] = "/home/env/jdk-21"
    env["CLASSPATH"] = ".:" + env["JAVA_HOME"] + "/lib"
    env["PATH"] = env["CLASSPATH"] + ":" + env["JAVA_HOME"] + "/bin:" + env["PATH"]

    command = ["joern-cli/joern-parse", input_path, "-o", output_path]

    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        print(f"[{id}]:error in joern parse")
        print(f"[{id}]: stdout: {result.stdout}")
        print(f"[{id}]: stderr: {result.stderr}")
        return False

def execute_joern_export(id, input_path, output_dir):
    env = os.environ.copy()
    env["JAVA_HOME"] = "/home/env/jdk-21"
    env["CLASSPATH"] = ".:" + env["JAVA_HOME"] + "/lib"
    env["PATH"] = env["CLASSPATH"] + ":" + env["JAVA_HOME"] + "/bin:" + env["PATH"]

    command = ["joern-cli/joern-export", input_path, "--repr", "cfg", "-o", output_dir]

    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        print(f"[{id}]:error in joern export")
        print(f"[Error]: stdout: {result.stdout}")
        print(f"[Error]: stderr: {result.stderr}")
        return False

def rebuild_graph(G):
    # build DFS
    def dfs_util(node, visited, stack):
        visited.add(node)
        for neighbor in G[node]:
            if neighbor not in visited:
                dfs_util(neighbor, visited, stack)
        stack.append(node)

    # run DFS
    visited = set()
    stack = []
    for node in G.nodes():
        if node not in visited:
            dfs_util(node, visited, stack)

    # Get reverse DFS
    dfs_order = list(reversed(stack))
    new_labels = {old_label: i for i, old_label in enumerate(dfs_order)}
    H = nx.relabel_nodes(G, new_labels, copy=True)
    for node in H.nodes(data=True):
        if 'label' in node[1]:
            del H.nodes[node[0]]['label']
    return H

def find_branchs(G):
    branch_nodes = set()
    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        if len(out_edges) > 1:
            branch_nodes.add(node)
            for _, target in out_edges:
                branch_nodes.add(target)
    return branch_nodes

def find_loops(G):
    LoopNodes = set()
    sccs = list(nx.strongly_connected_components(G))
    loop_heads = set()
    loop_tails = defaultdict(set)
    for scc in sccs:
        if len(scc) > 1:
            head = min(scc)#head = next(iter(scc))
            loop_heads.add(head)

            for tail in scc:
                if tail != head and (tail, head) in G.edges:
                    loop_tails[head].add(tail)

    for head in loop_heads:
            LoopNodes.add(head)
            for tail in loop_tails[head]:
                LoopNodes.add(tail)
    return LoopNodes    #loop_heads, loop_tails

def generate_cfg_json(id, dot_dir, func_name, out_dir, code_base_name):
    sys.setrecursionlimit(10000)
    files = os.listdir(dot_dir)
    find_file = None
    for file_name in files:
        file_path = os.path.join(dot_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fp:
                first_line = fp.readline()
                match = re.search(r'digraph\s+"' + re.escape(func_name) + r'"', first_line)
                if match:
                    find_file = file_path
                    break
                else:
                    continue
    if find_file:
        graph_from_file = read_dot(find_file)    # pydot.Dot(filename=find_file, format='dot')
        G = rebuild_graph(graph_from_file)
        try:
            LoopNodes = find_loops(G)
            branch_nodes = find_branchs(G)
        except Exception as e:
            print(f"\n[{id}]:can not find loop or branch, maybe recursionlimit: {code_base_name}")
            return
        out_data = {}
        out_data["edges"] = []
        out_data["features"] = {}
        for u,v in G.edges():
            out_data["edges"].append([u, v])
        nodes = sorted(G.nodes)
        other_feature_id = OtherNode_Feature
        for node in range(max(nodes)+1):
            if node in nodes:
                if node in LoopNodes:
                    out_data["features"][str(node)] = str(LoopNode_Feature)
                elif node in branch_nodes:
                    out_data["features"][str(node)] = str(BranchNode_Feature)
                else:
                    out_data["features"][str(node)] = str(other_feature_id)
                    other_feature_id = other_feature_id + 1

        # build cfg json file 
        out_file = os.path.join(out_dir, f"{id}.json")
        with open(out_file, 'w') as ofp:
            json.dump(out_data, ofp)
        new_graph_file = os.path.join(out_dir, f"{id}.dot")
        write_dot(G, new_graph_file)
    else:
        print(f"\n[{id}]:not find dot file in generate cfg json: {code_base_name}")
    return

def build_cfg_multi_thread(id, tmp_dir, cfg_dir, code_path, code_name):
    # id: id
    # tmp_dir: tmp/{class}/
    # cfg_dir: datasets/cfg_graph/{class}
    # code_path: datasets/code/{class}/xxx funcname.txt
    # code_name: xxx funcname.txt
    code_base_name = code_name.split('.')[0]    # "xxx funcname"
    parts = code_base_name.split(' ')           # funcname
    if len(parts)==2:
        func_name = parts[1]    
    else:
        func_name = parts[0]
    new_code_name = code_base_name + ".cpp"     # "xxx funcname.cpp"
    new_code_path = os.path.join(tmp_dir, new_code_name)    # "tmp/xxx funcname.cpp"
    if os.path.exists(new_code_path):
        os.remove(new_code_path)

    if os.path.exists(code_path):
        shutil.copy2(code_path, new_code_path)
    else:
        print(f"[{id}]:error copy code.txt 2 tmp/code.cpp -> {code_path}")
        return
    # joern parse
    code_bin_path = os.path.join(tmp_dir, f"{id}.bin")  # "tmp/id.bin"
    ret = execute_joern_parse(id, new_code_path, code_bin_path)
    if not ret:
        try:
            if os.path.exists(code_bin_path):
                os.remove(code_bin_path)
            os.remove(new_code_path)
        except Exception as e:
            pass
        return
    # joern export
    code_cfg_dir = os.path.join(tmp_dir, f"{id}")         # "tmp/id/"
    ret = execute_joern_export(id, code_bin_path, code_cfg_dir)
    if not ret:
        try:
            shutil.rmtree(code_cfg_dir)
        except Exception as e1:
            pass
        return
    try:
        if os.path.exists(code_bin_path):
            os.remove(code_bin_path)
        os.remove(new_code_path)
    except Exception as e2:
        pass
    # generate 0.json
    generate_cfg_json(id, code_cfg_dir, func_name, cfg_dir, code_base_name)

    try:
        shutil.rmtree(code_cfg_dir)
    except Exception as e2:
        pass
    return

def process_json_item(json_item, process_num, total_tasks, tmp_dir, cfg_dir, code_dir):
    try:
        code_name = json_item["func_file_path"].split('\\')[-1]
        code_path = os.path.join(code_dir, code_name)
    except Exception as e:
        print(f"Error read func file path: [{process_num}/{total_tasks}]")
        return
    build_cfg_multi_thread(process_num, tmp_dir, cfg_dir, code_path, code_name)
    return

def build_cfg_graph(class_name):
    # input
    input_json_file = f'datasets/json/{class_name}.json'
    code_dir = f"datasets/code/{class_name}"
    # output
    tmp_dir = f"tmp/{class_name}"
    cfg_dir = f"datasets/cfg_graph/{class_name}"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg_dir).mkdir(parents=True, exist_ok=True)

    input_json = []
    with open(input_json_file, 'r', encoding='utf-8') as fp:
        input_json = json.load(fp)

    total_cores = os.cpu_count()
    cores_to_use = math.ceil(total_cores * 0.8)
    # process_num = 0
    print(f"[{class_name}]: Start building CFG graph...")
    json_len = len(input_json)
    Parallel(n_jobs=cores_to_use)(  # n_jobs=-1
        delayed(process_json_item)(json_item, i, json_len, tmp_dir, cfg_dir, code_dir)
        for i, json_item in tqdm(enumerate(input_json), total=json_len)
    )
    print(f"[{class_name}]: Finish CFG build")
    return

# -----------------------------------  generate_metadata  ----------------------
def attempt_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
def has_notascii(text):
    pattern = re.compile(r'[^\x00-\x7F]')
    return bool(pattern.search(text))
def generate_metadata(class_name):
    input_json_file = f'datasets/json/{class_name}.json'
    code_dir = f"datasets/code/{class_name}"
    cfg_graph_dir = f"datasets/cfg_graph/{class_name}"
    metadata_file_path = f'index/{class_name}.json'

    metadata = []
    with open(input_json_file, 'r', encoding='utf-8') as fp:
        input_json = json.load(fp)
    
    refresh_cfg_json = False
    json_len = len(input_json)
    for process_num, json_item in enumerate(input_json):
        try:
            code_name = json_item["func_file_path"].split('\\')[-1]
            code_path = f"{code_dir}/{code_name}"
            cfg_json_path = f"{cfg_graph_dir}/{process_num}.json"
            if not os.path.exists(cfg_json_path):
                refresh_cfg_json = True
                continue
            json_data = {
                "Branch": json_item["If-Condition-Str"],
                "Loop": json_item["Loop-Condition-Str"],
                "Normal": json_item["Normal-Str"],
                "FuncName": code_name
            }
        except Exception as e:      
            print(f"[In generating metadata {process_num}/{json_len}]: error read json item.")
            attempt_delete_file(cfg_json_path)
            refresh_cfg_json = True
            continue
        
        data_all_content = json.dumps(json_data)
        if has_notascii(data_all_content):
        #if not all(ord(c) < 128 for c in data_all_content):#data_all_content.isascii():
            print(f"[In generating metadata {process_num}/{json_len}]: read not ascii.")
            attempt_delete_file(cfg_json_path)
            refresh_cfg_json = True
            continue
        # attempting read code 
        try:
            with open(code_path, 'r', encoding='utf-8') as fp:
                code_content = fp.read()
        except Exception as e:
            print(f"[In generating metadata {process_num}/{json_len}]: error read: {code_path}.")
            attempt_delete_file(cfg_json_path)
            refresh_cfg_json = True
            continue
        # add metadata
        metadata.append(json_data)
    
    # save metadata
    with open(metadata_file_path, 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=4)
    print(f"[{class_name}]metadata file save to: {metadata_file_path}")

    # reorder cfg json file
    if refresh_cfg_json:
        print("Refreshing cfg json...")
        files = [f for f in os.listdir(cfg_graph_dir) if os.path.isfile(os.path.join(cfg_graph_dir, f))]
        numeric_files = [f for f in files if f.endswith('.json') and f[:-5].isdigit()]
        numeric_files.sort(key=lambda f: int(f[:-5]))
        for i, filename in enumerate(numeric_files):
            old_filepath = os.path.join(cfg_graph_dir, filename)
            new_filename = f"{i}.json"
            new_filepath = os.path.join(cfg_graph_dir, new_filename)
            os.rename(old_filepath, new_filepath)
        print("Finish refresh")

# ------------------------------- build faiss index ----------------------------
def get_semantic_embedding(model, device, text):
    tokens_ids = model.tokenize([text],max_length=1023,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, max_func_embedding = model(source_ids)
    return max_func_embedding
    #return tokens_embeddings, max_func_embedding

def get_cfg_embeddings(input_path, dimensions, min_count, workers):
    g2vec = Graph2Vec(input_path=input_path, dimensions=dimensions, min_count=min_count, workers=workers)
    return g2vec.get_embeddings()

def build_classify_index(class_name):
    # define some path
    index_file_path = f'index/{class_name}.index'
    metadata_file_path = f'index/{class_name}.json'
    code_dir = f"datasets/code/{class_name}"
    cfg_graph_dir = f"datasets/cfg_graph/{class_name}"

    if os.path.exists(metadata_file_path):
        print(f"[{class_name}]: Generating CFG embeddings...")
        cfg_embeddings = get_cfg_embeddings(cfg_graph_dir, cfg_dimension, 1, 10)
        print(f"[{class_name}]: Finish CFG embedding")
        # build faiss index
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(cfg_dimension + semantic_dimension))
        # define semantic embedding model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("model/unixcoder-base-nine")
        model.to(device)

        with open(metadata_file_path, 'r', encoding='utf-8') as fp:
            input_json = json.load(fp)
        
        print(f"[{class_name}]: Inserting vector to faiss database...")
        json_len = len(input_json)
        for i, json_item in enumerate(input_json):
            code_name = json_item["FuncName"]
            code_path = f"{code_dir}/{code_name}"
            with open(code_path, 'r', encoding='utf-8') as fp:
                code_content = fp.read()
            semantic_vector = get_semantic_embedding(model, device, code_content)
            semantic_vector = semantic_vector.cpu().detach().numpy() if torch.is_tensor(semantic_vector) else semantic_vector
            if semantic_vector.ndim > 1:
                semantic_vector = semantic_vector[0]
            cfg_embedding = cfg_embeddings[i].cpu().detach().numpy() if torch.is_tensor(cfg_embeddings[i]) else cfg_embeddings[i]
            #print(cfg_embedding)
            #print("-------------------")
            #print(semantic_vector)
            combined_vector = np.concatenate((cfg_embedding, semantic_vector))
            index.add_with_ids(np.expand_dims(combined_vector, axis=0), np.array([i]))
            print(f"[{class_name}]: Inserting content: [{i}/{json_len}]")

        faiss.write_index(index, index_file_path)
        print(f"[{class_name}]: Finish build index vectors, the combined vectors are saved to: {index_file_path}")
    else:
        print(f"[{class_name}]: Not find metadata file, exit")
        return

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_score(e1, e2, f1, f2):
    # [e: faiss]: e1->struct, e2->semantic
    # [f: target]: f1->struct, f2->semantic
    cosine_struct = 1 - cosine(e1, f1)
    cosine_semantic = 1 - cosine(e2, f2)

    cosine_struct = sigmoid(weight_alpha*cosine_struct)
    cosine_semantic = sigmoid(-1*weight_beta*cosine_semantic)
    mixed_score = weight_mixed*cosine_struct + (1-weight_mixed)*cosine_semantic
    return mixed_score

def get_target_embeddings(funcfile, tmp_dir, cfg_dir):
    
    filename = os.path.basename(funcfile)
    id = 0

    print("Start building function CFG graph....")
    build_cfg_multi_thread(id, tmp_dir, cfg_dir, funcfile, filename)
    cfg_json_file = os.path.join(cfg_dir, f"{id}.json")
    print("Finish.")

    if os.path.exists(cfg_json_file):
        print("Start generating function embeddings...")
    else:
        print("Error: not find cfg json.")

    #  --------------- cfg embedding
    cfg_embeddings = get_cfg_embeddings(cfg_dir, cfg_dimension, 1, 1)
    if not cfg_embeddings or len(cfg_embeddings) == 0:
        print("Error: CFG embeddings is empty.")
        return None, None
    cfg_embedding = cfg_embeddings[0].cpu().detach().numpy() if torch.is_tensor(cfg_embeddings[0]) else cfg_embeddings[0]

    # ----------------- semantic embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder("model/unixcoder-base-nine")
    model.to(device)
    with open(funcfile, 'r', encoding='utf-8') as fp:
        code_content = fp.read()
    semantic_vector = get_semantic_embedding(model, device, code_content)
    semantic_vector = semantic_vector.cpu().detach().numpy() if torch.is_tensor(semantic_vector) else semantic_vector
    if semantic_vector.ndim > 1:
        semantic_vector = semantic_vector[0]
    
    # -------- return
    if cfg_embedding is None or (isinstance(cfg_embedding, np.ndarray) and cfg_embedding.size == 0):
        print("Error: CFG vector is empty.")
        return None, None
    if semantic_vector is None or (isinstance(semantic_vector, np.ndarray) and semantic_vector.size == 0):
        print("Error: Semantic vector is empty.")
        return None, None
    return cfg_embedding, semantic_vector

def build_all_index():
    class_list = ["video", "image", "network", "game", "database"]#["test"]#
    for class_name in class_list:
        build_cfg_graph(class_name)
        generate_metadata(class_name)
        build_classify_index(class_name)

def clear_directory_if_not_empty(directory_path):
    if not os.path.exists(directory_path):
        print(f"{directory_path} not exists")
        return

    if not os.listdir(directory_path):
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Can't delete {file_path}, for: {e}")


def select_metadata(outfile, funcfile, obfuse_class, num):
    #Path("out").mkdir(parents=True, exist_ok=True)
    index_dir = 'index'
    index_file = f'{index_dir}/{obfuse_class}.index'
    metadata_file = f'{index_dir}/{obfuse_class}.json'
    tmp_dir = 'obfus_json/tmp'
    cfg_dir = 'obfus_json/cfg'
    clear_directory_if_not_empty(cfg_dir)
    # get target function vectors
    f1, f2 = get_target_embeddings(funcfile, tmp_dir, cfg_dir)
    if len(f1)==0 or len(f2)==0:
        return
    # load index
    index = faiss.read_index(index_file)
    n_total = index.ntotal
    scores = []
    for i in range(n_total):
        combined_vector = index.reconstruct(i)
        e1 = combined_vector[:cfg_dimension]
        e2 = combined_vector[cfg_dimension:]
        score = calculate_score(e1,e2,f1,f2)
        scores.append([i, score])
    scores.sort(key=lambda x: x[1], reverse=True)
    print(scores)
    if num==0:
        top_scores = scores
    else:
        top_scores = scores[:num]
    selected_ids = [score[0] for score in top_scores]
    selected_metadata = []
    with open(metadata_file, 'r', encoding='utf-8') as fp:
        input_json = json.load(fp)
    for i in selected_ids:
        selected_metadata.append(input_json[i])
    with open(outfile, 'w') as fp:
        json.dump(selected_metadata, fp, ensure_ascii=False, indent=4)
    print(f"The results were saved in {outfile}")
    return selected_metadata


def build_insert(root_dir, metadata, max_num):
    insert_dir = os.path.join(root_dir, 'insert')
    if Path(insert_dir).is_dir():
        shutil.rmtree(insert_dir)
    os.mkdir(insert_dir)
    loop_list = []
    if_list = []
    normal_list = []
    loop_index = 0
    if_index = 0
    normal_index = 0
    for json_item in metadata:
        if loop_index<max_num or if_index<max_num or normal_index<max_num:
            if if_index < max_num:
                try:
                    if_con_str = json_item["Branch"]
                except Exception as e:
                    if_con_str = []
                if len(if_con_str) > 0:
                    if isinstance(if_con_str[0], dict):
                        for if_item in if_con_str:
                            try:
                                if_true = if_item["thenstr"]
                                if_false = if_item["elsestr"]
                            except Exception as e:
                                if_true = None
                                if_false = None
                            if if_true or if_false:
                                if_list.append({"iftrue":if_true, "iffalse": if_false})
                                if_index += 1
                    elif isinstance(if_con_str[0], str):
                        pass
            
            if loop_index < max_num:
                loop_con_str = json_item["Loop"]
                if len(loop_con_str) > 0:
                    if isinstance(loop_con_str[0], dict):
                        for loop_item in loop_con_str:
                            try:
                                loop_str = loop_item["loopstr"]
                            except Exception as e:
                                loop_str = None
                            if loop_str:
                                loop_list.append(loop_str)
                                loop_index += 1
                    elif isinstance(loop_con_str[0], str):
                        loop_str = loop_con_str
                        if loop_str:
                            loop_list.append(loop_str)
                            loop_index += 1
            
            if normal_index < max_num:
                if len(json_item["Normal"]) > 0:
                    if isinstance(json_item["Normal"][0], dict):
                        continue
                for normal_item in json_item["Normal"]:
                    if normal_item:
                        normal_list.append(normal_item)
                        normal_index += 1
    if_index = 0
    for item in if_list:
        file_path = os.path.join(insert_dir, f"if_else{if_index}.txt")
        true_out_list = []
        for true_item in item["iftrue"]:
            if true_item:
                true_item = true_item.replace('\n', '\\n').replace('\r', '\\r')
                true_item = true_item.replace('If-Condition-Str: ', '')
                true_item = true_item.replace('If-Condition-Str', '')
                true_out_list.append(true_item + '\n')
        false_out_list = []
        for false_item in item["iffalse"]:
            if false_item:
                false_item = false_item.replace('\n', '\\n').replace('\r', '\\r')
                false_item = false_item.replace('If-Condition-Str: ', '')
                false_item = false_item.replace('If-Condition-Str', '')
                false_out_list.append(false_item + '\n')
        if false_out_list:
            false_out_list[-1] = false_out_list[-1].rstrip('\n')
        
        if true_out_list or false_out_list:
            with open(file_path, 'w') as fp:
                fp.writelines(true_out_list)
                fp.write('\n')
                fp.writelines(false_out_list)
            if_index += 1
    
    loop_index = 0
    for item in loop_list:
        file_path = os.path.join(insert_dir, f"loop{loop_index}.txt")
        loop_out_list = []
        for loopstr in item:
            if loopstr:
                loopstr = loopstr.replace('Loop-Condition-Str: ', '')
                loopstr = loopstr.replace('Loop-Condition-Str', '')
                loopstr = loopstr.replace('\n', '\\n').replace('\r', '\\r')
                loop_out_list.append(loopstr + '\n')
        if loop_out_list:
            loop_out_list[-1] = loop_out_list[-1].rstrip('\n')
            with open(file_path, 'w') as fp:
                fp.writelines(loop_out_list)
            loop_index += 1
    num_content = f"{loop_index}\n{if_index}"
    num_file_path = os.path.join(insert_dir, "num.txt")
    with open(num_file_path, 'w') as fp:
        fp.write(num_content)
    no_file_path = os.path.join(insert_dir, "normal.txt")
    normal_out_list = []
    for item in normal_list:
        if item:
            normal_str = item.replace('\n', '\\n').replace('\r', '\\r')
            normal_str = normal_str.replace('Normal-Str: ', '')
            normal_str = normal_str.replace('Normal-Str', '')
            normal_out_list.append(normal_str + '\n')
    if normal_out_list:
        normal_out_list[-1] = normal_out_list[-1].rstrip('\n')
    with open(no_file_path, 'w') as fp:
        fp.writelines(normal_out_list)

# ----------------------------  main ----------------------------
'''
def main():
    funcfile = "input/aopt_6941 aopt_help.txt"
    outfile = "input/out.json"
    obfu_data = select_metadata(outfile, funcfile, "network", 0)
    #print(obfu_data)


main()
'''

#index = faiss.read_index("index/codebert_vectors.index")
#num_vectors = index.ntotal
#print(num_vectors)
#vectors = index.reconstruct_n(0, num_vectors)
#print(vectors)
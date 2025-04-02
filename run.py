from DQN import *
from core_gym_env import CoreEnv
from pathlib import Path
import shutil
import os
from multiprocessing import Process, Pipe, set_start_method
import numpy as np
from config import *
from time import sleep
from sem_obfus_core import set_obfus_type, run_record_pass
from get_obfus_json import select_metadata, build_insert


def create_gym_env(num, sample_name, funcname):
    train_envs = []
    
    base_dir = os.path.join('work', 'base')     # work/base
    sample_dir = os.path.join(base_dir, sample_name) # work/base/funcname
    insert_dir = os.path.join('work', 'insert') # work/insert
    tmp_dir = os.path.join('work', 'tmp')       # work/tmp
    tmp_insert_dir = os.path.join(tmp_dir, 'insert')
    if not os.path.exists(insert_dir):
        print("not find insert directory")
        return train_envs
    if Path(tmp_dir).is_dir():
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    if Path(tmp_insert_dir).is_dir():
        shutil.rmtree(tmp_insert_dir)
    shutil.copytree(insert_dir, tmp_insert_dir)
    sample_ll_path = os.path.join(sample_dir, "bin.ll")
    shutil.copy2(sample_ll_path, os.path.join(tmp_dir, "bin_src.ll"))
    obfus_elf_path = os.path.join(base_dir, "seobfus")
    shutil.copy2(obfus_elf_path, os.path.join(tmp_dir, "seobfus"))
    run_record_pass(tmp_dir, funcname)
    record_path = os.path.join(tmp_dir, "record.txt")
    tag_ll_path = os.path.join(tmp_dir, "bin.ll")
    if not os.path.exists(record_path) or not os.path.exists(tag_ll_path):
        print("run record_pass failed: no record.txt or bin.ll found")
        return train_envs
    
    #base_sh = ["build.sh", "get_bin_code.sh", "build_insert.sh", "get_insert_bin_code.sh"]
    gpu_count = torch.cuda.device_count()
    for it in range(num):
        gpu_id = it % gpu_count
        work_dir = os.path.join('work', str(it))
        if Path(work_dir).is_dir():
            shutil.rmtree(work_dir)
        os.mkdir(work_dir)

        for sh_name in ["get_bin_code.sh", "get_insert_bin_code.sh"]:
            sh_path = os.path.join(base_dir, sh_name)
            shutil.copy2(sh_path, os.path.join(work_dir, sh_name))
        for sh_name in ["build.sh", "build_insert.sh"]:
            sh_path = os.path.join(sample_dir, sh_name)
            shutil.copy2(sh_path, os.path.join(work_dir, sh_name))

        shutil.copy2(record_path, os.path.join(work_dir, "record.txt"))
        shutil.copy2(tag_ll_path, os.path.join(work_dir, "bin.ll"))
        # copy seobfus obfuscator
        shutil.copy2(obfus_elf_path, os.path.join(work_dir, "seobfus"))
        
        # Create env
        train_env = CoreEnv(work_dir, funcname, gpu_id)
        train_envs.append(train_env)
    return train_envs


def sub_process_env_step(env, conn, obfus_type):
    set_obfus_type(obfus_type)
    done = False
    while not done:
        action = conn.recv()
        res = env.step(action)
        done = res[2]
        conn.send(res)
    conn.close()

# ---------------------------------------------------------------- training ----------------------------------------------------------------
train_log_base_dir = 'train_log'
def write_train_log(train_log_dir, out_name, out_data):
    outfile = os.path.join(train_log_dir, out_name)
    with open(outfile, 'a') as f:
        f.write(out_data + '\n')

def an_train_episode(envs, dqn_agent, obfus_type, iteration, train_log_dir):
    num = len(envs)
    finished = [False for _ in range(num)]
    cur_states = [env.reset().reshape(1, 1, embedding_len) for env in envs]
    conns = [Pipe() for _ in range(num)]
    processes = [Process(target=sub_process_env_step, args=(envs[idx], conns[idx][1], obfus_type)) for idx in range(num)]
    for p in processes:
        p.start()
    while True:
        has_active = False
        actions = [-1 for _ in range(num)]
        for idx in range(num):
            if not finished[idx]:
                has_active = True
                action = dqn_agent.act(cur_states[idx])
                conns[idx][0].send(action)
                sleep(0.5)
                actions[idx] = action
        if not has_active:
            break
        for idx in range(num):
            if not finished[idx]:
                new_state, reward, done, log_dict = conns[idx][0].recv()
                new_state = new_state.reshape(1, 1, embedding_len)
                # record
                total_score = log_dict['total_score']
                if total_score:
                    out_data = f"{total_score} --> {reward}"
                else:
                    out_data = f"[] --> {reward}"
                out_name = f"{iteration}-{idx}.log"
                write_train_log(train_log_dir, out_name, out_data)
                if log_dict['reset']:  # error process
                    finished[idx] = True
                    continue
                else:
                    finished[idx] = done
                    if done:
                        conns[idx][0].close()
                dqn_agent.remember(cur_states[idx], actions[idx], reward, new_state, done)
                cur_states[idx] = new_state
    for _ in range(3):
        dqn_agent.replay()
    dqn_agent.target_train()
    for p in processes:
        p.join()

# generate train instance from config.py
def generate_train_instance():
    train_samples = []
    for samplename, func_list in train_samples_list.items():
        for funcname in func_list:
            func_type = train_func_types[funcname]
            for obfus_type in sem_types:
                if obfus_type == func_type:
                    continue
                train_samples.append({"sample_name":samplename, "func_name":funcname, "func_type":func_type, "obfus_type":obfus_type})
    return train_samples

# mixed similarity based obfus sem select
def build_insert_sem(sample_name, funcname, obfus_type):
    sample_dir = os.path.join('work', 'base')
    sample_dir = os.path.join(sample_dir, sample_name)
    funcfile = os.path.join(sample_dir, f"{funcname}.txt")
    outfile = "obfus_json/out.json"
    insert_data = select_metadata(outfile, funcfile, obfus_type, 0)
    build_insert('work', insert_data, 100)

def train():
    set_start_method('spawn')
    train_samples = generate_train_instance()
    samples_num = len(train_samples)
    print(f"Finish generate train instance, train sample num: [{samples_num}], show follow:")
    print(train_samples)
    print("====================================")

    model_save_path = 'checkpoints'             #os.path.join('checkpoints', f"{sample_name}-{obfus_type}")
    if Path(model_save_path).is_dir():
        shutil.rmtree(model_save_path)
    os.mkdir(model_save_path)
    if Path(train_log_base_dir).is_dir():
        shutil.rmtree(train_log_base_dir)
    os.mkdir(train_log_base_dir)
    num_iterations = 5
    current_iteration = 1
    dqn_agent = DQN(model_path=None)
    # training....
    for sample in train_samples:
        train_log_dir = os.path.join(train_log_base_dir, "{}-{}".format(sample["func_name"], sample["obfus_type"]))
        if Path(train_log_dir).is_dir():
            shutil.rmtree(train_log_dir)
        os.mkdir(train_log_dir)
        build_insert_sem(sample["sample_name"], sample["func_name"], sample["obfus_type"])
        train_envs = create_gym_env(5, sample["sample_name"], sample["func_name"])
        if not train_envs:
            print("[{}-{}-{}]: create env error".format(sample["sample_name"], sample["func_name"], sample["obfus_type"]))
            #print(f"[{sample["sample_name"]}-{sample["func_name"]}-{sample["obfus_type"]}]: create env error")
            continue
        for iteration in range(1, num_iterations+1):
            for env in train_envs:
                env.set_episod_count(current_iteration)
            an_train_episode(train_envs, dqn_agent, sample["obfus_type"], current_iteration, train_log_dir)
            if current_iteration % 5 == 0:
                dqn_agent.save_model(model_save_path, current_iteration)
            if current_iteration % 20 == 0:
                dqn_agent.reduce_lr()
            current_iteration += 1


# ---------------------------------------------------------------- evaluate ------------------------
evaluation_dir = 'evaluation'
def write_results(out_name, out_data):
    outfile = os.path.join(evaluation_dir, out_name)
    with open(outfile, 'w') as f:
        f.write(out_data + '\n')

def sub_process_eval_step(env, conn):
    done = False
    while not done:
        action = conn.recv()
        res = env.step(action, eval_tag=True)
        done = res[2]
        conn.send(res)
    conn.close()

def copy2out(sample_out_dir, iteration, idx, bin_code):
    out_name = f"{iteration}-{idx}"
    out_bin_file_name = f"{out_name}.exe"
    out_code_file_name = f"{out_name}.txt"
    out_bin_path = os.path.join(sample_out_dir, out_bin_file_name)
    out_code_path = os.path.join(sample_out_dir, out_code_file_name)
    in_bin_path = os.path.join('work', f"{idx}")
    in_bin_path = os.path.join(in_bin_path, 'bin_insert.exe')
    if os.path.exists(in_bin_path):
        shutil.copy2(in_bin_path, out_bin_path)
    else:
        print(f"[{iteration}-{idx}]: bin_insert.exe not found")
    if bin_code:
        with open(out_code_path, 'w') as fp:
            fp.write(bin_code)
    else:
        print(f"[{iteration}-{idx}]: code none")

def an_eval_episode(envs, dqn_agent, iteration, sample_out_dir):
    num = len(envs)
    finished = [False for _ in range(num)]
    cur_states = [env.reset().reshape(1, 1, embedding_len) for env in envs]
    conns = [Pipe() for _ in range(num)]
    processes = [Process(target=sub_process_eval_step, args=(envs[idx], conns[idx][1])) for idx in range(num)]
    for p in processes:
        p.start()
    while True:
        has_active = False
        actions = [-1 for _ in range(num)]
        for idx in range(num):
            if not finished[idx]:
                has_active = True
                action = dqn_agent.act(cur_states[idx], True)
                conns[idx][0].send(action)
                sleep(0.5)
                actions[idx] = action
        if not has_active:
            break
        for idx in range(num):
            if not finished[idx]:
                new_state, bin_code, done = conns[idx][0].recv()
                if new_state is None:
                    print("get error state")
                    finished[idx] = True
                    continue
                new_state = new_state.reshape(1, 1, embedding_len)
                finished[idx] = done
                if done:
                    copy2out(sample_out_dir, iteration, idx, bin_code)
                    conns[idx][0].close()
                cur_states[idx] = new_state
    for p in processes:
        p.join()

def generate_eval_instance():
    eval_samples = []
    for samplename, func_list in eval_samples_list.items():
        for funcname in func_list:
            func_type = eval_func_types[funcname][0]
            obfus_type = eval_func_types[funcname][1]
            eval_samples.append({"sample_name":samplename, "func_name":funcname, "func_type":func_type, "obfus_type":obfus_type})
    return eval_samples

def eval():
    set_start_method('spawn')
    eval_samples = generate_eval_instance()
    samples_num = len(eval_samples)
    print(f"Finish generate eval instance, eval sample num: [{samples_num}], show follow:")
    print(eval_samples)
    print("====================================")
    model_path = os.path.join('checkpoints', 'dqn_model-100.pt')
    out_dir = os.path.join('work', 'out')
    if Path(out_dir).is_dir():
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    
    num_iterations = 1
    current_iteration = 1
    dqn_agent = DQN(model_path=model_path)
    # eval....
    for sample in eval_samples:
        sample_out_dir = os.path.join(out_dir, "{}-{}".format(sample["func_name"], sample["obfus_type"]))
        if Path(sample_out_dir).is_dir():
            shutil.rmtree(sample_out_dir)
        os.mkdir(sample_out_dir)
        build_insert_sem(sample["sample_name"], sample["func_name"], sample["obfus_type"])
        eval_envs = create_gym_env(5, sample["sample_name"], sample["func_name"])
        if not eval_envs:
            print("[{}-{}-{}]: create env error".format(sample["sample_name"], sample["func_name"], sample["obfus_type"]))
            continue
        for iteration in range(1, num_iterations+1):
            for env in eval_envs:
                env.set_episod_count(current_iteration)
            an_eval_episode(eval_envs, dqn_agent, current_iteration, sample_out_dir)
            current_iteration += 1
            


if __name__ == "__main__":
    #train()
    eval()
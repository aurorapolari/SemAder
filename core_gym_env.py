import os.path

import gym
from gym.utils import seeding
from gym import utils
from gym import error, spaces
import numpy as np
from sem_obfus_core import *
from config import *

_dtype = np.float32


class CoreEnv(gym.core.Env):
    def __init__(self, work_dir, funcname, gpu_id):
        super(CoreEnv, self).__init__()
        self.work_dir = work_dir
        self.funcname = funcname
        self.gpu_id = gpu_id
        self.id = gpu_id
        
        self.num_obfu_point = 0
        self._action_set = action_set
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Space(shape=(embedding_len,), dtype=_dtype)
        self._episode_count = 0
        self.cur_obfu_point = 0
        

        self.insert_list = []
        self.mark_list = []
        
        # Initialize the embedding model
        print(f"[{self.work_dir}]: Initializing embedding model...")
        self.model_info = initial_model(self.gpu_id)
        # Get the record list
        self.record_list, self.num_obfu_point = get_record_list(self.work_dir)
        # Get the initial state space
        insert_record = self.record_list[self.cur_obfu_point]
        insert_parts = insert_record.split('|', 2)
        insert_record = insert_parts[0] + '|' + insert_parts[1] + '|' + '<INSERTION_POINT>'
        self.mark_list.append("2|" + insert_record)
        #self.insert_list.append("2|" + insert_record)
        self._original_state = get_state(self.model_info, self.work_dir, self.mark_list, self.funcname, self.id)     # Get the initial state
        self._state = self._original_state
        


    def set_episod_count(self, count):
        self._episode_count = count
    
    def _is_end(self):
        return self.cur_obfu_point >= self.num_obfu_point

    def _reset(self):
        self._state = self._original_state
        self.cur_obfu_point = 0
        self.insert_list.clear()
        self.mark_list.clear()
        return np.array(self._state, dtype=_dtype), 0.0, True, {'reset': True}
        
    
    def reset(self):
        return self._reset()[0]


    def step(self, action, eval_tag = False):
        if eval_tag is False:
            insert_record = self.record_list[self.cur_obfu_point]
            self.insert_list.append(str(action) + "|" + insert_record)
            self.mark_list = list(self.insert_list)
            progress_str = f"{self.cur_obfu_point + 1}/{self.num_obfu_point}"
            self.cur_obfu_point += 1
            if self._is_end():
                self.mark_list.append('')
            else:
                insert_record = self.record_list[self.cur_obfu_point]
                insert_parts = insert_record.split('|', 2)
                insert_record = insert_parts[0] + '|' + insert_parts[1] + '|' + '<INSERTION_POINT>'
                self.mark_list.append("2|" + insert_record)
            state, reward, json_data = apply_obfu(self.model_info, self.work_dir, self.mark_list, self.funcname, progress_str, self.id)
            #print(f"[{self.work_dir}: {self.cur_obfu_point + 1}/{self.num_obfu_point}]: [reward:{reward}]")
            
            if self._is_end():
                return state, reward, True, {'reset': False, 'total_score': json_data}
            else:
                return state, reward, False, {'reset': False, 'total_score': json_data}
        else:
            insert_record = self.record_list[self.cur_obfu_point]
            self.insert_list.append(str(action) + "|" + insert_record)
            self.mark_list = list(self.insert_list)
            progress_str = f"[{self.funcname}]:[{self.cur_obfu_point + 1}/{self.num_obfu_point}]"
            self.cur_obfu_point += 1
            if self._is_end():
                self.mark_list.append('')
            else:
                insert_record = self.record_list[self.cur_obfu_point]
                insert_parts = insert_record.split('|', 2)
                insert_record = insert_parts[0] + '|' + insert_parts[1] + '|' + '<INSERTION_POINT>'
                self.mark_list.append("2|" + insert_record)
            state, bin_code = apply_eval_obfu(self.model_info, self.work_dir, self.mark_list, self.funcname, progress_str, self.id)
            
            if self._is_end():
                return state, bin_code, True
            else:
                return state, bin_code, False
```
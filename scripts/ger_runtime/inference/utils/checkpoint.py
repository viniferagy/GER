import json
import os

def extract_best_metric(trainer_state_path, metric_name):
    with open(trainer_state_path, 'r') as file:
        trainer_state = json.load(file)
    
    # 提取所有的评估记录
    eval_records = [record for record in trainer_state['log_history'] if metric_name in record]
    
    # 如果没有找到评估记录，则返回None
    if not eval_records:
        return None
    
    # 选择具有最高评估指标的记录
    if "loss" in metric_name:
        best_record = min(eval_records, key=lambda x: x[metric_name])
    else:
        best_record = max(eval_records, key=lambda x: x[metric_name])
    
    # 提取最佳记录中的关键信息
    best_step = best_record['step']
    best_metric_value = best_record[metric_name]

    # get checkpoint dir
    if best_step == trainer_state['global_step']:
        checkpoint_dir = os.path.dirname(trainer_state_path)
    else:
        # best checkpoint is not the current checkpoint
        root_dir = os.path.dirname(trainer_state_path)
        if "checkpoint-" in root_dir.split('/')[-1]:
            root_dir = os.path.dirname(root_dir)
        checkpoint_dir = os.path.join(os.path.dirname(trainer_state_path), f'checkpoint-{best_step}')
    
    return {
        'step': best_step,
        'metric_value': best_metric_value,
        'checkpoint_dir': checkpoint_dir
    }

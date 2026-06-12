import os

import logging
from tqdm import tqdm
import re

from transformers import AutoTokenizer
import datasets
from datasets import Dataset

from .dataset_wrapper.wrapper import BasicWrapper
from .dataset_wrapper.WILocnessWrapper import WILocnessWrapper
from configs.config import get_data_dir

logger = logging.getLogger(__name__)

class GeneralDataset:
    def __init__(self, args, model_args, single_dataset_name) -> None:
        '''
        Provide a standard seq2seq interface for all supported datasets.
        '''
        self.args = args
        self.dataset_name = single_dataset_name
        self.data_dir = get_data_dir(single_dataset_name)
        assert single_dataset_name in args.datasets, 'Required dataset is not included in the `datasets` arguments'

        if self.dataset_name.lower() == 'wilocness':
            self.wrapper = WILocnessWrapper(args, self.data_dir)
        else:
            self.wrapper = BasicWrapper(args, self.data_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        except:
            self.tokenizer = None


    def get_dataset_map(self, split=None):
        '''
        Dynamic Load data split, if not identified, load all split.
        {'train': (Iterable)Dataset, 'valid': (Iterable)Dataset, 'test': (Iterable)Dataset}
        '''
        if split:
            assert split in ['train', 'valid', 'test']
            dataset_map = {}
            dataset_map[split] = self.wrapper.get_dataset(split=split)
            # dataset_map['valid'] = self._get_dataset(split='valid')
            for s in ['train', 'valid', 'test']:
                if s not in dataset_map:
                    dataset_map[s] = []
            if self.args.pre_split_length_for_infer and dataset_map["test"] != []:
                dataset_map['test'] = self.split_sentence_for_dataset(dataset_map['test'], dataset_flag='test')
            return dataset_map
        else:
            train_set = self.wrapper.get_dataset('train')
            val_set = self.wrapper.get_dataset('valid')
            test_set = self.wrapper.get_dataset('test')
            if self.args.pre_split_length_for_infer:
                test_set = self.split_sentence_for_dataset(test_set, dataset_flag='test')
            return {'train': train_set, 'valid': val_set, 'test': test_set}
    
    def get_standard_dataset_map(self, split=None):
        dataset_map = self.get_dataset_map(split=split)
        for split in dataset_map:
            if dataset_map[split]:
                old_dataset: Dataset = dataset_map[split]
                id_list = list(old_dataset['id'])
                id_list = [str(item) for item in id_list]
                text_list = list(old_dataset['text'])
                dataset_name = [self.dataset_name]*len(id_list)
                dataset_dict = {'id': id_list, 'from': dataset_name, 'text': text_list}
                if 'label' in old_dataset.column_names:
                    label_list = list(old_dataset['label'])
                    dataset_dict['label'] = label_list
                dataset_map[split] = datasets.Dataset.from_dict(dataset_dict)
        return dataset_map
        
    def split_sentence_for_dataset(self, loaded_dataset, dataset_flag):
        assert loaded_dataset, "Null Dataset"
        logger.info(f"Splitting sentences in {dataset_flag}. The id, text will be retained. id will be add a prefix for split order. label will remain original shape without split.")
        # if self.dataset_name == "mucgec":
        #     rePERIOD = re.compile(r'(?<=，|,|。|!|！|\?|？)(?!”)')
        # else:
        #     rePERIOD = re.compile(r'(?<=，|,)')
        if self.dataset_name in ["mucgec", "fangzhenggrammar"]:
            rePERIOD = re.compile(r'(?<=，|,|。|!|！|\?|？)(?!”)')
        elif self.dataset_name == "wilocness":
            rePERIOD = re.compile(r'(?<=\.|!|\?)(?!")')     # TODO: avoid split float number
        else:
            raise NotImplementedError()
        new_dataset = []
        max_len = self.args.pre_split_length_for_infer
        for item in tqdm(loaded_dataset):
            original_id = item["id"]
            line = item["text"]
            line = line.strip()
            line = re.split(rePERIOD, line)
            if line[-1] == '':
                line = line[:-1]
            idx = 0
            buff = ''
            for s in line:
                # if longer than max lenght than split it
                if len(self.tokenizer.encode(buff + s)) >= max_len and buff != '':
                    new_id = f"{original_id}#{idx}#{buff[-1] if buff.endswith((',', '，')) else 'P'}"
                    new_text = str(buff)
                    if "label" in item:
                        new_dataset.append({"id": new_id, "text": new_text, "label": item["label"]})
                    else:
                        new_dataset.append({"id": new_id, "text": new_text})
                    idx += 1
                    buff = s
                else:
                    buff += s
                # if not end with comma split it!
                if not buff.endswith((',', '，')) and self.dataset_name == "mucgec":
                    new_id = f"{original_id}#{idx}#P"
                    new_text = str(buff)
                    if "label" in item:
                        new_dataset.append({"id": new_id, "text": new_text, "label": item["label"]})
                    else:
                        new_dataset.append({"id": new_id, "text": new_text})
                    idx += 1
                    buff = ''
            if buff != '':
                new_id = f"{original_id}#{idx}#P"
                new_text = str(buff)
                if "label" in item:
                    new_dataset.append({"id": new_id, "text": new_text, "label": item["label"]})
                else:
                    new_dataset.append({"id": new_id, "text": new_text})


        dict_dataset = {"id": [item["id"] for item in new_dataset], "text": [item["text"] for item in new_dataset]}
        if "label" in new_dataset[0]:
            dict_dataset["label"] = [item["label"] for item in new_dataset]
        logger.info(f"Inputs length before merged: {len(loaded_dataset)}; After merged: {len(new_dataset)}")
        return datasets.Dataset.from_dict(dict_dataset)

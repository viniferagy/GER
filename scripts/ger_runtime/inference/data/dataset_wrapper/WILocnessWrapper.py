import os
import json
import datasets


class WILocnessWrapper:
    def __init__(self, args, directory) -> None:
        '''
        BEA 19 WI&Locness dataset.
        train, valid split available; test split is EQUAL to validation.
        '''
        self.args = args
        self.data_dir = directory
        self.split_files = {
            'train': os.path.join(self.data_dir, 'train.json'),
            'valid': os.path.join(self.data_dir, 'valid.json'),
            'test': os.path.join(self.data_dir, 'test.json'),
        }

    def _load_json_and_formatted(self, file_path):
        data = json.load(open(file_path))
        if type(data) == list:
            assert len(data) != 0
            new_data = {}
            if 'id' not in data[0]:
                new_data['id'] = list(range(0, len(data)))
            for key in data[0]:
                new_data[key] = [item[key] for item in data]
            return new_data
        else:
            raise NotImplementedError()

    def get_dataset(self, split=None)-> dict:
        assert split in ['train', 'valid', 'test']
        split_file = self.split_files[split]
        assert os.path.exists(split_file), f"WI+LOCNESS split `{split}` does not exist: {split_file}"
        json_data = self._load_json_and_formatted(split_file)
        return datasets.Dataset.from_dict(json_data)

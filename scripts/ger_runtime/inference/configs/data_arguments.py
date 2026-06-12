from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    Arguments which define the model and tokenizer to load.
    """
    datasets: str = field(
        default=None, 
        metadata={
            "help": "GER dataset mapping, e.g. 'conll14:wilocness'."
        }
    )
    icl_datasets: str = field(
        default=None, 
        metadata={
            "help": "Optional explicit train dataset names used by the GER ICL dataset loader."
        }
    )
    prompts: str = field(
        default=None,
        metadata={
            "help": (
                "prompt name saved in TEMPLATE in data/instructions/template.py. The num is either 1 or matched with dataset_name. (multiple prompt should be split by comma)"
            )
        }
    )
    streaming: bool = field(
        default=False,
        metadata={"help": ("Reserved compatibility flag; formal GER does not stream datasets.")}
    )
    pre_split_length_for_infer: bool = field(
        default=False,
        metadata={"help": ("Optional sentence splitting for long inference inputs.")}
    )
    infer_mode: str = field(
        default=None,
        metadata={
            "help": (
                "Inference split selector. None or `test` loads the test split; "
                "`eval`, `valid`, or `validation` loads the valid split; `train` "
                "loads the train split."
            )
        }
    )

    def to_dict(self):
        return dict(self.__dict__)
    
    def __post_init__(self):
        pass

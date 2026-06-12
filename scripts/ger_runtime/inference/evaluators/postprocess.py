import codecs
import json
import logging
import os
import re
import zipfile

import spacy


SPACY_BLANK_MAP = {
    "en_core_web_sm": "en",
    "de_core_news_sm": "de",
    "et_dep_ud_sm": "et",
    "ro_core_news_sm": "ro",
}

RETOKENIZATION_RULES = [
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]

logger = logging.getLogger(__name__)
_EN_TOKENIZER = None


def load_spacy_or_blank(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank(SPACY_BLANK_MAP[model_name])


def english_tokenizer():
    global _EN_TOKENIZER
    if _EN_TOKENIZER is None:
        _EN_TOKENIZER = load_spacy_or_blank("en_core_web_sm")
    return _EN_TOKENIZER


class PostProcessManipulator:
    en_tokenize = "en_tokenize"
    de_tokenize = "de_tokenize"
    et_tokenize = "et_tokenize"
    ro_tokenize = "ro_tokenize"
    replace_linebreaker = "replace_linebreaker"
    conll14 = "conll14"
    bea19 = "bea19"


class PostProcess:
    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = args.dir
        self.results_file = os.path.join(self.save_dir, "predictions.jsonl")
        self.results = [json.loads(item.strip()) for item in open(self.results_file, encoding="utf-8")]

        os.makedirs(self.save_dir, exist_ok=True)
        self._temp_key_transform()
        self._post_process_identification()

        self.post_process_func_map = {
            PostProcessManipulator.replace_linebreaker: self._replace_linebreaker,
            PostProcessManipulator.en_tokenize: self._en_retokenize,
            PostProcessManipulator.de_tokenize: self._de_retokenize,
            PostProcessManipulator.et_tokenize: self._et_retokenize,
            PostProcessManipulator.ro_tokenize: self._ro_retokenize,
            PostProcessManipulator.conll14: self._conll14,
            PostProcessManipulator.bea19: self._bea19,
        }

        self.allowed_dataset = {
            PostProcessManipulator.en_tokenize: {"wilocness"},
            PostProcessManipulator.de_tokenize: {"falko_merlin", "falko_merlin_train"},
            PostProcessManipulator.et_tokenize: {"estgec", "estgec_train"},
            PostProcessManipulator.ro_tokenize: {"rogec", "rogec_train"},
            PostProcessManipulator.conll14: {"conll14"},
            PostProcessManipulator.bea19: {"bea19"},
            PostProcessManipulator.replace_linebreaker: None,
        }

    def _temp_key_transform(self):
        converted = []
        for item in self.results:
            result = {"id": item["id"], "src": item["text"], "predict": item["prediction"]}
            if "label" in item:
                result["tgt"] = item["label"]
            converted.append(result)
        self.results = converted

    def _post_process_identification(self):
        dataset = self.args.dataset.lower()
        if dataset == "wilocness":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.en_tokenize]
        elif dataset in {"falko_merlin", "falko_merlin_train"}:
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.de_tokenize]
        elif dataset in {"estgec", "estgec_train"}:
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.et_tokenize]
        elif dataset in {"rogec", "rogec_train"}:
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.ro_tokenize]
        elif dataset == "conll14":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.conll14]
        elif dataset == "bea19":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.bea19]
        else:
            raise NotImplementedError(f"Unsupported GER dataset for runtime postprocess: {self.args.dataset}")

    @staticmethod
    def _reform_contractions(text):
        contractions = {"n't", "'d", "'ll", "'m", "'re", "'s", "'ve"}
        tokens = text.split()
        for idx in range(len(tokens)):
            if tokens[idx] in contractions and idx > 0:
                tokens[idx - 1] = tokens[idx - 1] + tokens[idx]
                tokens[idx] = ""
        return " ".join(token for token in tokens if token)

    @staticmethod
    def conll_postprocess(item):
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))", "'\\1", line)
        line = " ".join(token.text for token in english_tokenizer().tokenizer(line))
        for pattern, replacement in RETOKENIZATION_RULES:
            line = re.sub(pattern, replacement, line)
        item["predict"] = line
        return item

    @staticmethod
    def bea_postprocess(item):
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))", "'\\1", line)
        line = " ".join(token.text for token in english_tokenizer().tokenizer(line))
        line = re.sub(r"(?<=\d)\s+%", "%", line)
        line = re.sub(r"((?:have)|(?:has)) n't", "\\1n't", line)
        line = re.sub(r"^-", "- ", line)
        line = re.sub(r"\s+", " ", line)
        item["predict"] = line
        return item

    def _write_prediction_lines(self, filename, key="predict"):
        output_path = os.path.join(self.save_dir, filename)
        with codecs.open(output_path, "w", "utf-8") as handle:
            for item in self.results:
                handle.write(f"{str(item[key]).strip()}\n")
        return output_path

    def _conll14(self):
        self.results = [PostProcess.conll_postprocess(item) for item in self.results]
        self._write_prediction_lines("conll14.txt")

    def _bea19(self):
        self.results = [PostProcess.bea_postprocess(item) for item in self.results]
        if len(self.results) != 4477:
            raise AssertionError(f"BEA19 output should contain 4477 lines, got {len(self.results)}")
        bea19_path = self._write_prediction_lines("bea19.txt")
        with zipfile.ZipFile(os.path.join(self.save_dir, "bea19.zip"), mode="w") as zipf:
            zipf.write(bea19_path, "bea19.txt")

    def _replace_linebreaker(self):
        for item in self.results:
            item["predict"] = item["predict"].replace("\n", " ").replace("\r", " ").strip()
            if "rogec" in self.args.dataset:
                item["predict"] = item["predict"].replace("(", " ( ").replace("  ", " ").replace("( ", "(")

    def _get_retokenized_predictions(self, tokenizer_model):
        output_path = os.path.join(self.save_dir, f"{self.args.dataset}-output-retokenized.txt")
        with codecs.open(output_path, "w", "utf-8") as handle:
            for item in self.results:
                line = " ".join(token.text for token in tokenizer_model.tokenizer(item["predict"]))
                handle.write(f"{line.strip()}\n")

    def _en_retokenize(self):
        self._get_retokenized_predictions(english_tokenizer())

    def _de_retokenize(self):
        self._get_retokenized_predictions(load_spacy_or_blank("de_core_news_sm"))

    def _et_retokenize(self):
        self._get_retokenized_predictions(load_spacy_or_blank("et_dep_ud_sm"))

    def _ro_retokenize(self):
        self._get_retokenized_predictions(load_spacy_or_blank("ro_core_news_sm"))

    def basic_saving(self):
        processed_json = os.path.join(self.save_dir, f"{self.args.dataset}-processed.json")
        with codecs.open(processed_json, "w", "utf-8") as handle:
            json.dump(self.results, handle, ensure_ascii=False, indent=4)

        self._write_prediction_lines(f"{self.args.dataset}-output-processed.txt")

        retokenized_path = os.path.join(self.save_dir, f"{self.args.dataset}-output-retokenized.txt")
        if not os.path.exists(retokenized_path):
            self._write_prediction_lines(f"{self.args.dataset}-output-retokenized.txt")

        processed_txt = os.path.join(self.save_dir, f"{self.args.dataset}-processed.txt")
        with codecs.open(processed_txt, "w", "utf-8") as handle:
            for item in self.results:
                if "tgt" in item:
                    handle.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
                else:
                    handle.write("%s\t%s\n" % (item["src"], item["predict"]))

        logger.info(f"Results have been stored in {processed_json}.")

    def post_process_func(self):
        dataset = self.args.dataset.lower()
        for name in self.post_process:
            allowed_datasets = self.allowed_dataset[name]
            if allowed_datasets is not None and dataset not in allowed_datasets:
                raise NotImplementedError(f"Unsupported post process {name} for {self.args.dataset}.")
            self.post_process_func_map[name]()

    def post_process_and_save(self):
        self.post_process_func()
        self.basic_saving()

    def get_results(self):
        return self.results

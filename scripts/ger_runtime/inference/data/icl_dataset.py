import logging
import random
import os
import json
import math
import re
import string
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from .constants import BLANK_ITEM

from .dataset import GeneralDataset
from .instructions.template import ICLPromptTemplate

logger = logging.getLogger(__name__)


class InMemoryExamplePool:
    def __init__(self, icl_datasets, icl_config, cross_domain=False) -> None:
        assert icl_datasets, 'No dataset available for ICL.'
        self.icl_datasets = concatenate_datasets(icl_datasets)
        self.icl_config = icl_config
        self.cross_domain = cross_domain
        # self.errorneous_datasets = [item for item in self.icl_datasets if item['text'].strip() != item['label'].strip()]
        self.topk = icl_config.cross_domain_example_num if cross_domain else icl_config.in_domain_example_num
        self.strategy = icl_config.cross_domain_example_mode if cross_domain else icl_config.in_domain_example_mode

        self.prepare_examples()

        if cross_domain:
            logger.info(f"[examples] Cross-domain example pool prepared. Selection mode {self.strategy}")
        else:
            logger.info(f"[examples] In-domain example pool prepared. Selection mode {self.strategy}")

    def random_select(self):
        if self.topk <= 0:
            return []
        return random.choices(self.base.dataset, k=self.topk)
    
    # def random_error_select(self):
    #     return random.choices(self.errorneous_datasets, k=self.topk)

    def select(self, query):
        if self.strategy in ['random', 'default']:
            return self.random_select()
        # elif self.strategy == 'random_erroneous':
        #     return self.random_error_select()
        raise NotImplementedError(
            "This GER runtime supports precomputed --examples_dir retrieval "
            "and random/default in-memory examples only. Legacy text/relation/"
            "edit/bm25 internal retrieval is not part of the GER pipeline."
        )
    
    def prepare_examples(self):
        if self.strategy not in ["random", "default"] and self.topk > 0:
            raise NotImplementedError(
                "Legacy internal ICL retrieval was removed from the GER runtime. "
                "Use the GER representation retriever to write retrieval.jsonl "
                "and pass it through --examples_dir."
            )
        self.base = type("InMemoryExamples", (), {"dataset": self.icl_datasets})()
        




class ICLDataset:
    _errant_annotator = None
    _errant_failed = False

    @staticmethod
    def resolve_inference_split(infer_mode):
        mode = "" if infer_mode is None else str(infer_mode).strip().lower()
        if mode in {"", "test"}:
            return "test"
        if mode in {"eval", "valid", "validation"}:
            return "valid"
        if mode == "train":
            return "train"
        raise ValueError(
            f"Unsupported infer_mode={infer_mode!r}. "
            "Use one of: test, train, valid, validation, eval."
        )

    def __init__(self, data_args, model_args, icl_args) -> None:
        self.args = data_args
        self.model_args = model_args
        self.icl_args = icl_args
        self._hint_type_cache = {}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        except:
            self.tokenizer = None
        # check datasets and prompts
        # datasets example for ICL: wilocness:wilocness,nucle,fce;hsk:hsk,mucgec
        self.dataset_names = data_args.datasets.split(';')
        self.prompt_names = data_args.prompts.split(',')

        assert self.dataset_names is not None and self.prompt_names is not None

        logger.info(f"Dataset chosen: {self.dataset_names}")
        logger.info(f"Prompt chosen: {self.prompt_names}")

        assert len(self.dataset_names) == len(self.prompt_names) or len(self.prompt_names) == 1, "Unmatched length of dataset and prompt. They should have an equal num or prompt num is 1."

        # ICL datasets internal selection map
        self.icl_datasets_names_list = {}

        inference_dataset_names = []
        used_train_datasets = []

        # add dataset relations
        for dataset_mapping in self.dataset_names:
            assert len(dataset_mapping.split(':')) == 2, f"Dataset map relation is not correct. {dataset_mapping}"
            dataset_name, icl_datasets = dataset_mapping.split(':')
            icl_datasets = icl_datasets.split(',')

            assert dataset_name not in self.icl_datasets_names_list, f"{dataset_name} duplicated in the datasets input."
            self.icl_datasets_names_list[dataset_name] = icl_datasets

            inference_dataset_names.append(dataset_name)
            used_train_datasets.extend(icl_datasets)

        # train dataset dedup

        icl_train_datasets = list(set(used_train_datasets))
        # Load all source datasets as the optional cross-domain example pool.
        if data_args.icl_datasets:
            train_datasets_in_args = data_args.icl_datasets.split(',')
            if set(train_datasets_in_args) == set(used_train_datasets):
                logger.info("No contradiction in the arguments involved in ICL.")
                icl_train_datasets = list(set(used_train_datasets))
            elif set(train_datasets_in_args).issubset(set(used_train_datasets)):
                logger.info(f"Warning: Arguments `datasets` cover more train dataset, global ICL selection will be executed in values of datasets map in arguments.")
                icl_train_datasets = list(set(used_train_datasets))
            elif set(used_train_datasets).issubset(set(train_datasets_in_args)):
                logger.info(f"Warning: Arguments `icl_datasets` cover more train dataset, global ICL selection will be executed in icl_datasets in arguments.")
                icl_train_datasets = list(set(train_datasets_in_args))
            else:
                raise ValueError('Contradiction in the arguments `datasets` and `icl_datasets`.')
            
        # Inference dataset dedup
        assert len(inference_dataset_names) == len(list(set(inference_dataset_names))), 'The ICL datasets to infer contain duplicated item.'

        # General dataset mapping    
        self.train_datasets_map = {}
        for dataset_name in icl_train_datasets:
            self.train_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('train')['train']
        
        # Inference dataset mapping
        self.dataset_names = inference_dataset_names
        self.inference_datasets_map = {}
        inference_split = self.resolve_inference_split(data_args.infer_mode)
        for dataset_name in inference_dataset_names:
            self.inference_datasets_map[dataset_name] = GeneralDataset(
                data_args,
                model_args,
                dataset_name,
            ).get_standard_dataset_map(inference_split)[inference_split]
        

        # construct instruction template
        if len(self.prompt_names) == 1:
            self.prompt_names *= len(self.dataset_names)

        if icl_args.examples_dir and os.path.exists(icl_args.examples_dir):
            self._init_examples()
        else:
            self._init_in_memory_examples()
    
    def _init_examples(self):
        self.specific_examples = {}

        self.cross_domain_examples = None

        for dataset_name in self.icl_datasets_names_list:
            examples_file = os.path.join(self.icl_args.examples_dir, dataset_name, 'retrieval.jsonl')
            self.specific_examples[dataset_name] = [json.loads(item) for item in open(examples_file).readlines()]



    def _init_in_memory_examples(self):
        self.specific_examples = {}
        
        dataset_list = list(self.train_datasets_map.values())
        dataset_list = [ds for ds in dataset_list if ds and not (ds[0]['text'] == BLANK_ITEM['text'] and len(ds)==1)]
      
        # Init in-domain example pools.
        for dataset_name in self.icl_datasets_names_list:
            self.specific_examples[dataset_name] = InMemoryExamplePool(
                [self.train_datasets_map[key] for key in self.icl_datasets_names_list[dataset_name]], 
                self.icl_args,
                cross_domain=False
            )

        # Init cross-domain example pool.
        self.cross_domain_examples = InMemoryExamplePool(dataset_list, self.icl_args, cross_domain=True)

    def normalize_similarities(self, similarity_list):
        if not similarity_list:
            return []

        # 提取相似度值
        similarities = [item['similarity'] for item in similarity_list]
        
        # 获取最小和最大值
        min_val = min(similarities)
        max_val = max(similarities)
        
        # 检查是否所有值都相同
        if min_val == max_val:
            # 如果所有值都相同，直接将它们设为 1
            for item in similarity_list:
                item['similarity'] = 1.0
        else:
            # 归一化处理
            for item in similarity_list:
                normalized_similarity = (item['similarity'] - min_val) / (max_val - min_val)
                item['similarity'] = normalized_similarity

    def get_dedup_examples(self, examples, num_limit):
        if not examples:
            return examples
        
        # check keys in example and normalize similarity
        keys = examples[0].keys()

        if 'similarity' in keys:
            self.normalize_similarities(examples)

        if 'id' in keys and 'data_item_id' in keys and 'similarity' in keys:
            # reorder by dedup
            data_item_map = {}
            for example in examples:
                if example["data_item_id"] not in data_item_map:
                    data_item_map[example["data_item_id"]] = example
                    data_item_map[example["data_item_id"]]["times"] = 1
                else:
                    data_item_map[example["data_item_id"]]["similarity"] += example["similarity"]
                    data_item_map[example["data_item_id"]]["times"] += 1

            dedup_examples = list(data_item_map.values())
            dedup_examples = sorted(dedup_examples, key=lambda x:x["similarity"], reverse=True)
            if len(dedup_examples) >= num_limit:
                return dedup_examples
            else:
                logger.info(f"[WARNING IN ICL] After dedup, only {len(dedup_examples)} demonstration left, which do not meet the requirement {num_limit}.")
                # add duplicate examples until num_limit, cover more data_item_id
                while len(dedup_examples) < num_limit:
                    for data_item_id in data_item_map:
                        if data_item_map[data_item_id]["times"] != 1:
                            dedup_examples.append(data_item_map[data_item_id])
                            data_item_map[data_item_id]["times"] -= 1
                return dedup_examples
            
        return examples

    def _dynamic_example_limits(self, examples_data, max_topk):
        """Allocate per-row caps with min/max bounds and target dataset average."""
        if max_topk <= 0:
            return [0] * len(examples_data)

        min_topk = max(0, min(self.icl_args.dynamic_in_domain_example_min, max_topk))
        target_avg = max(0.0, float(self.icl_args.dynamic_in_domain_example_target_avg))
        target_total = int(round(min(target_avg, float(max_topk)) * len(examples_data)))

        available = [len(item.get("in_domain_examples", [])) for item in examples_data]
        lower = [min(min_topk, count) for count in available]
        upper = [min(max_topk, count) for count in available]
        lower_total = sum(lower)
        upper_total = sum(upper)
        target_total = max(lower_total, min(target_total, upper_total))
        remaining = target_total - lower_total

        limits = list(lower)
        if remaining <= 0:
            return limits

        capacities = [upper_i - lower_i for upper_i, lower_i in zip(upper, lower)]
        capacity_total = sum(capacities)
        if capacity_total <= 0:
            return limits

        raw_allocations = [
            (remaining * capacity / capacity_total) if capacity else 0.0
            for capacity in capacities
        ]
        additions = [
            min(capacity, int(math.floor(raw)))
            for capacity, raw in zip(capacities, raw_allocations)
        ]
        assigned = sum(additions)
        leftovers = remaining - assigned
        order = sorted(
            range(len(examples_data)),
            key=lambda idx: (
                raw_allocations[idx] - additions[idx],
                capacities[idx],
                available[idx],
                -idx,
            ),
            reverse=True,
        )
        for idx in order:
            if leftovers <= 0:
                break
            if additions[idx] < capacities[idx]:
                additions[idx] += 1
                leftovers -= 1

        return [limit + add for limit, add in zip(limits, additions)]

    def _truthy_env(self, key):
        return os.environ.get(key, "").strip() in {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}

    def _edited_neighbor_purity(self, examples):
        real_examples = [example for example in examples if example.get("id") != -1]
        if not real_examples:
            return 0.0
        edited = [
            example for example in real_examples
            if str(example.get("text", "")).strip() != str(example.get("label", "")).strip()
        ]
        return len(edited) / len(real_examples)

    def _short_hint_span(self, source, target, fallback):
        source = str(source).strip()
        target = str(target).strip()
        matcher = SequenceMatcher(a=source.split(), b=target.split(), autojunk=False)
        spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            old = " ".join(source.split()[i1:i2]).strip()
            new = " ".join(target.split()[j1:j2]).strip()
            if old or new:
                spans.append((old, new))
        if spans:
            old, new = max(spans, key=lambda pair: max(len(pair[0]), len(pair[1])))
            if old and new:
                return f"{old} -> {new}"
            return old or new
        if "\t" in fallback:
            fallback = fallback.split("\t", 1)[1]
        return fallback.strip()

    def _hint_span_allowed(self, span):
        if not self._truthy_env("GER_RETRIEVAL_HINT_STRICT"):
            return True
        if " -> " not in span:
            return False
        old, new = span.split(" -> ", 1)
        old_compact = "".join(old.split())
        new_compact = "".join(new.split())
        min_chars = int(os.environ.get("GER_RETRIEVAL_HINT_MIN_SIDE_CHARS", "3"))
        max_tokens = int(os.environ.get("GER_RETRIEVAL_HINT_MAX_SIDE_TOKENS", "5"))
        if len(old_compact) < min_chars or len(new_compact) < min_chars:
            return False
        if len(old.split()) > max_tokens or len(new.split()) > max_tokens:
            return False
        return True

    def _retrieval_span_hint_description(self, examples, max_hints):
        hints = []
        seen = set()
        for example in examples:
            if example.get("id") == -1:
                continue
            source = str(example.get("text", "")).strip()
            target = str(example.get("label", "")).strip()
            if not source or source == target:
                continue
            span = self._short_hint_span(source, target, str(example.get("key", "")))
            span = " ".join(span.split())
            if not span or span in seen or not self._hint_span_allowed(span):
                continue
            seen.add(span)
            hints.append(f"- Check whether a similar issue appears near: {span}")
            if len(hints) >= max_hints:
                break
        if not hints:
            return ""
        return "<hints>\n" + "\n".join(hints) + "\n</hints>"

    def _retrieval_hint_type_mode(self):
        return os.environ.get("GER_RETRIEVAL_HINT_TYPE_MODE", "auto").strip().lower() or "auto"

    def _is_english_retrieval_example(self, example):
        source_name = str(example.get("from", "")).lower()
        english_sources = (
            "bea",
            "conll",
            "fce",
            "lang8",
            "nucle",
            "wi+locness",
            "wilocness",
        )
        return any(name in source_name for name in english_sources)

    def _get_errant_annotator(self):
        if ICLDataset._errant_failed:
            return None
        if ICLDataset._errant_annotator is not None:
            return ICLDataset._errant_annotator
        try:
            import errant

            ICLDataset._errant_annotator = errant.load("en")
        except Exception as exc:
            ICLDataset._errant_failed = True
            logger.warning(f"[GER HINT] Failed to load ERRANT for H-type hints; falling back to heuristic labels: {exc}")
            return None
        return ICLDataset._errant_annotator

    def _errant_to_hint_type(self, errant_type):
        edit_type = str(errant_type or "").upper()
        if ":" in edit_type:
            edit_type = ":".join(edit_type.split(":")[1:])
        if "PUNCT" in edit_type:
            return "punctuation"
        if "SPELL" in edit_type:
            return "spelling"
        if "ORTH" in edit_type:
            return "capitalization or orthography"
        if "VERB:SVA" in edit_type:
            return "subject-verb agreement"
        if "VERB:TENSE" in edit_type:
            return "verb tense"
        if "VERB:FORM" in edit_type:
            return "verb form"
        if "VERB" in edit_type:
            return "verb usage"
        if "NOUN:NUM" in edit_type:
            return "noun number"
        if "NOUN:POSS" in edit_type:
            return "possessive noun"
        if "MORPH" in edit_type:
            return "word form"
        if "WO" in edit_type:
            return "word order"
        if "DET" in edit_type:
            return "determiner or article"
        if "PREP" in edit_type:
            return "preposition"
        if "PRON" in edit_type:
            return "pronoun"
        if "CONJ" in edit_type:
            return "conjunction"
        if "ADV" in edit_type:
            return "adverb"
        if "ADJ" in edit_type:
            return "adjective"
        if "NOUN" in edit_type:
            return "noun usage"
        return "word choice"

    def _tokenize_for_hint_type(self, text):
        return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)

    def _is_punctuation_text(self, text):
        text = str(text).strip()
        return bool(text) and all((not ch.isalnum()) and (not ch.isspace()) for ch in text)

    def _edit_similarity(self, old, new):
        return SequenceMatcher(a=old.lower(), b=new.lower(), autojunk=False).ratio()

    def _common_prefix_len(self, old, new):
        count = 0
        for old_ch, new_ch in zip(old.lower(), new.lower()):
            if old_ch != new_ch:
                break
            count += 1
        return count

    def _heuristic_to_hint_type(self, old, new):
        old = str(old).strip()
        new = str(new).strip()
        if not old and not new:
            return ""
        if self._is_punctuation_text(old) or self._is_punctuation_text(new):
            old_non_punct = "".join(ch for ch in old if ch not in string.punctuation).strip()
            new_non_punct = "".join(ch for ch in new if ch not in string.punctuation).strip()
            if not old_non_punct or not new_non_punct or old_non_punct == new_non_punct:
                return "punctuation"
        if "".join(old.split()).lower() == "".join(new.split()).lower() and old.split() != new.split():
            return "spacing or tokenization"
        if old.lower() == new.lower() and old != new:
            return "capitalization or orthography"

        old_tokens = self._tokenize_for_hint_type(old)
        new_tokens = self._tokenize_for_hint_type(new)
        old_words = [token for token in old_tokens if any(ch.isalnum() for ch in token)]
        new_words = [token for token in new_tokens if any(ch.isalnum() for ch in token)]
        if old_words and new_words and Counter(token.lower() for token in old_words) == Counter(token.lower() for token in new_words):
            if [token.lower() for token in old_words] != [token.lower() for token in new_words]:
                return "word order"

        function_words = {
            "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "with", "by",
            "and", "or", "but", "if", "that", "which", "who", "is", "are", "was", "were",
            "be", "been", "being", "do", "does", "did", "will", "would", "can", "could",
        }
        changed_words = {token.lower() for token in old_words + new_words}
        if changed_words and changed_words.issubset(function_words):
            if changed_words & {"a", "an", "the"}:
                return "determiner or article"
            if changed_words & {"of", "to", "in", "on", "at", "for", "from", "with", "by"}:
                return "preposition"
            return "function word"

        if len(old_words) == 1 and len(new_words) == 1:
            old_word = old_words[0]
            new_word = new_words[0]
            similarity = self._edit_similarity(old_word, new_word)
            prefix_len = self._common_prefix_len(old_word, new_word)
            min_word_len = min(len(old_word), len(new_word))
            if prefix_len >= min(4, max(2, min_word_len - 1)) and old_word[:1].lower() == new_word[:1].lower():
                return "word form"
            if similarity >= 0.76:
                return "spelling"
        return "word choice"

    def _span_from_errant_edit(self, edit):
        old = str(getattr(edit, "o_str", "") or "").strip()
        new = str(getattr(edit, "c_str", "") or "").strip()
        if old and new:
            return f"{old} -> {new}"
        return old or new

    def _primary_errant_hint_vote(self, source, target):
        annotator = self._get_errant_annotator()
        if annotator is None:
            return None
        try:
            original = annotator.parse(source)
            corrected = annotator.parse(target)
            edits = annotator.annotate(original, corrected)
        except Exception as exc:
            logger.debug(f"[GER HINT] ERRANT annotation failed for a retrieval neighbor: {exc}")
            return None
        candidates = []
        for edit in edits:
            hint_type = self._errant_to_hint_type(getattr(edit, "type", ""))
            span = " ".join(self._span_from_errant_edit(edit).split())
            if not hint_type or not span or not self._hint_span_allowed(span):
                continue
            candidates.append((hint_type, span, max(len(str(getattr(edit, "o_str", ""))), len(str(getattr(edit, "c_str", ""))))))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[2])[:2]

    def _primary_heuristic_hint_vote(self, source, target, fallback):
        span = " ".join(self._short_hint_span(source, target, fallback).split())
        if not span or not self._hint_span_allowed(span):
            return None
        if " -> " in span:
            old, new = span.split(" -> ", 1)
        else:
            old, new = span, ""
        hint_type = self._heuristic_to_hint_type(old, new)
        if not hint_type:
            return None
        return hint_type, span

    def _primary_hint_vote(self, example, mode):
        source = str(example.get("text", "")).strip()
        target = str(example.get("label", "")).strip()
        if not source or source == target:
            return None
        cache_key = (
            mode,
            str(example.get("from", "")),
            source,
            target,
            str(example.get("key", "")),
            self._truthy_env("GER_RETRIEVAL_HINT_STRICT"),
        )
        if cache_key in self._hint_type_cache:
            return self._hint_type_cache[cache_key]

        use_errant = mode in {"errant", "errant-only"} or (mode == "auto" and self._is_english_retrieval_example(example))
        vote = self._primary_errant_hint_vote(source, target) if use_errant else None
        if vote is None and mode != "errant-only":
            vote = self._primary_heuristic_hint_vote(source, target, str(example.get("key", "")))

        self._hint_type_cache[cache_key] = vote
        return vote

    def _retrieval_type_hint_description(self, examples, purity_min, max_hints):
        mode = self._retrieval_hint_type_mode()
        votes = []
        spans_by_type = defaultdict(list)
        for example in examples:
            if example.get("id") == -1:
                continue
            vote = self._primary_hint_vote(example, mode)
            if vote is None:
                continue
            hint_type, span = vote
            votes.append(hint_type)
            spans_by_type[hint_type].append((float(example.get("similarity", 0.0) or 0.0), span))

        if not votes:
            return ""
        counts = Counter(votes)
        total_votes = sum(counts.values())
        ranked_types = sorted(counts, key=lambda hint_type: (counts[hint_type], max(score for score, _ in spans_by_type[hint_type])), reverse=True)
        hints = []
        for hint_type in ranked_types:
            type_purity = counts[hint_type] / total_votes
            if type_purity < purity_min:
                continue
            best_span = max(spans_by_type[hint_type], key=lambda item: item[0])[1]
            hints.append(f"- Check for a possible {hint_type} issue near: {best_span}")
            if len(hints) >= max_hints:
                break
        if not hints:
            return ""
        return "<hints>\n" + "\n".join(hints) + "\n</hints>"

    def _retrieval_hint_description(self, examples):
        if not self._truthy_env("GER_ENABLE_RETRIEVAL_HINTS"):
            return ""
        purity_min = float(os.environ.get("GER_RETRIEVAL_HINT_PURITY_MIN", "0.6"))
        max_hints = max(0, int(os.environ.get("GER_RETRIEVAL_HINT_MAX", "2")))
        if max_hints <= 0:
            return ""
        mode = self._retrieval_hint_type_mode()
        if mode in {"off", "span", "lite", "h-lite"}:
            if self._edited_neighbor_purity(examples) < purity_min:
                return ""
            return self._retrieval_span_hint_description(examples, max_hints)
        return self._retrieval_type_hint_description(examples, purity_min, max_hints)

    def get_raw_datasets(self):
        datasets = []
        for inference_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            inference_dataset = self.inference_datasets_map[inference_dataset_name]
            datasets.append({"dataset_name": inference_dataset_name, "prompt_name": prompt_name, "dataset": inference_dataset})
        return datasets
    
    def get_datasets_using_retrieval_results(self):
        datasets = []
        for inference_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            inference_dataset = self.inference_datasets_map[inference_dataset_name]
            
            ## map the inference dataset into instruction dataset with ICL examples
            template = ICLPromptTemplate(prompt_name)
            in_domain_topk = self.icl_args.in_domain_example_num
            cross_domain_topk = self.icl_args.cross_domain_example_num
            
            # check dataset consistency between inference dataset and examples selection results
            examples_data = self.specific_examples[inference_dataset_name]
            assert len(inference_dataset) == len(examples_data)
            for item1, item2 in zip(inference_dataset, examples_data):
                assert item1['id'] == item2['id'], print(item1['id'], item2['id'])

            if self.icl_args.dynamic_in_domain_example_num:
                dynamic_limits = self._dynamic_example_limits(examples_data, in_domain_topk)
                avg_limit = sum(dynamic_limits) / len(dynamic_limits) if dynamic_limits else 0.0
                logger.info(
                    f"Select dynamic in-domain examples in ICL; min={min(dynamic_limits) if dynamic_limits else 0}, "
                    f"max={max(dynamic_limits) if dynamic_limits else 0}, avg={avg_limit:.3f}, cap={in_domain_topk}; "
                    f"{cross_domain_topk} cross-domain examples in ICL."
                )
            else:
                dynamic_limits = None
                logger.info(f"Select {in_domain_topk} in-domain examples in ICL; {cross_domain_topk} cross-domain examples in ICL.")

            # get examples
            instruction_items = []
            if_info_shown = False
            for idx, item in enumerate(examples_data):
                current_in_domain_topk = dynamic_limits[idx] if dynamic_limits is not None else in_domain_topk
                in_domain_examples = self.get_dedup_examples(item["in_domain_examples"], current_in_domain_topk)
                in_domain_examples = in_domain_examples[:current_in_domain_topk]
                cross_domain_examples = item["cross_domain_examples"][:cross_domain_topk]
                item['description'] = item["key_in_domain"]
                retrieval_hint_description = self._retrieval_hint_description(in_domain_examples)
                if retrieval_hint_description:
                    item['description'] = retrieval_hint_description
                for item_in in in_domain_examples:
                    item_in['description'] = item_in["key"]
                for item_cross in cross_domain_examples:
                    item_cross['description'] = "No grammatical or spelling error is found in this sentence."
                if not if_info_shown:
                    logger.info("[INFO] Description of the predicted text is key_in_domain; Descriptions of the retrieved in-domain items is key.")
                    if_info_shown = True
                instruction_item = self.get_instruction(examples=in_domain_examples+cross_domain_examples, test_item=item, template=template)
                new_item = {"id": item["id"], "text": item["text"]}
                if "label" in item:
                    new_item['label'] = item['label']
                new_item['sentence'] = instruction_item
                instruction_items.append(new_item)
        
            keys = ['id', 'text', 'label', 'sentence'] if 'label' in instruction_items[0] else ['id', 'text', 'sentence']
            
            instruction_dataset = Dataset.from_dict({key: [example.get(key) for example in instruction_items] for key in keys})
            datasets.append({"dataset_name": inference_dataset_name, "prompt_name": prompt_name, "dataset": instruction_dataset, "answer_start": template.get_answer_start(), "answer_end": template.get_answer_end()})
        return datasets


    def get_datasets(self):
        if self.icl_args.examples_dir:
            logger.info(f'[INFO] Input example data from {self.icl_args.examples_dir}')
            logger.info(f'[INFO] Due to the loaded examples, the settings of the example mode become invalid.')
            return self.get_datasets_using_retrieval_results()
        
        # normal retrieval
        logger.info('[INFO] No example data. Retrieve using original text when necessary.')
        datasets = []
        for inference_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            inference_dataset = self.inference_datasets_map[inference_dataset_name]
            ## map the inference dataset into instruction dataset with ICL examples
            template = ICLPromptTemplate(prompt_name)
            in_domain_topk = self.icl_args.in_domain_example_num
            cross_domain_topk = self.icl_args.cross_domain_example_num
            logger.info(f"Select {in_domain_topk} in-domain examples in ICL; {cross_domain_topk} cross-domain examples in ICL.")
            # get examples
            def _instruction_generate(item):
                in_domain_examples = self.specific_examples[inference_dataset_name].select(query=item["text"])
                cross_domain_examples = self.cross_domain_examples.select(query=item["text"])
                instruction_item = self.get_instruction(examples=in_domain_examples+cross_domain_examples, test_item=item, template=template)
                item['sentence'] = instruction_item
                return item
            # 使用map函数应用转换
            reserved_columns = ['id', 'text', 'label', 'sentence']
            remove_columns = [key for key in list(inference_dataset.column_names) if key not in reserved_columns]
            instruction_dataset = inference_dataset.map(_instruction_generate, remove_columns=remove_columns)
            datasets.append({"dataset_name": inference_dataset_name, "prompt_name": prompt_name, "dataset": instruction_dataset, "answer_start": template.get_answer_start(), "answer_end": template.get_answer_end()})
        return datasets
    

    def get_instruction(self, examples, test_item, template):
        converted_example_list = []
        for example_item in examples:
            value_dict = {'source': example_item['text'], 'target': example_item['label']}
            if 'description' in example_item:
                value_dict['description'] = example_item['description'].strip()
            converted_example_list.append(value_dict)
        
        if 'description' not in test_item:
            sys_prompt, instruction_sentence = template.format(examples_list=converted_example_list, source=test_item['text'])
        else:
            sys_prompt, instruction_sentence = template.format(examples_list=converted_example_list, source=test_item['text'], description=test_item['description'])

        dialogue = False
        if self.tokenizer == None or (self.icl_args.dialogue_form and self.tokenizer.chat_template):
            dialogue = True
        
        if dialogue:
            if self.tokenizer and 'System role not supported' in self.tokenizer.chat_template:
                instruction_item = [
                    {'role': "user", 'content': (sys_prompt + '\n' + instruction_sentence).strip()}
                ]
            else:
                instruction_item = [
                    {'role': "system", 'content': sys_prompt},
                    {'role': "user", 'content': instruction_sentence}
                ]
        else:
            instruction_item = (sys_prompt + '\n' + instruction_sentence).strip()
        
        return instruction_item

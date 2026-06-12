import argparse
import os
import json
import logging
import random
import time
from dataclasses import MISSING, dataclass, fields
from typing import Any, Union, get_args, get_origin
from tqdm import tqdm
from configs.data_arguments import DataConfig
from configs.icl_arguments import ICLConfig

from data.icl_dataset import ICLDataset
from llm.pipeline import TextGeneration
from data.check_jsonl import extract_matching_valid_lines, rewrite_jsonl_with_valid_lines
from utils.log import setup_log, log_config
from utils.random import init_seed



logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True


def is_truthy_env(name):
    return os.environ.get(name, "").strip() in {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def argument_type(annotation: Any):
    annotation = unwrap_optional(annotation)
    if annotation is bool:
        return parse_bool
    if annotation in {str, int, float}:
        return annotation
    return str


def add_dataclass_arguments(parser: argparse.ArgumentParser, config_cls: type) -> None:
    for item in fields(config_cls):
        option = f"--{item.name}"
        kwargs: dict[str, Any] = {"dest": item.name}
        if item.default is not MISSING:
            kwargs["default"] = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs["default"] = item.default_factory()  # type: ignore[misc]
        else:
            kwargs["required"] = True
        if unwrap_optional(item.type) is bool:
            kwargs["nargs"] = "?"
            kwargs["const"] = True
            kwargs["type"] = parse_bool
        else:
            kwargs["type"] = argument_type(item.type)
        help_text = item.metadata.get("help") if item.metadata else None
        if help_text:
            kwargs["help"] = help_text
        parser.add_argument(option, **kwargs)


def parse_args_and_config(argv: list[str] | None = None) -> tuple[ModelConfig, DataConfig, ICLConfig]:
    parser = argparse.ArgumentParser(description="Run GER in-context inference.")
    add_dataclass_arguments(parser, ModelConfig)
    add_dataclass_arguments(parser, DataConfig)
    add_dataclass_arguments(parser, ICLConfig)
    args = parser.parse_args(argv)
    values = vars(args)

    def build(config_cls: type):
        return config_cls(**{item.name: values[item.name] for item in fields(config_cls)})

    return build(ModelConfig), build(DataConfig), build(ICLConfig)


def write_shard_progress(results_save_dir, total, completed, done=False, stage="running"):
    progress_file = os.path.join(results_save_dir, "progress.json")
    tmp_file = progress_file + ".tmp"
    payload = {
        "total": int(total),
        "completed": int(completed),
        "done": bool(done),
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_file, progress_file)


def silence_console_logging():
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)


def gec_extract_answer(string, original_text, start, end):
    if start not in string or 'No errors found' in string:
        return original_text
    
    # Bloomz bug: the last character will not be included in the result
    if len(end) >= 5:
        end = end[:-1]

    if end not in string:
        return original_text
    answer = string.split(start)[1].split(end)[0]
    answer = answer.strip()

    if answer == '' or answer == ' ':
        return original_text
    return answer

def gec_split_answer(string:str, original_text:str, splitter:str):
    splits = string.split(splitter)
    if len(splits) >= 3:
        # normal situation
        answer = splits[1].strip()
        if 0.5 <= len(answer) / len(original_text) <= 2.0:
            return answer
        else:
            return original_text
    else:
        return original_text


def start_end_extraction(string, start, end):
    if start:
        if string.find(start) != -1:
            string = string.split(start, 1)[1]
    if end:
        if string.find(end) != -1:
            string = string.split(end, 1)[0]
    return string


def extract_strict_tagged_answer(string, start, end):
    if not start or not end:
        return None

    # Some ICL prompts prefill the opening answer tag. In that mode the model
    # may return only "answer</corrected sentence>" and then continue with more
    # tagged examples. Prefer the prefilled span before looking for later tags.
    end_idx = string.find(end)
    start_idx = string.find(start)
    if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
        answer = string[:end_idx].strip()
        if answer:
            return answer

    if start_idx == -1:
        return None

    content_start = start_idx + len(start)
    end_idx = string.find(end, content_start)
    if end_idx == -1:
        return None

    answer = string[content_start:end_idx].strip()
    if not answer:
        return None

    return answer


def postprocess(response, data_item, answer_start, answer_end, icl_config):
    # form result
    result_item = {'id': data_item['id'], 'text': data_item['text']}
    if '__original_index' in data_item:
        result_item['__original_index'] = data_item['__original_index']
    if 'label' in data_item:
        result_item['label'] = data_item["label"]

    if icl_config.postprocess == 'gec':
        # gec/fewshot postprocess
        answer = extract_strict_tagged_answer(response, answer_start, answer_end)
        if answer is None:
            legacy_response = answer_start + ' ' + response
            answer = gec_extract_answer(legacy_response, data_item['text'], answer_start, answer_end)
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'splitter':
        answer = gec_split_answer(response, data_item['text'], '```')
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'normal':
        answer = start_end_extraction(response, answer_start, answer_end)
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'no':
        pass
    else:
        raise NotImplementedError(f'Unknown postprocess mode: {icl_config.postprocess}')
    
    result_item['response'] = response
    result_item['sentence'] = data_item['sentence']

    return result_item


def get_generated_text(output):
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if isinstance(output, list):
        return output[0]['generated_text']
    return output['generated_text']


def normalize_pipeline_batch_outputs(outputs, batch_size):
    if batch_size == 1 and isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return [outputs]
    return outputs


def get_streaming_chunk_size(inference_batch_size):
    configured = os.environ.get("GER_STREAMING_CHUNK_SIZE", "").strip()
    if configured:
        chunk_size = int(configured)
    else:
        chunk_size = min(int(inference_batch_size), 8)
    return max(1, chunk_size)


def build_text_generation(model_config, icl_config):
    pipe = TextGeneration(
        model_name=model_config.model_name_or_path,
        max_new_tokens=icl_config.max_new_tokens,
        do_sample=icl_config.do_sample,
        temperature=icl_config.temperature,
        top_k=icl_config.top_k,
        top_p=icl_config.top_p,
        return_full_text=False,
        num_return_sequences=1,
        stop_string=None,
        batch_size=icl_config.inference_batch_size,
    )
    logger.info(f"Generation backend: {pipe.mode}")
    return pipe


def run_icl_inference(model_config, data_config, icl_config, pipe=None, *, configure_logging=True):
    init_seed(icl_config.seed)

    if configure_logging:
        setup_log(icl_config.output_dir)
        if is_truthy_env("GER_QUIET_CHILD_LOG"):
            silence_console_logging()

    ## Forced parameter modification
    if 'qalb' in data_config.datasets:
        logger.info("[WARNING] For QALB datasets, we will set error examples (current for cross_domain_example_num) to 0.")
        icl_config.cross_domain_example_num = 0

    logger.info(f"Model arguments:")
    log_config(model_config, logger)
    logger.info(f"Data arguments:")
    log_config(data_config, logger)
    logger.info(f"In-context learning arguments:")
    log_config(icl_config, logger)
    if icl_config.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {icl_config.num_shards}")
    if not 0 <= icl_config.shard_id < icl_config.num_shards:
        raise ValueError(f"shard_id must be in [0, {icl_config.num_shards}), got {icl_config.shard_id}")

    if pipe is None:
        pipe = build_text_generation(model_config, icl_config)
    else:
        logger.info(f"Generation backend: {pipe.mode}")

    # get dataset of ICL instructions
    dataset_controller = ICLDataset(data_args=data_config, model_args=model_config, icl_args=icl_config)

    # infer for every dataset
    for dataset in dataset_controller.get_datasets():
        data = dataset['dataset']
        answer_end = dataset['answer_end']
        answer_start = dataset['answer_start']
        if '__original_index' not in data.column_names:
            data = data.add_column('__original_index', list(range(len(data))))
        if icl_config.num_shards > 1:
            shard_indices = [idx for idx in range(len(data)) if idx % icl_config.num_shards == icl_config.shard_id]
            logger.info(
                f"Shard {icl_config.shard_id}/{icl_config.num_shards}: "
                f"processing {len(shard_indices)} of {len(data)} examples"
            )
            data = data.select(shard_indices)

        pipe.reset_stop_string(stop_string=answer_end)

        logger.info(f"ICL Infer on dataset {dataset['dataset_name']}, by prompt {dataset['prompt_name']}")
        os.makedirs(icl_config.output_dir, exist_ok=True)
        results_save_dir = os.path.join(icl_config.output_dir, dataset['dataset_name'])
        if icl_config.num_shards > 1:
            results_save_dir = os.path.join(results_save_dir, f'shard_{icl_config.shard_id}')
        os.makedirs(results_save_dir, exist_ok=True)
        logger.info(f"Results will be saved into {results_save_dir}")

        sentences = [item['sentence'] for item in data]
        logger.info("Data Example:\n" + str(random.choice(sentences)))

        jsonl_file = os.path.join(results_save_dir, 'predictions.jsonl')

        medium_res = extract_matching_valid_lines(data, jsonl_file)

        # API-based generation
        if pipe.mode == "API" and len(medium_res) != len(data):
            # generate for all data, cache in api_cache.json
            api_res_cache_file = os.path.join(results_save_dir, 'api_cache.json')
            results = pipe(data, api_res_cache_file)
            # convert to standard results
            with open(jsonl_file, 'w') as f:
                for i, item in enumerate(data):
                    assert results[i]["id"] == item["id"]
                    response = results[i]["result"].strip()
                    result_item = postprocess(response, item, answer_start, answer_end, icl_config)
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f.flush()


        # judge the result state now to continue on current results.
        medium_res = extract_matching_valid_lines(data, jsonl_file)
        writer = rewrite_jsonl_with_valid_lines(medium_res, jsonl_file)
        write_shard_progress(
            results_save_dir,
            len(data),
            len(medium_res),
            done=len(medium_res) >= len(data),
            stage="resume_checked" if len(medium_res) < len(data) else "done",
        )

        # Predict and save
        show_progress = icl_config.shard_id == 0 and not is_truthy_env("GER_DISABLE_CHILD_PROGRESS")
        if show_progress:
            print(len(medium_res), len(data))
            print(data[0])
            print(data[-1])
            print(data)
        if len(medium_res) < len(data):
            pending_data = data.select(range(len(medium_res), len(data)))
            write_shard_progress(results_save_dir, len(data), len(medium_res), stage="waiting_first_output")
            if pipe.mode != "vLLM":
                raise NotImplementedError("Only local vLLM streaming inference is supported here.")
            stream_chunk_size = get_streaming_chunk_size(icl_config.inference_batch_size)
            progress_bar = tqdm(
                total=len(pending_data),
                desc=dataset["dataset_name"],
                unit="ex",
                disable=not show_progress,
            )
            try:
                for chunk_start in range(0, len(pending_data), stream_chunk_size):
                    chunk_end = min(chunk_start + stream_chunk_size, len(pending_data))
                    chunk = pending_data.select(range(chunk_start, chunk_end))
                    chunk_inputs = [item["sentence"] for item in chunk]
                    write_shard_progress(
                        results_save_dir,
                        len(data),
                        len(medium_res) + chunk_start,
                        stage=f"generating_chunk_{chunk_start}_{chunk_end}",
                    )
                    outputs = normalize_pipeline_batch_outputs(pipe(chunk_inputs), len(chunk_inputs))
                    if len(outputs) != len(chunk_inputs):
                        raise RuntimeError(
                            f"Pipeline returned {len(outputs)} outputs for {len(chunk_inputs)} inputs "
                            f"in chunk {chunk_start}:{chunk_end}."
                        )
                    for local_idx, output in enumerate(outputs):
                        item = chunk[local_idx]
                        response = get_generated_text(output)
                        result_item = postprocess(response, item, answer_start, answer_end, icl_config)
                        writer.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        writer.flush()
                        completed = len(medium_res) + chunk_start + local_idx + 1
                        write_shard_progress(results_save_dir, len(data), completed, done=completed >= len(data), stage="generating")
                        progress_bar.update(1)
            finally:
                progress_bar.close()
        
        writer.close()
        write_shard_progress(results_save_dir, len(data), len(data), done=True, stage="done")


if __name__ == "__main__":
    model_config, data_config, icl_config = parse_args_and_config()
    run_icl_inference(model_config, data_config, icl_config)

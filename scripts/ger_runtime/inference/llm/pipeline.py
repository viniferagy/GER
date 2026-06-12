from typing import Any
import torch
import logging
import os

logger = logging.getLogger(__name__)

def is_truthy_value(value):
    return str(value).strip() in {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}

def _count_visible_cuda_devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


class VLLMTextGeneration:
    def __init__(
            self,
            model_name,
            max_new_tokens=512,
            do_sample=None,
            temperature=None,
            top_k=None,
            top_p=None,
            return_full_text=False,
            num_return_sequences=1,
            stop_string=None,
            batch_size=1,
            **kwargs
        ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "GER_LLM_BACKEND=vllm requires vLLM. Install it in the active "
                "environment, for example: uv pip install vllm"
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.return_full_text = return_full_text
        self.num_return_sequences = num_return_sequences
        self.stop_string = stop_string
        self.batch_size = batch_size
        self.SamplingParams = SamplingParams

        tensor_parallel_size = int(
            os.environ.get("GER_VLLM_TENSOR_PARALLEL_SIZE")
            or _count_visible_cuda_devices()
        )
        gpu_memory_utilization = float(os.environ.get("GER_VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        dtype = os.environ.get("GER_VLLM_DTYPE", "auto")
        max_model_len = os.environ.get("GER_VLLM_MAX_MODEL_LEN", "").strip()
        max_num_seqs = os.environ.get("GER_VLLM_MAX_NUM_SEQS", "").strip()
        max_num_batched_tokens = os.environ.get("GER_VLLM_MAX_NUM_BATCHED_TOKENS", "").strip()
        enforce_eager = is_truthy_value(os.environ.get("GER_VLLM_ENFORCE_EAGER", "0"))

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
        }
        if max_model_len:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_num_batched_tokens:
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)

        logger.info(
            "Initializing vLLM backend: model=%s tensor_parallel_size=%s "
            "gpu_memory_utilization=%s dtype=%s max_model_len=%s "
            "max_num_seqs=%s max_num_batched_tokens=%s enforce_eager=%s",
            model_name,
            tensor_parallel_size,
            gpu_memory_utilization,
            dtype,
            max_model_len or "<default>",
            max_num_seqs or "<default>",
            max_num_batched_tokens or "<default>",
            enforce_eager,
        )
        self.llm = LLM(**llm_kwargs)

    def reset_stop_string(self, stop_string):
        self.stop_string = stop_string

    def chat_template(self):
        try:
            return self.llm.get_tokenizer().chat_template
        except Exception:
            return True

    def _sampling_params(self):
        if self.do_sample:
            temperature = self.temperature if self.temperature is not None else 1.0
        else:
            temperature = 0.0
        kwargs = {
            "max_tokens": self.max_new_tokens,
            "n": self.num_return_sequences,
            "temperature": temperature,
        }
        if self.do_sample:
            if self.top_p is not None and self.top_p > 0:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        use_stop_string = is_truthy_value(os.environ.get("GER_VLLM_USE_STOP_STRING", "0"))
        if self.stop_string and use_stop_string:
            kwargs["stop"] = [self.stop_string]
            kwargs["include_stop_str_in_output"] = True
        try:
            return self.SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("include_stop_str_in_output", None)
            return self.SamplingParams(**kwargs)

    def _is_chat_input(self, inputs):
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], dict):
                return True
            if isinstance(inputs[0], list) and inputs[0] and isinstance(inputs[0][0], dict):
                return True
        return False

    def _is_single_input(self, inputs):
        if isinstance(inputs, str):
            return True
        return isinstance(inputs, list) and bool(inputs) and isinstance(inputs[0], dict)

    def _to_generation_dicts(self, outputs, single_input):
        converted = [{"generated_text": output.outputs[0].text} for output in outputs]
        return converted[0] if single_input and converted else converted

    def __call__(self, inputs) -> Any:
        sampling_params = self._sampling_params()
        single_input = self._is_single_input(inputs)
        if self._is_chat_input(inputs):
            messages = [inputs] if single_input else inputs
            try:
                outputs = self.llm.chat(
                    messages=messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except TypeError:
                outputs = self.llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        else:
            prompts = [inputs] if isinstance(inputs, str) else inputs
            outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return self._to_generation_dicts(outputs, single_input)


class TextGeneration:
    def __init__(
            self, 
            model_name,
            max_new_tokens=512,
            do_sample=None, 
            temperature=None,
            top_k=None, 
            top_p=None, 
            return_full_text=False, 
            num_return_sequences=1, 
            stop_string=None,
            batch_size=1,
            **kwargs
        ) -> None:
        self.base_name = model_name.split('/')[-1].strip()
        backend = os.environ.get("GER_LLM_BACKEND", "vllm").strip().lower()
        if backend in {"", "vllm"}:
            self.mode = "vLLM"
            self.generation_pipe = VLLMTextGeneration(
                model_name,
                max_new_tokens,
                do_sample,
                temperature,
                top_k,
                top_p,
                return_full_text,
                num_return_sequences,
                stop_string,
                batch_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported GER_LLM_BACKEND={backend!r}. Local inference only supports 'vllm'.")
        
    def reset_stop_string(self, stop_string):
        self.generation_pipe.reset_stop_string(stop_string)

    def chat_template(self):  
        return self.generation_pipe.chat_template()
        
    def __call__(self, inputs, cache_file=None):
        return self.generation_pipe(inputs)

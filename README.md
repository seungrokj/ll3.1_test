# ll3.1_test


## server setup

```shell
docker pull srjung/ll3.1_release_retune:latest

export TOKEN="YOUR TF TOKEN"
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd)/models/:/models -e HUGGINGFACE_HUB_CACHE=/models -e HF_TOKEN=$TOKEN -e VLLM_USE_TRITON_FLASH_ATTN=0 -e PYTORCH_TUNABLEOP_ENABLED=0 -e VLLM_TUNE_FILE=/app/vllm_retune_0812.csv -p 8000:8000 srjung/ll3.1_release_retune:latest python -m vllm.entrypoints.api_server --tensor-parallel-size 8 --enforce-eager --worker-use-ray --max-model-len=8192 --model meta-llama/Meta-Llama-3.1-405B-Instruct
```


## client setup: in another shell

```shell
curl http://localhost:8000/generate -H "Content-Type: application/json" -d '{ "prompt": "San Francisco City is ", "max_tokens": 200, "temperature": 0.9}' 
```

# Expected behavior

## server
```shell
INFO 07-25 03:59:46 llm_engine.py:161] Initializing an LLM engine (v0.4.3) with config: model='/model/Meta_Llama-3.1-405B-MP16_HF', speculative_config=None, tokenizer='/model/Meta_Llama-3.1-405B-MP16_HF', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=8, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=/model/Meta_Llama-3.1-405B-MP16_HF)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 07-25 04:00:10 selector.py:56] Using ROCmFlashAttention backend.
(RayWorkerWrapper pid=21448) INFO 07-25 04:00:10 selector.py:56] Using ROCmFlashAttention backend.
INFO 07-25 04:00:12 selector.py:56] Using ROCmFlashAttention backend.
INFO 07-25 04:01:16 model_runner.py:146] Loading model weights took 95.5194 GB
(RayWorkerWrapper pid=22543) INFO 07-25 04:01:44 model_runner.py:146] Loading model weights took 95.5194 GB
(RayWorkerWrapper pid=22697) INFO 07-25 04:00:12 selector.py:56] Using ROCmFlashAttention backend. [repeated 13x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(RayWorkerWrapper pid=22697) INFO 07-25 04:01:50 model_runner.py:146] Loading model weights took 95.5194 GB [repeated 6x across cluster]
INFO 07-25 04:02:33 distributed_gpu_executor.py:56] # GPU blocks: 34501, # CPU blocks: 2080
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO 07-25 04:03:59 async_llm_engine.py:553] Received request ecc91bf242964262a0cc05ca4d2b6c48: prompt: 'San Francisco City is ', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=200, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: None, lora_request: None.
INFO 07-25 04:04:04 async_llm_engine.py:124] Finished request ecc91bf242964262a0cc05ca4d2b6c48.
INFO:     172.17.0.1:36956 - "POST /generate HTTP/1.1" 200 OK
```

## client
```shell
curl http://localhost:8000/generate -H "Content-Type: application/json" -d '{ "prompt": "San Francisco City is ", "max_tokens": 200, "temperature": 0.9}'
{"text":["San Francisco City is 1 hour behind Santa Rosa City.\nSanta Rosa City, California, USA is located in PST (Pacific Standard Time) time zone and San Francisco City, California, USA is located in PDT (Pacific Daylight Time) time zone. There is an 1:0 hours time difference between Santa Rosa City and San Francisco City right now.\nPlease note that Daylight Saving Time (DST) / Summer Time is taken into account for the calculation of hour difference between Santa Rosa City and San Francisco City."]}
```

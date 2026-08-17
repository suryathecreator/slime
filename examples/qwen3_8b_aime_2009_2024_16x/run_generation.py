#!/usr/bin/env python3
"""Generate one draw/shard of Qwen3-8B AIME completions with durable commits."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pipeline


def stop_token_id(output: Any, finish_reason: str | None) -> int | None:
    raw = getattr(output, "stop_reason", None)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    if finish_reason == "stop":
        # Qwen3's configured EOS is <|im_end|>. Explicit non-EOS stop IDs are
        # surfaced by vLLM as stop_reason and are handled above.
        return 151645
    return None


def artifact(
    *,
    row: dict[str, Any],
    draw_index: int,
    shard_index: int,
    task_id: int,
    prompt: str,
    messages: list[dict[str, str]],
    rendered_prompt: str,
    input_tokens: int,
    response_budget: int,
    output: Any,
    tokenizer: Any,
    policy: dict[str, Any],
    contract: str,
    scorer: Any,
) -> dict[str, Any]:
    completion = output.outputs[0]
    generated_ids = [int(token) for token in (completion.token_ids or [])]
    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    response = scorer.clean_generated_text(decoded)
    finish_reason = str(completion.finish_reason) if completion.finish_reason else None
    observed_stop = stop_token_id(completion, finish_reason)
    cap_hit = finish_reason == "length" or len(generated_ids) >= response_budget
    trace = pipeline.score_accepted_answers(
        response,
        row["accepted_answer_integers"],
        finish_reason=finish_reason,
        stop_token_id=observed_stop,
        cap_hit=cap_hit,
        problem_id=row["problem_id"],
        scorer=scorer,
    )
    generation = policy["generation"]
    return {
        "accepted_answer_integers": row["accepted_answer_integers"],
        "answer_canonical": row["answer_canonical"],
        "answer_raw": row["answer_raw"],
        "artifact_schema_version": policy["artifact_schema_version"],
        "cap_hit": cap_hit,
        "configured_context_tokens": policy["context"]["max_model_len"],
        "contest": row["contest"],
        "contract_sha256": contract,
        "draw_index": draw_index,
        "effective_max_new_tokens": response_budget,
        "eval_index": row["eval_index"],
        "finish_reason": finish_reason,
        "generation_parameters": {
            "enable_thinking": generation["enable_thinking"],
            "min_p": generation["min_p"],
            "temperature": generation["temperature"],
            "top_k": generation["top_k"],
            "top_p": generation["top_p"],
        },
        "input_tokens": input_tokens,
        "messages": messages,
        "model_id": policy["model"]["id"],
        "model_revision": policy["model"]["revision"],
        "output_tokens": len(generated_ids),
        "part": row["part"],
        "problem": row["problem"],
        "problem_id": row["problem_id"],
        "problem_number": row["problem_number"],
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": pipeline.sha256_text(rendered_prompt),
        "response": response,
        "response_sha256": pipeline.sha256_text(response),
        "sample_id": f"{row['problem_id']}:draw-{draw_index:02d}",
        "scorer_trace": trace,
        "seed": pipeline.seed_for(draw_index, row["eval_index"], policy),
        "shard_index": shard_index,
        "source": row["source"],
        "source_record_sha256": row["source_record_sha256"],
        "stop_condition": (
            "endoftext"
            if observed_stop == 151643
            else "im_end"
            if observed_stop == 151645
            else "length_cap"
            if cap_hit
            else finish_reason
        ),
        "stop_token_id": observed_stop,
        "task_id": task_id,
        "year": row["year"],
    }


async def generate_pending(
    *,
    pending: list[tuple[dict[str, Any], str, list[dict[str, str]], str, list[int], int, Path]],
    args: argparse.Namespace,
    draw_index: int,
    shard_index: int,
    policy: dict[str, Any],
    contract: str,
    tokenizer: Any,
    scorer: Any,
) -> None:
    from vllm import SamplingParams, TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_policy = policy["vllm_engine"]
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(pipeline.model_root(policy)),
            tokenizer=str(pipeline.model_root(policy)),
            trust_remote_code=False,
            tensor_parallel_size=engine_policy["tensor_parallel_size"],
            dtype=policy["model"]["weight_dtype"],
            max_model_len=policy["context"]["max_model_len"],
            gpu_memory_utilization=engine_policy["gpu_memory_utilization"],
            max_num_seqs=engine_policy["max_num_seqs"],
            max_num_batched_tokens=engine_policy["max_num_batched_tokens"],
            enable_chunked_prefill=engine_policy["enable_chunked_prefill"],
            enable_prefix_caching=engine_policy["enable_prefix_caching"],
            safetensors_load_strategy="prefetch",
        )
    )
    print(
        f"QWEN3_AIME_ENGINE_READY task={args.task_id} draw={draw_index} "
        f"shard={shard_index} pending={len(pending)}",
        flush=True,
    )

    async def generate_one(
        item: tuple[dict[str, Any], str, list[dict[str, str]], str, list[int], int, Path]
    ) -> None:
        row, prompt, messages, rendered, input_ids, budget, target = item
        generation = policy["generation"]
        sampling = SamplingParams(
            n=1,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            top_k=generation["top_k"],
            min_p=generation["min_p"],
            max_tokens=budget,
            seed=pipeline.seed_for(draw_index, row["eval_index"], policy),
            stop_token_ids=policy["stop"]["accepted_token_ids"],
            ignore_eos=False,
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        final = None
        async for current in engine.generate(
            TokensPrompt(prompt_token_ids=input_ids),
            sampling,
            request_id=f"draw-{draw_index:02d}-eval-{row['eval_index']:04d}",
        ):
            final = current
        if final is None or not final.finished:
            raise RuntimeError(f"vLLM did not finish {row['problem_id']}")
        result = artifact(
            row=row,
            draw_index=draw_index,
            shard_index=shard_index,
            task_id=args.task_id,
            prompt=prompt,
            messages=messages,
            rendered_prompt=rendered,
            input_tokens=len(input_ids),
            response_budget=budget,
            output=final,
            tokenizer=tokenizer,
            policy=policy,
            contract=contract,
            scorer=scorer,
        )
        pipeline.atomic_text(target, pipeline.canonical_json(result) + "\n")
        print(
            f"QWEN3_AIME_SAMPLE_COMMIT sample={result['sample_id']} "
            f"correct={int(result['scorer_trace']['is_correct'])} "
            f"tokens={result['output_tokens']} cap={int(result['cap_hit'])} path={target}",
            flush=True,
        )

    try:
        await asyncio.gather(*(generate_one(item) for item in pending))
    finally:
        engine.shutdown()


def run(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer

    policy = pipeline.load_policy()
    contract = pipeline.contract_sha256(policy)
    corpus = pipeline.load_corpus()
    scorer = pipeline.import_scorer(policy)
    draw_index, start, end = pipeline.task_assignment(args.task_id)
    shard_index = args.task_id % 4
    root = pipeline.output_root(policy)
    model = pipeline.model_root(policy)
    if not (root / "preflight.json").is_file():
        raise RuntimeError("preflight artifact is missing")
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type != "qwen3" or int(config.max_position_embeddings) < 32768:
        raise RuntimeError("local model identity or context differs from policy")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for token, expected in zip(
        policy["stop"]["accepted_tokens"],
        policy["stop"]["accepted_token_ids"],
        strict=True,
    ):
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise RuntimeError(f"tokenizer ID changed for {token}")

    selected = corpus[start:end]
    if args.limit is not None:
        selected = selected[: args.limit]
    pending: list[
        tuple[dict[str, Any], str, list[dict[str, str]], str, list[int], int, Path]
    ] = []
    for row in selected:
        eval_index = int(row["eval_index"])
        target = pipeline.sample_path(root, draw_index, eval_index)
        if target.exists():
            if pipeline.compatible_sample(target, contract, draw_index, eval_index):
                continue
            raise RuntimeError(f"refusing incompatible existing sample: {target}")
        prompt = pipeline.prompt_for_problem(row["problem"], policy)
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        budget = int(policy["context"]["max_model_len"]) - len(input_ids)
        if budget <= 0:
            raise RuntimeError(
                f"prompt exceeds 32K context: {row['problem_id']} tokens={len(input_ids)}"
            )
        pending.append((row, prompt, messages, rendered, input_ids, budget, target))
    print(
        f"QWEN3_AIME_SHARD_START task={args.task_id} draw={draw_index} "
        f"shard={shard_index} range={start}:{end} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    asyncio.run(
        generate_pending(
            pending=pending,
            args=args,
            draw_index=draw_index,
            shard_index=shard_index,
            policy=policy,
            contract=contract,
            tokenizer=tokenizer,
            scorer=scorer,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

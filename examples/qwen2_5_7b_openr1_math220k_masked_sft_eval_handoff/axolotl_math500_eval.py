#!/usr/bin/env python3
"""Qwen2.5 configuration of the shared, audited Axolotl MATH-500 controller."""

from pathlib import Path

from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval as controller,
)

HANDOFF = Path(__file__).resolve().parent
controller.HANDOFF = HANDOFF
controller.DEFAULT_MANIFEST = HANDOFF / "available_eval_manifest.json"
controller.POLICY = HANDOFF / "axolotl_eval_policy.json"
controller.EXPECTED_VARIANTS = {
    "base_7b",
    "correct_only",
    "unmasked",
    "random_mask_25",
    "random_mask_50",
    "random_mask_70",
    "random_mask_80",
    "random_mask_90",
    "inverse_tau_0p20",
    "inverse_tau_0p05",
    "margin_mask",
    "prob_ratio_mask",
}
controller.EXPECTED_CHECKPOINT_COUNT = len(controller.EXPECTED_VARIANTS)
controller.APPROVED_BASE_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
controller.EXPECTED_BASE_TOKENIZER_FILES = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
}


def prepare_qwen2_tokenizer_overlay(repo_root, inventory):
    """Install the pinned Qwen2 base tokenizer, ignoring bundled Qwen3 files."""
    source_item = controller.item_for_variant(inventory, "base_7b")
    source_paths = controller.absolute_item_paths(repo_root, source_item)
    source_manifest = controller.load_json(source_paths["manifest"])
    source_hashes = {
        item["name"]: item["sha256"] for item in source_manifest["files"]
    }
    expected_hashes = inventory["base_model"]["tokenizer_files"]
    if {name: source_hashes.get(name) for name in expected_hashes} != expected_hashes:
        raise RuntimeError("base checkpoint does not contain the pinned Qwen2 tokenizer")

    ignored_tokenizers = {}
    for item in inventory["checkpoints"]:
        paths = controller.absolute_item_paths(repo_root, item)
        config = controller.load_json(paths["model"] / "config.json")
        if (
            config.get("model_type") != "qwen2"
            or config.get("architectures") != ["Qwen2ForCausalLM"]
            or int(config.get("vocab_size", -1)) != 152064
            or int(config.get("hidden_size", -1)) != 3584
            or int(config.get("num_hidden_layers", -1)) != 28
        ):
            raise RuntimeError(f"checkpoint is not Qwen2.5-7B: {item['variant']}")
        manifest = controller.load_json(paths["manifest"])
        hashes = {entry["name"]: entry["sha256"] for entry in manifest["files"]}
        ignored_tokenizers[item["variant"]] = {
            name: hashes.get(name) for name in expected_hashes
        }

    source = source_paths["model"]
    destination = controller.tokenizer_root(repo_root, inventory)
    for name, expected_hash in expected_hashes.items():
        if controller.sha256_file(source / name) != expected_hash:
            raise RuntimeError(f"base tokenizer file hash mismatch: {name}")
        controller.atomic_bytes(destination / name, (source / name).read_bytes())
    provenance = {
        "artifact_schema_version": 1,
        "base_model": {
            "hf_repo": inventory["base_model"]["hf_repo"],
            "revision": inventory["base_model"]["revision"],
        },
        "compatibility_transform": None,
        "files": {
            name: {
                "destination_sha256": controller.sha256_file(destination / name),
                "source_sha256": expected_hash,
            }
            for name, expected_hash in expected_hashes.items()
        },
        "ignored_checkpoint_tokenizers": {
            "files": ignored_tokenizers,
            "reason": (
                "The eleven training bundles contain Qwen3 tokenizer artifacts. "
                "Their configs and weights are Qwen2.5-7B; inference therefore "
                "uses only the hash-pinned Qwen2.5 base tokenizer overlay."
            ),
        },
        "source_checkpoint_manifest_sha256": source_manifest[
            "checkpoint_manifest_sha256"
        ],
        "source_files": {name: str(source / name) for name in expected_hashes},
    }
    controller.atomic_text(
        destination / "TOKENIZER_PROVENANCE.json",
        controller.canonical_json(provenance),
    )
    return destination


controller.prepare_tokenizer_overlay = prepare_qwen2_tokenizer_overlay


def verify_qwen2_checkpoints(args):
    repo_root = controller.repo_root_from_args(args)
    inventory = controller.load_inventory(repo_root, args.manifest)
    identities = {}
    for item in inventory["checkpoints"]:
        paths = controller.absolute_item_paths(repo_root, item)
        if args.full:
            identity = controller.checkpoint_manifests.verify(
                paths["manifest"], paths["model"]
            )
        else:
            identity = controller.quick_checkpoint_check(
                paths["model"], paths["manifest"]
            )
        manifest = controller.load_json(paths["manifest"])
        if manifest.get("variant") != item["variant"]:
            raise RuntimeError(f"{item['variant']} manifest variant mismatch")
        if item["variant"] == "base_7b":
            if manifest.get("family") != "shared" or manifest.get("final_iteration") is not None:
                raise RuntimeError("base_7b manifest is not the approved shared base")
        elif (
            manifest.get("family") != "2k"
            or int(manifest.get("final_iteration", -1)) != 9
        ):
            raise RuntimeError(f"{item['variant']} is not final 2K iteration 9")
        identities[item["variant"]] = identity
        print(
            f"CHECKPOINT_VERIFIED variant={item['variant']} "
            f"full={int(args.full)} sha256={identity}",
            flush=True,
        )
    if args.status:
        controller.atomic_text(
            Path(args.status),
            controller.canonical_json(
                {
                    "artifact_schema_version": 1,
                    "checked_at": controller.now_iso(),
                    "full_hash_verification": bool(args.full),
                    "identities": identities,
                }
            ),
        )


controller.verify_checkpoints = verify_qwen2_checkpoints


if __name__ == "__main__":
    controller.main()

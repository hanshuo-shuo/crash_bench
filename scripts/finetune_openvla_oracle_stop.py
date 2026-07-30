#!/usr/bin/env python
"""LoRA fine-tune OpenVLA on the oracle-stop replay dataset.

This is deliberately a small behavior-cloning baseline: near-wall oracle
interventions are labeled with LIBERO's zero-motion/open-gripper action and
off-path control observations retain the unchanged base policy's actions.  Only
``train.jsonl`` is accepted by this script; the held-out manifest is never opened
during training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import IGNORE_INDEX


def _register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OracleStopDataset(Dataset):
    def __init__(self, manifest: Path, processor, action_tokenizer: ActionTokenizer, base_checkpoint: str):
        if manifest.name != "train.jsonl":
            raise ValueError("trainer accepts only a manifest named train.jsonl")
        self.manifest = manifest
        self.root = manifest.parent
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("empty training manifest")
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.prompt_builder_fn = VicunaV15ChatPromptBuilder if "v01" in base_checkpoint else PurePromptBuilder
        self.stop_flags = np.asarray([bool(row["intervened"]) for row in self.rows])
        if self.stop_flags.all() or not self.stop_flags.any():
            raise ValueError("training data must contain stop and non-stop examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        action = np.asarray(row["normalized_action"], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"invalid normalized action at row {index}")

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {row['instruction']}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.processor.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)
        labels[: -(len(action) + 1)] = [IGNORE_INDEX] * (len(labels) - (len(action) + 1))
        return {
            "pixel_values": self.processor.image_processor.apply_transform(image),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        counts = np.bincount(self.stop_flags.astype(np.int64), minlength=2)
        weights = np.where(self.stop_flags, 1.0 / counts[1], 1.0 / counts[0])
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(self.rows),
            replacement=True,
            generator=generator,
        )


def _atomic_adapter_save(model, target: Path) -> None:
    tmp = target.parent / f".{target.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "adapter"
    merged_dir = output / "merged"
    metrics_path = output / "train_metrics.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    _register_openvla()

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.base_checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=min(args.lora_rank, 16),
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = OracleStopDataset(Path(args.train_manifest), processor, action_tokenizer, args.base_checkpoint)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    sampler = dataset.balanced_sampler(args.seed) if args.balance_stop_labels else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    iterator = iter(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(1, args.warmup_steps)
        return max(0.0, float(args.max_steps - step) / max(1, args.max_steps - args.warmup_steps))

    scheduler = LambdaLR(optimizer, lr_scale)
    print(
        json.dumps({
            "train_samples": len(dataset),
            "oracle_stop_samples": int(dataset.stop_flags.sum()),
            "reference_samples": int((~dataset.stop_flags).sum()),
            "max_steps": args.max_steps,
            "effective_batch": args.batch_size * args.grad_accumulation_steps,
        }, indent=2),
        flush=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with metrics_path.open("w") as metrics_file:
        for update_step in range(1, args.max_steps + 1):
            losses, accuracies = [], []
            for _ in range(args.grad_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = result.loss / args.grad_accumulation_steps
                loss.backward()
                losses.append(float(result.loss.detach().cpu()))
                with torch.no_grad():
                    shifted_labels = labels[:, 1:]
                    predictions = result.logits[:, :-1].argmax(dim=-1)
                    mask = shifted_labels > action_tokenizer.action_token_begin_idx
                    accuracy = ((predictions == shifted_labels) & mask).sum() / mask.sum().clamp_min(1)
                    accuracies.append(float(accuracy.detach().cpu()))

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            row = {
                "step": update_step,
                "loss": float(np.mean(losses)),
                "action_token_accuracy": float(np.mean(accuracies)),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if update_step == 1 or update_step % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if args.save_every and update_step % args.save_every == 0:
                _atomic_adapter_save(model, output / "adapter_latest")

    _atomic_adapter_save(model, adapter_dir)
    processor.save_pretrained(adapter_dir)
    summary = {
        "schema_version": 1,
        "base_checkpoint": args.base_checkpoint,
        "train_manifest": str(Path(args.train_manifest)),
        "train_manifest_sha256": _sha256(Path(args.train_manifest)),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "method": "LoRA behavior cloning of oracle-stop filtered OpenVLA actions",
        "heldout_used_for_training": False,
        "hyperparameters": {
            key: getattr(args, key)
            for key in (
                "max_steps", "batch_size", "grad_accumulation_steps", "learning_rate",
                "weight_decay", "warmup_steps", "lora_rank", "lora_dropout", "seed",
                "balance_stop_labels",
            )
        },
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if args.skip_merge:
        print(f"smoke run complete; adapter saved to {adapter_dir}; merge skipped", flush=True)
        return

    # Free GPU state, then merge on CPU so the result is directly loadable by OpenVLAPolicy.
    del optimizer, scheduler, loader, iterator, model
    gc.collect()
    torch.cuda.empty_cache()
    print("merging LoRA adapter into a standalone checkpoint on CPU", flush=True)
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.config.oracle_recovery = summary
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(merged_dir)
    (merged_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved merged recovery-finetuned OpenVLA checkpoint to {merged_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--train-manifest", default="results/oracle_recovery/dataset_v1/train.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accumulation-steps", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--balance-stop-labels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-merge", action="store_true",
                    help="save the adapter only (intended for one-step GPU smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if min(args.max_steps, args.batch_size, args.grad_accumulation_steps) < 1:
        raise SystemExit("max steps and batch sizes must be positive")
    train(args)


if __name__ == "__main__":
    main()

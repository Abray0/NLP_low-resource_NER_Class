"""
Named Entity Recognition Pipeline for Arabic (Low-Resource Language)
Uses XLM-RoBERTa with WikiANN dataset
Transfer learning approach with cross-lingual knowledge transfer
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline as hf_pipeline,
)
from datasets import load_dataset
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Label schema for WikiANN NER
# ─────────────────────────────────────────────
WIKIANN_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]
ID2LABEL = {i: l for i, l in enumerate(WIKIANN_LABELS)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}


@dataclass
class PipelineConfig:
    """Configuration for the NER pipeline."""
    model_name: str = "xlm-roberta-base"           # Pretrained multilingual model
    dataset_name: str = "wikiann"                   # WikiANN for NER
    language: str = "ar"                            # Target: Arabic
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./outputs/ner_model"
    seed: int = 42
    few_shot_samples: Optional[int] = None


class WikiANNDataset(Dataset):
    """Torch Dataset wrapper for WikiANN tokenized data."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


class NERPipeline:
    """
    End-to-end NER pipeline:
      1. Load pretrained XLM-R (100+ language cross-lingual model)
      2. SentencePiece tokenization (subword, language-agnostic)
      3. Fine-tune with task-specific classification head
      4. Evaluate with seqeval F1 / precision / recall
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.metrics_log: List[Dict] = []

    # ── Step 1: Load pretrained model & tokenizer ──────────────────────────
    def load_model(self):
        logger.info(f"Loading pretrained model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(WIKIANN_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        return self

    # ── Step 2: Load & preprocess WikiANN Arabic data ─────────────────────
    def load_data(self):
        logger.info(f"Loading WikiANN [{self.config.language}] dataset…")
        raw = load_dataset(self.config.dataset_name, self.config.language)

        if self.config.few_shot_samples:
            logger.info(f"Few-shot mode: {self.config.few_shot_samples} training samples")
            raw["train"] = raw["train"].shuffle(seed=self.config.seed).select(
                range(min(self.config.few_shot_samples, len(raw["train"])))
            )

        tokenized = raw.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=raw["train"].column_names,
        )
        self.datasets = tokenized
        logger.info(
            f"Data splits → train: {len(tokenized['train'])}, "
            f"val: {len(tokenized['validation'])}, test: {len(tokenized['test'])}"
        )
        return self

    def _tokenize_and_align_labels(self, examples):
        """
        SentencePiece subword tokenization with label alignment.
        Subword tokens that are not the first piece of a word get label -100 (ignored in loss).
        """
        tokenized = self.tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=self.config.max_length,
            is_split_into_words=True,
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != prev_word_id:
                    labels.append(ner_tags[word_id])
                else:
                    labels.append(-100)
                prev_word_id = word_id
            all_labels.append(labels)
        tokenized["labels"] = all_labels
        return tokenized

    # ── Step 3: Fine-tune ──────────────────────────────────────────────────
    def train(self):
        logger.info("Starting fine-tuning…")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        seqeval = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            true_preds = [
                [ID2LABEL[pred] for pred, lbl in zip(prediction, label) if lbl != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [ID2LABEL[lbl] for pred, lbl in zip(prediction, label) if lbl != -100]
                for prediction, label in zip(predictions, labels)
            ]
            results = seqeval.compute(predictions=true_preds, references=true_labels)
            metric_entry = {
                "precision": results["overall_precision"],
                "recall":    results["overall_recall"],
                "f1":        results["overall_f1"],
                "accuracy":  results["overall_accuracy"],
            }
            self.metrics_log.append(metric_entry)
            return metric_entry

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            seed=self.config.seed,
            logging_steps=50,
            report_to="none",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
        logger.info("Fine-tuning complete.")
        return self

    # ── Step 4: Evaluate ───────────────────────────────────────────────────
    def evaluate(self) -> Dict:
        logger.info("Evaluating on test set…")
        results = self.trainer.evaluate(self.datasets["test"])
        logger.info(f"Test results: {results}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save test metrics
        with open(os.path.join(self.config.output_dir, "test_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Save full training history from trainer logs
        history = self.trainer.state.log_history
        with open(os.path.join(self.config.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        self.plot_training(history, results)
        return results

    def plot_training(self, history: List[Dict], test_results: Dict):
        """Generate and save training charts."""
        charts_dir = os.path.join(self.config.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Extract per-epoch metrics
        train_loss, eval_loss, eval_f1, eval_precision, eval_recall, epochs = [], [], [], [], [], []
        for entry in history:
            if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
                train_loss.append(entry["loss"])
            if "eval_f1" in entry:
                eval_loss.append(entry.get("eval_loss", 0))
                eval_f1.append(entry["eval_f1"] * 100)
                eval_precision.append(entry.get("eval_precision", 0) * 100)
                eval_recall.append(entry.get("eval_recall", 0) * 100)
                epochs.append(int(round(entry["epoch"])))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("white")
        plt.rcParams.update({"text.color": "#e5e7eb", "axes.labelcolor": "#9ca3af",
                              "xtick.color": "#6b7280", "ytick.color": "#6b7280"})

        colors = {"train": "#7c6af7", "f1": "#fbbf24", "precision": "#00d4aa", "recall": "#f97316"}

        # Chart 1: Loss curve
        ax = axes[0]
        ax.set_facecolor("#f8f8f8")
        if train_loss:
            ax.plot(range(1, len(train_loss)+1), train_loss, color=colors["train"],
                    marker="o", linewidth=2, markersize=5, label="Train Loss")
        if eval_loss:
            ax.plot(epochs, eval_loss, color=colors["f1"],
                    marker="s", linewidth=2, markersize=5, label="Val Loss")
        ax.set_title("Loss Curve", color="white", fontweight="bold")
        ax.set_xlabel("Step / Epoch"); ax.set_ylabel("Loss")
        ax.legend(facecolor="white", edgecolor="#cccccc")
        ax.grid(color="#dddddd", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        # Chart 2: F1 / Precision / Recall per epoch
        ax = axes[1]
        ax.set_facecolor("#f8f8f8")
        if epochs:
            ax.plot(epochs, eval_f1,        color=colors["f1"],        marker="o", linewidth=2, markersize=5, label="F1")
            ax.plot(epochs, eval_precision, color=colors["precision"],  marker="s", linewidth=2, markersize=5, label="Precision")
            ax.plot(epochs, eval_recall,    color=colors["recall"],     marker="^", linewidth=2, markersize=5, label="Recall")
        ax.set_title("NER Metrics per Epoch", color="white", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.legend(facecolor="white", edgecolor="#cccccc")
        ax.grid(color="#dddddd", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        # Chart 3: Final test metrics bar
        ax = axes[2]
        ax.set_facecolor("#f8f8f8")
        metrics = {
            "F1":        test_results.get("eval_f1", 0) * 100,
            "Precision": test_results.get("eval_precision", 0) * 100,
            "Recall":    test_results.get("eval_recall", 0) * 100,
            "Accuracy":  test_results.get("eval_accuracy", 0) * 100,
        }
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      color=["#fbbf24","#00d4aa","#f97316","#7c6af7"], alpha=0.9)
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", color="#111111", fontsize=10, fontweight="bold")
        ax.set_title("Test Set Results", color="white", fontweight="bold")
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110)
        ax.grid(axis="y", color="#ffffff10", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        plt.suptitle("NER Pipeline — Arabic (XLM-RoBERTa)", color="#111111", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(charts_dir, "ner_training_results.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Charts saved to {out_path}")

    # ── Inference ──────────────────────────────────────────────────────────
    def get_inference_pipeline(self):
        """Returns an HuggingFace pipeline ready for inference."""
        return hf_pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, text: str) -> List[Dict]:
        """Run NER on an Arabic text string."""
        pipe = self.get_inference_pipeline()
        entities = pipe(text)
        return [
            {
                "entity_group": ent["entity_group"],
                "word": ent["word"],
                "score": round(float(ent["score"]), 4),
                "start": ent["start"],
                "end": ent["end"],
            }
            for ent in entities
        ]

    # ── Save / Load ────────────────────────────────────────────────────────
    def save(self, path: Optional[str] = None):
        save_dir = path or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[PipelineConfig] = None):
        cfg = config or PipelineConfig(output_dir=path)
        instance = cls(cfg)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForTokenClassification.from_pretrained(path)
        return instance


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
def run_full_pipeline(few_shot: bool = False):
    config = PipelineConfig(
        few_shot_samples=100 if few_shot else None
    )
    pipeline = (
        NERPipeline(config)
        .load_model()
        .load_data()
        .train()
    )
    metrics = pipeline.evaluate()
    pipeline.save()
    return pipeline, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arabic NER Pipeline")
    parser.add_argument("--few-shot", action="store_true", help="Use 100-sample few-shot training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./outputs/ner_model")
    args = parser.parse_args()

    config = PipelineConfig(
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        few_shot_samples=100 if args.few_shot else None,
    )
    pipeline = NERPipeline(config).load_model().load_data().train()
    pipeline.evaluate()
    pipeline.save()

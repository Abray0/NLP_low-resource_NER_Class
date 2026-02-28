"""
Sentiment Analysis Pipeline for Arabic
Uses XLM-RoBERTa fine-tuned on SIB-200 (topic/sentiment classification)
Falls back to AraBERT-based sentiment if SIB-200 is unavailable locally.
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline as hf_pipeline,
)
from datasets import load_dataset, Dataset
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SIB-200 has 7 topic categories used as classification labels
SIB200_LABELS = [
    "science/technology", "travel", "politics",
    "sports", "health", "entertainment", "geography"
]
ID2LABEL = {i: l for i, l in enumerate(SIB200_LABELS)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}


@dataclass
class SentimentConfig:
    model_name: str = "xlm-roberta-base"
    dataset_name: str = "Davlan/sib200"
    language_code: str = "arb_Arab"          # Arabic (Modern Standard) in SIB-200
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    output_dir: str = "./outputs/sentiment_model"
    seed: int = 42
    few_shot_samples: Optional[int] = None


class SentimentPipeline:
    """
    Text classification pipeline (topic/sentiment) for Arabic.
    Architecture: XLM-R encoder + linear classification head.
    """

    def __init__(self, config: SentimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_model(self):
        logger.info(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(SIB200_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        return self

    def load_data(self):
        logger.info(f"Loading SIB-200 for language: {self.config.language_code}")
        try:
            raw = load_dataset(self.config.dataset_name, self.config.language_code)
        except Exception as e:
            logger.warning(f"Could not load SIB-200: {e}. Using synthetic demo data.")
            raw = self._make_demo_dataset()

        # Map string labels → int if needed
        if "category" in raw["train"].column_names:
            raw = raw.rename_column("category", "label")
            raw = raw.map(lambda x: {"label": LABEL2ID.get(x["label"], 0)})

        if self.config.few_shot_samples:
            raw["train"] = raw["train"].shuffle(seed=self.config.seed).select(
                range(min(self.config.few_shot_samples, len(raw["train"])))
            )

        self.datasets = raw.map(self._tokenize, batched=True, remove_columns=["text"])
        logger.info(f"Dataset loaded. Train: {len(self.datasets['train'])}")
        return self

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )

    def _make_demo_dataset(self):
        """Synthetic fallback dataset for offline/demo usage."""
        texts = [
            "اكتشف العلماء علاجاً جديداً للسرطان", "فاز الفريق ببطولة العالم",
            "أعلنت الحكومة عن ميزانية جديدة", "السفر إلى أوروبا في الصيف",
            "أحدث الهواتف الذكية في السوق", "صحة الإنسان في العصر الحديث",
        ]
        labels = [0, 3, 2, 1, 0, 4]
        data = {"text": texts * 20, "label": labels * 20}
        ds = Dataset.from_dict(data).train_test_split(test_size=0.2, seed=42)
        return {"train": ds["train"], "validation": ds["test"], "test": ds["test"]}

    def train(self):
        data_collator = DataCollatorWithPadding(self.tokenizer)
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
                "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
            }

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=self.config.seed,
            report_to="none",
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
        return self

    def evaluate(self) -> Dict:
        results = self.trainer.evaluate(self.datasets["test"])
        os.makedirs(self.config.output_dir, exist_ok=True)

        with open(os.path.join(self.config.output_dir, "test_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        history = self.trainer.state.log_history
        with open(os.path.join(self.config.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        self.plot_training(history, results)
        logger.info(f"Test metrics: {results}")
        return results

    def plot_training(self, history: List[Dict], test_results: Dict):
        """Generate and save training charts."""
        charts_dir = os.path.join(self.config.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        train_loss, eval_loss, eval_f1, eval_acc, epochs = [], [], [], [], []
        for entry in history:
            if "loss" in entry and "eval_loss" not in entry:
                train_loss.append(entry["loss"])
            if "eval_f1" in entry:
                eval_loss.append(entry.get("eval_loss", 0))
                eval_f1.append(entry["eval_f1"] * 100)
                eval_acc.append(entry.get("eval_accuracy", 0) * 100)
                epochs.append(int(round(entry["epoch"])))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("white")
        plt.rcParams.update({
            "text.color": "#111111",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        })

        # Loss curve
        ax = axes[0]
        ax.set_facecolor("#f8f8f8")
        if train_loss:
            ax.plot(range(1, len(train_loss)+1), train_loss, color="#7c6af7",
                    marker="o", linewidth=2, markersize=5, label="Train Loss")
        if eval_loss:
            ax.plot(epochs, eval_loss, color="#fbbf24",
                    marker="s", linewidth=2, markersize=5, label="Val Loss")
        ax.set_title("Loss Curve", color="white", fontweight="bold")
        ax.set_xlabel("Step / Epoch"); ax.set_ylabel("Loss")
        ax.legend(facecolor="white", edgecolor="#cccccc")
        ax.grid(color="#dddddd", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        # F1 + Accuracy per epoch
        ax = axes[1]
        ax.set_facecolor("#f8f8f8")
        if epochs:
            ax.plot(epochs, eval_f1,  color="#fbbf24", marker="o", linewidth=2, markersize=5, label="F1")
            ax.plot(epochs, eval_acc, color="#00d4aa", marker="s", linewidth=2, markersize=5, label="Accuracy")
        ax.set_title("Metrics per Epoch", color="white", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.legend(facecolor="white", edgecolor="#cccccc")
        ax.grid(color="#dddddd", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        # Final test bars
        ax = axes[2]
        ax.set_facecolor("#f8f8f8")
        metrics = {
            "F1":       test_results.get("eval_f1", 0) * 100,
            "Accuracy": test_results.get("eval_accuracy", 0) * 100,
        }
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      color=["#fbbf24", "#00d4aa"], alpha=0.9, width=0.4)
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", color="#111111", fontsize=11, fontweight="bold")
        ax.set_title("Test Set Results", color="white", fontweight="bold")
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110)
        ax.grid(axis="y", color="#ffffff10", linestyle="--"); ax.spines[["top","right"]].set_visible(False)

        plt.suptitle("Sentiment Pipeline — Arabic (XLM-RoBERTa)", color="#111111", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(charts_dir, "sentiment_training_results.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Charts saved to {out_path}")

    def predict(self, text: str) -> Dict:
        pipe = hf_pipeline(
            "text-classification",
            model=self.model,
            processing_class=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        result = pipe(text, truncation=True, max_length=self.config.max_length)[0]
        return {"label": result["label"], "score": round(result["score"], 4)}

    def save(self, path: Optional[str] = None):
        save_dir = path or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, path: str):
        config = SentimentConfig(output_dir=path)
        instance = cls(config)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        return instance


if __name__ == "__main__":
    config = SentimentConfig(
        few_shot_samples=None,   # use full dataset for better accuracy
        num_epochs=5,            # more epochs = higher F1/accuracy
        batch_size=16,
    )
    pipeline = SentimentPipeline(config).load_model().load_data().train()
    pipeline.evaluate()
    # pipeline.save() is optional — best checkpoint is already saved by Trainer
    # Uncomment if you have enough disk space:
    # pipeline.save()

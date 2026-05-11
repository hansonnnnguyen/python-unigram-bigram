# DistilBERT model with IMDb database
# before running install and virtual environment:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install datasets transformers torch scikit-learn pandas
# pip install "accelerate>=1.1.0"
# pip install "transformers[torch]"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb")

# Use smaller subsets first if your computer is slow
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(300))
# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 5. Training settings
training_args = TrainingArguments(
    output_dir="./tmp",
    eval_strategy="no",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",
    disable_tqdm=False
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

# 7. Train
trainer.train()
results = trainer.predict(test_dataset)
logits = results.predictions
labels = results.label_ids

predictions = np.argmax(logits, axis=1)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
acc = accuracy_score(labels, predictions)

print("\nDistilBERT Results:")
print("accuracy:", acc)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

# Output:
# DistilBERT Results:
# accuracy: 0.82
# precision: 0.8529411764705882
# recall: 0.7733333333333333
# f1: 0.8111888111888111
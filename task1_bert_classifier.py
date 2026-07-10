import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1. Load the AG News Dataset from Hugging Face
print("--- Loading AG News Dataset ---")
dataset = load_dataset("fancyzhx/ag_news")
# Using a smaller subset for faster training on standard hardware
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(3000))
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(500))

# 2. Tokenization & Preprocessing
print("--- Initializing BERT Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


print("--- Preprocessing and Tokenizing Data ---")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Load BERT Model (AG News has 4 classes)
print("--- Loading BERT Base Model ---")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# 4. Set Up Evaluation Metrics (Accuracy & F1-Score)
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": acc, "f1": f1}


# 5. Define Training Arguments
print("--- Configuring Training Parameters ---")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 7. Fine-tune the Model
print("--- Starting BERT Fine-Tuning ---")
trainer.train()

# 8. Evaluate and Print Final Metrics
print("--- Evaluating Model Performance ---")
eval_results = trainer.evaluate()
print(f"\nFinal Evaluation Results: {eval_results}\n")

# 9. Save the Fine-Tuned Model and Tokenizer for deployment
print("--- Saving Fine-Tuned Model for Live Deployment ---")
model_save_path = "./saved_bert_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved successfully to {model_save_path}!")
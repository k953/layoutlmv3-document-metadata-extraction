import torch
from datasets import Dataset
from transformers import (
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from dataset import load_dataset, LABELS, id2label, label2id

train_data = load_dataset("train.csv", "train_images")

if len(train_data) == 0:
    raise ValueError("❌ No training data found")

train_dataset = Dataset.from_list(train_data)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./model",

    per_device_train_batch_size=1,   # small dataset → better memorization
    num_train_epochs=18,             # 50 nahi ❌
    learning_rate=3e-5,              # stable for LayoutLMv3

    warmup_ratio=0.1,                # very important
    logging_steps=5,

    save_strategy="no",              # last model enough
    report_to="none",                # warning remove

    fp16=torch.cuda.is_available()
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator
)

trainer.train()
model.save_pretrained("./model")

print("✅ Training complete")

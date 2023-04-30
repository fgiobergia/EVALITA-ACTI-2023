import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer
import evaluate

def compute_metrics(eval_pred):
    (logits, _), labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # predictions = torch.max(logits, axis=1).indices
    return f1.compute(predictions=predictions, references=labels, average="macro")

def build_tokenizer_func(tokenizer):
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)
    return tokenize_func

def load_data_for_task(tokenizer, task="A", load_val=False):
    df = pd.read_csv(f"subtask{task}_train.csv", index_col=0)
    if task == "B":
        df.drop(columns=["topic"], inplace=True)
    df.columns = ["text", "labels"]
    
    if load_val:
        df_train, df_val = train_test_split(df, train_size=.8)
    else:
        df_train = df
    
    df_test = pd.read_csv(f"subtask{task}_test.csv", index_col=0)
    df_test.columns = ["text"]

    ds_train = Dataset.from_pandas(df_train, split="train")
    if load_val:
        ds_val = Dataset.from_pandas(df_val, split="test")
    ds_test = Dataset.from_pandas(df_test, split="test")
    
    return (ds_train, ds_val, ds_test) if load_val == True else (ds_train, ds_test)

import torch
task = "B"
config = {
    "A": {
        "num_labels": 2,
        "n_epochs": 8
    },
    "B": {
        "num_labels": 4,
        "n_epochs": 10
    }
}
model = AutoModelForSequenceClassification.from_pretrained("morenolq/bart-it", num_labels=config[task]["num_labels"])
tokenizer = AutoTokenizer.from_pretrained("morenolq/bart-it")
ds_train, ds_val, ds_test = load_data_for_task(tokenizer, task, load_val=True)

tok_func = build_tokenizer_func(tokenizer)
ds_train_tok = ds_train.map(tok_func, batched=True)
ds_val_tok = ds_val.map(tok_func, batched=False)
ds_test_tok = ds_test.map(tok_func, batched=False)

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=config[task]["n_epochs"],
)

f1 = evaluate.load("f1")
trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=ds_train_tok,
    eval_dataset=ds_val_tok,
    compute_metrics=compute_metrics,
)

trainer.train()

with torch.no_grad():
    model.eval()
    pred = model(torch.tensor(ds_test_tok["input_ids"]).to("cuda:0"), torch.tensor(ds_test_tok["attention_mask"]).to("cuda:0"))
    model.train()
    y_pred = pred.logits.argmax(axis=1).cpu().detach().numpy()

    df_test = pd.read_csv(f"subtask{task}_test.csv", index_col=0)
    df_test["Expected"] = y_pred

    df_test.to_csv(f"output_{task}.csv", index="Id", columns=["Expected"])
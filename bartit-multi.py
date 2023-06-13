import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer
import evaluate

import torch.nn as nn
import torch
from transformers.utils import ModelOutput

class MultiTaskModel(nn.Module):
    def __init__(self, encoder, max_size=200):
        super().__init__()
        self.encoder = encoder
        self.fc_binary = nn.Linear(768, 768)
        self.head_binary = nn.Linear(768, 1)

        self.fc_categ = nn.Linear(768, 768)
        self.head_categ = nn.Linear(768, 4)

        self.encoder._init_weights(self.fc_binary)
        self.encoder._init_weights(self.fc_categ)
        self.encoder._init_weights(self.head_binary)
        self.encoder._init_weights(self.head_categ)

        self.loss_binary = nn.BCEWithLogitsLoss()
        self.loss_categ = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x[:, 0] # use only [CLS]

        x_binary = self.fc_binary(x)
        x_binary = torch.tanh(x_binary) # relu?
        x_binary = self.head_binary(x_binary)

        x_categ = self.fc_categ(x)
        x_categ = torch.tanh(x_categ)
        x_categ = self.head_categ(x_categ)

        x = torch.cat([x_binary, x_categ], axis=-1)

        if labels is not None:
            # loss = self.loss(x, labels)
            zero_cat = (labels[:, 0] == 0) | (labels[:, 1] == 4)
            loss = self.loss_binary(x_binary.flatten(), labels[:, 0].float()) + self.loss_categ(x_categ[~zero_cat], labels[~zero_cat, 1])
            return ModelOutput({"loss": loss, "logits": x })

        return ModelOutput({ "logits": x })

from transformers import Trainer

# class MultiTaskTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss_categ = nn.CrossEntropyLoss()
#         self.loss_binary = nn.BCEWithLogitsLoss()
        
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # implement custom logic here
#         output = model(inputs["input_ids"], inputs["attention_mask"])
        
#         zero_cat = (inputs["labels"][:, 0] == 0) | (inputs["labels"][:, 1] == 4)

#         loss_bin = self.loss_binary(output.get("logits")[:, 0], inputs["labels"][:, 0].float())
#         loss_cat = self.loss_categ(output.get("logits")[~zero_cat, 1:], inputs["labels"][~zero_cat, 1])

#         lmbda = 1.
#         loss = loss_bin + lmbda * loss_cat
        
#         if return_outputs:
#             return loss, output
#         return loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from sklearn.metrics import roc_curve, f1_score
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    meaningful_categ = labels[:, 1] != 4
    
    bin_labels = labels[~meaningful_categ, 0]
    categ_labels = labels[meaningful_categ, 1]
    
    fpr, tpr, thresh = roc_curve(labels[:,0], sigmoid(logits[:, 0]))
    best_thresh = thresh[((1-tpr)**2 + fpr**2).argmin()]
    
    bin_pres = sigmoid(logits[:, 0]) > best_thresh
    categ_preds = np.argmax(logits[meaningful_categ, 1:], axis=-1)
    
    # f1A = f1.compute(predictions=bin_pres, references=bin_labels, average="macro")["f1"]
    # f1B = f1.compute(predictions=categ_preds, references=categ_labels, average="macro")["f1"]
    f1A = f1_score(labels[:,0], bin_pres, average="macro")
    f1B = f1_score(categ_labels, categ_preds, average="macro")
    f1_overall = .6 * f1A + .4 * f1B

    return {
        "f1_A": f1A,
        "f1_B": f1B,
        "f1": f1_overall,
        "thresh": best_thresh,
    }

def build_tokenizer_func(tokenizer):
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)
    return tokenize_func

def load_data_for_task(tokenizer, load_val=False):
    df_A = pd.read_csv("subtaskA_train.csv", index_col=0)
    df_B = pd.read_csv("subtaskB_train.csv", index_col=0)
    df_B.drop(columns=["topic"], inplace=True)

    df_A.columns = ["text", "labels_binary"]
    df_B.columns = ["text", "labels_categ"]

    df_A["labels_categ"] = 4
    df_B["labels_binary"] = 1

    # df = pd.concat([df_A, df_B], axis=0).sample(frac=1., replace=False, random_state=42)
    df = df_A.merge(df_B, on="text", how="outer").sample(frac=1., replace=False, random_state=42)
    
    df["labels_binary"] = ((df["labels_binary_x"] == 1) | (df["labels_binary_y"] == 1)).astype(int)
    df["labels_categ"] = df["labels_categ_y"].map(lambda x: 4 if np.isnan(x) else x).astype(int)
    

    df["labels"] = df.apply(lambda x: [x["labels_binary"], x["labels_categ"]], axis=1)
    # df.drop(columns=["labels_binary", "labels_categ"], inplace=True)
    df = df[["text", "labels"]]
    
    if load_val:
        df_train, df_val = train_test_split(df, train_size=.8)
    else:
        df_train = df
    
    df_test_A = pd.read_csv("subtaskA_test.csv", index_col=0)
    df_test_B = pd.read_csv("subtaskB_test.csv", index_col=0)
    df_test = pd.concat([df_test_A, df_test_B], axis=0)
    df_test.columns = ["text"]

    ds_train = Dataset.from_pandas(df_train, split="train")
    if load_val:
        ds_val = Dataset.from_pandas(df_val, split="test")
    ds_test = Dataset.from_pandas(df_test, split="test")
    
    return (ds_train, ds_val, ds_test) if load_val == True else (ds_train, ds_test)

import torch

n_epochs = 15

model_name = "morenolq/bart-it-fanpage"
encoder = AutoModel.from_pretrained(model_name)
model = MultiTaskModel(encoder=encoder)
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds_train, ds_val, ds_test = load_data_for_task(tokenizer, load_val=True)

tok_func = build_tokenizer_func(tokenizer)
ds_train_tok = ds_train.map(tok_func, batched=True)
ds_val_tok = ds_val.map(tok_func, batched=False)
ds_test_tok = ds_test.map(tok_func, batched=False)

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=n_epochs,
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
    # y_pred = pred.logits.argmax(axis=1).cpu().detach().numpy()

    df_test_A = pd.read_csv("subtaskA_test.csv", index_col=0)
    df_test_B = pd.read_csv("subtaskB_test.csv", index_col=0)
    
    df_test_A["Expected"] = (torch.sigmoid(pred.logits[:len(df_test_A), 0]) > .5).int().tolist()
    df_test_B["Expected"] = pred.logits[len(df_test_A):, 1:].argmax(axis=1).tolist()

    df_test_A.to_csv("output_A.csv", index="Id", columns=["Expected"])
    df_test_B.to_csv("output_B.csv", index="Id", columns=["Expected"])
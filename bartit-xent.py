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

from model import ConvModel

class MultiTaskModel(nn.Module):
    def __init__(self, encoder, extra_vecs=None, max_size=200, use_conv=True, text_features=None):
        super().__init__()
        self.encoder = encoder
        
        emb_size = 0
        self.use_text_features = False
        if self.encoder is not None:
            emb_size += 768 # assuming 768
        if use_conv and extra_vecs is not None:
            emb_size += 16
            self.conv_model = ConvModel(extra_vecs.shape[0], extra_vecs, text_features)
            if text_features is not None:
                self.use_text_features = True
        else:
            self.conv_model = None

        self.fc_categ = nn.Linear(emb_size, 768)
        self.head_categ = nn.Linear(768, 5)

        if self.encoder:
            self.encoder._init_weights(self.fc_categ)
            self.encoder._init_weights(self.head_categ)
        
        self.loss_categ = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, x_tokens, x_text, labels=None, **kwargs):
        if self.encoder is not None:
            x_enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x_enc = x_enc[:, 0] # use only [CLS]
        else:
            x_enc = torch.zeros((input_ids.shape[0], 0)).cuda()
        
        if self.conv_model is not None:
            x_conv = self.conv_model(x_tokens, x_text if self.use_text_features else None)
        else:
            x_conv = torch.zeros((input_ids.shape[0], 0)).cuda()
        
        x = torch.hstack([x_enc , x_conv])

        x = self.fc_categ(x)
        x = torch.tanh(x)
        x = self.head_categ(x)

        if labels is not None:
            # loss = self.loss(x, labels)
            
            loss_known = self.loss_categ(x[labels != 5], labels[labels!=5])
            loss_unknown = self.loss_categ(x[labels==5], x[labels==5, :4].argmax(axis=1))

            lmbda = 0.5
            # loss = self.loss_binary(x_binary.flatten(), labels[:, 0].float()) + lmbda * self.loss_categ(x_categ[~zero_cat], labels[~zero_cat, 1])
            
            loss = loss_known + lmbda * loss_unknown
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
    
    y_bin_true = labels != 4
    y_bin_pred = logits.argmax(axis=1) != 4
    
    y_cat_true = labels[labels < 4]
    y_cat_pred = logits[labels < 4, :4].argmax(axis=1)
    
    f1A = f1_score(y_bin_true, y_bin_pred, average="macro")
    f1B = f1_score(y_cat_true, y_cat_pred, average="macro")
    f1_overall = .6 * f1A + .4 * f1B

    return {
        "f1_A": f1A,
        "f1_B": f1B,
        "f1": f1_overall,
    }
def build_tokenizer_func(tokenizer, max_len=200):
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    return tokenize_func

from tfidf import load_vocabs, get_vec

from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


    
def count_upper(s):
    return sum([ x.isupper() for x in s ])

def encode_df(df, tokenizer, tokens_map, max_len = None):
    tokenized = []
    upper = []
    for sent in df.text:
        toks = tokenizer.tokenize(unidecode(sent).lower())
        toks_up = tokenizer.tokenize(unidecode(sent))
        tokenized.append([ tokens_map.get(tok, len(tokens_map)) for tok in toks ])
        upper.append([ count_upper(tok) / len(tok) for tok in toks_up ])
    
    if max_len is not None:
        max_len = max_len if max_len != "compute" else max(map(len, tokenized))
        tokenized = [ tokens[:max_len] + [len(tokens_map)] * (max_len - len(tokens)) for tokens in tokenized ]
        upper = [ up[:max_len] + [0] * (max_len - len(up)) for up in upper ]
    return tokenized, upper

def load_data_for_task(tokenizer, load_val=False):
    df_A = pd.read_csv("subtaskA_train.csv", index_col=0)
    df_B = pd.read_csv("subtaskB_train.csv", index_col=0)
    df_B.drop(columns=["topic"], inplace=True)

    df_A.columns = ["text", "labels"]
    df_B.columns = ["text", "labels"]
    
    df_A["labels"] += 4 # "4" for "no conspiracy", "5" for conspiracy, but unknown
    
    df = pd.concat([ df_A, df_B ]).groupby("text").min("labels").sample(frac=1., replace=False, random_state=42).reset_index()
    
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
    
    # Now, load stuff for the conv model
    tokenizer = RegexpTokenizer(r'\w+')

    tok_freq = {}
    for sent in df.text:
        toks = tokenizer.tokenize(unidecode(sent).lower())
        for tok in toks:
            tok_freq[tok] = tok_freq.get(tok, 0) + 1

    threshold = 0# 0 => take all!
    # tokens = { k: v for k, v in tok_freq.items() if v > threshold }
    tokens = sorted([ k for k, v in tok_freq.items() if v > threshold ])
    tokens_map = { k: v for v, k in enumerate(tokens) }

    n_tokens = len(tokens_map)

    max_len = 660 # precomputed
    tok, up = encode_df(df_train, tokenizer, tokens_map, max_len=max_len)
    # X_train, upper_train = torch.tensor().cuda()
    ds_train = ds_train.add_column("x_tokens", tok)
    ds_train = ds_train.add_column("x_text", up)
    
    
    if load_val:
        # X_val = torch.tensor(encode_df(df_val, tokenizer,  tokens_map, max_len=max_len)).cuda()
        tok, up = encode_df(df_val, tokenizer,  tokens_map, max_len=max_len)
        ds_val = ds_val.add_column("x_tokens", tok)
        ds_val = ds_val.add_column("x_text", up)

    # X_test = torch.tensor(encode_df(df_test, tokenizer,  tokens_map, max_len=max_len)).cuda()
    tok, up = encode_df(df_test, tokenizer,  tokens_map, max_len=max_len)
    ds_test = ds_test.add_column("x_tokens", tok)
    ds_test = ds_test.add_column("x_text", up)
    
    vocabs_tfidf, mat_tfidf = load_vocabs()
    mat_info = np.vstack([ get_vec(t, vocabs_tfidf, mat_tfidf) for t in tokens ])

    return (mat_info, ds_train, ds_val, ds_test) if load_val == True else (mat_info, ds_train, ds_test)

import torch

if __name__ == "__main__":
    torch.random.manual_seed(42)
    n_epochs = 11

    model_name = "morenolq/bart-it-fanpage"
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_val = True
    
    if load_val:
        mat_info, ds_train, ds_val, ds_test = load_data_for_task(tokenizer, load_val=True)
    else:
        mat_info, ds_train, ds_test = load_data_for_task(tokenizer, load_val=False)
    
    model = MultiTaskModel(encoder=encoder, extra_vecs=mat_info, text_features=1)

    max_len = 400
    tok_func = build_tokenizer_func(tokenizer, max_len)
    ds_train_tok = ds_train.map(tok_func, batched=True)
    if load_val:
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
        # eval_dataset=ds_val_tok,
        eval_dataset=ds_val_tok if load_val else ds_train_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    with torch.no_grad():
        model.eval()
        
        batch_size=32
        pred = []
        
        for i in range(0, len(ds_test_tok), batch_size):
            ds = ds_test_tok[i:i+batch_size]
            p = model(**{ k: torch.tensor(x).cuda() for k, x in ds.items() if k in ["attention_mask", "input_ids", "labels", "x_tokens", "x_text"] }).logits
            pred.append(p)
        
        pred = torch.vstack(pred)
        # pred = model(torch.tensor(ds_test_tok["input_ids"]).to("cuda:0"), torch.tensor(ds_test_tok["attention_mask"]).to("cuda:0"))
        model.train()
        # y_pred = pred.logits.argmax(axis=1).cpu().detach().numpy()

        df_test_A = pd.read_csv("subtaskA_test.csv", index_col=0)
        df_test_B = pd.read_csv("subtaskB_test.csv", index_col=0)

        df_test_A["Expected"] = (torch.sigmoid(pred[:len(df_test_A), 0]) > .8).int().tolist()
        df_test_B["Expected"] = pred[len(df_test_A):, 1:].argmax(axis=1).tolist()

        df_test_A.to_csv("output_A.csv", index="Id", columns=["Expected"])
        df_test_B.to_csv("output_B.csv", index="Id", columns=["Expected"])
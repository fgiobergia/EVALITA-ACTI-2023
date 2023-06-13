import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer

import torch.nn as nn
import torch
from transformers.utils import ModelOutput

from model import ConvModel

class MultiTaskModel(nn.Module):
    """
    encoder: BertModel-like encoder
    extra_vecs: additional vectors extracted and used for the 1-d conv
    use_conv: whether to use 1-d convolution
    text_features: additional token-based features that can be passed to the convolution (differently
        from extra_vecs, these vectors are directly added, whereas extra_vecs works as a lookup table)
        (number of text_features or None)
    lmbda: coefficient for loss function
    n_tokens: number of total tokens that should be used for the 1-d conv (size of embedding in conv)
    """
    def __init__(self, encoder, extra_vecs=None, use_conv=True, text_features=None, lmbda=1., n_tokens=None, n_extra=0):
        super().__init__()
        self.encoder = encoder
        
        emb_size = n_extra
        self.n_extra = n_extra
        self.lmbda = lmbda
        self.use_text_features = False
        if self.encoder is not None:
            emb_size += 768 # assuming 768
        if use_conv:
            emb_size += 16
            self.conv_model = ConvModel(n_tokens, extra_vecs, text_features)
            if text_features is not None:
                self.use_text_features = True
        else:
            self.conv_model = None
        # if text_features is not None:
        #     emb_size += text_features
        self.fc_binary = nn.Linear(emb_size, 768)
        self.head_binary = nn.Linear(768, 1)

        self.fc_categ = nn.Linear(emb_size, 768)
        self.head_categ = nn.Linear(768, 4)

        if self.encoder:
            self.encoder._init_weights(self.fc_binary)
            self.encoder._init_weights(self.fc_categ)
            self.encoder._init_weights(self.head_binary)
            self.encoder._init_weights(self.head_categ)
        
        

        self.loss_binary = nn.BCEWithLogitsLoss()
        self.loss_categ = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, x_tokens, x_text, x_extra, labels=None, **kwargs):
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
        if self.n_extra:
            assert x_extra.shape[1] == self.n_extra
            x = torch.hstack([ x, x_extra ])

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
            loss = self.loss_binary(x_binary.flatten(), labels[:, 0].float()) + self.lmbda * self.loss_categ(x_categ[~zero_cat], labels[~zero_cat, 1])
            return ModelOutput({"loss": loss, "logits": x })

        return ModelOutput({ "logits": x })

from transformers import Trainer

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
    best_thresh = .5
    
    bin_pres = sigmoid(logits[:, 0]) > best_thresh
    categ_preds = np.argmax(logits[meaningful_categ, 1:], axis=-1)
    
    f1A = f1_score(labels[:,0], bin_pres, average="macro")
    f1B = f1_score(categ_labels, categ_preds, average="macro")
    f1_overall = .6 * f1A + .4 * f1B

    return {
        "f1_A": f1A,
        "f1_B": f1B,
        "f1": f1_overall,
        "thresh": best_thresh,
    }

def build_tokenizer_func(tokenizer, max_len=200):
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    return tokenize_func

from tfidf import load_vocabs, get_vec, load_w2v

from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


    
def count_upper(s):
    return sum([ x.isupper() for x in s ]) / len(s)

def count_punct(s):
    return (s.count("!") + s.count("?")) / len(s)

def encode_df(df, tokenizer, tokens_map, max_len = None):
    tokenized = []
    upper = []
    for sent in df.text:
        toks = tokenizer.tokenize(unidecode(sent).lower())
        toks_up = tokenizer.tokenize(unidecode(sent))
        tokenized.append([ tokens_map.get(tok, len(tokens_map)) for tok in toks ])
        upper.append([ count_upper(tok) for tok in toks_up ])
    
    if max_len is not None:
        max_len = max_len if max_len != "compute" else max(map(len, tokenized))
        tokenized = [ tokens[:max_len] + [len(tokens_map)] * (max_len - len(tokens)) for tokens in tokenized ]
        upper = [ up[:max_len] + [0] * (max_len - len(up)) for up in upper ]
    return tokenized, upper

def load_data_for_task(tokenizer, load_val=False):
    df_A = pd.read_csv("subtaskA_train.csv", index_col=0)
    df_B = pd.read_csv("subtaskB_train.csv", index_col=0)
    df_B.drop(columns=["topic"], inplace=True)

    df_A.columns = ["text", "labels_binary"]
    df_B.columns = ["text", "labels_categ"]

    df_A["labels_categ"] = 4
    df_B["labels_binary"] = 1

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
    ds_train = ds_train.add_column("x_extra", [ [ count_punct(x) ] for x in df_train.text ])
    
    
    if load_val:
        # X_val = torch.tensor(encode_df(df_val, tokenizer,  tokens_map, max_len=max_len)).cuda()
        tok, up = encode_df(df_val, tokenizer,  tokens_map, max_len=max_len)
        ds_val = ds_val.add_column("x_tokens", tok)
        ds_val = ds_val.add_column("x_text", up)
        ds_val = ds_val.add_column("x_extra", [ [ count_punct(x) ] for x in df_val.text ])

    # X_test = torch.tensor(encode_df(df_test, tokenizer,  tokens_map, max_len=max_len)).cuda()
    tok, up = encode_df(df_test, tokenizer,  tokens_map, max_len=max_len)
    ds_test = ds_test.add_column("x_tokens", tok)
    ds_test = ds_test.add_column("x_text", up)
    ds_test = ds_test.add_column("x_extra", [ [ count_punct(x) ] for x in df_test.text ])
    
    dfA = df_train.loc[df_train["labels"].map(lambda x: x[1] == 4)]
    dfB = df_train.loc[df_train["labels"].map(lambda x: x[1] != 4)]
    dfA["conspiratorial"] = dfA["labels"].map(lambda x: x[0])
    dfB["conspiracy"] = dfB["labels"].map(lambda x: x[1])
    dfA = dfA[["text", "conspiratorial"]]
    dfB = dfB[["text", "conspiracy"]]
    vocabs_tfidf, mat_tfidf = load_vocabs(
        # dfA = dfA.rename(columns={"text": "comment_text"}),
        # dfB = dfB.rename(columns={"text": "comment_text"})
    )
    mat_info = np.vstack([ get_vec(t, vocabs_tfidf, mat_tfidf) for t in tokens ])

    # mat_sent = load_w2v(tokens)
    # mat_info = np.hstack([ mat_info, mat_sent ])

    return (mat_info, tokens, ds_train, ds_val, ds_test) if load_val == True else (mat_info, tokens, ds_train, ds_test)

import torch
import transformers

if __name__ == "__main__":
    torch.random.manual_seed(42)
    transformers.set_seed(42)
    n_epochs = 11

    model_name = "morenolq/bart-it"
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_val = True
    
    if load_val:
        mat_info, tokens, ds_train, ds_val, ds_test = load_data_for_task(tokenizer, load_val=True)
    else:
        mat_info, tokens, ds_train, ds_test = load_data_for_task(tokenizer, load_val=False)

        
    max_len = 200
    tok_func = build_tokenizer_func(tokenizer, max_len)
    ds_train_tok = ds_train.map(tok_func, batched=True)
    if load_val:
        ds_val_tok = ds_val.map(tok_func, batched=False)
    ds_test_tok = ds_test.map(tok_func, batched=False)

    lmbda = 1.
    model = MultiTaskModel(encoder=encoder, extra_vecs=mat_info, text_features=1, lmbda=lmbda, n_tokens=len(tokens), n_extra=0)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=n_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok if load_val else ds_train_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

#     with torch.no_grad():
#         model.eval()
        
#         batch_size=32
#         pred = []
        
#         for i in range(0, len(ds_test_tok), batch_size):
#             ds = ds_test_tok[i:i+batch_size]
#             p = model(**{ k: torch.tensor(x).cuda() for k, x in ds.items() if k in ["attention_mask", "input_ids", "labels", "x_tokens", "x_text"] }).logits
#             pred.append(p)
        
#         pred = torch.vstack(pred)
#         # pred = model(torch.tensor(ds_test_tok["input_ids"]).to("cuda:0"), torch.tensor(ds_test_tok["attention_mask"]).to("cuda:0"))
#         model.train()
#         # y_pred = pred.logits.argmax(axis=1).cpu().detach().numpy()

#         df_test_A = pd.read_csv("subtaskA_test.csv", index_col=0)
#         df_test_B = pd.read_csv("subtaskB_test.csv", index_col=0)

#         df_test_A["Expected"] = (torch.sigmoid(pred[:len(df_test_A), 0]) > .8).int().tolist()
#         df_test_B["Expected"] = pred[len(df_test_A):, 1:].argmax(axis=1).tolist()

#         df_test_A.to_csv("output_A.csv", index="Id", columns=["Expected"])
#         df_test_B.to_csv("output_B.csv", index="Id", columns=["Expected"])
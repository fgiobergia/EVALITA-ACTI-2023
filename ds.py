import datasets

def _info(self):
    features = datasets.Features(
        {
            "id": datasets.Value("int32"),
            "comment_text": dataset.Value("string"),
            "conspirational": dataset.Value("bool")
        }
    )
    
    di = datasets.DatasetInfo(
        description="None", 
        features=features,
        supervised_keys=None,
        homepage="",
        citation=""
    )
    return di

def _generate_examples(self, filepath):
    df = pd.read_csv(filepath, index_col=0)
    for row_id, row in df.iterrows():
        yield row_id, {
            "id": row_id,
            "comment_text": row.comment_text,
            "conspirational": row.conspirational
        }
        
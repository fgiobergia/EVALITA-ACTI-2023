import torch 
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, n_tokens, extra_vecs=None, text_features=None):
        super().__init__()
        emb_size = 768
        conv1_channels = 64
        proj_size = 16
        
        # assuming custom tokenizer, use padding_idx = n_tokens + 1 for the filler
        padding_idx = n_tokens
        n_tokens += 1
        
        self.emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size, padding_idx=padding_idx)
        
        if extra_vecs is not None:
            # make sure we are passing a matrix shaped correctly
            # (+1 to account for the +1 that is being passed when n_tokens != 52000
            assert n_tokens == extra_vecs.shape[0] + 1
            self.extra_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=extra_vecs.shape[1], padding_idx=padding_idx)
            with torch.no_grad():
                self.extra_emb.weight[:-1] = torch.tensor(extra_vecs)
            
            emb_size += extra_vecs.shape[1]
        else:
            self.extra_emb = None
        
        if text_features is not None:
            emb_size += text_features
        
        
        
        self.conv1 = nn.Conv1d(emb_size, conv1_channels, stride=1, kernel_size=3)
        self.conv2 = nn.Conv1d(conv1_channels, 32, stride=1, kernel_size=5)
        self.conv3 = nn.Conv1d(32, 8, stride=3, kernel_size=5)
        
        self.mp = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.fc1 = nn.Linear(56, proj_size)
        
    def forward(self, x, x_text=None):
        x1 = self.emb(x).permute(0, 2, 1)
        if self.extra_emb is not None:
            x = torch.cat([ x1, self.extra_emb(x).permute(0,2,1) ], axis=1)
        else:
            x = x1
       
        if x_text is not None:
            # x shape = (batch, channel, len)
            x = torch.cat([ x, x_text.reshape(x_text.shape[0], 1, -1) ], axis=1)
        
        x = torch.relu(self.mp(self.conv1(x)))
        x = torch.relu(self.mp(self.conv2(x)))
        x = torch.relu(self.mp(self.conv3(x)))
        
        x = x.reshape(x.shape[0], -1)
        
        return self.fc1(x) # headless
import torch
import math
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,seq_len:int,hidden_state:int)->torch.tensor:
        pos = torch.zeros(seq_len,hidden_state)
        max_pos = torch.arange(seq_len).unsqueeze(1)
        fun = torch.exp(torch.arange(0, hidden_state, 2).float() * (-math.log(10000.0) / hidden_state))
        angle = max_pos * fun
        pos[:,0::2] = torch.sin(angle)
        pos[:,1::2] = torch.cos(angle)
        return pos


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_states,no_of_heads):
        super().__init__()
        self.hidden = hidden_states
        self.heads = no_of_heads
        self.head_dim = hidden_states // no_of_heads
        self.q = nn.Linear(self.hidden,self.hidden,bias=False)
        self.k = nn.Linear(self.hidden,self.hidden,bias=False)
        self.v = nn.Linear(self.hidden,self.hidden,bias=False)
        self.w = nn.Linear(self.hidden,self.hidden,bias=False)

    def forward(self,source,mask=None):
        batch,seq_len,_=source.shape
        q = self.q(source)
        k = self.k(source)
        v = self.v(source)
        q_h = q.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        k_h = k.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        v_h = v.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        k_t = torch.transpose(k_h,-2,-1)
        scores = (q_h@k_t) / math.sqrt(self.head_dim)
        if mask is not None:
            mask_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask_matrix,float('-inf'))
        pred_q_k = torch.softmax(scores,dim=-1)
        output = (pred_q_k @ v_h)
        output = torch.transpose(output,1,2).reshape(batch,seq_len,-1)
        final_output = self.w(output)
        return final_output , pred_q_k

class SingleGPT(nn.Module):
    def __init__(self, hidden_states,no_of_heads=8):
        super().__init__()
        self.hidden_states = hidden_states
        self.no_of_heads = no_of_heads
        self.norm1 = nn.LayerNorm(hidden_states)
        self.norm2 = nn.LayerNorm(hidden_states)
        self.masked_multi_head = MultiHeadAttention(hidden_states=self.hidden_states,no_of_heads=self.no_of_heads)
        self.l1 = nn.Linear(hidden_states,4*hidden_states,bias=True)
        self.l2 = nn.Linear(4*hidden_states,hidden_states,bias=True)

    def forward(self,source:torch.tensor,mask=None):
        z_dash      = source + self.masked_multi_head(self.norm1(source), mask)[0]
        final_output = z_dash + self.l2(torch.relu(self.l1(self.norm2(z_dash))))
        return final_output


class MiniGPT(nn.Module):
    def __init__(self,vocab_size ,no_of_components,hidden_states, no_of_heads,mask=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_states)
        self.out = nn.Linear(hidden_states, vocab_size)
        self.no_of_components = no_of_components
        self.hidden_states = hidden_states
        self.no_of_heads = no_of_heads
        self.mask = mask
        self.pos_enc = PositionEncoding()
        self.gpts = nn.ModuleList()
        for _ in range(self.no_of_components):
            self.gpts.append(SingleGPT(self.hidden_states,no_of_heads))

    def forward(self,source: torch.tensor):
        source = self.embedding(source)
        _,seq_len,_ = source.shape
        positional_encoding = self.pos_enc(seq_len,self.hidden_states)
        source = source+positional_encoding
        for gpt in self.gpts:
            source = gpt(source,mask=self.mask)
        source = self.out(source)
        return source

    def evaluate(self, dataloader, loss_fn):
        self.eval()

        device = next(self.parameters()).device

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device).long()
                y = y.to(device).long()

                logits = self(x)

                logits = logits.reshape(-1, logits.size(-1))
                targets = y.reshape(-1)

                loss = loss_fn(logits, targets)

                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

        return total_loss / total_tokens


    def fit(self, train_loader, val_loader, epochs, loss_fn, optimizer):
        self.train()
        device = next(self.parameters()).device
        history = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            total_loss = 0.0
            total_tokens = 0
            for x, y in train_loader:

                x = x.to(device).long()
                y = y.to(device).long()

                optimizer.zero_grad()

                # Forward pass
                logits = self(x)

                logits = logits.reshape(-1, logits.size(-1))
                targets = y.reshape(-1)

                # Calculate loss
                loss = loss_fn(logits, targets)

                # Backpropagation
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

            # Average training loss
            train_loss = total_loss / total_tokens

            # Validation loss
            val_loss = self.evaluate(val_loader, loss_fn)

            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        return history


if __name__ == "__main__":
    model = MiniGPT(vocab_size=100, no_of_components=6, hidden_states=128, no_of_heads=8, mask=True)
    x = torch.randint(0, 100, (2, 6))
    print(model(x))
    print(model(x).shape)
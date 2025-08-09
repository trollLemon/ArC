from torch import nn




""" constants """
color_dim = 3

class ColorEmbedding(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(color_dim, hidden_dim) # -> B,N,hidden_dim

    def forward(self,x):
        return self.projection(x)

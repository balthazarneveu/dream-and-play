import torch
import torch.nn as nn
import numpy as np


def sinusoidal_positional_encoding(positions, d_model):

    """
    positions: torch.Tensor of shape (batch_size, seq_len), the positions to encode
    d_model: int, the dimension of the model
    """
    
    seq_len = positions.shape[1]
    # Fréquences définies par la dimension du modèle [1, d_model]
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    
    positions = positions.unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Calcul des encodages sinusoïdaux
    sinusoidal_encoding = torch.zeros((positions.shape[0], positions.shape[1], d_model))  # (batch_size, seq_len, d_model)
    sinusoidal_encoding[:, :, 0::2] = torch.sin(positions * div_term)  # Sinus sur les indices pairs
    sinusoidal_encoding[:, :, 1::2] = torch.cos(positions * div_term)  # Cosinus sur les indices impairs

    return sinusoidal_encoding

class SelfAttention(nn.Module):
    
    def __init__(self, embed_size, heads):
    
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert(self.head_dim*heads == embed_size), "Embed size needs to be divisible by heads"

        # compute the values, keys and queries for all heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
    
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand(N, 1, query_len, key_len)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy/ (self.head_dim ** 0.5), dim = 3) # normalize accross the key_len

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)

        return out, attention

class MoveTransformerBlock(nn.Module):
    
    def __init__(self, embed_size, heads, dropout, forward_expansion) :
        
        super(MoveTransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        
        attention, attention_matrix = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out, attention_matrix


class MoveEncoderPoint(nn.Module):
    
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, timeframes,
                  n_coords, n_points):

        super(MoveEncoderPoint, self).__init__()

        self.embed_size = embed_size
        self.timeframes = timeframes
        self.n_coords = n_coords
        self.n_points = n_points

        self.coords_embeddings = nn.Linear(n_coords*timeframes, embed_size)
        self.points_embeddings = nn.Embedding(n_points+1, embed_size, padding_idx = n_points)

        self.layers = nn.ModuleList([MoveTransformerBlock(embed_size, heads, dropout, forward_expansion)
                                     for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, point_id, coords, attention_mask):

        # generate time ids: shape [batch_size, n_points*timeframes] 
        # each row in the format [0, 1, 2, ..., timeframes-1, 0, 1, 2, ..., timeframes-1, ...]
        out = self.dropout(self.relu(self.points_embeddings(point_id))+\
                            self.coords_embeddings(coords)
                            )

        attention_matrices = []
        for layer in self.layers:
            out, attention_matrix = layer(out, out, out, attention_mask)
            attention_matrices.append(attention_matrix)

        return out, attention_matrices

class MoveEncoderPose(nn.Module):
    
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout,
                  n_pose_features):

        super(MoveEncoderPose, self).__init__()

        self.embed_size = embed_size
        
        self.pose_features_embeddings = nn.Linear(n_pose_features, embed_size)
        
        self.layers = nn.ModuleList([MoveTransformerBlock(embed_size, heads, dropout, forward_expansion)
                                     for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, pose_features, positions, attention_mask):

        # generate time ids: shape [batch_size, n_points*timeframes] 
        # each row in the format [0, 1, 2, ..., timeframes-1, 0, 1, 2, ..., timeframes-1, ...]
        positions -= positions[:, -1].unsqueeze(1)
        out = self.dropout(self.pose_features_embeddings(pose_features)+\
                           sinusoidal_positional_encoding(positions, self.embed_size)
                            )

        attention_matrices = []
        for layer in self.layers:
            out, attention_matrix = layer(out, out, out, attention_mask)
            attention_matrices.append(attention_matrix)

        return out, attention_matrices
    
if __name__ == "__main__":
    # Exemple d'utilisation
    batch_size = 8
    positions = torch.randn(batch_size, 39)
    d_model = 64  # Dimension du modèle
    positional_encodings = sinusoidal_positional_encoding(positions, d_model)
    print(positional_encodings.shape)  # (50, 128)


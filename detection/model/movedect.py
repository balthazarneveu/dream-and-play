import torch
import torch.nn as nn

from transformers.modeling_outputs import MaskedLMOutput

from movedect_utils import MoveEncoder


TIMEFRAMES = 20 # 10 fps
N_COORDS = 2
N_POINTS = 50
N_CLASSES = 5

class TransformerForMoveDetection(nn.Module):

    """ 
    A transformer-based model for ...
    """

    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, timeframes = TIMEFRAMES,
                  n_coords = N_COORDS, n_points = N_POINTS, n_classes = N_CLASSES):

        super(TransformerForMoveDetection, self).__init__()

        self.timeframes = timeframes
        self.n_coords = n_coords
        self.n_points = n_points

        self.move_encoder = MoveEncoder(embed_size, num_layers, heads, forward_expansion, dropout, timeframes,
                  n_coords, n_points)

        self.decoder = nn.Linear(embed_size, n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, coords, attention_mask):

        """
        inputs_ids: torch.Tensor of shape (batch_size, n_points), vector of points IDs
        labels: torch.Tensor of shape (batch_size), vector of move (collection of frames) labels
        coords: torch.Tensor of shape (batch_size, n_points, n_coords*timeframes), vector of points coordinates
        accross timeframes
        attention_mask: torch.Tensor of shape (batch_size, n_points), mask for non visible points
        """
        
        points_embeddings, attention_matrices = self.move_encoder(input_ids, coords, attention_mask)

        # pooling of the points_embeddings (B, N_points, D)
        points_embeddings = points_embeddings.max(dim=1).values # (B, D)

        output = self.decoder(points_embeddings) # (B, n_classes)
        
        loss = self.criterion(output, labels)

        return MaskedLMOutput(loss = loss,
                              logits = output,
                              hidden_states = points_embeddings,
                              attentions=attention_matrices)

if __name__ == '__main__':

    batch_size = 8
    
    input_ids = torch.randint(0, 50, (batch_size, N_POINTS))
    labels = torch.randint(0, N_CLASSES, (batch_size,))
    coords = torch.randn(batch_size, N_POINTS, N_COORDS*TIMEFRAMES)
    attention_mask = torch.randint(0, 2, (batch_size, N_POINTS))

    model = TransformerForMoveDetection(embed_size = 64, num_layers = 1, heads = 2, forward_expansion = 4, dropout = 0.1)
    
    with torch.no_grad():
        output = model(input_ids, labels, coords, attention_mask)

        print(f"output.shape: {output.logits.shape}\n loss: {output.loss.item()}")
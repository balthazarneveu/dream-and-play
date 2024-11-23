import torch
import torch.nn as nn

from transformers.modeling_outputs import MaskedLMOutput

from model.movedect_utils import MoveEncoderPoint, MoveEncoderPose


TIMEFRAMES = 20 # 10 fps
N_COORDS = 2
N_POINTS = 50
N_CLASSES = 5
N_POSE_FEATURES = 33*3

class TransformerForPointMoveDetection(nn.Module):

    """ 
    A transformer-based model for ...
    """

    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, timeframes = TIMEFRAMES,
                  n_coords = N_COORDS, n_points = N_POINTS, n_classes = N_CLASSES):

        super(TransformerForPointMoveDetection, self).__init__()

        self.timeframes = timeframes
        self.n_coords = n_coords
        self.n_points = n_points

        self.move_encoder = MoveEncoderPoint(embed_size, num_layers, heads, forward_expansion, dropout, timeframes,
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

class TransformerForPoseMoveDetection(nn.Module):

    """ 
    A transformer-based model for ...
    """

    def __init__(self, embed_size, num_layers, heads, forward_expansion=4, 
                 dropout = 0.1, n_pose_features = N_POSE_FEATURES, n_classes = N_CLASSES):

        super(TransformerForPoseMoveDetection, self).__init__()

        
        self.move_encoder = MoveEncoderPose(embed_size, num_layers, heads, forward_expansion, dropout, n_pose_features)

        self.decoder = nn.Linear(embed_size, n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, labels, pose_features, positions, attention_mask):

        """
        labels: torch.Tensor of shape (batch_size), vector of move (collection of frames) labels
        pose_features: torch.Tensor of shape (batch_size, n_pose, n_pose_features), tensor of features of each
        pose in the move
        positions: torch.Tensor of shape (batch_size, n_pose), vector of time positions of each pose
        attention_mask: torch.Tensor of shape (batch_size, n_pose), mask for non visible points
        """
        
        pose_embeddings, attention_matrices = self.move_encoder(pose_features, positions, attention_mask)

        # pooling of the pose_embeddings (B, N_poses, D)
        pose_embeddings = pose_embeddings.max(dim=1).values # (B, D)

        output = self.decoder(pose_embeddings) # (B, n_classes)
        
        loss = None
        if labels is not None:
            loss = self.criterion(output, labels)

        return MaskedLMOutput(loss = loss,
                              logits = output,
                              hidden_states = pose_embeddings,
                              attentions=attention_matrices)

class TransformerForPoseMoveDetectionV2(nn.Module):

    """ 
    A transformer-based model for ...
    """

    def __init__(self, embed_size, num_layers, heads, forward_expansion=4, 
                 dropout = 0.1, n_pose_features = N_POSE_FEATURES, n_classes = N_CLASSES):

        super(TransformerForPoseMoveDetectionV2, self).__init__()

        self.n_classes = n_classes

        self.move_encoder = MoveEncoderPose(embed_size, num_layers, heads, forward_expansion, dropout, n_pose_features)

        self.decoder = nn.Linear(embed_size, n_classes)

        self.criterion = nn.CrossEntropyLoss(ignore_index = -100)

    def forward(self, labels, pose_features, positions, attention_mask):

        """
        labels: torch.Tensor of shape (batch_size), vector of move (collection of frames) labels
        pose_features: torch.Tensor of shape (batch_size, n_pose, n_pose_features), tensor of features of each
        pose in the move
        positions: torch.Tensor of shape (batch_size, n_pose), vector of time positions of each pose
        attention_mask: torch.Tensor of shape (batch_size, n_pose), mask for non visible points
        """
        batch_size = pose_features.shape[0]

        pose_embeddings, attention_matrices = self.move_encoder(pose_features, positions, attention_mask) #(B, N_poses, D)

        output = self.decoder(pose_embeddings) # (B, n_classes)
        
        loss = None
        if labels is not None:
            loss = self.criterion(output.view(-1, self.n_classes), labels.view(-1))

        return MaskedLMOutput(loss = loss,
                              logits = output,
                              hidden_states = pose_embeddings,
                              attentions=attention_matrices)

if __name__ == '__main__':

    batch_size = 8
    
    """
    # for point level pose encoding
    input_ids = torch.randint(0, 50, (batch_size, N_POINTS))
    labels = torch.randint(0, N_CLASSES, (batch_size,))
    coords = torch.randn(batch_size, N_POINTS, N_COORDS*TIMEFRAMES)
    attention_mask = torch.randint(0, 2, (batch_size, N_POINTS))

    model = TransformerForPointMoveDetection(embed_size = 64, num_layers = 1, heads = 2, forward_expansion = 4, dropout = 0.1)
    """

    # for pose level pose encoding
    n_pose = 39
    n_pose_features = 33*3 # 34 joints with 3 coordinates each
    labels = torch.randint(0, N_CLASSES, (batch_size,))
    pose_features = torch.randn(batch_size, n_pose, n_pose_features)
    positions = torch.randn(batch_size, n_pose)
    attention_mask = torch.randint(0, 2, (batch_size, n_pose))

    model = TransformerForPoseMoveDetection(embed_size = 64, 
                                            num_layers = 1, 
                                            heads = 2, 
                                            forward_expansion = 4, 
                                            dropout = 0.1, 
                                            n_pose_features = n_pose_features)
    
    with torch.no_grad():
        output = model(labels,pose_features, positions, attention_mask)

        print(f"output.shape: {output.logits.shape}\n loss: {output.loss.item()}")
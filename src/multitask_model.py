# PyTorch provides tensors and neural-network functionality.
import torch

# nn contains neural-network layers such as Embedding, Linear,
# and TransformerEncoder.
import torch.nn as nn

# Functional utilities include embedding normalization.
import torch.nn.functional as F


class TransformerHead(nn.Module):
    """
    A task-specific transformer output head.

    It converts the transformed task token into one scalar output.
    """

    def __init__(
        self,
        model_dim,
        hidden_dim=128,
    ):
        # Initialize the parent nn.Module class.
        super().__init__()

        # Define the task output head.
        self.net = nn.Sequential(
            # Normalize the transformer representation.
            nn.LayerNorm(model_dim),

            # Convert it into a hidden representation.
            nn.Linear(model_dim, hidden_dim),

            # Introduce nonlinearity.
            nn.ReLU(),

            # Process the hidden representation again.
            nn.Linear(hidden_dim, hidden_dim),

            # Introduce another nonlinearity.
            nn.ReLU(),

            # Produce one scalar output.
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Run the transformer representation through the output head.
        output = self.net(x)

        # Remove the final size-one dimension.
        #
        # Shape changes:
        #   [batch_size, 1] -> [batch_size]
        return output.squeeze(-1)


class MultiTaskWorldModel(nn.Module):
    """
    Transformer model with:
        one shared entity embedding table
        one shared transformer encoder
        one distance task token
        one nearest-neighbor task token
        one distance output head
        one nearest-neighbor output head
    """

    def __init__(
        self,
        num_points,
        emb_dim=32,
        hidden_dim=128,
        normalize_embeddings=False,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    ):
        # Initialize the parent nn.Module class.
        super().__init__()

        # A transformer divides the embedding dimension among its heads.
        if emb_dim % num_heads != 0:
            raise ValueError(
                "emb_dim must be divisible by num_heads"
            )

        # Create one learned vector for every world point.
        #
        # Shape:
        #   [num_points, emb_dim]
        self.emb = nn.Embedding(
            num_points,
            emb_dim,
        )

        # Store whether every entity embedding should be normalized
        # to have Euclidean length 1.
        self.normalize_embeddings = (
            normalize_embeddings
        )

        # Create one learned token for the distance task.
        #
        # Shape:
        #   [1, 1, emb_dim]
        self.distance_token = nn.Parameter(
            torch.empty(
                1,
                1,
                emb_dim,
            )
        )

        # Create one learned token for the nearest-neighbor task.
        #
        # Shape:
        #   [1, 1, emb_dim]
        self.nearest_token = nn.Parameter(
            torch.empty(
                1,
                1,
                emb_dim,
            )
        )

        # Define one transformer encoder block.
        transformer_layer = nn.TransformerEncoderLayer(
            # Every token contains emb_dim values.
            d_model=emb_dim,

            # Split attention across several heads.
            nhead=num_heads,

            # Size of the feed-forward network inside each transformer layer.
            dim_feedforward=hidden_dim,

            # Dropout probability.
            dropout=dropout,

            # Nonlinearity inside the transformer layer.
            activation="relu",

            # Inputs use:
            #   [batch_size, sequence_length, emb_dim]
            batch_first=True,
        )

        # Stack the requested number of transformer layers.
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers,
        )

        # Task-specific distance predictor.
        self.distance_head = TransformerHead(
            model_dim=emb_dim,
            hidden_dim=hidden_dim,
        )

        # Task-specific nearest-neighbor classifier.
        self.nearest_head = TransformerHead(
            model_dim=emb_dim,
            hidden_dim=hidden_dim,
        )

        # Initialize entity embeddings with small random values.
        nn.init.normal_(
            self.emb.weight,
            std=0.02,
        )

        # Initialize the distance task token.
        nn.init.normal_(
            self.distance_token,
            std=0.02,
        )

        # Initialize the nearest task token.
        nn.init.normal_(
            self.nearest_token,
            std=0.02,
        )

    def encode(self, indices):
        """
        Convert point indices into learned embeddings.
        """

        # Look up every requested embedding.
        embeddings = self.emb(indices)

        # Optionally force every embedding to have Euclidean length 1.
        if self.normalize_embeddings:
            embeddings = F.normalize(
                embeddings,
                dim=-1,
            )

        return embeddings

    def pair_representation(
        self,
        i,
        j,
        task_token,
    ):
        """
        Convert two batches of point indices into one transformer
        representation.

        Each sequence contains three tokens:

            [task token, point i, point j]

        No positional embeddings are added, so the two entity tokens
        are treated symmetrically.
        """

        # Look up embeddings for the first point in every pair.
        embedding_i = self.encode(i)

        # Look up embeddings for the second point in every pair.
        embedding_j = self.encode(j)

        # Read the current batch size.
        batch_size = i.shape[0]

        # Copy the learned task token once for every example.
        #
        # Shape:
        #   [1, 1, emb_dim]
        #       ->
        #   [batch_size, 1, emb_dim]
        expanded_task_token = task_token.expand(
            batch_size,
            -1,
            -1,
        )

        # Add a sequence dimension to each entity embedding.
        #
        # Shape:
        #   [batch_size, emb_dim]
        #       ->
        #   [batch_size, 1, emb_dim]
        embedding_i = embedding_i.unsqueeze(1)
        embedding_j = embedding_j.unsqueeze(1)

        # Construct the transformer input sequence.
        #
        # Shape:
        #   [batch_size, 3, emb_dim]
        tokens = torch.cat(
            [
                expanded_task_token,
                embedding_i,
                embedding_j,
            ],
            dim=1,
        )

        # Process all three tokens with the transformer.
        #
        # Shape:
        #   [batch_size, 3, emb_dim]
        transformed_tokens = self.transformer(
            tokens
        )

        # Use the transformed task token as the pair representation.
        #
        # Shape:
        #   [batch_size, emb_dim]
        return transformed_tokens[:, 0, :]

    def forward_distance(self, i, j):
        """
        Predict normalized distance for each point pair.
        """

        # Create the transformer pair representation using
        # the distance task token.
        pair = self.pair_representation(
            i,
            j,
            self.distance_token,
        )

        # Run the representation through the distance-specific head.
        return self.distance_head(pair)

    def forward_nearest(self, i, j):
        """
        Predict nearest-neighbor logits for each point pair.

        This returns logits, not probabilities.
        """

        # Create the transformer pair representation using
        # the nearest-neighbor task token.
        pair = self.pair_representation(
            i,
            j,
            self.nearest_token,
        )

        # Run the representation through the nearest-specific head.
        return self.nearest_head(pair)

    def forward(self, task, i, j):
        """
        General task-routing method.
        """

        # Route distance examples to the distance head.
        if task == "distance":
            return self.forward_distance(i, j)

        # Route nearest examples to the nearest head.
        if task == "nearest":
            return self.forward_nearest(i, j)

        # Reject unsupported task names.
        raise ValueError(
            f"Unknown task: {task}"
        )
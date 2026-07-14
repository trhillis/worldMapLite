# PyTorch provides tensors and neural-network functionality.
import torch

# nn contains neural-network layers such as Embedding and Linear.
import torch.nn as nn

# Functional utilities include embedding normalization.
import torch.nn.functional as F


class PairFeatures(nn.Module):
    """
    Convert two entity embeddings into symmetric pair features.

    Symmetric means swapping i and j produces the same representation.

    This is appropriate because:
        distance(i, j) == distance(j, i)
        nearest(i, j) is treated symmetrically in this setup
    """

    def forward(self, embedding_i, embedding_j):
        # Absolute difference describes how different the embeddings are
        # in each dimension.
        absolute_difference = torch.abs(
            embedding_i - embedding_j
        )

        # Elementwise product describes where the embeddings have
        # similar signs and magnitudes.
        elementwise_product = (
            embedding_i * embedding_j
        )

        # Squared difference emphasizes larger coordinate differences.
        squared_difference = (
            embedding_i - embedding_j
        ) ** 2

        # Join all pair features into one longer vector.
        return torch.cat(
            [
                absolute_difference,
                elementwise_product,
                squared_difference,
            ],
            dim=-1,
        )


class MLPHead(nn.Module):
    """
    A task-specific multilayer perceptron.

    It converts pair features into one scalar output.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
    ):
        # Initialize the parent nn.Module class.
        super().__init__()

        # Define the task head as a sequence of layers.
        self.net = nn.Sequential(
            # Convert the pair features into a hidden representation.
            nn.Linear(input_dim, hidden_dim),

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
        # Run the input through the MLP.
        output = self.net(x)

        # Remove the final size-one dimension.
        #
        # Shape changes:
        #   [batch_size, 1] -> [batch_size]
        return output.squeeze(-1)


class MultiTaskWorldModel(nn.Module):
    """
    Model with:
        one shared entity embedding table
        one distance head
        one nearest-neighbor head
    """

    def __init__(
        self,
        num_points,
        emb_dim=32,
        hidden_dim=128,
        normalize_embeddings=False,
    ):
        # Initialize the parent nn.Module class.
        super().__init__()

        # Create one learned vector for every world point.
        #
        # Shape:
        #   [num_points, emb_dim]
        self.emb = nn.Embedding(
            num_points,
            emb_dim,
        )

        # Store whether every embedding should be normalized
        # to have length 1.
        self.normalize_embeddings = (
            normalize_embeddings
        )

        # PairFeatures concatenates three emb_dim-sized vectors.
        pair_dim = emb_dim * 3

        # Shared transformation from two embeddings into pair features.
        self.pair_features = PairFeatures()

        # Task-specific distance predictor.
        self.distance_head = MLPHead(
            input_dim=pair_dim,
            hidden_dim=hidden_dim,
        )

        # Task-specific nearest-neighbor classifier.
        self.nearest_head = MLPHead(
            input_dim=pair_dim,
            hidden_dim=hidden_dim,
        )

        # Initialize embeddings with small random values.
        nn.init.normal_(
            self.emb.weight,
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

    def pair_representation(self, i, j):
        """
        Convert two batches of point indices into pair features.
        """

        # Look up embeddings for the first point in every pair.
        embedding_i = self.encode(i)

        # Look up embeddings for the second point in every pair.
        embedding_j = self.encode(j)

        # Build the shared symmetric pair representation.
        return self.pair_features(
            embedding_i,
            embedding_j,
        )

    def forward_distance(self, i, j):
        """
        Predict normalized distance for each point pair.
        """

        # Create shared pair features.
        pair = self.pair_representation(i, j)

        # Run them through the distance-specific head.
        return self.distance_head(pair)

    def forward_nearest(self, i, j):
        """
        Predict nearest-neighbor logits for each point pair.

        This returns logits, not probabilities.
        """

        # Create shared pair features.
        pair = self.pair_representation(i, j)

        # Run them through the nearest-specific head.
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
        raise ValueError(f"Unknown task: {task}")
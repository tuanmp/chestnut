import torch
import torch.nn.functional as F

from chestnut.utils.ml_utils import make_mlp

from ..graph_construction_stage import MetricLearningBase

_ENCODING_SCHEMES = ("none", "pe", "rff")

class FourierInputEncoder(torch.nn.Module):
    """Scheme-driven Fourier/positional input encoder.

    Produces, per encoded channel, `sin(x * bands)` and `cos(x * bands)`, optionally
    concatenating the raw channel and passing through any non-encoded channels
    unchanged. No normalization is applied here.
    """

    def __init__(self, scheme: str, bands: torch.Tensor, encode_indices: list[int], include_raw: bool):
        super().__init__()
        self.scheme = scheme
        self.encode_indices = sorted(list(set(encode_indices)))
        self.include_raw = bool(include_raw)
        self.register_buffer("bands", bands)  # (1, k) or (n_encoded, m)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass_indices = [i for i in range(x.shape[-1]) if i not in self.encode_indices]
        x_encoded = x[..., self.encode_indices]  # (..., n_encoded)
        angles = x_encoded.unsqueeze(-1) * self.bands  # (..., n_encoded, n_bands)

        parts = []
        if self.include_raw:
            parts.append(x_encoded)
        parts.append(torch.sin(angles).flatten(-2))
        parts.append(torch.cos(angles).flatten(-2))
        parts.append(x[..., pass_indices])
        return torch.cat(parts, dim=-1)


def build_input_encoder(scheme: str, 
                        encode_indices: list[int], 
                        include_raw: bool,
                        encode_config: dict):
    """Build the configured Fourier/positional input encoder.

    Input scaling is expected to come from the stage's `node_scales`: the encoder
    applies no normalization of its own, so frequencies are expressed in whatever
    units the features arrive in (post-`node_scales`).

    Args:
        scheme: One of `"none"`, `"pe"`, or `"rff"`.
        encode_indices: Input channel indices to encode; unused by `"none"`.
        include_raw: Concatenate the raw values of the encoded channels
            alongside their sin/cos features.
        encode_config: Scheme parameters:
            - `encoding_k`: number of frequency bands for `"pe"` and `"rff"`
              (default 8).
            - `encoding_sigma`: frequency scale for `"rff"` (default 1.0).
            - `encoding_seed`: random seed for `"rff"` (default 0).

    Returns:
        The encoder module. `"none"` returns `torch.nn.Identity()`.
    """
    scheme = str(scheme).lower()
    if scheme not in _ENCODING_SCHEMES:
        raise ValueError(
            f"Unknown input_encoding '{scheme}'; expected one of {_ENCODING_SCHEMES}"
        )

    if scheme == "none":
        return torch.nn.Identity()

    n_encoded = len(encode_indices)

    if scheme == "pe":
        k = encode_config.get("encoding_k", 8)
        if k < 1:
            raise ValueError("encoding_k must be >= 1")
        bands = 2.0 ** torch.arange(k).float().reshape(1, -1) * torch.pi
    else:  # rff
        m = int(encode_config.get("encoding_k", 8))
        if m < 1:
            raise ValueError("encoding_k must be >= 1")
        sigma = float(encode_config.get("encoding_sigma", 1.0))
        seed = int(encode_config.get("encoding_seed", 0))
        generator = torch.Generator().manual_seed(seed)
        bands = 2.0 * torch.pi * sigma * torch.randn(n_encoded, m, generator=generator)

    encoder = FourierInputEncoder(
        scheme=scheme,
        bands=bands,
        encode_indices=encode_indices,
        include_raw=include_raw,
    )

    return encoder


class FourierDoubleMetricLearning(MetricLearningBase):
    """Double metric learning with a Fourier/positional input encoder."""
    rearrange_edges: bool = False

    def __init__(
        self,
        emb_hidden: int,
        emb_dim: int,
        activation: str,
        nb_layer: int,
        r_train: float,
        r_val: float,
        knn: int,
        knn_val: int,
        weighting: list[dict],
        margin: float,
        points_per_batch: int,
        randomisation: float,
        input_encoding: str,
        encode_indices: list[int],
        include_raw: bool,
        encode_config: dict,
    ):

        self.input_encoding = input_encoding
        self.encode_indices = encode_indices
        self.include_raw = include_raw
        self.encode_config = encode_config

        if input_encoding.lower() != 'none':
            assert len(encode_indices) > 0, "encode_indices must be a non-empty list of integers"

        super().__init__(
            emb_hidden=emb_hidden,
            emb_dim=emb_dim,
            activation=activation,
            nb_layer=nb_layer,
            r_train=r_train,
            r_val=r_val,
            knn=knn,
            knn_val=knn_val,
            weighting=weighting,
            margin=margin,
            points_per_batch=points_per_batch,
            randomisation=randomisation,
        )

    def build_network(self):
        self.input_encoder = build_input_encoder(
            scheme=self.input_encoding,
            encode_indices=self.encode_indices,
            include_raw=self.include_raw,
            encode_config=self.encode_config,
        )

        self.network = make_mlp(
            None,
            [self.emb_hidden] * self.nb_layer,
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

        self.src_decoder = make_mlp(
            self.emb_hidden,
            [self.emb_dim],
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

        self.tgt_decoder = make_mlp(
            self.emb_hidden,
            [self.emb_dim],
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

    def forward(self, x):
        x_encoded = self.input_encoder(x)
        hidden = self.network(x_encoded)
        return (
            F.normalize(self.src_decoder(hidden)),
            F.normalize(self.tgt_decoder(hidden)),
        )
        
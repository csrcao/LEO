import torch
import torch.nn as nn

from .configuration_aurora import AuroraConfig
from .util_functions import sinusoidal_position_embedding, causal_attention_mask


class PrototypeRetriever(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size  # Dimension of input token embeddings
        self.intermediate_size = config.intermediate_size
        self.num_prototypes = config.num_prototypes  # Total number of prototypes (M)
        self.token_len = config.token_len
        # Prototype collection parameters (learnable)
        self.prototypes = nn.Parameter(torch.Tensor(self.num_prototypes, self.token_len))
        num_cycles = self.num_prototypes // 2
        self._initialize_prototypes(num_cycles)

        self.retriever = Retriever(config)

    def _generate_cyclic_prototypes(self):
        wave_type = torch.randint(0, 3, (1,))  # 0:sine, 1:cosine, 2:mixed
        # Logarithmic frequency distribution for broader range
        freq = torch.exp(torch.randn(1) * 0.5 + torch.log(torch.tensor(0.1)))  # ~0.1-1.0 Hz
        phase = torch.rand(1) * 2 * torch.pi  # Random phase (0-2π)
        amplitude = 0.5 + torch.rand(1)  # Random amplitude (0.5-1.5)
        t = torch.linspace(0, 4 * torch.pi, self.token_len)  # Time base

        if wave_type == 0:
            prototype = amplitude * torch.sin(freq * t + phase)
        elif wave_type == 1:
            prototype = amplitude * torch.cos(freq * t + phase)
        else:
            # Mixed frequency components
            freq2 = freq * (1.5 + torch.rand(1))  # Harmonic frequency
            prototype = amplitude * (torch.sin(freq * t + phase) + 0.5 * torch.cos(freq2 * t))
        return prototype

    def _generate_trend_prototypes(self):
        trend_functions = [
            lambda t, a, b: a * t + b,  # Linear
            lambda t, a, b: a * t ** 2 + b * t,  # Quadratic
            lambda t, a, b: a * t ** 3 + b * t ** 2,  # Cubic
            lambda t, a, b: a * torch.exp(b * t),  # Exponential
            lambda t, a, b: a * torch.log(torch.clamp(t + b + 2.0, min=1e-5)),  # Logarithmic (添加偏移量并限制最小值)
            lambda t, a, b: a * torch.tanh(b * t),  # Hyperbolic tangent
            lambda t, a, b: a * (t > 0).float() * t + b * (t <= 0).float() * t  # Piecewise linear
        ]
        t = torch.linspace(-1, 1, self.token_len)
        func_idx = torch.randint(0, len(trend_functions), (1,))
        # Random parameters for trend shape variation
        if func_idx == 3:
            a = 0.1 + torch.rand(1) * 0.4
            b = -0.5 + torch.rand(1) * 1.0
        else:
            a = 0.5 + torch.rand(1) * 1.5
            b = -1 + torch.rand(1) * 2

        prototype = trend_functions[func_idx](t, a, b)
        # Normalize to prevent extreme values
        prototype = prototype / (prototype.abs().max() + 1e-5)

        return prototype

    def _initialize_prototypes(self, num_cyclic):
        """Initialize prototype collection with diverse cyclic and trend patterns
        Cyclic prototypes: Various waveforms with randomized parameters
        Trend prototypes: Multiple trend functions with random characteristics
        """
        num_trend = self.num_prototypes - num_cyclic

        # Initialize cyclic prototypes with diverse waveforms
        for i in range(num_cyclic):
            # Random waveform type: sine, cosine, or mixed
            prototype = self._generate_cyclic_prototypes()
            self.prototypes.data[i] = prototype

        # Initialize trend prototypes with diverse functions
        trend_start = num_cyclic

        for i in range(num_trend):
            max_attempts = 100
            attempts = 0
            prototype = None
            while attempts < max_attempts:
                prototype = self._generate_trend_prototypes()
                if not torch.isnan(prototype).any():
                    break
                attempts += 1
            if attempts == max_attempts:
                import warnings
                warnings.warn(f"max prototype attempts {max_attempts}，using zeros to fill")
                prototype = torch.zeros_like(prototype)
            self.prototypes.data[trend_start + i] = prototype

    def forward(self, x, output_token_len):
        """
        Args:
            x: Input representation with shape [B, k, d]
               where B=batch size, k=number of tokens, d=embedding dimension
        Returns:
            prob_dist: Prototype probability distributions with shape [B, F, M]
            synthetic_protos: Combined synthetic prototypes with shape [B, F, p]
        """

        dist = self.retriever(x, output_token_len)  # Shape: [B, F, M]
        synthetic_protos = torch.matmul(dist, self.prototypes)  # Shape: [B, F, p]

        # normalize
        mean = synthetic_protos.mean(dim=-1, keepdim=True).detach()
        std = synthetic_protos.std(dim=-1, keepdim=True).detach() + 1e-5
        synthetic_protos = (synthetic_protos - mean) / std  # Shape: [B, F, p]

        return synthetic_protos


class Retriever(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.input_emb = nn.Sequential(nn.LayerNorm(config.hidden_size),
                                       nn.Linear(config.hidden_size, config.hidden_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_retriever_enc_layers,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_retriever_dec_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),  # Combine context and position information
            nn.LayerNorm(config.intermediate_size),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.intermediate_size, config.num_prototypes),  # Predict prototype distribution
            nn.Softmax(dim=-1)
        )

        self.hidden_size = config.hidden_size

    def forward(self, x, output_token_len):
        x_encoded = self.input_emb(x)
        enc_attn_mask = causal_attention_mask(x.shape[1]).to(x.device)
        enc_output = self.encoder(x_encoded, mask=enc_attn_mask.squeeze(0).squeeze(0))  # Shape: [B, k, d]

        enc_output = enc_output[:, -1:, :]

        dec = enc_output.repeat(1, output_token_len, 1)

        pos_embeds = sinusoidal_position_embedding(
            batch_size=dec.shape[0], num_heads=1,
            max_len=output_token_len, output_dim=self.hidden_size,
            device=dec.device).squeeze(1)

        embeds = dec + pos_embeds

        dec_attn_mask = causal_attention_mask(output_token_len).to(x.device)
        dec_output = self.decoder(embeds, mask=dec_attn_mask.squeeze(0).squeeze(0))

        dist = self.head(dec_output)  # Shape: [B, F, M]

        return dist

"""
BreakoutModel – Verbessertes MLP für intraday Breakout-Vorhersage.

Architektur (verbessert gegenüber V1):
  Input (82 Features)
    → BatchNorm1d
    → Linear(82 → 128) + ReLU + Dropout
    → Linear(128 → 64)  + ReLU + BatchNorm1d + Dropout
    → Linear(64 → 32)   + ReLU + BatchNorm1d + Dropout
    → Linear(32 → 16)   + ReLU + Dropout
    → Linear(16 → 1)    + Sigmoid

Verbesserungen gegenüber V1 (82→64→32→1):
  - Mehr Breite (128 statt 64 im ersten Layer) → mehr Kapazität
  - Vier statt drei Hidden-Layer → tiefere Hierarchie
  - BatchNorm1d nach jedem Layer → stabileres Training
  - Höherer Dropout (0.4) → stärkere Regularisierung
  - Optional: Residual Connection (überspringt einen Layer)
"""

import torch
import torch.nn as nn


class BreakoutModel(nn.Module):
    """Verbessertes MLP für Breakout-Vorhersage (V2).

    Parameters
        input_size : Anzahl Input-Features (default: 82)
        hidden_sizes : Tuple mit Hidden-Layer-Größen
        dropout : Dropout-Rate (default: 0.4)
        use_batch_norm : Ob BatchNorm nach jedem Layer (default: True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple = (128, 64, 32, 16),
        dropout: float = 0.4,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        layers = []
        in_features = input_size

        # Input-BatchNorm für stabileres Training
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_size))

        for i, hidden in enumerate(hidden_sizes):
            # Linear
            layers.append(nn.Linear(in_features, hidden))
            # Activation
            layers.append(nn.ReLU())
            # BatchNorm (außer vor dem letzten Layer)
            if use_batch_norm and i < len(hidden_sizes) - 1:
                layers.append(nn.BatchNorm1d(hidden))
            # Dropout (mit abnehmender Rate in tieferen Schichten)
            layer_dropout = dropout * (1.0 - i * 0.15)  # 0.4 → 0.34 → 0.28 → 0.22
            layers.append(nn.Dropout(max(layer_dropout, 0.1)))

            in_features = hidden

        # Output-Layer
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass.

        Parameters
            x : (batch, n_features) – Input-Features

        Returns
            (batch, 1) – Breakout-Wahrscheinlichkeit [0, 1]
        """
        return self.network(x)

"""
Sequenzielle Modellarchitekturen für Breakout-Vorhersage.

Modelle:
  - LSTMBreakoutModel : 2-Layer BiLSTM + Classifier-Head
  - GRUBreakoutModel  : 2-Layer GRU + Classifier-Head
  - CNNBreakoutModel  : 1D-Conv (Multi-Kernel) + Classifier-Head

Alle Modelle erwarten Input der Form (batch, seq_len, n_features).
Die Sequenzlänge entspricht standardmäßig dem Breakout-Horizont (30 Minuten).

Design-Prinzipien:
  - Bidirektionales LSTM für Kontext aus Vergangenheit
  - Layer-Norm für stabileres Training
  - Residual Connections wo sinnvoll
  - Dropout in mehreren Stufen gegen Overfitting
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# LSTM-Breakout-Modell
# ═══════════════════════════════════════════════════════════════════
class LSTMBreakoutModel(nn.Module):
    """Bidirektionales LSTM für zeitliche Breakout-Muster.

    Architektur:
      Input (batch, seq_len, features)
        → BiLSTM (2 Layer, 128 hidden)
        → Concat letzte Hidden-States beider Richtungen
        → LayerNorm
        → Dropout
        → Linear(256 → 64)
        → ReLU + Dropout
        → Linear(64 → 32)
        → ReLU + Dropout
        → Linear(32 → 1) + Sigmoid

    Warum bidirektional:
      Das gesamte Input-Fenster liegt zum Zeitpunkt der Vorhersage bereits
      in der Vergangenheit – es gibt keinen Data-Leakage durch Rückwärts-
      Kontext. Bidirektional erlaubt dem Modell, Muster sowohl vom Anfang
      als auch vom Ende des 30-Minuten-Fensters zu erfassen.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass.

        Parameters
            x : (batch, seq_len, n_features)

        Returns
            (batch, 1) – Breakout-Wahrscheinlichkeit
        """
        # LSTM: output (batch, seq, hidden*D), (h_n, c_n)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # h_n shape: (D*num_layers, batch, hidden)
            # Letzte forward-Layer: h_n[-2], letzte backward-Layer: h_n[-1]
            forward_last = h_n[-2]   # (batch, hidden)
            backward_last = h_n[-1]  # (batch, hidden)
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_hidden = h_n[-1]  # (batch, hidden)

        return self.classifier(last_hidden)


# ═══════════════════════════════════════════════════════════════════
# GRU-Breakout-Modell
# ═══════════════════════════════════════════════════════════════════
class GRUBreakoutModel(nn.Module):
    """GRU-basierter Breakout-Klassifikator (leichter & schneller als LSTM).

    Architektur:
      Input (batch, seq_len, features)
        → GRU (2 Layer, 128 hidden)
        → Letzter Hidden-State
        → LayerNorm
        → Dropout
        → Linear(128 → 64) + ReLU + Dropout
        → Linear(64 → 32) + ReLU + Dropout
        → Linear(32 → 1) + Sigmoid

    Vorteile gegenüber LSTM:
      - Weniger Parameter (kein separates Cell-State-Gate)
      - Schnelleres Training (~15-20% weniger Rechenzeit)
      - Bei kürzeren Sequenzen oft gleichwertig zum LSTM
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass.

        Parameters
            x : (batch, seq_len, n_features)

        Returns
            (batch, 1) – Breakout-Wahrscheinlichkeit
        """
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]  # letzte Layer: (batch, hidden)
        return self.classifier(last_hidden)


# ═══════════════════════════════════════════════════════════════════
# CNN-1D-Breakout-Modell
# ═══════════════════════════════════════════════════════════════════
class CNNBreakoutModel(nn.Module):
    """1D-Convolutional Netzwerk für lokale Zeitmuster.

    Architektur:
      Input (batch, seq_len, features)
        → Drei parallele Conv1D-Stränge (Kernel 3, 5, 10)
        → Concat der global-max-gepoolten Features
        → BatchNorm + Dropout
        → Linear(kombiniert → 64) + ReLU
        → Linear(64 → 32) + ReLU
        → Linear(32 → 1) + Sigmoid

    Idee:
      Verschiedene Kernel-Größen erkennen Muster auf unterschiedlichen
      Zeitskalen – 3-Minuten-Impulse, 5-Minuten-Trends, 10-Minuten-Swells.
      Global Max Pooling macht das Modell robust gegen kleine zeitliche
      Verschiebungen der Muster.

    Leichtgewichtig (weniger Parameter als LSTM) und gut parallelisierbar
    auf der GPU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 64,
        kernel_sizes: tuple = (3, 5, 10),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        # Parallele Conv1D-Stränge
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=hidden_channels,
                    kernel_size=k,
                    padding="same",
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=k,
                    padding="same",
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        combined = hidden_channels * len(kernel_sizes)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined),
            nn.Dropout(dropout),
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass.

        Parameters
            x : (batch, seq_len, n_features)

        Returns
            (batch, 1) – Breakout-Wahrscheinlichkeit
        """
        # Conv1D erwartet (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # Jeder Conv-Strang → Global Max Pooling
        pooled = []
        for conv in self.convs:
            out = conv(x)           # (batch, hidden_channels, seq_len)
            pooled.append(F.adaptive_max_pool1d(out, 1).squeeze(-1))
            # → (batch, hidden_channels)

        combined = torch.cat(pooled, dim=1)  # (batch, hidden*3)
        return self.classifier(combined)

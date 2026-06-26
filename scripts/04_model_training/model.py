import torch
import torch.nn as nn


class BreakoutModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Drei Schichten: immer kleiner werdend
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),  # Eingabe -> 64 Neuronen
            nn.ReLU(),  # Aktivierungsfunktion
            nn.Dropout(0.3),  # 30% der Neuronen zufällig abschalten (gegen Overfitting)

            nn.Linear(64, 32),  # 64 -> 32 Neuronen
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1),  # 32 -> 1 Ausgabe
            nn.Sigmoid()  # Ausgabe zwischen 0 und 1 (Wahrscheinlichkeit)
        )

    def forward(self, x):
        return self.network(x)
import torch
import torch.nn as nn

class CNN_TransformerModel(nn.Module):
    def __init__(self):
        super(CNN_TransformerModel, self).__init__()

        # CNN module
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=3328, out_channels=2048, kernel_size=3, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.3, batch_first=True)

        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.3)

        self.ffn = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1024)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)     # Convert input dimensions to [batch, channels, sequence_length]
        x = self.conv_layers(x)
        x = x.transpose(1, 2)     # turn back [batch, sequence_length, channels]

        x, attention = self.attention(x, x, x)
        x = self.dropout(self.norm(x + x))
        inter = x

        x = self.dropout(self.norm(self.ffn(x) + x))
        return self.class_head(x), inter


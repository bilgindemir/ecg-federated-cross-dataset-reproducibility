from torch import nn

class CNN(nn.Module):
    def __init__(self, architecture="manuscript_bn"):
        super().__init__()
        if architecture == "manuscript_bn":
            # Explicit reconstruction of Table 4, not a recovered old model.
            layers = []; incoming = 12
            for outgoing, kernel in [(32, 7), (64, 5), (128, 3)]:
                layers += [nn.Conv1d(incoming, outgoing, kernel, padding=kernel//2), nn.BatchNorm1d(outgoing), nn.ReLU(), nn.MaxPool1d(2)]
                incoming = outgoing
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1)); self.head = nn.Linear(128, 1)
        else: raise ValueError("Unknown architecture")

    def forward(self, x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)

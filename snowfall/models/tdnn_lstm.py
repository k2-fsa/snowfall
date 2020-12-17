from torch import Tensor
from torch import nn

from snowfall.models import AcousticModel


class TdnnLstm1a(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(self, num_features: int, num_classes: int, subsampling_factor: int = 3) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=self.subsampling_factor,  # <---- stride=3: subsampling_factor!
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
        )
        self.lstm = nn.LSTM(500, 500)
        self.dropout = nn.Dropout(0.5)
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(in_channels=500,
                      out_channels=2000,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=2000, affine=False),
            nn.Conv1d(in_channels=2000,
                      out_channels=num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x, _ = self.lstm(x.permute(2, 0, 1))  # (B, F, T) -> (T, B, F)
        x = x.permute(1, 2, 0)  # (T, B, F) -> (B, F, T)
        x = self.dropout(x)
        x = self.tdnn2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

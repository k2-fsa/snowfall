from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from snowfall.models import AcousticModel
from snowfall.training.diagnostics import measure_weight_norms


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


class TdnnLstm1b(AcousticModel):
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
                      stride=self.subsampling_factor,  # <---- stride: subsampling_factor!
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
        )
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=500, hidden_size=500, num_layers=1)
            for _ in range(5)
        ])
        self.lstm_bnorms = nn.ModuleList([
            nn.BatchNorm1d(num_features=500, affine=False)
            for _ in range(5)
        ])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=500, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(2, 0, 1)  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(1, 0)  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(
            self,
            tb_writer: SummaryWriter,
            global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            'train/weight_l2_norms',
            measure_weight_norms(self, norm='l2'),
            global_step=global_step
        )
        tb_writer.add_scalars(
            'train/weight_max_norms',
            measure_weight_norms(self, norm='linf'),
            global_step=global_step
        )


class PredictOrderingModule(nn.Module):
    """
     This module tries to predict the ordering of some embeddings taken some
     timesteps in the future.  It takes in 2 inputs:
         predictor: (B, T, num_channels_predictor)
         to_order:  (B, T, num_channels_in)

     and returns a value of shape (B, num_frames) which consists of log-probabilities;
     this can be summed treated as the negative of a loss function.

     What it does is: for each time t in `predictor`, it takes the input `to_order` at times:

        t + delay,
        t + delay + skip,
        t + delay + 2*skip,
         ...
        t + delay + 2*skip*(num_frames - 1)

     and without knowing which time step each of those inputs was from, it tries to predict
     the (relative) time step, so essentially it is trying to predict the ordering
     among the `num_frames` different future inputs.  Of course this only makes sense
     if 'in' contains only finite past/future context, and 'pred' only contains
     past but not future context; otherwise the task becomes too trivial.

    """
    def __init__(self, num_channels_predictor: int , num_channels_to_order: int,
                 delay: int, num_frames_to_order: int, skip: int,
                 hidden_size: int = 256, num_hidden_layers: int = 0):
        """
        num_channels_predictor: The number of channels of the input used to predict the
             ordering of the other input.  (This input feature may have infinite left context,
             like an LSTM, but must have only finite right context, e.g. no greater than
             'delay').
        num_channels_to_order:  The number of channels of the input whose ordering we
             predict.
        delay:  The delay between the frame of `predictor` that we predict from, and the
                first of several frames of `to_order` that we predict the ordering of.
       num_frames_to_order: The number of different frames whose ordering by position we
                have to predict; must be >1.
        skip:  The interval (in t steps) between different frames whose positions
                we are trying to predict.
        hidden_size:  If num_hidden_layers > 0,  the model, in addition to a trilinear component, will
                include a feedforward network whose input is pairs of (pred, in) vectors;
                this is the size of the hidden dimension.
        num_hidden_layers:  Number of hidden layers in feedforward component of this
                prediction
        """
        super(PredictOrderingModule, self).__init__()
        self.num_channels_predictor = num_channels_predictor
        self.num_channels_to_order = num_channels_to_order
        self.delay = delay
        self.num_frames_to_order = num_frames_to_order
        assert num_frames_to_order > 1
        self.skip = skip
        self.tot_latency = delay + skip * (num_frames_to_order - 1)
        self.hidden_size = hidden_size

        self.trilinear = nn.Parameter(torch.zeros(num_frames_to_order,
                                                  num_channels_to_order,
                                                  num_channels_predictor))
        layers = []
        if num_hidden_layers > 0:
            # We user a mini-network to predict this as well..
            self.from_o = nn.Conv1d(in_channels=num_channels_to_order,
                                    out_channels=hidden_size, kernel_size=1)
            self.from_p = nn.Conv1d(in_channels=num_channels_predictor,
                                    out_channels=hidden_size, kernel_size=1)
            layers = []
            for i in range(num_hidden_layers):
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(hidden_size))
                next_size = hidden_size if i + 1 < num_hidden_layers else num_frames_to_order
                layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=next_size,
                                        kernel_size=1))
            self.layers = nn.Sequential(*layers)
        else:
            self.register_parameter('from_o', None)
            self.register_parameter('from_p', None)

    def forward(self, predictor: Tensor, to_order: Tensor) -> Tensor:
        """
        Forward function.
        Args:
            predictor:   of shape (B, num_channels_predictor, T), the input that is used
                        to predict the ordering of the frames
             to_order:  of shape (B, num_channels_to_order, T), the input whose future
                        frames we need to predict the ordering of
        Return:
             Returns the log-likelihoods of shape (num_frames_to_order,) of the average
             log-likelihoods of the correct orderings; you can treat the negative
             mean of this, possibly weighted by the number of frames if you desire,
             as a portion of the loss function.
        """
        (B, np, T) = predictor.shape
        (B_, no, T_) = to_order.shape
        nf = self.num_frames_to_order
        assert B == B_ and T == T_
        assert np == self.num_channels_predictor
        assert no == self.num_channels_to_order


        tot_latency = self.delay + (self.skip * (nf - 1))
        T_reduced = T - tot_latency
        if T_reduced <= 0:
            return torch.zeros(nf, device=predictor.device)
        # Reduce `predictor` and `to_order` to just the frames that we need.
        predictor = predictor[:,:,0:T_reduced]
        to_order = to_order[:,:,self.delay:]


        # (B, nf, T_reduced, no) = matmul((B, 1, T_reduced, np), (1, nf, np, no))
        predictor_product = torch.matmul(predictor.transpose(1, 2).unsqueeze(1),
                                       self.trilinear.transpose(1, 2).unsqueeze(0))
        assert predictor_product.shape == (B, nf, T_reduced, no)

        (batch_stride, channel_stride, t_stride) = to_order.stride()
        to_order_view = to_order.as_strided((B, nf, T_reduced, no),
                                            (batch_stride, t_stride * self.skip, t_stride, channel_stride))

        # (B, nf, nf, T_reduced) =
        #               torch.inner((B, nf, 1, T_reduced, no),
        #                           (B, 1, nf, T_reduced, no))
        # The 1st `nf` in the shape corresponds to the actual order, the
        # 2nd corresponds to the predicted order.
        # Version 1.7.2 doesn't support inner..
        #inner_prods = torch.inner(to_order_view.unsqueeze(2),
        #                          predictor_product.unsqueeze(1))


        # (B, nf_ref, T_reduced, no), (B, nf_hyp, T_reduced, no) -> (B, nf_ref, jf_hyp, T_reduced)
        inner_prods = torch.einsum('ijlm,iklm->ijkl', to_order_view, predictor_product)


        if self.from_o is not None:
            hs = self.hidden_size
            to_order_proj = self.from_o(to_order)
            predictor_proj = self.from_p(predictor)
            predictor_proj = predictor_proj.unsqueeze(1)
            assert predictor_proj.shape == (B, 1, hs, T_reduced)

            (batch_stride, channel_stride, time_stride) = to_order_proj.stride()
            to_order_proj = to_order_proj.as_strided((B, nf, hs, T_reduced),
                                                     (batch_stride, time_stride * self.skip,
                                                      channel_stride, time_stride))
            # (B, nf, hs, T_reduced)
            tot_proj = (to_order_proj + predictor_proj).reshape(B * nf, hs, T_reduced)
            # (B, nf, nf, T_reduced)
            output = self.layers(tot_proj)
            inner_prods += output.reshape(B, nf, nf, T_reduced)

        # after the log_softmax, the log-probs are now normalized to sum to one.
        # This is of shape: (B, nf, nf, T_reduced), where the 1st nf corresponds
        # to the real order and the 2nd nf corresponds to the predicted order.
        # We care about the diagonal of the (nf, nf) part.
        log_probs = torch.nn.functional.log_softmax(inner_prods, dim=2)
        assert log_probs.shape == (B, nf, nf, T_reduced)
        (stride_b, frame_stride1, frame_stride2, t_stride) = log_probs.stride()

        correct_logprobs = log_probs.as_strided((B, nf, T_reduced),
                                                (stride_b, frame_stride1 + frame_stride2, t_stride))
        # ans is of shape (nf,); it's the mean logprob, per predicted position
        # (presumably the earlier positions will have a higher, i.e. closer-to-zero
        # log-prob, as they are easier.
        ans = correct_logprobs.mean(2).mean(0)
        return ans



class TdnnLstm1c(AcousticModel):
    """
    TdnnLstm1c augments the TDNN-LSTM with auxiliary objective that tries to predict
    the ordering of some future segments.

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
                      stride=self.subsampling_factor,  # <---- stride: subsampling_factor!
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
        )

        self.predictor = PredictOrderingModule(num_channels_predictor=500,
                                               num_channels_to_order=500,
                                               delay=2, skip=3,
                                               num_frames_to_order=8,
                                               num_hidden_layers=2,
                                               hidden_size=196)

        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=500, hidden_size=500, num_layers=1)
            for _ in range(5)
        ])
        self.lstm_bnorms = nn.ModuleList([
            nn.BatchNorm1d(num_features=500, affine=False)
            for _ in range(5)
        ])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=500, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            (result, aux_likes)
            where `result` is the CTC output, of shape (batch_size, number_of_classes, input_length);
            and `aux_likes` is a tensor of shape (8,)==(num_frames_to_order,)
             representing average log-likes of the ordering
            of predicted frames, that can be negated and scaled and used in the loss
            function.


        """
        x = tdnn = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(2, 0, 1)  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections

        # x is currently (T, B, F), permute(1, 2, 0) gives (B, F, T)
        aux = self.predictor(x.permute(1, 2, 0), tdnn)

        x = x.transpose(1, 0)  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall

        x = nn.functional.log_softmax(x, dim=1)
        return x, aux

    def write_tensorboard_diagnostics(
            self,
            tb_writer: SummaryWriter,
            global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            'train/weight_l2_norms',
            measure_weight_norms(self, norm='l2'),
            global_step=global_step
        )
        tb_writer.add_scalars(
            'train/weight_max_norms',
            measure_weight_norms(self, norm='linf'),
            global_step=global_step
        )





def test_predict_ordering():
    num_channels_predictor = 128
    num_channels_to_order = 256
    delay = 10
    num_frames_to_order = 6
    skip = 4
    hidden_size = 256
    num_hidden = 1

    m = PredictOrderingModule(num_channels_predictor, num_channels_to_order,
                              delay, num_frames_to_order, skip,
                              hidden_size, num_hidden)

    B = 10
    T = 150
    predictor = torch.randn(B, num_channels_predictor, T)
    to_order = torch.randn(B, num_channels_to_order, T)
    p = m(predictor, to_order)
    print("p = ", p)

def test_tdnn_lstm_1c():
    m = TdnnLstm1c(80, 1000, 3)
    B = 5
    num_features = 80
    input_length = 500
    x = torch.randn(B, num_features, input_length)
    y, aux = m(x)
    print("aux = ", aux)


def main():
    test_predict_ordering()
    test_tdnn_lstm_1c()


if __name__ == '__main__':
    main()

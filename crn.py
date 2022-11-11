import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, channels):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x2, x4):
        residual = x
        x = x.permute(0,1,3,2)
        x2 = torch.cat((x2[:, :, :, :-2], x2[:, :, :, 1:-1], x2[:, :, :, 2:]), 2)
        x2 = x2[:, :, :, ::2]
        x2 = x2.permute(0,1,3,2)

        x4 = torch.cat((x4[:, :, :, :-6], \
                        x4[:, :, :, 1:-5], \
                        x4[:, :, :, 2:-4], \
                        x4[:, :, :, 3:-3], \
                        x4[:, :, :, 4:-2], \
                        x4[:, :, :, 5:-1], \
                        x4[:, :, :, 6:]), 2)
        x4 = x4[:, :, :, ::4]
        x4 = x4.permute(0,1,3,2)
        
        x = self.linear(torch.cat((x, x2, x4), 3))
        x = x.permute(0,1,3,2)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        x = self.activation(x)
        return x

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16) # 257 -> 128
        self.conv_block_1_resolution2 = CausalConvBlock(1, 16) # 129 -> 64
        self.conv_block_1_resolution4 = CausalConvBlock(1, 16) # 65 -> 32
        self.linear_block_1 = LinearBlock(128 + 64*3 + 32*7, 128, 16)

        self.conv_block_2 = CausalConvBlock(16, 32) # 128 -> 63
        self.conv_block_2_resolution2 = CausalConvBlock(16, 32) # 64 -> 31
        self.conv_block_2_resolution4 = CausalConvBlock(16, 32) # 32 -> 15
        self.linear_block_2 = LinearBlock(63 + 31*3 + 15*7, 63, 32)

        self.conv_block_3 = CausalConvBlock(32, 64) # 63 -> 31
        self.conv_block_3_resolution2 = CausalConvBlock(32, 64) # 31 -> 15
        self.conv_block_3_resolution4 = CausalConvBlock(32, 64) # 15 -> 7
        self.linear_block_3 = LinearBlock(31 + 15*3 + 7*7, 31, 64)

        self.conv_block_4 = CausalConvBlock(64, 128) # 31 -> 15
        self.conv_block_4_resolution2 = CausalConvBlock(64, 128) # 15 -> 7
        self.conv_block_4_resolution4 = CausalConvBlock(64, 128) # 7 -> 3
        self.linear_block_4 = LinearBlock(15 + 7*3 + 3*7, 15, 128)

        self.conv_block_5 = CausalConvBlock(128, 256) # 15 -> 7
        self.conv_block_5_resolution2 = CausalConvBlock(128, 256) # 7 -> 3
        self.conv_block_5_resolution4 = CausalConvBlock(128, 256) # 3 -> 1
        self.linear_block_5 = LinearBlock(7 + 3*3 + 1*7, 7, 256)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=1792, hidden_size=1792, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x, x2, x4):

        self.lstm_layer.flatten_parameters()

        device = x2.device
        batch_x2, channelgs_x2, bins_x2, frames_x2 = x2.shape
        x2 = torch.cat((x2, torch.zeros(batch_x2, channelgs_x2, bins_x2, 2).to(device)), 3)
        batch_x4, channelgs_x4, bins_x4, frames_x4 = x4.shape
        x4 = torch.cat((x4, torch.zeros(batch_x4, channelgs_x4, bins_x4, 6).to(device)), 3)

        e_1 = self.conv_block_1(x)
        e_1_resolution2 = self.conv_block_1_resolution2(x2)
        e_1_resolution4 = self.conv_block_1_resolution4(x4)
        e_1 = self.linear_block_1(e_1, e_1_resolution2, e_1_resolution4)

        e_2 = self.conv_block_2(e_1)
        e_2_resolution2 = self.conv_block_2_resolution2(e_1_resolution2)
        e_2_resolution4 = self.conv_block_2_resolution4(e_1_resolution4)
        e_2 = self.linear_block_2(e_2, e_2_resolution2, e_2_resolution4)

        e_3 = self.conv_block_3(e_2)
        e_3_resolution2 = self.conv_block_3_resolution2(e_2_resolution2)
        e_3_resolution4 = self.conv_block_3_resolution4(e_2_resolution4)
        e_3 = self.linear_block_3(e_3, e_3_resolution2, e_3_resolution4)

        e_4 = self.conv_block_4(e_3)
        e_4_resolution2 = self.conv_block_4_resolution2(e_3_resolution2)
        e_4_resolution4 = self.conv_block_4_resolution4(e_3_resolution4)
        e_4 = self.linear_block_4(e_4, e_4_resolution2, e_4_resolution4)

        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]
        e_5_resolution2 = self.conv_block_5_resolution2(e_4_resolution2)
        e_5_resolution4 = self.conv_block_5_resolution4(e_4_resolution4)
        e_5 = self.linear_block_5(e_5, e_5_resolution2, e_5_resolution4)

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        return d_5 * x


if __name__ == '__main__':
    layer = CRN()
    a = torch.rand(2, 1, 161, 200)
    print(layer(a).shape)

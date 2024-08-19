import jax.numpy as jnp
import equinox as eqx
from modules import ConvSC, Inception


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(eqx.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = eqx.nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(eqx.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = eqx.nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = eqx.nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        # Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.dec[-1](jnp.concatenate([hid, enc1], axis=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(eqx.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [
            Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
        ]
        for i in range(1, N_T - 1):
            enc_layers.append(
                Inception(
                    channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
                )
            )
        enc_layers.append(
            Inception(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        )

        dec_layers = [
            Inception(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        ]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            Inception(
                2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            # z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
            z = self.dec[i](jnp.concatenate([z, skips[-i]], axis=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(eqx.Module):
    def __init__(
        self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8
    ):
        super(SimVP, self).__init__()
        num_timesteps, num_channels, Height, Width = shape_in
        self.enc = Encoder(num_channels, hid_S, N_S)
        self.hid = Mid_Xnet(num_timesteps * hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, num_channels, N_S)

    def forward(self, x_raw):
        B, num_timesteps, num_channels, Height, W = x_raw.shape
        x = x_raw.view(B * num_timesteps, num_channels, Height, Width)

        embed, skip = self.enc(x)
        _, num_channels_, Height_, Width_ = embed.shape

        z = embed.view(B, num_timesteps, num_channels_, Height_, Width_)
        hid = self.hid(z)
        hid = hid.reshape(B * num_timesteps, num_channels_, Height_, Width_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, num_timesteps, num_channels, Height, Width)
        return Y
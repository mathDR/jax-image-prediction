import equinox as eqx
from equinox import nn
import jax
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array
from .modules import ConvSC, gInception_ST
from typing import Tuple


def sampling_generator(N: int, reverse: bool = False) -> list:
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(eqx.Module):
    """3D Encoder for SimVP"""

    enc: list

    def __init__(
        self,
        key: jax.random.PRNGKey,
        C_in: int,
        C_hid: int,
        N_S: int,
        spatio_kernel: int,
    ) -> None:
        super(Encoder, self).__init__()
        samplings = sampling_generator(N_S)
        keys = jax.random.split(key, N_S)
        self.enc = eqx.nn.Sequential(
            [
                ConvSC(
                    keys[0],
                    C_in,
                    C_hid,
                    spatio_kernel,
                    downsampling=samplings[0],
                ),
                *[
                    ConvSC(
                        k,
                        C_hid,
                        C_hid,
                        spatio_kernel,
                        downsampling=s,
                    )
                    for k, s in zip(keys[1:], samplings[1:])
                ],
            ]
        )

    def __call__(self, x: Array) -> Array:
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(eqx.Module):
    """3D Decoder for SimVP"""

    readout: eqx.nn.Conv2d
    dec: list

    def __init__(
        self,
        key: jax.random.PRNGKey,
        C_hid: int,
        C_out: int,
        N_S: int,
        spatio_kernel: int,
    ) -> None:
        super(Decoder, self).__init__()
        samplings = sampling_generator(N_S, reverse=True)
        keys = jax.random.split(key, N_S + 1)
        self.dec = eqx.nn.Sequential(
            [
                *[
                    ConvSC(
                        k,
                        C_hid,
                        C_hid,
                        spatio_kernel,
                        upsampling=s,
                    )
                    for k, s in zip(keys[:-2], samplings[:-1])
                ],
                ConvSC(
                    keys[-2],
                    C_hid,
                    C_hid,
                    spatio_kernel,
                    upsampling=samplings[-1],
                ),
            ]
        )

        self.readout = eqx.nn.Conv2d(
            C_hid,
            C_out,
            1,
            key=keys[-1],
        )

    def __call__(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        return self.readout(Y)


class MidIncepNet(eqx.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    N2: int
    enc: eqx.nn.Sequential
    dec: eqx.nn.Sequential

    def __init__(
        self,
        key: jax.random.PRNGKey,
        channel_in: int,
        channel_hid: int,
        N2: int,
        incep_ker: list[int] = [3, 5, 7, 11],
        groups: int = 8,
        **kwargs,
    ):
        super(MidIncepNet, self).__init__()
        assert (
            N2 >= 2 and len(incep_ker) > 1
        ), "Incorrect N2 and incep_ker in MidInceptNet"
        self.N2 = N2
        keys = jax.random.split(key, 2 * N2)
        enc_layers = [
            gInception_ST(
                keys[0],
                channel_in,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(
                    keys[i],
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        enc_layers.append(
            gInception_ST(
                keys[N2 - 1],
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        dec_layers = [
            gInception_ST(
                keys[N2],
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(
                    keys[N2 + i],
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            gInception_ST(
                keys[-1],
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        self.enc = eqx.nn.Sequential(enc_layers)
        self.dec = eqx.nn.Sequential(dec_layers)

    def __call__(self, x: Array) -> Array:
        T, C, H, W = x.shape
        x = x.reshape(T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](jnp.concatenate([z, skips[-i]], axis=0))

        y = z.reshape(T, C, H, W)
        return y


class SimVP_Model(eqx.Module):
    enc: Encoder
    hid: MidIncepNet
    dec: Decoder

    def __init__(
        self,
        key: jax.random.PRNGKey,
        in_shape: Tuple,
        hid_S: int = 16,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        model_type: str = "gSTA",
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        **kwargs,
    ) -> None:
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2 ** (N_S / 2)), int(
            W / 2 ** (N_S / 2)
        )  # downsample 1 / 2**(N_S/2)
        keys = jax.random.split(key, 3)
        self.enc = Encoder(
            keys[0],
            C,
            hid_S,
            N_S,
            spatio_kernel_enc,
        )
        self.dec = Decoder(
            keys[2],
            hid_S,
            C,
            N_S,
            spatio_kernel_dec,
        )
        self.hid = MidIncepNet(keys[1], T * hid_S, hid_T, N_T)

    def __call__(self, x_raw: Array, **kwargs) -> Array:
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B * T, C, H, W)
        embed, skip = vmap(self.enc)(x)

        _, C_, H_, W_ = embed.shape
        z = embed.reshape(B, T, C_, H_, W_)
        hid = vmap(self.hid)(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = vmap(self.dec)(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

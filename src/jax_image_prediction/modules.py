"""Modules for use in SimVP models."""
import equinox as eqx
from einops import rearrange
from functools import partial
import jax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from typing import List, Sequence, Union


# x = jnp.arange(5 * 8 * 8 * 3).reshape(5, 8, 8, 3)
# pixel_shuffle(x) = print(x.shape) # (5, 4, 4, 12)
# pixel_unshuffle(pixel_shuffle(x)) = print(x.shape) # (5, 8, 8, 3)
class PixelShuffle(eqx.Module):
    scale_factor: int
    layer: partial

    def __init__(self, scale_factor: int) -> None:
        self.scale_factor = scale_factor
        self.layer = partial(
            rearrange,
            pattern="... (c b1 b2) h w -> ... c (h b1) (w b2)",
            b1=self.scale_factor,
            b2=self.scale_factor,
        )

    def __call__(self, x: Array, key: jax.random.PRNGKey = None) -> Array:
        return self.layer(x)


class BasicConv2d(eqx.Module):
    act_norm: bool
    conv: list
    norm: eqx.nn.GroupNorm
    act: jax.nn.silu

    def __init__(
        self,
        key: jax.random.PRNGKey,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = 0,
        dilation: Union[str, int, Sequence[int]] = 1,
        upsampling: bool = False,
        act_norm: bool = False,
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = eqx.nn.Sequential(
                [
                    *[
                        eqx.nn.Conv2d(
                            in_channels,
                            out_channels * 4,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            key=key,
                        ),
                        PixelShuffle(2),
                    ]
                ]
            )
        else:
            self.conv = eqx.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                key=key,
            )

        self.norm = eqx.nn.GroupNorm(2, out_channels)
        self.act = jax.nn.silu

    def __call__(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(eqx.Module):
    conv: eqx.nn.Conv

    def __init__(
        self,
        key: jax.random.PRNGKey,
        C_in: int,
        C_out: int,
        kernel_size: int = 3,
        downsampling=False,
        upsampling=False,
        act_norm: bool = True,
    ) -> None:
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(
            key,
            C_in,
            C_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
        )

    def __call__(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(eqx.Module):
    act_norm: bool
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    act: jax.nn.leaky_relu

    def __init__(
        self,
        key: jax.random.PRNGKey,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act_norm: bool = False,
    ):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            key=key,
        )
        self.norm = eqx.nn.GroupNorm(groups, out_channels)
        self.act = jax.nn.leaky_relu

    def __call__(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class gInception_ST(eqx.Module):
    """A IncepU block for SimVP"""

    conv1: eqx.nn.Conv2d
    layers: eqx.nn.Sequential

    def __init__(
        self,
        key: jax.random.PRNGKey,
        C_in: int,
        C_hid: int,
        C_out: int,
        incep_ker: List[int] = [3, 5, 7, 11],
        groups: int = 8,
    ):
        super(gInception_ST, self).__init__()
        keys = jax.random.split(key, 1 + len(incep_ker))
        self.conv1 = eqx.nn.Conv2d(
            C_in, C_hid, kernel_size=1, stride=1, padding=0, key=keys[0]
        )

        layers = []
        for i, ker in enumerate(incep_ker):
            layers.append(
                GroupConv2d(
                    keys[1 + i],
                    C_hid,
                    C_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True,
                )
            )
        self.layers = eqx.nn.Sequential(layers)

    def __call__(self, x: Array) -> Array:
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

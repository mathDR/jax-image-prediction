import matplotlib.pyplot as plt
from jax_image_prediction import (
    SimVP_Model,
    SimVP_Model_No_Hid,
    shift_image,
    compute_loss,
    evaluate,
    train,
)

import math

import equinox as eqx
import jax
from jax import vmap
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import torch
from jaxtyping import Array, Float, Int, PyTree

imgs = np.load("/Users/daniel.marthaler/dev/SimVP/data/moving_mnist/mnist_test_seq.npy")
from dataloader_moving_mnist import load_data

# model parameters
in_shape = [10, 1, 64, 64]
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = "gSTA"
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4

# training
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
TOTAL_STEPS = 1500
TOTAL_EPOCH = 5
PRINT_EVERY = 30
SEED = 42
val_batch_size = 16
num_workers = 8
drop_path = 0
sched = "onecycle"


def main():
    key = jax.random.PRNGKey(SEED)

    train_loader, vali_loader, test_loader, data_mean, data_std = load_data(
        BATCH_SIZE, BATCH_SIZE, "/Users/daniel.marthaler/dev/SimVP/data", num_workers
    )
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.numpy()
    y_batch = y_batch.numpy()

    model = SimVP_Model_No_Hid(
        key,
        tuple(in_shape),
        hid_S,
        hid_T,
        N_S,
        N_T,
        spatio_kernel_enc=spatio_kernel_enc,
        spatio_kernel_dec=spatio_kernel_dec,
    )

    loss = eqx.filter_jit(compute_loss)
    cosine_decay_scheduler = optax.cosine_decay_schedule(
        LEARNING_RATE, decay_steps=TOTAL_STEPS, alpha=0.95
    )
    optim = optax.adamw(learning_rate=cosine_decay_scheduler)

    model, opt_state_saved = train(
        model,
        train_loader,
        test_loader,
        optim,
        loss,
        TOTAL_STEPS,
        PRINT_EVERY,
        first_call=True,
        opt_state_saved=None,
    )

    y_pred = model(x_batch)
    breakpoint()
    # jnp.save(y_pred, "y_pred")
    # jnp.save(x_batch, "x_batch")

    j = 2
    k = 5
    l = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.imshow(y_pred[j, k, l, :, :])
    ax2.imshow(x_batch[j, k, l, :, :])
    plt.show()


if __name__ == "__main__":
    main()

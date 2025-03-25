import equinox as eqx
import optax
import torch
from jaxtyping import Array, Float, PyTree


def compute_loss(model, x, y):
    pred_y = model(x)
    # Trains with respect to huber loss
    # return optax.losses.huber_loss(pred_y, y).sum()
    return optax.losses.l2_loss(pred_y, y).sum()


def evaluate(model, loss, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    for x, y in testloader:
        x = x.numpy().astype(float)
        # y = y.numpy().astype(float)
        # Note that all the JAX operations happen inside `loss` ,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, x)
    return avg_loss / len(testloader)


def train(
    model,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss,
    steps: int,
    print_every: int,
    first_call: bool = True,
    opt_state_saved: PyTree = None,
):
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    if first_call:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
    else:
        opt_state = opt_state_saved

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model,
        opt_state: PyTree,
        x: Float[Array, "batch 10 1 64 64"],
        y: Float[Array, "batch 10 1 64 64"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy().astype(float)
        y = y.numpy().astype(float)
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss = evaluate(model, loss, testloader)
            print(
                f"{step}, train_loss={train_loss.item()}, test_loss={test_loss.item()}"
            )
    return model, opt_state

"""
Nicholas M. Boffi
6/19/25

Example usage of the EDM2 UNet architecture with positional embeddings
"""

import edm2_net
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

if __name__ == "__main__":
    # initialize the model
    model = edm2_net.PrecondUNet(
        img_resolution=32,
        img_channels=3,
        label_dim=10,
        sigma_data=0.5,
        logvar_channels=128,
        use_bfloat16=True,
        unet_kwargs={
            "model_channels": 128,
            "channel_mult": [2, 2, 2],
            "num_blocks": 3,
            "attn_resolutions": [16, 8],
            "use_fourier": False,
            "block_kwargs": {"dropout": 0.13},
        },
    )

    # note the pytorch (NCHW) convention
    prng_key = jax.random.PRNGKey(42)
    ex_input = jax.random.normal(prng_key, (1, 3, 32, 32))
    ex_t = jnp.array([0.0])
    ex_label = jax.nn.one_hot(0, num_classes=10).reshape((1, -1))

    # initialize the model
    params = model.init(
        {"params": prng_key},
        ex_t,
        ex_input,
        ex_label,
        train=False,
        calc_weight=True,
    )
    print(f"Number of parameters: {ravel_pytree(params)[0].size}")

    # note need to project to sphere due to jax functional style
    # also needs to happen after every gradient step in a training loop
    params = edm2_net.project_to_sphere(params)

    # apply to example input
    print(model.apply(params, ex_t, ex_input, ex_label, train=False, calc_weight=True))

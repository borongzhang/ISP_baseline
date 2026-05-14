import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


def build_permutation_indices(L, l):
    """Returns dictionary of permutation indices at level 'l'.
    Ht_l = blkdiag(l) × column_permutation
    """
    # indices inside each 4^(L-l) × 4^(L-l) block
    delta = 4 ** (L - l - 1)

    # [0, Δ, 2Δ, 3Δ, 0, Δ, 2Δ, 3Δ, 0, Δ, 2Δ, 3Δ, … ]
    tmp = np.tile(np.arange(4) * delta, delta)

    # + [0, 0, 0, 0, 1, 1, 1, 1, … , Δ, Δ, Δ, Δ]
    tmp += np.repeat(np.arange(delta), 4)

    # indices for the entire block diagonal matrix
    tmp = np.tile(tmp, 4**l)
    tmp += np.repeat(np.arange(4**l) * (4 ** (L - l)), 4 ** (L - l))

    return jnp.asarray(tmp)


def build_switch_indices(L):
    """Returns permutation indices for patches at the switch layer."""
    tmp = np.arange(2**L) * (2**L)

    tmp = np.tile(tmp, 2**L)
    tmp += np.repeat(np.arange(2**L), 2**L)

    return jnp.asarray(tmp)


class V(nn.Module):
    r: int

    @nn.compact
    def __call__(self, x):
        n, s = x.shape[-2], x.shape[-1]

        init_fn = nn.initializers.glorot_uniform()
        vr1 = self.param("vr1", init_fn, (n, s, self.r))
        vi1 = self.param("vi1", init_fn, (n, s, self.r))
        vr2 = self.param("vr2", init_fn, (n, s, self.r))
        vi2 = self.param("vi2", init_fn, (n, s, self.r))

        x_re, x_im = x[:, 0, :, :], x[:, 1, :, :]
        y_re_1 = jnp.einsum("baj,ajk->bak", x_re, vr1)
        y_re_2 = jnp.einsum("baj,ajk->bak", x_im, vi1)
        y_re = y_re_1 + y_re_2

        y_im_1 = jnp.einsum("baj,ajk->bak", x_re, vi2)
        y_im_2 = jnp.einsum("baj,ajk->bak", x_im, vr2)
        y_im = y_im_1 + y_im_2

        y = jnp.stack([y_re, y_im], axis=1)

        return y


class H(nn.Module):
    L: int
    l: int

    def setup(self):
        self.perm_idx = build_permutation_indices(self.L, self.l)

    @nn.compact
    def __call__(self, x):
        m = x.shape[-2] // 4
        s = x.shape[-1] * 4

        # Define weights
        init_fn = nn.initializers.glorot_uniform()
        hr1 = self.param("hr1", init_fn, (m, s, s))
        hi1 = self.param("hi1", init_fn, (m, s, s))
        hr2 = self.param("hr2", init_fn, (m, s, s))
        hi2 = self.param("hi2", init_fn, (m, s, s))

        # Apply permutations
        x = x.take(self.perm_idx, axis=-2)

        # Reshape operation
        x = x.reshape((-1, 2, m, s))
        # Split real and imaginary parts for processing
        x_re, x_im = x[:, 0, :, :], x[:, 1, :, :]

        y_re_1 = jnp.einsum("baj,ajk->bak", x_re, hr1)
        y_re_2 = jnp.einsum("baj,ajk->bak", x_im, hi1)
        y_re = y_re_1 + y_re_2

        y_im_1 = jnp.einsum("baj,ajk->bak", x_re, hi2)
        y_im_2 = jnp.einsum("baj,ajk->bak", x_im, hr2)
        y_im = y_im_1 + y_im_2

        y = jnp.stack([y_re, y_im], axis=1)

        n = m * 4
        r = s // 4
        y = y.reshape((-1, 2, n, r))

        return y


class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        n, s = x.shape[-2], x.shape[-1]

        init_fn = nn.initializers.glorot_uniform()
        mr1 = self.param("mr1", init_fn, (n, s, s))
        mi1 = self.param("mi1", init_fn, (n, s, s))
        mr2 = self.param("mr2", init_fn, (n, s, s))
        mi2 = self.param("mi2", init_fn, (n, s, s))

        x_re, x_im = x[:, 0, :, :], x[:, 1, :, :]

        y_re_1 = jnp.einsum("baj,ajk->bak", x_re, mr1)
        y_re_2 = jnp.einsum("baj,ajk->bak", x_im, mi1)
        y_re = y_re_1 + y_re_2

        y_im_1 = jnp.einsum("baj,ajk->bak", x_re, mi2)
        y_im_2 = jnp.einsum("baj,ajk->bak", x_im, mr2)
        y_im = y_im_1 + y_im_2

        y = jnp.stack([y_re, y_im], axis=1)

        return y


class G(nn.Module):
    L: int
    l: int

    def setup(self):
        self.perm_idx = build_permutation_indices(self.L, self.l)

    @nn.compact
    def __call__(self, x):
        m = x.shape[-2] // 4
        s = x.shape[-1] * 4
        t = s * (self.L - self.l + 1) // 3

        # Define weights
        init_fn = nn.initializers.glorot_uniform()
        gr1 = self.param("gr1", init_fn, (m, s, t))
        gi1 = self.param("gi1", init_fn, (m, s, t))
        gr2 = self.param("gr2", init_fn, (m, s, t))
        gi2 = self.param("gi2", init_fn, (m, s, t))

        # Reshape operation
        x = x.reshape((-1, 2, m, s))
        # Split real and imaginary parts for processing
        x_re, x_im = x[:, 0, :, :], x[:, 1, :, :]

        y_re_1 = jnp.einsum("baj,ajk->bak", x_re, gr1)
        y_re_2 = jnp.einsum("baj,ajk->bak", x_im, gi1)
        y_re = y_re_1 + y_re_2

        y_im_1 = jnp.einsum("baj,ajk->bak", x_re, gi2)
        y_im_2 = jnp.einsum("baj,ajk->bak", x_im, gr2)
        y_im = y_im_1 + y_im_2

        y = jnp.stack([y_re, y_im], axis=1)

        n = m * 4
        r = t // 4
        y = y.reshape((-1, 2, n, r))

        y = y.take(self.perm_idx, axis=-2)

        return y


class U(nn.Module):
    s: int

    @nn.compact
    def __call__(self, x):
        n, r = x.shape[-2], x.shape[-1]

        init_fn = nn.initializers.glorot_uniform()
        ur1 = self.param("ur1", init_fn, (n, r, self.s))
        ui1 = self.param("ui1", init_fn, (n, r, self.s))
        ur2 = self.param("ur2", init_fn, (n, r, self.s))
        ui2 = self.param("ui2", init_fn, (n, r, self.s))

        x_re, x_im = x[:, 0, :, :], x[:, 1, :, :]

        y_re_1 = jnp.einsum("baj,ajk->bak", x_re, ur1)
        y_re_2 = jnp.einsum("baj,ajk->bak", x_im, ui1)
        y_re = y_re_1 + y_re_2

        y_im_1 = jnp.einsum("baj,ajk->bak", x_re, ui2)
        y_im_2 = jnp.einsum("baj,ajk->bak", x_im, ur2)
        y_im = y_im_1 + y_im_2

        y = jnp.stack([y_re, y_im], axis=1)

        return y


class WideBNetModel(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    NUM_CNN: int
    idx_morton_to_flatten: jnp.ndarray

    def setup(self):
        self.nx = (2**self.L) * self.s
        self.switch_idx = build_switch_indices(self.L)
        self.convs = [
            nn.Conv(features=6, kernel_size=(2, 2), padding="SAME")
            for _ in range(self.NUM_CNN - 1)
        ]
        self.final_conv = nn.Conv(features=1, kernel_size=(2, 2), padding="SAME")

    @nn.compact
    def __call__(self, inputs):
        y = inputs[..., 0]
        y = jnp.reshape(y, (-1, 2, 4**self.L, self.s**2))
        y = V(self.r)(y)

        for l in range(self.L - 1, int(self.L / 2) - 1, -1):
            d = self.L - l  # layer depth in butterfly
            y_l = inputs[..., d]
            y_l = jnp.reshape(y_l, (-1, 2, 4**l, 4**d * self.s**2))
            y_l = V(self.r)(y_l)
            y_l = y_l.repeat(4**d, axis=-2)
            y = jnp.concatenate([y, y_l], axis=-1)
            y = H(self.L, l)(y)

        y = y.take(self.switch_idx, axis=-2)

        for m in range(self.NUM_RESNET):
            y_tmp = y + M()(nn.relu(M()(y)))
            y = y + y_tmp

            if not (m + 1) == self.NUM_RESNET:
                y = nn.relu(y)

        for l in range(self.L // 2, self.L):
            y = G(self.L, l)(y)

        y = U(self.s**2)(y)

        y = y[:, 0, ...]
        y = jnp.reshape(y, (-1, self.nx**2))
        y = y.take(self.idx_morton_to_flatten, axis=-1)

        y = jnp.reshape(y, (-1, self.nx, self.nx, 1))

        for conv_layer in self.convs:
            y = conv_layer(y)
            y = nn.relu(y)

        y = self.final_conv(y)

        return y[:, :, :, 0]

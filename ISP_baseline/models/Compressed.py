import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn


def build_permutation_indices(L, l):
    """
    Compute permutation indices used for grouping neighboring blocks before applying H layers.

    Args:
        L (int): The overall level determining the size of the data.
        l (int): The current level used to compute spacing and offsets.

    Returns:
        jnp.ndarray: A 1D array of indices for reordering tensor dimensions.
    """
    # Calculate the spacing between indices at the current level.
    delta = 2 ** (L - l - 1)

    # Create a tiled pattern over a small block.
    tmp = np.tile(np.arange(2) * delta, delta)
    # Adjust indices by repeating numbers over sub-blocks.
    tmp += np.repeat(np.arange(delta), 2)
    # Tile the pattern to cover blocks at level l.
    tmp = np.tile(tmp, 2**l)
    # Offset indices so that each block is correctly positioned.
    tmp += np.repeat(np.arange(2**l) * (2 ** (L - l)), 2 ** (L - l))
    return jnp.asarray(tmp)


def build_switch_indices(L):
    """
    Compute indices to redistribute blocks according to the transformation represented by x -> M*xM.

    Args:
        L (int): The overall level determining the block size.

    Returns:
        jnp.ndarray: A 1D array of indices for reordering after transformation.
    """
    # Halve the level to work with sub-blocks.
    L = L // 2
    # Calculate base offsets for each block row.
    tmp = np.arange(2**L) * (2**L)
    # Tile the row offsets across all blocks.
    tmp = np.tile(tmp, 2**L)
    # Add column indices within each block.
    tmp += np.repeat(np.arange(2**L), 2**L)
    return jnp.asarray(tmp)


class V(nn.Module):
    r: int  # Output rank (number of features) for the transformation.

    @nn.compact
    def __call__(self, x):
        """
        Applies x -> V*xV.

        Args:
            x (jnp.ndarray): Input tensor with shape (..., n, s, 2), where the last
                             dimension represents [real, imaginary] components.

        Returns:
            jnp.ndarray: Transformed tensor with combined complex components.
        """
        # Extract spatial dimensions.
        n, s = x.shape[1], x.shape[2]

        # Initialize parameters for the transformation.
        init_fn = nn.initializers.glorot_uniform()
        # Parameters for transforming the real part.
        vr1 = self.param("vr1", init_fn, (n, s, self.r))
        vi1 = self.param("vi1", init_fn, (n, s, self.r))
        # Parameters for transforming the imaginary part (first branch).
        vr2 = self.param("vr2", init_fn, (n, s, self.r))
        vi2 = self.param("vi2", init_fn, (n, s, self.r))
        # Parameters for transforming the imaginary part (second branch).
        vr3 = self.param("vr3", init_fn, (n, s, self.r))
        vi3 = self.param("vi3", init_fn, (n, s, self.r))
        # Parameters for transforming the real part (second branch).
        vr4 = self.param("vr4", init_fn, (n, s, self.r))
        vi4 = self.param("vi4", init_fn, (n, s, self.r))

        # Split the input into real and imaginary components.
        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum("...iaj,ajk->...iak", x_re, vr1)
        y_re_1 = jnp.einsum("abj...i,bjk->abk...i", y_re_1, vr1)

        y_re_2 = jnp.einsum("...iaj,ajk->...iak", x_re, vi1)
        y_re_2 = jnp.einsum("abj...i,bjk->abk...i", y_re_2, vi1)

        y_re_3 = jnp.einsum("...iaj,ajk->...iak", x_im, vi2)
        y_re_3 = jnp.einsum("abj...i,bjk->abk...i", y_re_3, vr2)

        y_re_4 = jnp.einsum("...iaj,ajk->...iak", x_im, vr2)
        y_re_4 = jnp.einsum("abj...i,bjk->abk...i", y_re_4, vi2)

        # Sum the contributions for the real part.
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        y_im_1 = jnp.einsum("...iaj,ajk->...iak", x_im, vr3)
        y_im_1 = jnp.einsum("abj...i,bjk->abk...i", y_im_1, vr3)

        y_im_2 = jnp.einsum("...iaj,ajk->...iak", x_im, vi3)
        y_im_2 = jnp.einsum("abj...i,bjk->abk...i", y_im_2, vi3)

        y_im_3 = jnp.einsum("...iaj,ajk->...iak", x_re, vi4)
        y_im_3 = jnp.einsum("abj...i,bjk->abk...i", y_im_3, vr4)

        y_im_4 = jnp.einsum("...iaj,ajk->...iak", x_re, vr4)
        y_im_4 = jnp.einsum("abj...i,bjk->abk...i", y_im_4, vi4)

        # Sum the contributions for the imaginary part.
        y_im = y_im_1 + y_im_2 + y_im_3 + y_im_4

        # Stack the computed real and imaginary parts into a complex tensor.
        y = jnp.stack([y_re, y_im], axis=-1)
        return y


class H(nn.Module):
    perm_idx: jnp.ndarray  # Precomputed permutation indices for reordering

    @nn.compact
    def __call__(self, x):
        """
        Applies x -> H*xH.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Transformed tensor.
        """
        # Infer dimensions: 'm' becomes half of the current dimension and 's' doubles.
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Initialize parameters for four transformation branches.
        init_fn = nn.initializers.glorot_uniform()
        hr1 = self.param("hr1", init_fn, (m, s, s))
        hi1 = self.param("hi1", init_fn, (m, s, s))
        hr2 = self.param("hr2", init_fn, (m, s, s))
        hi2 = self.param("hi2", init_fn, (m, s, s))
        hr3 = self.param("hr3", init_fn, (m, s, s))
        hi3 = self.param("hi3", init_fn, (m, s, s))
        hr4 = self.param("hr4", init_fn, (m, s, s))
        hi4 = self.param("hi4", init_fn, (m, s, s))

        # Permute the tensor dimensions using the precomputed indices.
        x = x.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)

        # Reshape the tensor to split it into blocks:
        # New shape: (batch, m, s, m, s, 2) where the last dimension holds [real, imaginary].
        x = x.reshape((-1, m, s, m, s, 2))
        x_re, x_im = x[..., 0], x[..., 1]

        # Process the real part with two sets of operations.
        y_re_1 = jnp.einsum("...iaj,ajk->...iak", x_re, hr1)
        y_re_1 = jnp.einsum("abj...i,bjk->abk...i", y_re_1, hr1)
        y_re_2 = jnp.einsum("...iaj,ajk->...iak", x_re, hi1)
        y_re_2 = jnp.einsum("abj...i,bjk->abk...i", y_re_2, hi1)
        y_re_3 = jnp.einsum("...iaj,ajk->...iak", x_im, hi2)
        y_re_3 = jnp.einsum("abj...i,bjk->abk...i", y_re_3, hr2)
        y_re_4 = jnp.einsum("...iaj,ajk->...iak", x_im, hr2)
        y_re_4 = jnp.einsum("abj...i,bjk->abk...i", y_re_4, hi2)
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        # Process the imaginary part similarly.
        y_im_1 = jnp.einsum("...iaj,ajk->...iak", x_im, hr3)
        y_im_1 = jnp.einsum("abj...i,bjk->abk...i", y_im_1, hr3)
        y_im_2 = jnp.einsum("...iaj,ajk->...iak", x_im, hi3)
        y_im_2 = jnp.einsum("abj...i,bjk->abk...i", y_im_2, hi3)
        y_im_3 = jnp.einsum("...iaj,ajk->...iak", x_re, hi4)
        y_im_3 = jnp.einsum("abj...i,bjk->abk...i", y_im_3, hr4)
        y_im_4 = jnp.einsum("...iaj,ajk->...iak", x_re, hr4)
        y_im_4 = jnp.einsum("abj...i,bjk->abk...i", y_im_4, hi4)
        y_im = y_im_1 + y_im_2 + y_im_3 + y_im_4

        # Stack real and imaginary parts back into a single tensor.
        y = jnp.stack([y_re, y_im], axis=-1)

        # Reshape to merge the block dimensions back to the overall tensor shape.
        n = m * 2  # Reconstructed spatial dimension.
        r = s // 2  # Reconstructed resolution.
        y = y.reshape((-1, n, r, n, r, 2))
        return y


class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        """
        Applies x -> M*xM.

        Args:
            x (jnp.ndarray): Input tensor with complex channels.

        Returns:
            jnp.ndarray: Transformed tensor.
        """
        # Extract dimensions: n (spatial size) and r (resolution factor).
        n, r = x.shape[1], x.shape[2]

        # Initialize parameters for the transformation.
        init_fn = nn.initializers.glorot_uniform()
        mr1 = self.param("mr1", init_fn, (n, r, r))
        mi1 = self.param("mi1", init_fn, (n, r, r))
        mr2 = self.param("mr2", init_fn, (n, r, r))
        mi2 = self.param("mi2", init_fn, (n, r, r))
        mr3 = self.param("mr3", init_fn, (n, r, r))
        mi3 = self.param("mi3", init_fn, (n, r, r))
        mr4 = self.param("mr4", init_fn, (n, r, r))
        mi4 = self.param("mi4", init_fn, (n, r, r))

        # Split the input into real and imaginary parts.
        x_re, x_im = x[..., 0], x[..., 1]

        # Process the real part with two branches.
        y_re_1 = jnp.einsum("...iaj,ajk->...iak", x_re, mr1)
        y_re_1 = jnp.einsum("abj...i,bjk->abk...i", y_re_1, mr1)
        y_re_2 = jnp.einsum("...iaj,ajk->...iak", x_re, mi1)
        y_re_2 = jnp.einsum("abj...i,bjk->abk...i", y_re_2, mi1)
        y_re_3 = jnp.einsum("...iaj,ajk->...iak", x_im, mi2)
        y_re_3 = jnp.einsum("abj...i,bjk->abk...i", y_re_3, mr2)
        y_re_4 = jnp.einsum("...iaj,ajk->...iak", x_im, mr2)
        y_re_4 = jnp.einsum("abj...i,bjk->abk...i", y_re_4, mi2)
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        # Process the imaginary part similarly.
        y_im_1 = jnp.einsum("...iaj,ajk->...iak", x_im, mr3)
        y_im_1 = jnp.einsum("abj...i,bjk->abk...i", y_im_1, mr3)
        y_im_2 = jnp.einsum("...iaj,ajk->...iak", x_im, mi3)
        y_im_2 = jnp.einsum("abj...i,bjk->abk...i", y_im_2, mi3)
        y_im_3 = jnp.einsum("...iaj,ajk->...iak", x_re, mi4)
        y_im_3 = jnp.einsum("abj...i,bjk->abk...i", y_im_3, mr4)
        y_im_4 = jnp.einsum("...iaj,ajk->...iak", x_re, mr4)
        y_im_4 = jnp.einsum("abj...i,bjk->abk...i", y_im_4, mi4)
        y_im = y_im_1 + y_im_2 + y_im_3 + y_im_4

        # Stack the processed real and imaginary parts.
        y = jnp.stack([y_re, y_im], axis=-1)
        return y


class G(nn.Module):
    perm_idx: jnp.ndarray  # Precomputed permutation indices for final reordering

    @nn.compact
    def __call__(self, x):
        """
        Applies x -> G*xG.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Transformed and permuted tensor.
        """
        # Infer new dimensions: 'm' is half and 's' is doubled.
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Initialize parameters for transformation.
        init_fn = nn.initializers.glorot_uniform()
        gr1 = self.param("gr1", init_fn, (m, s, s))
        gi1 = self.param("gi1", init_fn, (m, s, s))
        gr2 = self.param("gr2", init_fn, (m, s, s))
        gi2 = self.param("gi2", init_fn, (m, s, s))
        gr3 = self.param("gr3", init_fn, (m, s, s))
        gi3 = self.param("gi3", init_fn, (m, s, s))
        gr4 = self.param("gr4", init_fn, (m, s, s))
        gi4 = self.param("gi4", init_fn, (m, s, s))

        # Reshape the input into blocks.
        x = x.reshape((-1, m, s, m, s, 2))
        x_re, x_im = x[..., 0], x[..., 1]

        # Process the real part.
        y_re_1 = jnp.einsum("...iaj,ajk->...iak", x_re, gr1)
        y_re_1 = jnp.einsum("abj...i,bjk->abk...i", y_re_1, gr1)
        y_re_2 = jnp.einsum("...iaj,ajk->...iak", x_re, gi1)
        y_re_2 = jnp.einsum("abj...i,bjk->abk...i", y_re_2, gi1)
        y_re_3 = jnp.einsum("...iaj,ajk->...iak", x_im, gi2)
        y_re_3 = jnp.einsum("abj...i,bjk->abk...i", y_re_3, gr2)
        y_re_4 = jnp.einsum("...iaj,ajk->...iak", x_im, gr2)
        y_re_4 = jnp.einsum("abj...i,bjk->abk...i", y_re_4, gi2)
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        # Process the imaginary part.
        y_im_1 = jnp.einsum("...iaj,ajk->...iak", x_im, gr3)
        y_im_1 = jnp.einsum("abj...i,bjk->abk...i", y_im_1, gr3)
        y_im_2 = jnp.einsum("...iaj,ajk->...iak", x_im, gi3)
        y_im_2 = jnp.einsum("abj...i,bjk->abk...i", y_im_2, gi3)
        y_im_3 = jnp.einsum("...iaj,ajk->...iak", x_re, gi4)
        y_im_3 = jnp.einsum("abj...i,bjk->abk...i", y_im_3, gr4)
        y_im_4 = jnp.einsum("...iaj,ajk->...iak", x_re, gr4)
        y_im_4 = jnp.einsum("abj...i,bjk->abk...i", y_im_4, gi4)
        y_im = y_im_1 + y_im_2 + y_im_3 + y_im_4

        # Combine the results into a complex tensor.
        y = jnp.stack([y_re, y_im], axis=-1)

        # Reshape to merge block dimensions.
        n, r = m * 2, s // 2
        y = y.reshape((-1, n, r, n, r, 2))
        # Apply final permutation to reorder tensor dimensions.
        y = y.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)
        return y


class U(nn.Module):
    s: int  # Size parameter used to scale up the output resolution

    @nn.compact
    def __call__(self, x):
        """
        Applies x -> U*xU.

        Args:
            x (jnp.ndarray): Input tensor with complex channels.

        Returns:
            jnp.ndarray: Real-valued output tensor reshaped to (batch, nx, nx).
        """
        # Extract dimensions.
        n, r = x.shape[1], x.shape[2]
        nx = n * self.s  # Final spatial resolution

        # Initialize parameters for the transformation.
        init_fn = nn.initializers.glorot_uniform()
        ur1 = self.param("ur1", init_fn, (n, r, self.s))
        ui1 = self.param("ui1", init_fn, (n, r, self.s))
        ur2 = self.param("ur2", init_fn, (n, r, self.s))
        ui2 = self.param("ui2", init_fn, (n, r, self.s))
        ur3 = self.param("ur3", init_fn, (n, r, self.s))
        ui3 = self.param("ui3", init_fn, (n, r, self.s))
        ur4 = self.param("ur4", init_fn, (n, r, self.s))
        ui4 = self.param("ui4", init_fn, (n, r, self.s))

        # Split input into real and imaginary parts.
        x_re, x_im = x[..., 0], x[..., 1]

        # Apply successive transformations to compute the real output.
        y_re_1 = jnp.einsum("...iaj,ajk->...iak", x_re, ur1)
        y_re_1 = jnp.einsum("abj...i,bjk->abk...i", y_re_1, ur1)
        y_re_2 = jnp.einsum("...iaj,ajk->...iak", x_re, ui2)
        y_re_2 = jnp.einsum("abj...i,bjk->abk...i", y_re_2, ui2)
        y_re_3 = jnp.einsum("...iaj,ajk->...iak", x_im, ui3)
        y_re_3 = jnp.einsum("abj...i,bjk->abk...i", y_re_3, ur3)
        y_re_4 = jnp.einsum("...iaj,ajk->...iak", x_im, ur4)
        y_re_4 = jnp.einsum("abj...i,bjk->abk...i", y_re_4, ui4)
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        # Reshape the result into the final output grid.
        return y_re.reshape((-1, nx, nx))


class Fstar(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    cart_mat: jnp.ndarray  # Matrix for converting from polar to Cartesian coordinates.
    r_index: jnp.ndarray  # Index array for selecting specific parts of the input.

    def setup(self):
        # Compute overall dimensions.
        self.n = 2**self.L
        self.nx = (2**self.L) * self.s
        self.neta = (2**self.L) * self.s

        self.V = V(self.r)
        self.Hs = [
            H(build_permutation_indices(self.L, l))
            for l in range(self.L - 1, self.L // 2 - 1, -1)
        ]
        self.Ms = [M() for _ in range(self.NUM_RESNET)]
        self.Gs = [
            G(build_permutation_indices(self.L, l)) for l in range(self.L // 2, self.L)
        ]
        self.U = U(self.s)
        self.switch_idx = build_switch_indices(self.L)

    def __call__(self, inputs):
        """
        Applies the full Fstar transformation pipeline, including:
          1. Selection and reshaping of input.
          2. Sequential application of V, H, M, and G modules.
          3. Final upsampling and conversion from polar to Cartesian coordinates.

        Args:
            inputs (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Final transformed output in Cartesian coordinates.
        """
        # Select relevant parts of the input using r_index and reshape.
        y = inputs.take(self.r_index, axis=1)
        y = jnp.reshape(y, (-1, self.n, self.s, self.n, self.s, 2))

        # Apply the V transformation.
        y = self.V(y)
        # Sequentially apply each H transformation.
        for h in self.Hs:
            y = h(y)

        # Reorder blocks using the switch indices. Apply the series of M modules in a residual (skip-connection) fashion.
        y = y.take(self.switch_idx, axis=1).take(self.switch_idx, axis=3)

        for m in self.Ms:
            # For the last M module, apply directly; for others, add a residual connection with ReLU.
            y = m(y) if m is self.Ms[-1] else y + nn.relu(m(y))

        # Sequentially apply each G module.
        for g in self.Gs:
            y = g(y)
        # Final upsampling and transformation.
        y = self.U(y)

        # Extract the diagonal from the spatial dimensions to collapse redundancy.
        y = jnp.diagonal(y, axis1=1, axis2=2)
        output_polar = jnp.reshape(y, (-1, self.nx**2, 1))

        # Define a helper function to convert from polar to Cartesian coordinates.
        def polar_to_cart(x):
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))

        # Apply the conversion to each sample in the batch.
        return jax.vmap(polar_to_cart)(output_polar)


class CompressedModel(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    NUM_CONV: int
    cart_mat: jnp.ndarray  # Matrix for polar to Cartesian conversion.
    r_index: jnp.ndarray  # Index array for selecting relevant input data.

    def setup(self):
        self.fstar_layer0 = Fstar(
            L=self.L,
            s=self.s,
            r=self.r,
            NUM_RESNET=self.NUM_RESNET,
            cart_mat=self.cart_mat,
            r_index=self.r_index,
        )
        self.fstar_layer1 = Fstar(
            L=self.L,
            s=self.s,
            r=self.r,
            NUM_RESNET=self.NUM_RESNET,
            cart_mat=self.cart_mat,
            r_index=self.r_index,
        )
        self.fstar_layer2 = Fstar(
            L=self.L,
            s=self.s,
            r=self.r,
            NUM_RESNET=self.NUM_RESNET,
            cart_mat=self.cart_mat,
            r_index=self.r_index,
        )

        self.convs = [
            nn.Conv(features=6, kernel_size=(3, 3), padding="SAME")
            for _ in range(self.NUM_CONV)
        ]
        self.final_conv = nn.Conv(features=1, kernel_size=(3, 3), padding="SAME")

    def __call__(self, inputs):
        """
        The forward pass of the CompressedModel:
          1. Processes three input channels through separate Fstar layers.
          2. Concatenates the outputs.
          3. Passes the concatenated tensor through a series of convolutional layers,
             concatenating intermediate features.
          4. Produces a final single-channel output.

        Args:
            inputs (jnp.ndarray): Input tensor with at least 4 dimensions.

        Returns:
            jnp.ndarray: The final output of the model.
        """
        # Process each channel separately using Fstar layers.
        y0 = self.fstar_layer0(inputs[:, :, :, 0])
        y1 = self.fstar_layer1(inputs[:, :, :, 1])
        y2 = self.fstar_layer2(inputs[:, :, :, 2])

        # Concatenate outputs along the channel dimension.
        y = jnp.concatenate([y0, y1, y2], axis=-1)

        # Apply a series of convolutional layers with ReLU activations.
        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = nn.relu(tmp)
            # Concatenate new features with existing ones.
            y = jnp.concatenate([y, tmp], axis=-1)

        # Final convolution to combine features into a single channel.
        y = self.final_conv(y)

        # Return the output, removing the trailing singleton channel dimension.
        return y[:, :, :, 0]

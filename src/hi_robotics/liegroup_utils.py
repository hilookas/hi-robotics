# Python Native LieGroup operations (SO3, SE3, RxSO3, Sim3) for PyTorch

# Copy from: https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/pytorch3d/transforms/se3.py
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Pose3.cpp

import torch


def so3_vee(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(torch.abs(h + h.permute(0, 2, 1)).max()) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def so3_hat(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        phi: Batch of vectors of shape `(minibatch, 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3, 3)` where each matrix is of the form:
            `[      0 -phi_z  phi_y ]
             [  phi_z      0 -phi_x ]
             [ -phi_y  phi_x      0 ]`

    Raises:
        ValueError if `phi` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = phi.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    # h = torch.zeros((N, 3, 3), dtype=phi.dtype, device=phi.device)

    # x, y, z = phi.unbind(1)

    # h[:, 0, 1] = -z
    # h[:, 0, 2] = y
    # h[:, 1, 0] = z
    # h[:, 1, 2] = -x
    # h[:, 2, 0] = -y
    # h[:, 2, 1] = x

    # return h

    rx, ry, rz = phi[..., 0], phi[..., 1], phi[..., 2]
    zeros = torch.zeros(phi.shape[:-1], dtype=phi.dtype, device=phi.device)
    return torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(phi.shape + (3,))


def so3_expmap(phi: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `phi`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`phi`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    Args:
        phi: Batch of vectors of shape `(minibatch, 3)`.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `phi` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    _, dim = phi.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    theta = torch.norm(phi, dim=-1)

    B = torch.sinc(theta / torch.pi)
    C = torch.where(theta * theta == 0, 0, (1 - torch.cos(theta)) / (theta**2))
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W

    return I + B * W + C * WW

def so3_logmap(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)`.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    if R.size(-1) != 3 or R.size(-2) != 3:
        raise ValueError(f"Invalid rotation R shape {R.shape}.")

    sin_theta_axis = torch.stack([ # vee # sin(theta) * e
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) * 0.5
    theta = torch.atan2(
        torch.norm(sin_theta_axis, p=2, dim=-1) * 2.,
        torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (R + torch.eye(3, dtype=R.dtype, device=R.device).unsqueeze(0))
    n_norm = torch.norm(n, dim=-1, keepdim=True)

    return torch.where(torch.isclose(theta, theta.new_full((1,), torch.pi)),
        theta * (n / n_norm)[:,torch.argmax(n_norm)],
        torch.where(torch.isclose(theta, torch.zeros_like(theta)),
            torch.zeros(3, dtype=R.dtype, device=R.device),
            sin_theta_axis / torch.sinc(theta / torch.pi)
        ),
    )

# Test cases for edge cases
# so3_logmap(torch.tensor([[[-1,0,0],[0,-1,0],[0,0,1.]]]))

# Copy from: https://github.com/princeton-vl/lietorch/blob/e7df86554156b36846008d8ddbcc4d8521a16554/lietorch/include/rxso3.h
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Similarity3.cpp
# See: https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

EPS = 1e-6


def se3_expmap(log_transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[phi | log_translation]`,
    i.e. a concatenation of two 3D vectors `phi` and `log_translation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ so3_hat(phi) log_translation ]
                         [   0 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||phi|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_logmap(se3_exponential_map(log_transform)) == log_transform
    ```

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., 3:]
    phi = log_transform[..., :3]

    R = so3_expmap(phi)

    # A helper function that computes the "V" matrix from [1], Sec 9.4.2. [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    A = torch.tensor(1., dtype=phi.dtype, device=phi.device).unsqueeze(0)
    B = torch.where(torch.abs(theta) < EPS,
        torch.tensor(0.5, dtype=phi.dtype, device=phi.device).unsqueeze(0),
        (1. - torch.cos(theta)) / (theta**2),
    )
    C = torch.where(torch.abs(theta) < EPS,
        torch.tensor(1. / 6., dtype=phi.dtype, device=phi.device).unsqueeze(0),
        (theta - torch.sin(theta)) / (theta**3),
    )
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V = A * I + B * W + C * WW # shape == (B,3,3)

    # translation is V @ T
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform


def se3_logmap(transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[phi | log_translation]`,
    i.e. a concatenation of two 3D vectors `phi` and `log_translation`.

    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [phi | log_translation]`
    is done as follows:
        ```
        log_transform = log(transform)
        phi = so3_vee(log_transform[:3, :3])
        log_translation = log_transform[:3, 3]
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].

    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_expmap(se3_logmap(transform)) == transform
    ```

    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.

    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.

    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, 3, :3], torch.zeros_like(transform[:, 3, :3])):
        raise ValueError("All elements of `transform[:, 3, :3]` should be 0.")

    # log_rot is just so3_logmap of the upper left 3x3 block
    R = transform[:, :3, :3]
    phi = so3_logmap(R)

    # log_translation is V^-1 @ T
    T = transform[:, :3, 3]

    theta = torch.norm(phi, dim=-1)

    A = torch.tensor(1., dtype=phi.dtype, device=phi.device).unsqueeze(0)
    B = torch.tensor(-0.5, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    C = torch.where(torch.abs(theta) < EPS,
        torch.tensor(1. / 12., dtype=phi.dtype, device=phi.device).unsqueeze(0),
        (theta * torch.sin(theta) + 2. * torch.cos(theta) - 2.) / (2. * theta**2 * (torch.cos(theta) - 1.)), # theta接近0的时候，
    )
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V_inv = A * I + B * W + C * WW

    log_translation = (V_inv @ T[:, :, None])[:, :, 0]

    return torch.cat((phi, log_translation), dim=1)


def se3_inv(A: torch.Tensor) -> torch.Tensor:
    """
    se3_inv

    Faster version of torch.linalg.inv for SE3 matrices.
    ( R t )^-1   ( R^T -R^T*t )
    ( 0 1 )    = ( 0    1     )

    :param A: A.shape == (..., 4, 4)
    :type A: torch.Tensor
    :return: A_inv.shape == (..., 4, 4)
    :rtype: Tensor
    """

    R = A[:, :3, :3]
    t = A[:, :3, 3]
    R_inv = R.permute(0, 2, 1)
    t_inv = -torch.bmm(R_inv, t[:, :, None])[:, :, 0]
    A_inv = torch.zeros_like(A)
    A_inv[:, :3, :3] = R_inv
    A_inv[:, :3, 3] = t_inv
    A_inv[:, 3, 3] = 1.0
    return A_inv


def sim3_inv(A: torch.Tensor) -> torch.Tensor:
    """
    sim3_inv

    Faster version of torch.linalg.inv for Sim3 matrices (similarity transformation).
    Sim3矩阵结构：(sR  t)，其中s是缩放因子，R是3x3正交旋转矩阵，t是3维平移向量
                  (0^T 1)
    逆矩阵公式：  ( (1/s)R^T   -R^T*t/s )
                  ( 0^T        1        )

    :param A: Sim3矩阵，shape == (..., 4, 4)
    :type A: torch.Tensor
    :return: Sim3逆矩阵，shape == (..., 4, 4)
    :rtype: Tensor
    """
    sR = A[..., :3, :3]  # (..., 3, 3)  s*R部分
    t = A[..., :3, 3]    # (..., 3)     平移向量

    norm_sR = torch.norm(sR, dim=(-2, -1))  # (...,) 计算每个sR的Frobenius范数
    s = norm_sR / torch.sqrt(torch.tensor(3.0, device=sR.device, dtype=sR.dtype))

    EPS = 1e-8

    R = sR / (s[..., None, None] + EPS)  # (..., 3, 3)  还原纯旋转矩阵
    R_inv = R.permute(*tuple(range(R.ndim-2)), -1, -2)  # (..., 3, 3)  R^T，适配任意批量维度

    inv_s = 1.0 / (s + EPS)
    sR_inv = inv_s[..., None, None] * R_inv  # (..., 3, 3)

    t_inv = -torch.matmul(sR_inv, t[..., None])[..., 0]  # (..., 3)

    A_inv = torch.zeros_like(A)
    A_inv[..., :3, :3] = sR_inv
    A_inv[..., :3, 3] = t_inv
    A_inv[..., 3, 3] = 1.0

    return A_inv


# Copy from: https://github.com/princeton-vl/lietorch/blob/e7df86554156b36846008d8ddbcc4d8521a16554/lietorch/include/rxso3.h
# Which is copied from: https://github.com/strasdat/Sophus/blob/d0b7315a0d90fc6143defa54596a3a95d9fa10ec/sophus/so3.hpp
# MAGIC CODE!
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Similarity3.cpp

EPS = 1e-6


def one_minus_cos(theta: torch.Tensor):
    # 1. - torch.cos(theta)
    return 2. * (torch.sin(0.5 * theta) ** 2) # Better numeric stability


def sim3_logmap(T: torch.Tensor):
    # To get the logmap, calculate phi and sigma, then solve for u as shown by Ethan at
    # See: https://www.ethaneade.org/latex2html/lie/node29.html
    # See: https://www.ethaneade.com/lie.pdf
    # See: https://www.cis.upenn.edu/~jean/interp-SIM.pdf
    s = torch.norm(T[:, :3, :3], dim=(1,2)) / torch.sqrt(torch.tensor(3.0, dtype=T.dtype, device=T.device).unsqueeze(0)) # shape == (B,)
    R = T[:, :3, :3] / s.unsqueeze(-1).unsqueeze(-1) # shape == (B,3,3)
    t = T[:, :3, 3] # shape == (B,3)

    sigma = torch.log(s) # shape == (B,)
    phi = so3_logmap(R) # shape == (B,3)

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    A = torch.where(torch.abs(sigma) < EPS,
        1. - 0.5 * sigma,
        sigma / torch.expm1(sigma),
    )
    B = torch.where(torch.abs(sigma) < EPS,
        torch.tensor(-0.5, dtype=phi.dtype, device=phi.device).unsqueeze(0),
        torch.where(torch.abs(theta) < EPS,
            (-sigma * (torch.expm1(sigma) + 1.) + (torch.expm1(sigma) + 1.) - 1.) / (torch.expm1(sigma) * torch.expm1(sigma)),
            (theta * (torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta)) - theta - sigma * (torch.expm1(sigma) + 1.) * torch.sin(theta)) / (theta * ((torch.expm1(sigma) + 1.)**2 - 2. * (torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta)) + 1.)),
        ),
    )
    C = torch.where(torch.abs(sigma) < EPS,
        torch.where(torch.abs(theta) < EPS,
            torch.tensor(1. / 12., dtype=phi.dtype, device=phi.device).unsqueeze(0),
            (theta * torch.sin(theta) + 2. * (1. - one_minus_cos(theta)) - 2.) / (2. * theta**2 * ((1. - one_minus_cos(theta)) - 1.)),
        ),
        torch.where(torch.abs(theta) < EPS,
            ((torch.expm1(sigma) + 1.)**2 * sigma - 2. * (torch.expm1(sigma) + 1.)**2 + (torch.expm1(sigma) + 1.) * sigma + 2. * (torch.expm1(sigma) + 1.)) / (2. * (torch.expm1(sigma) + 1.)**3 - 6 * (torch.expm1(sigma) + 1.)**2 + 6 * (torch.expm1(sigma) + 1.) - 2.),
            -(torch.expm1(sigma) + 1.) * (theta * (torch.expm1(sigma) + 1.) * torch.sin(theta) - theta * torch.sin(theta) + sigma * (torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta)) - (torch.expm1(sigma) + 1.) * sigma + sigma * (1. - one_minus_cos(theta)) - sigma) / (theta**2 * ((torch.expm1(sigma) + 1.)**2 - 2. * (torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta)) + 1.) * torch.expm1(sigma)),
        ),
    )
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V_inv = A * I + B * W + C * WW

    rho = (V_inv @ t.unsqueeze(-1)).squeeze(-1) # shape == (B,3)

    v = torch.concat([ sigma.unsqueeze(-1), phi, rho ], dim=-1)
    return v


def sim3_expmap(v: torch.Tensor):
    N, dim1 = v.shape
    if dim1 != 7:
        raise ValueError("Input tensor shape has to be Nx7.")

    sigma = v[:,0] # shape == (B,)
    phi = v[:,1:4] # shape == (B,3)
    rho = v[:,4:7] # shape == (B,3)

    R = so3_expmap(phi) # shape == (B,3,3)

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    A = torch.where(torch.abs(sigma) < EPS,
        torch.tensor(1., dtype=phi.dtype, device=phi.device).unsqueeze(0),
        torch.expm1(sigma) / sigma,
    )
    B = torch.where(torch.abs(sigma) < EPS,
        torch.where(torch.abs(theta) < EPS,
            torch.tensor(0.5, dtype=phi.dtype, device=phi.device).unsqueeze(0),
            (1. - (1. - one_minus_cos(theta))) / (theta**2),
        ),
        torch.where(torch.abs(theta) < EPS,
            ((sigma - 1.) * (torch.expm1(sigma) + 1.) + 1.) / (sigma**2),
            ((torch.expm1(sigma) + 1.) * torch.sin(theta) * sigma + (1. - (torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta))) * theta) / (theta * (theta**2 + sigma**2)),
        ),
    )
    C = torch.where(torch.abs(sigma) < EPS,
        torch.where(torch.abs(theta) < EPS,
            torch.tensor(1. / 6., dtype=phi.dtype, device=phi.device).unsqueeze(0),
            (theta - torch.sin(theta)) / (theta**3),
        ),
        torch.where(torch.abs(theta) < EPS,
            ((torch.expm1(sigma) + 1.) * 0.5 * sigma**2 + (torch.expm1(sigma) + 1.) - 1. - sigma * (torch.expm1(sigma) + 1.)) / (sigma**3),
            (torch.expm1(sigma) / sigma - (((torch.expm1(sigma) + 1.) * (1. - one_minus_cos(theta)) - 1.) * sigma + (torch.expm1(sigma) + 1.) * torch.sin(theta) * theta) / (theta**2 + sigma**2)) / (theta**2),
        ),
    )
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V = A * I + B * W + C * WW # shape == (B,3,3)

    t = (V @ rho.unsqueeze(-1)).squeeze(-1) # shape == (B,3)
    s = torch.exp(sigma) # shape == (B,)

    T = torch.zeros((N, 4, 4), dtype=v.dtype, device=v.device)
    T[:, :3, :3] = R * s.unsqueeze(-1).unsqueeze(-1)
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


if __name__ == "__main__":
    if True:
        # SE3 test
        import pytransform3d.transformations as pt
        import pytransform3d.rotations as pr
        import numpy as np

        T = pt.random_transform(np.random.default_rng(None))
        print("original", T)
        print()
        print("pytransform3d", pt.exponential_coordinates_from_transform(T))
        print("pytorch3d", se3_logmap(torch.tensor(T[None,]).float()))
        print()
        print("pytransform3d", pt.transform_from_exponential_coordinates(pt.exponential_coordinates_from_transform(T)))
        print("pytorch3d", se3_expmap(se3_logmap(torch.tensor(T[None,]).float())))
        print()
        print(se3_inv(torch.tensor(T)[None,].float()))
        print(pt.invert_transform(T))

        import ipdb; ipdb.set_trace()

    if True:
        # Sim3 test
        import lietorch

        print(lietorch.Sim3.exp(torch.tensor([[0,0,0,0,0,0,0.]])).matrix())
        #                  scale  rot   trans              trans   rot scale
        print(torch.tensor([[0., 0,0,0, 0,3,0]]))
        print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix()))
        print(torch.tensor([[0., 0,2,0, 0,3,0]]))
        print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix()))
        print(torch.tensor([[1., 0,0,0, 0,3,0]]))
        print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix()))
        print(torch.tensor([[1., 0,2,0, 0,3,0]]))
        print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix()))
        print()
        print(torch.tensor([[1., 0,0,0, 0,0,2]]))
        print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix()))
        print()
        print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix())
        print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix())))
        print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix())
        print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix())))
        print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix())
        print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix())))
        print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix())
        print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix())))
        print()
        print(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix())
        print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix())))

        import ipdb; ipdb.set_trace()

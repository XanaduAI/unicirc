import numpy as np


def count_clifford(theta_opt, verbose=False, atol=1e-6):
    r"""
    Function to compute Clifford, non-Clifford and zero angles from rotation angles

    Args:
        theta_opt (TensorLike): Rotation angles
        verbose (bool): Whether or not to output the angles and their assignments to debugging and sanity-checking
        atol (float): absolute tolerance when comparing values to Clifford angles and zeros

    **Example**

    >>> theta_opt = np.array([np.pi, np.pi/2, np.pi/4, 1e-16])
    >>> cliffs, non_cliffs, zeros = count_clifford(theta_opt, verbose=True)
    Clifford: 2.0
    Clifford: 1.0
    non-Clifford: 0.5
    Zero: 0.0
    >>> cliffs, non_cliffs, zeros
    (2, 1, 1)

    """

    params = np.mod(theta_opt + 2 * np.pi, 2 * np.pi)
    params = params / (np.pi / 2)

    cliffs = 0
    non_cliffs = 0
    zeros = 0

    for p in params:
        if any(np.isclose(p, val, atol=atol) for val in [0.0, 4.0]):
            print(f"Zero: {p}") if verbose else None
            zeros += 1
        elif any(np.isclose(p, val, atol=atol) for val in [1.0, 2.0, 3.0]):
            print(f"Clifford: {p}") if verbose else None
            cliffs += 1
        else:
            print(f"non-Clifford: {p}") if verbose else None
            non_cliffs += 1

    return (cliffs, non_cliffs, zeros)

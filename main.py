import numpy as np


def first_derivative(f, x0, h=0.1, kind='central'):
    """Evaluates the first derivative of a function at a point x0.

    Parameters
    ----------
    f : callable
        Function to take the derivative of
    x0 : float
        Point to take the derivative at
    h : float (default: 0.1)
        Step size
    kind : str (default: 'central')
        Type of differentiation ('forward', 'backward', or 'central')

    Returns
    -------
    float
        Tangent of the function at x0"""

    if kind == 'forward':
        tangent = (f(x0+h) - f(x0)) / h
    elif kind == 'backward':
        tangent = (f(x0) - f(x0-h)) / h
    elif kind == 'central':
        tangent = (f(x0+h) - f(x0-h)) / (2*h)
    else:
        raise TypeError(f'"{kind}" is not a valid or supported form of differentiation')
    return tangent


def second_derivative_direct(f, x0, h=0.1):
    """Evaluates the second derivative of a function at a point x0.

        Parameters
        ----------
        f : callable
            Function to take the second derivative of
        x0 : float
            Point to take the second derivative at
        h : float (default: 0.1)
            Step size

        Returns
        -------
        float
            Second derivative of the function at x0"""
    return (f(x0+h) - 2*f(x0) + f(x0-h)) / (h**2)


def get_lagrange_polynomial(pivots, i):
    """Given a list of pivots and a function, this generates the lagrange polynomial Li
    (leaving the ith pivot out)

    Parameters
    ----------
    f : callable
        Function to generate the lagrange polynomial on
    pivots : np.array
        Array of pivots to use for the lagrange polynomial
    i : int
        Which pivot to leave out

    Returns
    -------
    np.poly1d
        The lagrange polynomial around the given pivot."""
    # Error handling for i
    try:
        assert i >= 0
        assert type(i) == int
    except AssertionError:
        raise ValueError(f'i needs to be >= 0 and an integer (is {i})')

    n = len(pivots)
    # Take all the pivots except i.
    lagrange_pivots = np.delete(pivots, i)
    # Set up the polynomial itself
    lagrange_polynomial = np.poly1d(lagrange_pivots, r=True)

    # Generate factor to divide by
    base = np.full(n - 1, pivots[i])
    changed = (base - lagrange_pivots)
    factor = changed.prod()

    lagrange_polynomial /= factor

    return lagrange_polynomial


# todo Pim: Verify these results
def get_lagrange_list(x1, x2, n=2, pivots=None):
    """Generate lagrange polynomials of f in n evenly spaced pivot points.

    Parameters
    ----------
    f : callable
        Function to generate the lagrange polynomials of
    x1 : float
        Left bound of polynomial
    x2 : float
        Right bound of polynomial
    n : int (default: 2)
        There are a total of n pivot points
    pivots : np.array (default: None)
        Optionally you can manually select pivots by specifying them in this argument.
        Ex. get_lagrange_list(1, 2, pivots=[1, 1.2, 2]) will use 1, 1.2 and 2 as pivots.

    Returns
    -------
    list
        A list containing all the lagrange polynomials."""
    # Error handling for when you have less than the 2 endpoints as pivot
    try:
        assert n >= 2
        assert type(n) == int
    except AssertionError:
        raise ValueError(f'n needs to be >= 2 and an integer (is {n})')

    # Allow user to input pivots too
    if pivots is None:
        # Set up all the pivots (equidistant from one another)
        pivots = np.linspace(start=x1, stop=x2, num=n)
    else:
        n = len(pivots)

    # Set up the lagrange polynomials
    lagrange_polynomials = [get_lagrange_polynomial(pivots, i) for i in range(n)]

    return lagrange_polynomials


def get_derivation_matrix(f, x1, x2, n=2, k=1):
    """Gives the matrix associated with the k-th derivative of f using n evenly spaced pivots

    Parameters
    ----------
    f : callable
        Function to generate the lagrange polynomials of
    x1 : float
        Left bound of polynomial
    x2 : float
        Right bound of polynomial
    n : int (default: 2)
        There are a total of n pivot points
    k : int (default: 1)
        The order of the derivative we are taking

    Returns
    -------
    np.array
        The discrete derivation matrix for the k-th derivative of this function in these points
    """
    # Check that the order of the derivative is larger than or equal to one
    try:
        assert k >= 1
        assert type(k) == int
    except AssertionError:
        raise ValueError(f'k needs to be >= 1 and an integer (is {k})')

    # Get lagrange polynomials and list of pivot points
    pivots = np.linspace(start=x1, stop=x2, num=n)
    lagrange_polynomials = get_lagrange_list(x1=x1, x2=x2, pivots=pivots)

    # Take the k-th derivative of every lagrange polynomial
    lagrange_poly_derivs = [poly.deriv(k) for poly in lagrange_polynomials]

    # Fill in the derivation matrix
    deriv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            deriv_matrix[i, j] = lagrange_poly_derivs[i](pivots[j])  # Evaluate polynomial at pivots[j].

    return deriv_matrix


if __name__ == '__main__':
    # Define function and grid proportions
    f = lambda x: x ** 7
    n = 5
    x1 = 0
    x2 = 4

    first_deriv_matrix = get_derivation_matrix(f=f, x1=x1, x2=x2, n=n, k=1)
    second_deriv_matrix = get_derivation_matrix(f=f, x1=x1, x2=x2, n=n, k=2)

    print(first_deriv_matrix * first_deriv_matrix)
    print(second_deriv_matrix)
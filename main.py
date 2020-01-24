import numpy as np
import pandas as pd

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


def get_lagrange_list(pivots=None, x1=0, x2=1, n=2):
    """Generate lagrange polynomials of f in n evenly spaced pivot points.

    Required to pass either pivots or x1, x2, and n.

    Parameters
    ----------
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


def get_derivation_matrix(pivots=None, x1=0, x2=1, n=2, k=1):
    """Gives the matrix associated with the k-th derivative of f using n evenly spaced pivots

    Required to pass either pivots or x1, x2, and n.

    Parameters
    ----------
    x1 : float
        Left bound of polynomial
    x2 : float
        Right bound of polynomial
    n : int (default: 2)
        There are a total of n pivot points
    pivots : np.array
        Array of pivots to use. Only used to supply unevenly spaced pivots.
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

    # Allow for uneven pivots if pivots is defined
    if pivots is None:
        pivots = np.linspace(start=x1, stop=x2, num=n)
    else:
        n = len(pivots)

    # Get lagrange polynomials
    lagrange_polynomials = get_lagrange_list(x1=x1, x2=x2, pivots=pivots)

    # Take the k-th derivative of every lagrange polynomial
    lagrange_poly_derivs = [poly.deriv(k) for poly in lagrange_polynomials]

    # Fill in the derivation matrix
    deriv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            deriv_matrix[i, j] = lagrange_poly_derivs[j](pivots[i])  # Evaluate polynomial at pivots[i].

    return deriv_matrix


if __name__ == '__main__':
    # Make lists with different number of pivots and orders of derivatives at which to evaluate the difference
    num_pivots_list = [3 * i for i in range(5, 13)]
    k_list = [i for i in range(2,10)]

    # Create empty matrix to store results in
    results_matrix = np.zeros((len(k_list), len(num_pivots_list)))

    # Iterate over different number of pivots and order of derivative
    for i in range(len(k_list)):
        for j in range(len(num_pivots_list)):
            # Calculate the difference measure five times and save in list, so it can be averaged
            results = np.zeros(10)
            for run in range(10):
                # Create a list of randomly spaced pivots
                pivots = np.random.random_sample(num_pivots_list[j])

                # Calculate the k-th derivative matrix by taking the first derivative matrix to the power k
                k_deriv_from_product = np.linalg.matrix_power(
                                                    get_derivation_matrix(pivots=pivots, k=1),
                                                    k_list[i])
                # Calculate the k-th derivative matrix directly
                k_deriv_direct = get_derivation_matrix(pivots=pivots, k=k_list[i])

                # Take the normalised norm of the difference between the matrices and add to results.
                results[run] = np.linalg.norm(k_deriv_from_product - k_deriv_direct) / np.linalg.norm(k_deriv_direct)
            # Take the average value of the relative difference measure between the matrices
            # Then take the 10-log and round to the nearest integer, and save the result
            results_matrix[i,j] = round(np.log10(np.mean(results)))

    # Print the latex code that generates the table
    df = pd.DataFrame(results_matrix, columns=[num_pivots_list], index=[k_list], dtype=int)
    print(df)
    print(df.to_latex())
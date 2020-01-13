import numpy as np
import itertools


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


# todo What is the function for??
def get_lagrange_polynomial(f, pivots, i):
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
    np.array
        An array signifying the polynomial where 1 + 2x + 3x^2 gets encoded as [1, 2, 3]"""
    # Error handling for i
    try:
        assert i >= 0
        assert type(i) == int
    except AssertionError:
        raise ValueError(f'i needs to be >= 0 and an integer (is {i})')

    n = len(pivots)
    # Take all the pivots except i.
    lagrange_pivots = np.delete(pivots, i) * -1  # Pivots made negative to simplify computation
    # Set up the polynomial itself
    lagrange_polynomial = np.zeros(n)

    # Calculate coefficients for polynomial
    for j in range(n):
        # Calculate coefficients
        for combination in itertools.combinations(lagrange_pivots, n - j - 1):
            coefficient = np.array(combination).prod()
            lagrange_polynomial[j] += coefficient
    # Set coefficient of highest power to 1
    lagrange_polynomial[-1] = 1

    # Generate factor to divide by
    base = np.full(n - 1, pivots[i])
    changed = (base + lagrange_pivots)  # Remember that all the lagrange_pivots are negative
    factor = changed.prod()

    lagrange_polynomial /= factor

    return lagrange_polynomial


# todo What is the function for??
def get_lagrange_array(f, x1, x2, n=2):
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

    Returns
    -------
    np.array
        An array containing all the lagrange polynomials."""
    # Error handling for when you have less than the 2 endpoints as pivot
    try:
        assert n >= 2
        assert type(n) == int
    except AssertionError:
        raise ValueError('n needs to be >= 2 and an integer (is {n})')

    # Set up all the pivots (equidistant from one another)
    pivots = np.linspace(start=x1, stop=x2, num=n)

    # Set up the lagrange polynomials
    lagrange_polynomials = [get_lagrange_polynomial(f, pivots, i) for i in range(n)]

    return lagrange_polynomials


def differentiate_polynomial(polynomial):
    """Takes the exact derivative of a polynomial

    Parameters
    ----------
    polynomial : np.array
        A polynomial encoded as  1 + 2x^2 + 3x^3 == [1, 2, 3] (length n)

    Returns
    -------
    np.array
        The derivative of the supplied polynomial. (length n-1)
    """

    # Initialize an empty derivative
    derivative = np.zeros(len(polynomial) - 1)

    # Go through the polynomial to take the derivative term by term
    for i, _ in enumerate(derivative):
        derivative[i] = polynomial[i+1] * (i+1)

    return derivative


def evaluate_polynomial(polynomial,x):
    """Gives the value of a polynomial at point x

    Parameters
    ----------
    polynomial : np.array
        A polynomial encoded as  1 + 2x^2 + 3x^3 == [1, 2, 3] (length n)
    x : float
        Point at which to evaluate the polynomial

    Returns
    -------
    Float
        The value of the polynomial at point x
    """

    # Initiate variable to sum the value of the polynomial
    value = 0

    # Sum all terms of the polynomial
    for power in range(len(polynomial)):
        value += polynomial[power]*x**power

    return value


def get_derivation_matrix(f,x1,x2,n=2,k=1):
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
    #Check that the order of the derivative is larger than or equal to one
    try:
        assert k >= 1
        assert type(k) == int
    except AssertionError:
        raise ValueError('k needs to be >= 1 and an integer (is {n})')

    # Get lagrange polynomials and list of pivot points
    pivots = np.linspace(start=x1, stop=x2, num=n)
    lagrange_polynomials = get_lagrange_array(f=f, x1=x1, x2=x2, n=n)

    # Take the k-th derivative of every lagrange polynomial
    derivatives_lagrange_polynomials = lagrange_polynomials
    for i in range(k):
        derivatives_lagrange_polynomials = [differentiate_polynomial(polynomial) for polynomial in derivatives_lagrange_polynomials]

    # Fill in the derivation matrix
    deriv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            deriv_matrix[i, j] = evaluate_polynomial(derivatives_lagrange_polynomials[i], pivots[j])


    return deriv_matrix


if __name__ == '__main__':
    # Define function and grid proportions
    f = lambda x: x**7
    n = 5
    x1 = 0
    x2 = 4

    first_deriv_matrix = get_derivation_matrix(f=f,x1=x1,x2=x2,n=n,k=1)
    second_deriv_matrix = get_derivation_matrix(f=f,x1=x1,x2=x2,n=n,k=2)

    print(first_deriv_matrix*first_deriv_matrix)
    print(second_deriv_matrix)
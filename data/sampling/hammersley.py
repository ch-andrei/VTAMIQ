# source: https://github.com/IDAES/idaes-pse/blob/main/idaes/surrogate/pysmo/sampling.py

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
import numpy as np

__prime_numbers__ = [
    112253, 112261, 112279, 112289, 112291, 112297, 112303, 112327, 112331, 112337
]


def base_conversion(a, b):
    """
    Function converts integer a from base 10 to base b
        Args:
            a(int): Number to be converted, base 10
            b(int): Base required
        Returns:
            string_representation(list): List containing strings of individual digits of "a" in the new base "b"
    Examples: Convert (i) 5 to base 2 and (ii) 57 to base 47
        >>  base_conversion(5, 2)
        >> ['1', '0', '1']
        >>  base_conversion(57, 47)
        >> ['1', '10']
    """

    string_representation = []
    if a < b:
        string_representation.append(str(a))
    else:
        while a > 0:
            a, c = (a // b, a % b)
            string_representation.append(str(c))
        string_representation = (string_representation[::-1])
    return string_representation


def prime_base_to_decimal(num, base):
    """
    ===============================================================================================================
    Function converts a fractional number "num" in base "base" to base 10. Reverses the process in base_conversion
    Note: The first string element is ignored, since this would be zero for a fractional number.
        Args:
            num(list): Number in base b to be converted. The number must be represented as a list containing individual digits of the base, with the first entry as zero.
            b(int): Original base
        Returns:
            decimal_equivalent(float): Fractional number in base 10
    Examples:
    Convert 0.01 (base 2) to base 10
        >>  prime_base_to_decimal(['0', '0', '1'], 2)  # Represents 0.01 in base 2
        >> 0.25
    Convert 0.01 (base 20) to base 10
        >>  prime_base_to_decimal(['0', '0', '1'], 20)  # Represents 0.01 in base 20
        >> 0.0025
    ================================================================================================================
    """
    binary = num
    decimal_equivalent = 0
    # Convert fractional part decimal equivalent
    for i in range(1, len(binary)):
        decimal_equivalent += int(binary[i]) / (base ** i)
    return decimal_equivalent


def hammersley(no_samples, prime_base):
    """
    ===============================================================================================================
    Function which generates the first no_samples elements of the Halton or Hammersley sequence based on the prime number prime_base
    The steps for generating the first no_samples of the sequence are as follows:
    1. Create a list of numbers between 0 and no_samples --- nums = [0, 1, 2, ..., no_samples]
    2. Convert each element in nums into its base form based on the prime number prime_base, reverse the base digits of each number in num
    3. Add a decimal point in front of the reversed number
    4. Convert the reversed numbers back to base 10
        Args:
            no_samples(int): Number of Halton/Hammersley sequence elements required
            prime_base(int): Current prime number to be used as base
        Returns:
            sequence_decimal(NumPy Array): 1-D array containing the first no_samples elements of the sequence based on prime_base
    Examples:
    First three elements of the Halton sequence based on base 2
        >>  data_sequencing(self, 3, 2)
        >> [0, 0.5, 0.75]
    ================================================================================================================
    """
    pure_numbers = np.arange(0, no_samples)
    bitwise_rep = []
    reversed_bitwise_rep = []
    sequence_bitwise = []
    sequence_decimal = np.zeros((no_samples, 1))
    for i in range(0, no_samples):
        base_rep = base_conversion(pure_numbers[i], prime_base)
        bitwise_rep.append(base_rep)
        reversed_bitwise_rep.append(base_rep[::-1])
        sequence_bitwise.append(['0.'] + reversed_bitwise_rep[i])
        sequence_decimal[i, 0] = prime_base_to_decimal(sequence_bitwise[i], prime_base)
    sequence_decimal = sequence_decimal.reshape(sequence_decimal.shape[0], )
    return sequence_decimal


def halton(n, b):
    m, d = 0, 1
    samples = np.zeros(n)
    for i in range(n):
        x = d - m
        if x == 1:
            m = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            m = (b + 1) * y - x
        samples[i] = m / d
    return samples


def halton2d(n):
    hamx = halton(n, 3)
    hamy = halton(n, 2)
    return np.concatenate([hamx, hamy], axis=0).reshape(2, -1)


from matplotlib import pyplot as plt

# n = 2132
# ham = hammersley_plane(n)
#
# plt.plot(ham[..., 0], ham[..., 1], 'o', markersize=0.75)
# plt.show()

n = 3000
ham = halton2d(n)

print(ham.shape)

ham1 = ham[0]
ham2 = ham[1]

ham11 = ham1[:n//3]
ham12 = ham2[:n//3]

ham21 = ham1[n//3:2*n//3]
ham22 = ham2[n//3:2*n//3]

ham31 = ham1[2*n//3:]
ham32 = ham2[2*n//3:]

plt_figure = lambda: plt.gca().set_aspect(1)

plt.figure()
plt_figure()
plt.plot(ham11, ham12, 'ro', markersize=0.5)

plt.figure()
plt_figure()
plt.plot(ham21, ham22, 'go', markersize=0.5)

plt.figure()
plt_figure()
plt.plot(ham31, ham32, 'bo', markersize=0.5)

r = np.random.rand(n//3, 2)
plt.figure()
plt_figure()
plt.plot(r[..., 0], r[..., 1], 'ko', markersize=0.5)

plt.show()

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as stats
import pytest

from . import distributions
from . import generators
from .utils import (
    all_supported_configs,
    generate_random_numbers,
    per_thread_skipahead,
    per_thread_skipahead_sequence,
    prepare_states,
    prepare_states_and_generate,
)

"""
This set of tests checks random device APIs.

Running the tests requires compiling a setup kernel for every generator, and random number
generation kernel for all combinations of generator, distribution, dtype and group size. As
a result, due to lazy jit compilation, the tests will be running slower at the beginning and
then become much faster.
"""


@pytest.mark.parametrize("nsamples", (200,))
@pytest.mark.parametrize("threads,blocks", ((15, 17),))
@pytest.mark.parametrize(
    "distribution,dtype_name,group_size,generator",
    all_supported_configs(
        distributions.Poisson(1),
        distributions.Poisson(10),
        distributions.Poisson(15),
        distributions.LogNormal(),
        distributions.LogNormal(-2, 0.5),
        distributions.Uniform(),
        distributions.Normal(),
    ),
)
def test_distribution(distribution, dtype_name, generator, nsamples, threads, blocks, group_size):
    """
    Tests if the generated distribution is as expected.
    """
    generated = prepare_states_and_generate(
        distribution=distribution,
        dtype_name=dtype_name,
        generator=generator,
        nsamples=nsamples,
        seed=12345,
        threads=threads,
        blocks=blocks,
        group_size=group_size,
    )
    if distribution.is_discrete():

        def test(x):
            # Chi2 test
            bin_edges = distribution.ppf(np.linspace(0, 1, 10)) + 1
            bin_edges[bin_edges == np.inf] = np.max(x) + 1
            bin_edges[bin_edges == -np.inf] = np.min(x) - 1
            bin_edges = np.unique(bin_edges.astype(int))
            freq, _ = np.histogram(x, bin_edges)
            expected_freq = distribution.cdf(bin_edges[1:] - 0.5) - distribution.cdf(bin_edges[:-1] - 0.5)
            expected_freq *= len(x)
            expected_freq[-1] = len(x) - np.sum(expected_freq[:-1])
            return stats.chisquare(freq, expected_freq).pvalue

    else:
        # Kolmogorov-Smirnov test
        test = lambda x: stats.kstest(x, distribution.cdf).pvalue

    # We assess the match both for values generated by a single thread (sequence_match)
    # and for values generated by multiple threads (grid_match).
    sequence_match = stats.combine_pvalues(np.apply_along_axis(test, 0, generated))
    grid_match = stats.combine_pvalues(np.apply_along_axis(test, 1, generated))

    # We use p-value 0.005 to reduce the amount of false positives.
    print("p-values", sequence_match.pvalue, grid_match.pvalue)
    assert sequence_match.pvalue > 0.005
    assert grid_match.pvalue > 0.005


@pytest.mark.parametrize("nsamples", (20,))
@pytest.mark.parametrize("threads,blocks", ((38, 2), (1, 1)))
@pytest.mark.parametrize(
    "distribution,dtype_name,group_size,generator",
    all_supported_configs(
        distributions.Poisson(12), distributions.LogNormal(2, 1.1), distributions.Uniform(), distributions.Normal()
    ),
)
def test_seeds(distribution, dtype_name, generator, nsamples, threads, blocks, group_size):
    """
    Tests if seeding works, i.e. same seeds results in same sequences and different seeds
    result in different sequences.
    """

    def generate_with_seed(seed):
        return prepare_states_and_generate(
            distribution=distribution,
            dtype_name=dtype_name,
            generator=generator,
            nsamples=nsamples,
            seed=seed,
            threads=threads,
            blocks=blocks,
            group_size=group_size,
        )

    a1 = generate_with_seed(123)
    b = generate_with_seed(456)
    a2 = generate_with_seed(123)

    assert np.all(a1 == a2)
    if not distribution.is_discrete():
        assert np.all(a1 != b)  # Accidental equality of two floats is almost impossible
    else:
        assert np.sum(a1 != b) > a1.size // 2


@pytest.mark.parametrize(
    "threads,blocks",
    (
        (3, 4),
        (67, 3),
        (1, 200),
    ),
)
@pytest.mark.parametrize(
    "generator",
    [pytest.param(g(), id=g().name()) for g in generators.GENERATORS if g().supports_skipahead()],
)
def test_skipahead(generator, threads, blocks):
    """
    Tests if seeding works, i.e. same seeds results in same sequences and different seeds
    result in different sequences.
    """

    seed = 765
    nthreads = threads * blocks
    states1 = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks)
    states2 = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks)
    states3 = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks, offset=17)

    def gen(states, n):
        return generate_random_numbers(
            states=states,
            distribution=distributions.Uniform(),
            dtype_name="float",
            nsamples=n,
            threads=threads,
            blocks=blocks,
        )

    def skip(states, ns):
        return per_thread_skipahead(blocks=blocks, threads=threads, states=states, ns=ns)

    def gen_all():
        return (
            gen(states1, 1).flatten(),
            gen(states2, 1).flatten(),
            gen(states3, 1).flatten(),
        )

    a1, a2, a3 = gen_all()  # Positions: (0, 0, 17)
    assert np.all(a1 == a2)
    assert np.all(a1 != a3)

    gen(states1, 17)
    b1, b2, b3 = gen_all()  # Positions: (17, 0, 17)
    assert np.all(b1 == b3)

    even_mask = np.arange(nthreads) % 2 == 0

    skip(states2, even_mask * 17)  # Advance even threads in states2
    c1, c2, c3 = gen_all()  # Positions: (17, 17|0, 17)
    assert np.all(c1 == c3)
    assert np.all(c2[even_mask] == c1[even_mask])
    assert np.all(c2[~even_mask] == c2[~even_mask])

    skip(states2, (1 - even_mask) * 17)  # Advance odd threads in states2
    d1, d2, d3 = gen_all()  # Positions: (17, 17, 17)
    assert np.all(d1 == d3)
    assert np.all(d1 == d2)

    skip(states1, np.ones(nthreads))
    skip(states2, np.ones(nthreads))
    gen(states3, 1)
    e1, e2, e3 = gen_all()  # Positions: (18, 18, 18)
    assert np.all(e1 == e3)
    assert np.all(e1 == e2)


@pytest.mark.parametrize(
    "threads,blocks",
    (
        (2, 2),
        (12, 30),
        (256, 1),
    ),
)
@pytest.mark.parametrize(
    "generator",
    [pytest.param(g(), id=g().name()) for g in generators.GENERATORS if g().supports_skipahead_subsequence()],
)
def test_skipahead_sequence(generator, threads, blocks):
    """
    Tests if seeding works, i.e. same seeds results in same sequences and different seeds
    result in different sequences.
    """

    seed = 100
    nthreads = threads * blocks
    states1 = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks)
    states2 = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks)

    def gen(states):
        return generate_random_numbers(
            states=states,
            distribution=distributions.Uniform(),
            dtype_name="double",
            nsamples=1,
            threads=threads,
            blocks=blocks,
        ).flatten()

    def gen_all():
        return gen(states1), gen(states2)

    def skip_sequence(states, ns):
        return per_thread_skipahead_sequence(generator=generator, blocks=blocks, threads=threads, states=states, ns=ns)

    a1, a2 = gen_all()
    assert np.all(a1 == a2)  # (0:0, 1:0, 2:0, ...) (0:0, 1:0, 2:0, ...)

    skip_sequence(states2, np.ones(nthreads))
    b1, b2 = gen_all()
    assert np.all(b1 != b2)  # (0:1, 1:1, 2:1, ...) (1:1, 2:1, 1:1, ...)
    assert np.all(b1[1:] == b2[:-1])

    skip_sequence(states1, nthreads - np.arange(nthreads))
    skip_sequence(states2, nthreads - np.arange(nthreads) - 1)
    c1, c2 = gen_all()
    assert np.all(c1 == c2)  # (n:2, n:2, n:2, ...) (n:2, n:2, n:2, ...)

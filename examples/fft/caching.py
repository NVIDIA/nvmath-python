# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ['fft']

import logging

import nvmath

# A simple class to allow caching and resource management
class FFTCache(dict):
    def free(self):
        """Release all resources owned in the cache"""
        for fft_obj in self.values():
            fft_obj.free()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.free()

# A simple illustration of creating and using a cached FFT operation.
def fft(a, axes=None, direction=None, options=None, prolog=None, epilog=None, stream=None, cache={}):
    """
    A cached version of FFT, taking a cache argument in addition the the regular arguments for fft(). The stateful
    objects are cached in the provided cache, and reused.

    Args:
        cache: an object to use as the cache that satisfies `typing.Mapping` concept.

    Note:
        User is responsible for explicitly free all resources stored in `cache` after no longer needed.
        If a native `dict` object is used to store the cache, the resources can be released via:

        >>> for f in cache.values():
        >>>    f.free()

        Alternatively, users may use the `FFTCache` class above.
        Resources can be cleaned by a call the the `free` method or will be automatically released if used in a context manager.
    """
    logger = logging.getLogger()

    package = stream.__class__.__module__.split('.')[0]
    stream_ptr = stream.ptr if package == 'cupy' else stream.cuda_stream if package == 'torch' else stream

    key = nvmath.fft.FFT.create_key(a, axes=axes, options=options, prolog=prolog, epilog=epilog)

    # Get object from cache if it already exists, or create a new one and add it to the cache.
    if (key, stream_ptr) in cache:
        logger.info("Cache HIT: using planned object.")
        # The planned object is already cached, so retrieve it.
        f = cache[key, stream_ptr]
        # Set new operand in object.
        f.reset_operand(a, stream=stream)
    else:
        # Create a new stateful object, plan the operation, and cache the  object.
        f = cache[key, stream_ptr] = nvmath.fft.FFT(a, axes=axes, options=options, stream=stream)
        f.plan(prolog=prolog, epilog=epilog, stream=stream)
        logger.info("Cache MISS: creating and caching a planned FFT object.")

    # Execute the FFT on the cached object.
    r = f.execute(direction=direction, stream=stream)

    # Reset operand to None to discard internal reference, allowing memory to be recycled.
    f.reset_operand(stream=stream)

    return r

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of functions for parsing Einsum expressions.
"""

from collections import Counter
from itertools import chain
from functools import lru_cache

from ...internal import formatters, tensor_wrapper
from .data import WHITESPACE_UNICODE


DISALLOWED_LABELS = {".", "-", ">"}
native_to_str = lambda native: "'" + "".join(s if s is not Ellipsis else "..." for s in native) + "'"


def select_morpher(mapper=None):
    """
    Select appropriate function for mode label representation based on string format.
    """
    if mapper is None:
        return native_to_str

    return lambda s: native_to_str(mapper(s))


class ModeLabelMapper:
    """
    Map mode labels, with special treatment for Ellipsis characters.
    """

    def __init__(self, _map):
        """
        Args:
            _map = dict-like object to map mode labels.
        """
        self._map = _map

    def __call__(self, sequence):
        return tuple(s if s is Ellipsis else self._map[s] for s in sequence)


def parse_single(single):
    """
    Parse single operand mode labels considering ellipsis.
    Leading or trailing whitespace, if present, is removed.
    """
    whitespace = WHITESPACE_UNICODE
    subexpr = single.strip(whitespace).split("...")
    n = len(subexpr)
    expr = [[Ellipsis]] * (2 * n - 1)
    expr[::2] = subexpr

    return tuple(chain(*expr))


def check_single(single):
    """
    Check for disallowed characters used as mode labels for a single operand.
    """
    whitespace = WHITESPACE_UNICODE
    for s in single:
        if s is Ellipsis:
            continue
        if s in whitespace or s in DISALLOWED_LABELS:
            return False

    return True


@lru_cache(maxsize=128)
def parse_einsum_str(expr):
    """
    Parse einsum expression in string format, retaining ellipses if present.

    Return operand as well as output mode labels if explicit form or None for
    implicit form.
    """
    inputs, output, *rest = expr.split("->") if "->" in expr else (expr, None)
    if rest:
        raise ValueError("""Invalid expression.
It is not permitted to specify more than one '->' in the Einstein summation
expression.""")

    inputs = list(parse_single(_input) for _input in inputs.split(","))  # noqa: C400
    if output is not None:
        output = parse_single(output)

    checks = [check_single(_input) for _input in inputs]
    if not all(checks):
        incorrect = [
            f"{location}: {native_to_str(inputs[location])}" for location, predicate in enumerate(checks) if predicate is False
        ]
        incorrect = formatters.array2string(incorrect)
        message = f"""Incorrect term.
Whitespace characters and characters from the set {DISALLOWED_LABELS} cannot be
used as mode labels in a summation expression.
The incorrectly specified terms as a sequence of "position: term" are:
\n{incorrect}"""
        raise ValueError(message)

    return inputs, output


def check_ellipses(user_inputs, morpher):
    """
    Check ellipsis specification for validity.

    Args:
        user_inputs: Einsum expression in "neutral format"
            (sequence of sequences) before mapping.
        morpher: A callable that transforms a term in neutral format
            (sequence) to string format.
    """

    checks = [user_input.count(Ellipsis) <= 1 for user_input in user_inputs]
    if not all(checks):
        incorrect = [
            f"{location}: {morpher(user_inputs[location])}" for location, predicate in enumerate(checks) if predicate is False
        ]
        incorrect = formatters.array2string(incorrect)
        message = f"""Incorrect ellipsis use.
There must not be more than one ellipsis present in each term.
The incorrectly specified terms as a sequence of "position: term" are:
\n{incorrect}"""
        raise ValueError(message)


def check_einsum_with_operands(user_inputs, operands, morpher):
    """
    Check that the number of modes in each Einsum term is consistent
    with the shape of the corresponding operand.

    Args:
        operands: Wrapped operands.
        user_inputs: Einsum expression in "neutral format"
            (sequence of sequences) before mapping.
        morpher: A callable that transforms a term in neutral format
            (sequence) to string format.
    """

    checks = [
        len(i) - 1 <= len(o.shape) if Ellipsis in i else len(i) == len(o.shape)
        for i, o in zip(user_inputs, operands, strict=False)
    ]
    if not all(checks):
        mismatch = [
            f"{location}: {morpher(user_inputs[location])} <=> {operands[location].shape}"
            for location, predicate in enumerate(checks)
            if predicate is False
        ]
        mismatch = formatters.array2string(mismatch)
        message = f"""Term-operand shape mismatch.
The number of mode labels in each term of the expression must match the shape
of the corresponding operand.
The mismatch as a sequence of "position: mode labels in term <=> operand shape"
is: \n{mismatch}"""
        raise ValueError(message)


def map_modes(user_inputs, user_output, num_extra_labels, morpher):
    """
    Map modes in user-defined inputs and output to ordinals, leaving ellipsis
    for later processing. Create extra mode labels in anticipation of ellipsis
    replacement. Create the forward as well as inverse maps.

    Args:
        user_inputs: Einsum expression in "neutral format"
            (sequence of sequences) before mapping.
        user_output: The output mode labels before mapping as a sequence or None.
        num_extra_labels: The number of extra mode labels to generate
            to use in ellipsis expansion later.
        morpher: A callable that transforms a term in neutral format
            (sequence) to string format.

    Returns:
        tuple: A 5-tuple containing (mapped input, mapped output, forward map,
            reverse map, largest label).
    """

    ordinal = 0
    mode_map_user_to_ord = {}
    for modes in user_inputs:
        for mode in modes:
            if mode not in mode_map_user_to_ord:
                mode_map_user_to_ord[mode] = ordinal
                ordinal += 1

    mode_map_user_to_ord.update((f"__{i}__", i) for i in range(ordinal, ordinal + num_extra_labels))
    label_end = ordinal + num_extra_labels

    mode_map_ord_to_user = {v: k for k, v in mode_map_user_to_ord.items()}

    inputs = list(tuple(m if m is Ellipsis else mode_map_user_to_ord[m] for m in modes) for modes in user_inputs)  # noqa: C400

    output = None
    if user_output is not None:
        extra = set(user_output) - set(mode_map_user_to_ord.keys()) - {Ellipsis}
        if extra:
            output_modes = morpher(user_output)
            message = f"""Extra modes in output.
The specified output modes {output_modes} contain the extra modes: {extra}"""
            raise ValueError(message)
        output = tuple(m if m is Ellipsis else mode_map_user_to_ord[m] for m in user_output)

    return inputs, output, mode_map_user_to_ord, mode_map_ord_to_user, label_end


def create_size_dict(inputs, operands):
    """
    Create size dictionary (mode label to extent map) capturing the extent of
    each mode.

    Args:
        inputs: Einsum expression in "neutral format" (sequence of sequences)
            after relabelling modes.
        operands: Wrapped operands.

    Returns:
        size_dict: size dictionary.
    """

    size_dict = {}
    for i, _input in enumerate(inputs):
        for m, mode in enumerate(_input):
            shape = operands[i].shape
            if mode in size_dict:
                if size_dict[mode] == 1:  # Handle broadcasting
                    size_dict[mode] = shape[m]
                elif size_dict[mode] != shape[m] and shape[m] != 1:
                    message = f"""Extent mismatch.
The extent ({shape[m]}) of mode {m} for operand {i} does not match the extent
({size_dict[mode]}) of the same mode found in previous operand(s)."""
                    raise ValueError(message)
            else:
                size_dict[mode] = shape[m]

    return size_dict


def infer_output_mode_labels(inputs, mode_map_ord_to_user=None):
    """
    Infer output mode labels (those that appear exactly once).

    Args:
        inputs: Einsum expression in "neutral format" (sequence of sequences).
            If `mode_map_ord_to_user` is provided, the mode labels correspond to
            ordinals, otherwise they correspond to user labels.
        mode_map_ord_to_user: the map from ordinals to user labels.
    """
    mode_label_freq = Counter(chain(*inputs))
    del mode_label_freq[Ellipsis]

    key = None if mode_map_ord_to_user is None else lambda m: mode_map_ord_to_user[m]
    return tuple(sorted((m for m, c in mode_label_freq.items() if c == 1), key=key))


def process_ellipses(inputs, output, operands, label_end, mode_map_ord_to_user, mapping_morpher):
    """
    Replace ellipses by generated mode labels, using 'label_end' and aligning
    shapes from the right. Infer or update output mode labels.

    Args:
        inputs: Einsum expression in "neutral format" (sequence of sequences)
            after relabelling modes.
        output: The output mode labels after relabelling as a sequence or None.
        operands: Wrapped operands.
        label_end: One past the largest mode label (int), including modes
            resulting from Ellipsis expansion.
        mode_map_ord_to_user: the map from ordinals to user labels.
        mapping_morpher: A callable that transforms a term in neutral format
            (sequence) to string format, while converting internal labels to
            user labels.

    Returns:
        tuple: a 2-tuple (inputs, output) after ellipsis expansion and inferring
            output mode labels if needed.
    """

    inferred = False
    if output is None:
        output = infer_output_mode_labels(inputs, mode_map_ord_to_user)
        inferred = True

    shortest, longest = label_end, 0
    for i, _input in enumerate(inputs):
        if Ellipsis not in _input:
            continue

        n = len(operands[i].shape) - (len(_input) - 1)
        assert n >= 0, "Internal error"

        s = _input.index(Ellipsis)
        shortest, longest = min(shortest, n), max(longest, n)
        inputs[i] = _input[:s] + tuple(range(label_end - n, label_end)) + _input[s + 1 :]

    if not inferred:
        count = output.count(Ellipsis)
        if count > 1:
            message = f"""Incorrect ellipsis use.
The output term cannot have more than one ellipsis.
Specified term = {mapping_morpher(output)}"""
            raise ValueError(message)
        if count == 1:  # Replace ellipsis by the longest sequence of labels.
            s = output.index(Ellipsis)
            output = output[:s] + tuple(range(label_end - longest, label_end)) + output[s + 1 :]
        else:  # If all ellipses expand to the same number of mode labels, the latter are reduced.
            if shortest != longest:
                message = f"""Ellipsis length mismatch for reduction.
The ellipses specified in the expression do not expand to the same number of
mode labels and thus cannot be reduced. The expanded number of dimensions
ranges from {shortest} to {longest}."""
                raise ValueError(message)
    else:  # The mode labels corresponding to ellipsis expansion followed by the inferred mode labels.
        output = tuple(range(label_end - longest, label_end)) + output

    return inputs, output


def parse_einsum(expr, *operands):
    """
    Parse the generalized Einstein summation expression in string format. Any
    hashable and comparable object is accepted for mode label specification, and
    unicode strings are accepted. If the output is not provided (implicit form
    or missing output sublist), it will be inferred from the expression.

    Returns wrapped operands, mapped inputs and output, size dictionary based on
    internal mode numbers, and the forward as well as the reverse mode maps.
    """

    # Parse einsum keeping ellipses.

    inputs, output = parse_einsum_str(expr)

    num_operand, num_input = len(operands), len(inputs)
    if num_operand != num_input:
        message = f"""Operand-term mismatch.
The number of operands ({num_operand}) must match the number of inputs
({num_input}) specified in the Einsum expression."""
        raise ValueError(message)

    morpher = select_morpher()

    if num_operand < 1:
        message = f"""The number of operands must be either 2 (binary contraction) or 3 (ternary contraction),
found {num_operand}."""
        raise ValueError(message)

    # First wrap operands.
    operands = tensor_wrapper.wrap_operands(operands)

    # Preliminary checks, before mode label remapping.

    ellipses = any(Ellipsis in _input for _input in inputs)

    # Ensure at most one ellipsis per operand.
    if ellipses:
        check_ellipses(inputs, morpher)

    # Ensure that ellipsis is not present only in the output.
    if not ellipses and output is not None and Ellipsis in output:
        message = f"""Invalid ellipsis specification.
The output term {morpher(output)} contains ellipsis while none of the input
terms do."""
        raise ValueError(message)

    # Ensure that the number of modes is consistent with the operand shape.
    check_einsum_with_operands(inputs, operands, morpher)

    # Calculate the maximum number of extra mode labels that will be needed.
    num_extra_labels = max(len(o.shape) for o in operands) if ellipses else 0

    # Map data to ordinals for cutensornet.
    inputs, output, mode_map_user_to_ord, mode_map_ord_to_user, label_end = map_modes(inputs, output, num_extra_labels, morpher)

    has_user_output = output is not None

    mapper = ModeLabelMapper(mode_map_ord_to_user)
    mapping_morpher = select_morpher(mapper)

    # Ellipsis expansion.
    if ellipses:
        inputs, output = process_ellipses(inputs, output, operands, label_end, mode_map_ord_to_user, mapping_morpher)
    elif output is None:
        output = infer_output_mode_labels(inputs, mode_map_ord_to_user)

    # Create mode-extent map based on internal mode numbers.
    size_dict = create_size_dict(inputs, operands)

    return operands, inputs, output, has_user_output, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, ellipses


def parse_elementary_einsum(inputs, output, a, b, c):
    morpher = select_morpher()

    # Preliminary checks, before mode label remapping.
    ellipses = any(Ellipsis in _input for _input in inputs)

    # Ensure at most one ellipsis per operand.
    if ellipses:
        check_ellipses(inputs, morpher)

    # Ensure that ellipsis is not present only in the output.
    if not ellipses and output is not None and Ellipsis in output:
        message = f"""Invalid ellipsis specification.
The output term {morpher(output)} contains ellipsis while none of the input
terms do."""
        raise ValueError(message)

    # Ensure that the number of modes is consistent with the operand shape.
    num_inputs = len(inputs)
    if num_inputs == 2:
        operands = [a, b]
    elif num_inputs == 3:
        operands = [a, b, c]
    else:
        raise ValueError(f"Invalid number of inputs: {num_inputs}")

    check_einsum_with_operands(inputs, operands, morpher)

    # Calculate the maximum number of extra mode labels that will be needed.
    num_extra_labels = max(len(o.shape) for o in operands) if ellipses else 0

    # Map data to ordinals for cutensornet.
    inputs, output, mode_map_user_to_ord, mode_map_ord_to_user, label_end = map_modes(inputs, output, num_extra_labels, morpher)

    has_user_output = output is not None

    mapper = ModeLabelMapper(mode_map_ord_to_user)
    mapping_morpher = select_morpher(mapper)

    # Ellipsis expansion.
    if ellipses:
        inputs, output = process_ellipses(inputs, output, operands, label_end, mode_map_ord_to_user, mapping_morpher)
    elif output is None:
        output = infer_output_mode_labels(inputs, mode_map_ord_to_user)

    # Create mode-extent map based on internal mode numbers.
    size_dict = create_size_dict(inputs, operands)

    return inputs, output, has_user_output, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, ellipses

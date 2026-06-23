"""Reading and turning the safety dial.

One linear direction in a small instruct model's residual stream that both
*reads* (predicts) and *writes* (controls) the model's decision to refuse,
measured across several models and safeguard domains.

The package is organised as a resumable pipeline:

    extraction -> generation -> judging -> metrics -> figures

The numeric core (``stats``, ``extraction``, ``monitor``, ``metrics``) is pure
NumPy and unit-tested without a GPU. The ``model`` and ``judge`` modules touch
the GPU and the Anthropic API respectively.
"""

__version__ = "0.1.0"

{# Autosummary template for TorchUST.

   - Only from_torch is documented because it is the primary public entry point;
     other inherited members are internal or covered elsewhere.
#}
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-inherited-members:
   :members: from_torch

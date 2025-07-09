.. module:: nvmath.bindings.cudss

cuDSS (:mod:`nvmath.bindings.cudss`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuDSS documentation
<https://docs.nvidia.com/cuda/cudss/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

    AlgType
    ConfigParam
    cuDSSError
    DataParam
    IndexBase
    Layout
    MatrixFormat
    MatrixType
    MatrixViewType
    OpType
    Phase
    PivotType
    Status

Functions
*********

.. autosummary::
   :toctree: generated/

    check_status
    config_create
    config_destroy
    config_get
    config_set
    create
    data_create
    data_destroy
    data_get
    data_set
    destroy
    execute
    get_config_param_dtype
    get_data_param_dtype
    get_device_mem_handler
    get_property
    matrix_create_batch_csr
    matrix_create_batch_dn
    matrix_create_csr
    matrix_create_dn
    matrix_destroy
    matrix_get_batch_csr
    matrix_get_batch_dn
    matrix_get_csr
    matrix_get_dn
    matrix_get_format
    matrix_set_batch_csr_pointers
    matrix_set_batch_values
    matrix_set_csr_pointers
    matrix_set_values
    set_comm_layer
    set_device_mem_handler
    set_stream
    set_threading_layer

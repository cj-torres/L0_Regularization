# L0 Regularizer
PyTorch adaptation of Louizos (2017) to wrap PyTorch modules.

This code provides a method for wrapping an existing PyTorch module and regularizing it according to the L0 norm. It is an adaptation and extension of code from the repository [here](https://github.com/AMLab-Amsterdam/L0_regularization). The code works in the following manner:

1. The supplied module is copied.
2. The copied module has its existing parameters copied into a new parameter dictionary.
3. Parameters used for masking are generated for each copied parameter in the dictionary.
4. The parameters are then deleted from the copied module.
5. During a forward pass the parameters are masked and then plugged into the module as regular tensors.

This method allows for the generic adaptation of any PyTorch module, though it may brook further performance improvements.

An important divergence from Louizos that allows it to be generalized is that each parameter is optimized separately. In other words, it optimizes for connection, not neural, dropout. It also suffers from an inability to do large batches while also sampling each dropout independently. Nevertheless, the code has proven useful for analyzing the computational complexity of different data given an architecture.

I plan to revisit this code in the future and implement some or all of the following:
* A method to exclude parameters from regularization
* A method allowing batch processing with independent sampling
* Support for initializing parameters with different distributions


# Using

Initializing requires a `module` of type `torch.nn.Module` and `lam` of type `float`, the module is used as explained above, and `lam` is the L0 regularization constant. Optional parameters include `weight_decay` (L2 regularization constant) and `droprate_init` (likelihood of masking at initialization). Other parameters are detailed in Louizos (2017) but have to do with the dropout sampling method and shouldn't need tweaking.

After initialization use your new module in the following way:

* `.forward()` should function as your module did initially
* `.regularization()` will return the combined L0 and L2 norm (avoid using L2 norm of existing PyTorch optimizers)
* `.count_l0` will return the expected value of the number of retained parameters during a forward pass
* `.count_l2` will return the expected cost of encoding the parameters (sum of squares of expected values AFTER masking)

The last two are useful for measurements, but `.regularization()` is the backprop supporting function.

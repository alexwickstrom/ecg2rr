from scipy import sparse 
import numpy as np

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms
def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))

def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=None, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.
        .. versionadded:: 0.20
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.
        For object dtyped data, only np.nan is checked and not np.inf.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.
    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.
        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
            removed in 0.23.
    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.
    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # warn_on_dtype deprecation
    if warn_on_dtype is not None:
        warnings.warn(
            "'warn_on_dtype' is deprecated in version 0.21 and will be "
            "removed in 0.23. Don't set `warn_on_dtype` to remove this "
            "warning.",
            DeprecationWarning, stacklevel=2)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        dtypes_orig = np.array(array.dtypes)
        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*array.dtypes)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""
    # @@@ don't need; also provided by scipy (OK)
    # commented out if issparse(), since we will only use full arrays

    # If np.array(..) gives ComplexWarning, then we convert the warning
    # to an error. This is needed because specifying a non complex
    # dtype to the function converts complex to real dtype,
    # thereby passing the test made in the lines following the scope
    # of warnings context manager.
    with warnings.catch_warnings(): # OK@@@@@@@
        try:
            warnings.simplefilter('error', ComplexWarning)
            if dtype is not None and np.dtype(dtype).kind in 'iu':
                # Conversion float -> int should not contain NaN or
                # inf (numpy#14412). We cannot use casting='safe' because
                # then conversion float -> int would be disallowed.
                array = np.asarray(array, order=order)
                if array.dtype.kind == 'f':
                    _assert_all_finite(array, allow_nan=False,
                                       msg_dtype=dtype)
                array = array.astype(dtype, casting="unsafe", copy=False)
            else:
                array = np.asarray(array, order=order, dtype=dtype)
        except ComplexWarning:
            raise ValueError("Complex data not supported\n"
                             "{}\n".format(array))

    # It is possible that the np.array(..) gave no warning. This happens
    # when no dtype conversion happened, for example dtype = None. The
    # result is that np.array(..) produces an array of complex dtype
    # and we need to catch and raise exception for such cases.
    _ensure_no_complex_data(array) # @@@@@@@@@@@@@@@ OK

    if ensure_2d:
        # If input is scalar raise error
        if array.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(array))
        # If input is 1D raise error
        if array.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(array))

    # in the future np.flexible dtypes will be handled like object dtypes
    if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
        warnings.warn(
            "Beginning in version 0.22, arrays of bytes/strings will be "
            "converted to decimal numbers if dtype='numeric'. "
            "It is recommended that you convert the array to "
            "a float dtype before using it in scikit-learn, "
            "for example by using "
            "your_array = your_array.astype(np.float64).",
            FutureWarning, stacklevel=2)

    # make sure we actually converted to numeric:
    if dtype_numeric and array.dtype.kind == "O":
        array = array.astype(np.float64)
    if not allow_nd and array.ndim >= 3:
        raise ValueError("Found array with dim %d. %s expected <= 2."
                         % (array.ndim, estimator_name))
    # @@@@@@@@@@@@@@@  ????
    if force_all_finite:
        _assert_all_finite(array,
                           allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning, stacklevel=2)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array

# sklearn preprocessing
def normalize(X, norm='l2', axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).
    Read more in the :ref:`User Guide <preprocessing_normalization>`.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.
    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).
    return_norm : boolean, default False
        whether to return the computed norms
    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.
    norms : array, shape [n_samples] if axis=1 else [n_features]
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.
    See also
    --------
    Normalizer: Performs normalization using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)
    # @@@@@@@@@@@@@@@@@@@@@  OK?? (_ensure_no_complex_data)
    X = check_array(X, sparse_format, copy=copy,
                    estimator='the normalize function', dtype=FLOAT_DTYPES)
    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        if return_norm and norm in ('l1', 'l2'):
            raise NotImplementedError("return_norm=True is not implemented "
                                      "for sparse matrices with norm 'l1' "
                                      "or norm 'l2'")
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
        elif norm == 'max':
            _, norms = min_max_axis(X, 1)
            norms_elementwise = norms.repeat(np.diff(X.indptr))
            mask = norms_elementwise != 0
            X.data[mask] /= norms_elementwise[mask]
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)
        elif norm == 'l2':
            norms = row_norms(X)
        elif norm == 'max':
            norms = np.max(X, axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
    	return X

def optimize_memory_usage(data):
    """
    Optimize memory usage by converting data types where applicable.

    Args:
        data (numpy.ndarray): The input data array to optimize.

    Returns:
        numpy.ndarray: The optimized data array.
    """
    # Example optimization: Convert float64 to float32 if applicable
    if data.dtype == np.float64:
        return data.astype(np.float32)
    return data


def efficient_data_handling(data, max_samples=None):
    """
    Handle data efficiently by limiting the number of samples processed.

    Args:
        data (numpy.ndarray): The input data array.
        max_samples (int, optional): The maximum number of samples to process.

    Returns:
        numpy.ndarray: The processed data array.
    """
    if max_samples is not None and data.shape[0] > max_samples:
        return data[:max_samples]
    return data


def clear_memory():
    """
    Clear unused variables and force garbage collection to free memory.
    """
    import gc
    gc.collect()
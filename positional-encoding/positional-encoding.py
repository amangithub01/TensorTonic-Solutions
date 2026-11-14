import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len, dtype=float)[:, np.newaxis]

    two_i = np.arange(0, d_model, 2, dtype=float)[np.newaxis, :]

    div_term = np.power(base, two_i / d_model)

    angles = pos / div_term


    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles)


    pe[:, 1::2] = np.cos(angles[:, : d_model // 2])

    return pe
    # pass
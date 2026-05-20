import numpy as np
from scipy.fft import dctn, idctn


QUANT_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float32)

QUANT_MATRIX_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_quant_matrix(channel: str, qf: float = 1.0) -> np.ndarray:

    base = QUANT_MATRIX_LUMA if channel == 'Y' else QUANT_MATRIX_CHROMA
    return np.maximum(1, np.round(base * qf)).astype(np.float32)




def dct_block(block: np.ndarray) -> np.ndarray:
    centered = block.astype(np.float32) - 128.0
    return dctn(centered, norm='ortho')


def idct_block(dct_coeffs: np.ndarray) -> np.ndarray:

    pixels = idctn(dct_coeffs, norm='ortho')
    return np.clip(pixels + 128.0, 0, 255)


def quantize(dct_coeffs: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
   
    return np.round(dct_coeffs / quant_matrix).astype(np.int16)


def dequantize(quantized: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
 
    return (quantized * quant_matrix).astype(np.float32)




def encode_channel(channel: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    H, W = channel.shape

    H_pad = (H + 7) // 8 * 8
    W_pad = (W + 7) // 8 * 8
    padded = np.zeros((H_pad, W_pad), dtype=np.float32)
    padded[:H, :W] = channel

    nb_h = H_pad // 8
    nb_w = W_pad // 8
    coeffs = np.zeros((nb_h, nb_w, 8, 8), dtype=np.int16)

    for bh in range(nb_h):
        for bw in range(nb_w):
            block = padded[bh*8:(bh+1)*8, bw*8:(bw+1)*8]
            dct_c = dct_block(block)
            coeffs[bh, bw] = quantize(dct_c, quant_matrix)

    return coeffs


def decode_channel(coeffs: np.ndarray, quant_matrix: np.ndarray,
                   original_H: int, original_W: int) -> np.ndarray:

    nb_h, nb_w = coeffs.shape[:2]
    H_pad = nb_h * 8
    W_pad = nb_w * 8
    reconstructed = np.zeros((H_pad, W_pad), dtype=np.float32)

    for bh in range(nb_h):
        for bw in range(nb_w):
            dq = dequantize(coeffs[bh, bw], quant_matrix)
            reconstructed[bh*8:(bh+1)*8, bw*8:(bw+1)*8] = idct_block(dq)

    return reconstructed[:original_H, :original_W]


def encode_iframe(preprocessed: dict, qf: float = 1.0) -> dict:

    Y      = preprocessed['Y']
    Cb_sub = preprocessed['Cb_sub']
    Cr_sub = preprocessed['Cr_sub']
    H, W   = preprocessed['shape']

    qm_y  = get_quant_matrix('Y',  qf)
    qm_c  = get_quant_matrix('C',  qf)

    return {
        'type':     'I',
        'Y_coeffs':  encode_channel(Y,      qm_y),
        'Cb_coeffs': encode_channel(Cb_sub, qm_c),
        'Cr_coeffs': encode_channel(Cr_sub, qm_c),
        'shape':    (H, W),
        'qf':       qf,
    }


def decode_iframe(encoded: dict) -> dict:

    H, W = encoded['shape']
    qf   = encoded['qf']
    Hc   = (H + 1) // 2
    Wc   = (W + 1) // 2

    qm_y = get_quant_matrix('Y', qf)
    qm_c = get_quant_matrix('C', qf)

    Y      = decode_channel(encoded['Y_coeffs'],  qm_y, H,  W)
    Cb_sub = decode_channel(encoded['Cb_coeffs'], qm_c, Hc, Wc)
    Cr_sub = decode_channel(encoded['Cr_coeffs'], qm_c, Hc, Wc)

    return {
        'Y':      Y,
        'Cb_sub': Cb_sub,
        'Cr_sub': Cr_sub,
        'shape':  (H, W)
    }

import numpy as np
from part2_iframe import (encode_channel, decode_channel,
                          get_quant_matrix, dct_block, quantize,
                          dequantize, idct_block)

MACROBLOCK_SIZE = 16   

def block_matching(current_block: np.ndarray,
                   ref_frame: np.ndarray,
                   origin_y: int, origin_x: int,
                   search_range: int) -> tuple[int, int]:

    H, W = ref_frame.shape
    bs   = MACROBLOCK_SIZE
    best_mse = float('inf')
    best_dy, best_dx = 0, 0

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ry = origin_y + dy
            rx = origin_x + dx

            if ry < 0 or rx < 0 or ry + bs > H or rx + bs > W:
                continue

            ref_block = ref_frame[ry:ry+bs, rx:rx+bs]
            diff = current_block.astype(np.float32) - ref_block.astype(np.float32)
            mse  = np.mean(diff ** 2)

            if mse < best_mse:
                best_mse    = mse
                best_dy, best_dx = dy, dx

    return best_dy, best_dx

def encode_residual_block(residual: np.ndarray,
                          quant_matrix: np.ndarray) -> np.ndarray:
 
    coeffs = np.zeros((2, 2, 8, 8), dtype=np.int16)
    for i in range(2):
        for j in range(2):
            sub = residual[i*8:(i+1)*8, j*8:(j+1)*8]
           
            dct_c = dctn_raw(sub)
            coeffs[i, j] = quantize(dct_c, quant_matrix)
    return coeffs


def decode_residual_block(coeffs: np.ndarray,
                          quant_matrix: np.ndarray) -> np.ndarray:
    
    residual = np.zeros((16, 16), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            dq = dequantize(coeffs[i, j], quant_matrix)
            residual[i*8:(i+1)*8, j*8:(j+1)*8] = idctn_raw(dq)
    return residual


def dctn_raw(block: np.ndarray) -> np.ndarray:
   
    from scipy.fft import dctn
    return dctn(block.astype(np.float32), norm='ortho')


def idctn_raw(coeffs: np.ndarray) -> np.ndarray:
   
    from scipy.fft import idctn
    return idctn(coeffs.astype(np.float32), norm='ortho')



def encode_pframe(preprocessed: dict,
                  ref_reconstructed: dict,
                  search_range: int = 8,
                  qf: float = 1.0) -> dict:

    Y_cur  = preprocessed['Y']
    Y_ref  = ref_reconstructed['Y']
    H, W   = preprocessed['shape']
    bs     = MACROBLOCK_SIZE

    qm_y = get_quant_matrix('Y', qf)
    qm_c = get_quant_matrix('C', qf)

    H_pad = (H + bs - 1) // bs * bs
    W_pad = (W + bs - 1) // bs * bs

    Y_cur_pad = np.zeros((H_pad, W_pad), dtype=np.float32)
    Y_ref_pad = np.zeros((H_pad, W_pad), dtype=np.float32)
    Y_cur_pad[:H, :W] = Y_cur
    Y_ref_pad[:H, :W] = Y_ref

    nb_bh = H_pad // bs
    nb_bw = W_pad // bs

    mvs       = []  
    residuals = []   

    for bh in range(nb_bh):
        for bw in range(nb_bw):
            oy, ox = bh * bs, bw * bs
            curr_block = Y_cur_pad[oy:oy+bs, ox:ox+bs]

            dy, dx = block_matching(curr_block, Y_ref_pad, oy, ox, search_range)
            mvs.append((dy, dx))

            ry, rx  = oy + dy, ox + dx
            pred_block = Y_ref_pad[ry:ry+bs, rx:rx+bs]
            residual   = curr_block.astype(np.float32) - pred_block.astype(np.float32)

            res_encoded = encode_residual_block(residual, qm_y)
            residuals.append(res_encoded)

    Cb_coeffs = encode_channel(preprocessed['Cb_sub'], qm_c)
    Cr_coeffs = encode_channel(preprocessed['Cr_sub'], qm_c)

    return {
        'type':       'P',
        'Y_mvs':      mvs,
        'Y_residuals': residuals,
        'Cb_coeffs':  Cb_coeffs,
        'Cr_coeffs':  Cr_coeffs,
        'shape':      (H, W),
        'nb_bh':      nb_bh,
        'nb_bw':      nb_bw,
        'qf':         qf,
    }


def decode_pframe(encoded: dict, ref_reconstructed: dict) -> dict:

    H, W   = encoded['shape']
    qf     = encoded['qf']
    bs     = MACROBLOCK_SIZE
    Hc, Wc = (H + 1) // 2, (W + 1) // 2

    qm_y = get_quant_matrix('Y', qf)
    qm_c = get_quant_matrix('C', qf)

    H_pad = encoded['nb_bh'] * bs
    W_pad = encoded['nb_bw'] * bs

    Y_ref_pad = np.zeros((H_pad, W_pad), dtype=np.float32)
    Y_ref_pad[:H, :W] = ref_reconstructed['Y']

    Y_reconstructed = np.zeros((H_pad, W_pad), dtype=np.float32)

    for idx, ((dy, dx), res_enc) in enumerate(
            zip(encoded['Y_mvs'], encoded['Y_residuals'])):
        bh = idx // encoded['nb_bw']
        bw = idx  % encoded['nb_bw']
        oy, ox = bh * bs, bw * bs

   
        ry, rx = oy + dy, ox + dx
        pred_block = Y_ref_pad[ry:ry+bs, rx:rx+bs]

      
        residual = decode_residual_block(res_enc, qm_y)

    
        Y_reconstructed[oy:oy+bs, ox:ox+bs] = (
            np.clip(pred_block + residual, 0, 255))

    Y_out  = Y_reconstructed[:H, :W]
    Cb_sub = decode_channel(encoded['Cb_coeffs'], qm_c, Hc, Wc)
    Cr_sub = decode_channel(encoded['Cr_coeffs'], qm_c, Hc, Wc)

    return {
        'Y':      Y_out,
        'Cb_sub': Cb_sub,
        'Cr_sub': Cr_sub,
        'shape':  (H, W)
    }

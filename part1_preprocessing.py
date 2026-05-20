
import numpy as np

def rgb_to_ycbcr(frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    frame = frame_rgb.astype(np.float32)
    R, G, B = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5    * B + 128.0
    Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128.0

    # Clamp dans [0, 255]
    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return Y, Cb, Cr


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:

    Y  = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0

    R = Y + 1.40200 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.77200 * Cb

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb




def subsample_420(Cb: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    Cb_sub = (Cb[0::2, 0::2] + Cb[1::2, 0::2] +
              Cb[0::2, 1::2] + Cb[1::2, 1::2]) / 4.0
    Cr_sub = (Cr[0::2, 0::2] + Cr[1::2, 0::2] +
              Cr[0::2, 1::2] + Cr[1::2, 1::2]) / 4.0
    return Cb_sub, Cr_sub


def upsample_420(Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                 target_h: int, target_w: int) -> tuple[np.ndarray, np.ndarray]:

    from PIL import Image as PILImage

    def _upsample(channel):
        img = PILImage.fromarray(channel.astype(np.float32))
        img = img.resize((target_w, target_h), PILImage.BILINEAR)
        return np.array(img, dtype=np.float32)

    return _upsample(Cb_sub), _upsample(Cr_sub)



def preprocess_frame(frame_rgb: np.ndarray) -> dict:

    H, W = frame_rgb.shape[:2]
    Y, Cb, Cr = rgb_to_ycbcr(frame_rgb)
    Cb_sub, Cr_sub = subsample_420(Cb, Cr)

    return {
        'Y':      Y,
        'Cb_sub': Cb_sub,
        'Cr_sub': Cr_sub,
        'shape':  (H, W)
    }


def reconstruct_frame(data: dict) -> np.ndarray:

    H, W = data['shape']
    Cb_full, Cr_full = upsample_420(data['Cb_sub'], data['Cr_sub'], H, W)
    return ycbcr_to_rgb(data['Y'], Cb_full, Cr_full)

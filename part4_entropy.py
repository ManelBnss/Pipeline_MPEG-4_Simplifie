import struct
import pickle
import heapq
import numpy as np
from collections import Counter

MAGIC = b'MPG4'


def rle_encode(symbols):

    if not symbols:
        return []
    encoded = []
    current = symbols[0]
    count   = 1
    for sym in symbols[1:]:
        if sym == current:
            count += 1
        else:
            encoded.append((current, count))
            current = sym
            count   = 1
    encoded.append((current, count))
    return encoded


def rle_decode(encoded):
    symbols = []
    for sym, count in encoded:
        symbols.extend([sym] * count)
    return symbols


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(symbols):
    freq = Counter(symbols)
    if len(freq) == 1:
        sym  = list(freq.keys())[0]
        root = HuffmanNode(None, freq[sym])
        root.left = HuffmanNode(sym, freq[sym])
        return root

    heap = [HuffmanNode(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left  = heapq.heappop(heap)
        right = heapq.heappop(heap)
        p     = HuffmanNode(None, left.freq + right.freq)
        p.left, p.right = left, right
        heapq.heappush(heap, p)
    return heap[0]


def build_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else '0'
        return codes
    build_codes(node.left,  prefix + '0', codes)
    build_codes(node.right, prefix + '1', codes)
    return codes


def _rebuild_tree(codes):
    root = HuffmanNode(None, 0)
    for symbol, code in codes.items():
        node = root
        for bit in code:
            if bit == '0':
                if node.left  is None: node.left  = HuffmanNode(None, 0)
                node = node.left
            else:
                if node.right is None: node.right = HuffmanNode(None, 0)
                node = node.right
        node.symbol = symbol
    return root


def huffman_encode(symbols, codes):
    bitstring = ''.join(codes[s] for s in symbols)
    padding   = (8 - len(bitstring) % 8) % 8
    bitstring += '0' * padding
    result = bytearray(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return bytes(result), padding


def huffman_decode(data, padding, tree, nb_symbols):
    bitstring = ''.join(f'{b:08b}' for b in data)
    if padding > 0:
        bitstring = bitstring[:-padding]
    symbols, node = [], tree
    for bit in bitstring:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            symbols.append(node.symbol)
            node = tree
            if len(symbols) == nb_symbols:
                break
    return symbols


def _flatten_frame(encoded):
    syms = []
    syms.append(ord(encoded['type']))
    H, W = encoded['shape']
    syms.extend([H, W, int(encoded['qf'] * 100)])

    def append_rle(arr):
        flat = [int(x) for x in arr.flatten().tolist()]
        rle  = rle_encode(flat)
        syms.append(len(rle))
        for s, c in rle:
            syms.extend([s, c])

    if encoded['type'] == 'I':
        for k in ['Y_coeffs', 'Cb_coeffs', 'Cr_coeffs']:
            append_rle(encoded[k])
    else:
        mvs = encoded['Y_mvs']
        syms.append(len(mvs))
        for dy, dx in mvs:
            syms.extend([dy + 128, dx + 128])
        for res in encoded['Y_residuals']:
            append_rle(res)
        for k in ['Cb_coeffs', 'Cr_coeffs']:
            append_rle(encoded[k])
        syms.extend([encoded['nb_bh'], encoded['nb_bw']])
    return syms


def _unflatten_frame(syms, pos):
    frame_type = chr(syms[pos]); pos += 1
    H = syms[pos]; pos += 1
    W = syms[pos]; pos += 1
    qf = syms[pos] / 100.; pos += 1
    enc = {'type': frame_type, 'shape': (H, W), 'qf': qf}

    def read_ch(pos, shape):
        n = syms[pos]; pos += 1
        rle = [(syms[pos+i*2], syms[pos+i*2+1]) for i in range(n)]
        pos += n * 2
        return np.array(rle_decode(rle), dtype=np.int16).reshape(shape), pos

    def sh(h, w):
        return (-((-h)//8), -((-w)//8), 8, 8)

    Hc, Wc = (H+1)//2, (W+1)//2

    if frame_type == 'I':
        enc['Y_coeffs'],  pos = read_ch(pos, sh(H, W))
        enc['Cb_coeffs'], pos = read_ch(pos, sh(Hc, Wc))
        enc['Cr_coeffs'], pos = read_ch(pos, sh(Hc, Wc))
    else:
        nb_mvs = syms[pos]; pos += 1
        mvs = [(syms[pos+i*2]-128, syms[pos+i*2+1]-128) for i in range(nb_mvs)]
        pos += nb_mvs * 2
        enc['Y_mvs'] = mvs
        residuals = []
        for _ in range(nb_mvs):
            res, pos = read_ch(pos, (2, 2, 8, 8))
            residuals.append(res)
        enc['Y_residuals'] = residuals
        enc['Cb_coeffs'], pos = read_ch(pos, sh(Hc, Wc))
        enc['Cr_coeffs'], pos = read_ch(pos, sh(Hc, Wc))
        enc['nb_bh'] = syms[pos]; pos += 1
        enc['nb_bw'] = syms[pos]; pos += 1
    return enc, pos


def write_bin(frames_encoded, output_path):
    all_syms = []
    for enc in frames_encoded:
        all_syms.extend(_flatten_frame(enc))

    offset = min(all_syms)
    if offset < 0:
        all_syms = [s - offset for s in all_syms]

    tree  = build_huffman_tree(all_syms)
    codes = build_codes(tree)
    huff_bytes, padding = huffman_encode(all_syms, codes)

    meta = pickle.dumps({'codes': codes, 'nb_symbols': len(all_syms),
                         'nb_frames': len(frames_encoded), 'offset': offset}, protocol=4)

    header = (MAGIC +
              struct.pack('<I', len(frames_encoded)) +
              struct.pack('<I', len(meta)) +
              struct.pack('<I', len(huff_bytes)) +
              struct.pack('<B', padding))

    with open(output_path, 'wb') as f:
        f.write(header + meta + huff_bytes)
    return len(header) + len(meta) + len(huff_bytes)


def read_bin(input_path):
    with open(input_path, 'rb') as f:
        raw = f.read()
    pos = 0
    if raw[pos:pos+4] != MAGIC:
        raise ValueError("Fichier .bin invalide")
    pos += 4
    nb_frames = struct.unpack('<I', raw[pos:pos+4])[0]; pos += 4
    meta_size = struct.unpack('<I', raw[pos:pos+4])[0]; pos += 4
    huff_size = struct.unpack('<I', raw[pos:pos+4])[0]; pos += 4
    padding   = struct.unpack('<B', raw[pos:pos+1])[0]; pos += 1
    meta      = pickle.loads(raw[pos:pos+meta_size]);   pos += meta_size
    huff_data = raw[pos:pos+huff_size]

    tree     = _rebuild_tree(meta['codes'])
    all_syms = huffman_decode(huff_data, padding, tree, meta['nb_symbols'])
    offset   = meta['offset']
    if offset < 0:
        all_syms = [s + offset for s in all_syms]

    frames, pos_sym = [], 0
    for _ in range(nb_frames):
        enc, pos_sym = _unflatten_frame(all_syms, pos_sym)
        frames.append(enc)
    return frames, {'nb_frames': nb_frames, 'file_size': len(raw)}


def compute_original_size(frames_rgb):
    return sum(f.nbytes for f in frames_rgb)

def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size if compressed_size > 0 else 0.0

def print_stats(original_size, compressed_size, nb_iframes, nb_pframes):
    ratio = compression_ratio(original_size, compressed_size)
    print("\n" + "═"*50)
    print("  STATISTIQUES DE COMPRESSION")
    print("═"*50)
    print(f"  Taille originale    : {original_size:>10,} octets")
    print(f"  Taille compressée   : {compressed_size:>10,} octets")
    print(f"  Ratio de compression: {ratio:>10.2f}×")
    print(f"  I-frames            : {nb_iframes}")
    print(f"  P-frames            : {nb_pframes}")
    print("═"*50)
    print(f"  Codage entropique   : RLE + Huffman (manuel)")
    print(f"  RLE     → supprime les zéros consécutifs après DCT")
    print(f"  Huffman → codes courts aux symboles fréquents")
    print("═"*50)

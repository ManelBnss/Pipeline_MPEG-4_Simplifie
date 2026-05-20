import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_metrics(frames_original: list[np.ndarray],
                    frames_reconstructed: list[np.ndarray],
                    frames_encoded: list[dict],
                    original_size: int,
                    compressed_size: int) -> dict:
 
    psnr_values = []
    for orig, rec in zip(frames_original, frames_reconstructed):
        psnr_values.append(psnr(orig, rec))

    nb_iframes = sum(1 for f in frames_encoded if f['type'] == 'I')
    nb_pframes = sum(1 for f in frames_encoded if f['type'] == 'P')
    frame_types = [f['type'] for f in frames_encoded]

    ratio = original_size / compressed_size if compressed_size > 0 else 0

    return {
        'psnr_values':       psnr_values,
        'psnr_mean':         float(np.mean(psnr_values)),
        'compression_ratio': ratio,
        'original_size':     original_size,
        'compressed_size':   compressed_size,
        'nb_iframes':        nb_iframes,
        'nb_pframes':        nb_pframes,
        'frame_types':       frame_types,
    }


def visualize_pipeline(frames_original: list[np.ndarray],
                       frames_reconstructed: list[np.ndarray],
                       frames_encoded: list[dict],
                       preprocessed_list: list[dict],
                       metrics: dict,
                       save_path: str = "pipeline_visualization.png"):
  
    from scipy.fft import dctn

    fig = plt.figure(figsize=(22, 26), facecolor='#0d1117')
    fig.suptitle('Pipeline MPEG-4 Simplifié — Visualisation Complète',
                 fontsize=18, color='white', fontweight='bold', y=0.99)

    gs = GridSpec(6, 6, figure=fig, hspace=0.55, wspace=0.35,
                  top=0.96, bottom=0.03, left=0.04, right=0.97)

    TITLE_COLOR  = '#58a6ff'
    TEXT_COLOR   = '#c9d1d9'
    PANEL_BG     = '#161b22'
    BORDER_COLOR = '#30363d'

    def styled_title(ax, title):
        ax.set_title(title, color=TITLE_COLOR, fontsize=9, fontweight='bold', pad=6)

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

    n_show = min(6, len(frames_original))
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i])
        style_ax(ax)
        ax.imshow(frames_original[i])
        ftype = frames_encoded[i]['type'] if i < len(frames_encoded) else '?'
        color = '#ff7b72' if ftype == 'I' else '#56d364'
        ax.set_title(f'Frame {i} [{ftype}]', color=color, fontsize=8,
                     fontweight='bold', pad=4)
        ax.axis('off')

    fig.text(0.01, 0.845, '① Frames\nOriginaux', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    pp = preprocessed_list[0]
    Y, Cb_sub, Cr_sub = pp['Y'], pp['Cb_sub'], pp['Cr_sub']

    channels = [
        (frames_original[0], 'Frame original RGB', 'viridis', None),
        (Y,   'Canal Y (Luminance)',  'gray',    (0, 255)),
        (Cb_sub, 'Canal Cb (↓2) Chroma Bleu', 'Blues', None),
        (Cr_sub, 'Canal Cr (↓2) Chroma Rouge','Reds',  None),
    ]
    for i, (data, title, cmap, vrange) in enumerate(channels):
        ax = fig.add_subplot(gs[1, i*1 + (0 if i < 2 else 0)])
        style_ax(ax)
        if i == 0:
            ax.imshow(data)
        else:
            kwargs = {}
            if vrange:
                kwargs = {'vmin': vrange[0], 'vmax': vrange[1]}
            im = ax.imshow(data, cmap=cmap, **kwargs)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='', shrink=0.8)
        styled_title(ax, title)
        ax.axis('off')

    ax = fig.add_subplot(gs[1, 4:])
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)
    ax.axis('off')
    Hc, Wc = Cb_sub.shape
    H, W   = Y.shape
    info = (f"Sous-échantillonnage 4:2:0\n\n"
            f"Y  : {H}×{W} pixels\n"
            f"Cb : {Hc}×{Wc} pixels (×0.25)\n"
            f"Cr : {Hc}×{Wc} pixels (×0.25)\n\n"
            f"Gain mémoire :\n"
            f"RGB    = {H*W*3} octets\n"
            f"YCbCr  = {H*W + Hc*Wc*2} octets\n"
            f"Ratio  = {H*W*3/(H*W + Hc*Wc*2):.2f}×")
    ax.text(0.1, 0.5, info, transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=8.5, va='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937',
                      edgecolor=BORDER_COLOR))
    styled_title(ax, '② Espace colorimétrique YCbCr')

    fig.text(0.01, 0.695, '② YCbCr\nSub-sample', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    fig.text(0.01, 0.543, '③ DCT &\nQuantif.', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    block_raw = Y[16:24, 16:24].astype(np.float32)
    block_centered = block_raw - 128.0
    block_dct = dctn(block_centered, norm='ortho')

    from part2_iframe import get_quant_matrix, quantize, dequantize, idct_block
    qm = get_quant_matrix('Y', 1.0)
    block_quant = quantize(block_dct, qm)
    block_deq   = dequantize(block_quant, qm)
    block_rec   = idct_block(block_deq)

    stages_3 = [
        (block_raw,   'Pixels bruts (8×8)', 'gray',      (0, 255)),
        (block_dct,   'Coefficients DCT',   'RdBu_r',    None),
        (block_quant, 'Après Quantification','coolwarm', None),
        (block_deq,   'Déquantifié',         'RdBu_r',    None),
        (block_rec,   'Reconstruit (IDCT)',  'gray',      (0, 255)),
    ]

    mse_block = np.mean((block_raw - block_rec) ** 2)
    for i, (data, title, cmap, vrange) in enumerate(stages_3):
        ax = fig.add_subplot(gs[2, i])
        style_ax(ax)
        kwargs = {}
        if vrange:
            kwargs = {'vmin': vrange[0], 'vmax': vrange[1]}
        im = ax.imshow(data, cmap=cmap, interpolation='nearest', **kwargs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        styled_title(ax, title)
        if title == 'Après Quantification':
            for row in range(8):
                for col in range(8):
                    val = data[row, col]
                    ax.text(col, row, str(val), ha='center', va='center',
                            fontsize=4, color='white' if abs(val) > 5 else 'black')
        ax.axis('off')

    ax = fig.add_subplot(gs[2, 5])
    style_ax(ax)
    ax.axis('off')
    ax.text(0.5, 0.5,
            f"MSE bloc = {mse_block:.4f}\n"
            f"PSNR bloc = {10*np.log10(255**2/max(mse_block,1e-10)):.1f} dB\n\n"
            f"Non-zéros:\n{np.count_nonzero(block_quant)}/64 coeff.",
            transform=ax.transAxes, color=TEXT_COLOR, fontsize=9,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937',
                      edgecolor=BORDER_COLOR))
    styled_title(ax, 'Qualité du bloc')

    fig.text(0.01, 0.390, '④ Motion\nVectors', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    pframe_idx = next((i for i, f in enumerate(frames_encoded)
                       if f['type'] == 'P'), None)

    ax_mv = fig.add_subplot(gs[3, :3])
    style_ax(ax_mv)
    if pframe_idx is not None:
        ax_mv.imshow(frames_original[pframe_idx], alpha=0.7)
        enc_p = frames_encoded[pframe_idx]
        bs    = 16
        nb_bw = enc_p['nb_bw']
        for idx, (dy, dx) in enumerate(enc_p['Y_mvs']):
            bh = idx // nb_bw
            bw = idx  % nb_bw
            cy = bh * bs + bs // 2
            cx = bw * bs + bs // 2
            if dy != 0 or dx != 0:
                ax_mv.annotate('',
                    xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#f0e68c',
                                    lw=1.2))
        styled_title(ax_mv, f'④ Vecteurs de mouvement — P-frame {pframe_idx}')
    else:
        ax_mv.text(0.5, 0.5, 'Pas de P-frame trouvé',
                   ha='center', va='center', color=TEXT_COLOR)
        styled_title(ax_mv, '④ Vecteurs de mouvement')
    ax_mv.axis('off')

    ax_hist = fig.add_subplot(gs[3, 3])
    ax_hist.set_facecolor(PANEL_BG)
    for spine in ax_hist.spines.values():
        spine.set_edgecolor(BORDER_COLOR)

    if pframe_idx is not None:
        mvs = enc_p['Y_mvs']
        dxs = [mv[1] for mv in mvs]
        dys = [mv[0] for mv in mvs]
        ax_hist.hist(dxs, bins=15, color='#58a6ff', alpha=0.7, label='dx')
        ax_hist.hist(dys, bins=15, color='#ff7b72', alpha=0.7, label='dy')
        ax_hist.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR,
                       labelcolor=TEXT_COLOR)
        ax_hist.tick_params(colors=TEXT_COLOR)
    styled_title(ax_hist, 'Distribution MV')

    ax_res = fig.add_subplot(gs[3, 4:])
    style_ax(ax_res)
    if pframe_idx is not None and pframe_idx > 0:
        Y_cur = preprocessed_list[pframe_idx]['Y']
        Y_ref = preprocessed_list[pframe_idx - 1]['Y']
        residual_map = np.abs(Y_cur.astype(np.float32) - Y_ref.astype(np.float32))
        im = ax_res.imshow(residual_map, cmap='hot', vmin=0, vmax=60)
        plt.colorbar(im, ax=ax_res, fraction=0.046, shrink=0.8)
    styled_title(ax_res, '⑤ Carte des résidus |Y_cur - Y_ref|')
    ax_res.axis('off')

    fig.text(0.01, 0.237, '⑤ Recons-\ntruction', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    n_show2 = min(6, len(frames_reconstructed))
    for i in range(n_show2):
        ax = fig.add_subplot(gs[4, i])
        style_ax(ax)
        ax.imshow(frames_reconstructed[i])
        p = metrics['psnr_values'][i]
        ax.set_title(f'F{i} PSNR={p:.1f}dB', color=TEXT_COLOR,
                     fontsize=7.5, pad=4)
        ax.axis('off')

    fig.text(0.01, 0.085, '⑥ Métriques\nGlobales', color=TITLE_COLOR,
             fontsize=8, va='center', fontweight='bold')

    ax_psnr = fig.add_subplot(gs[5, :3])
    ax_psnr.set_facecolor(PANEL_BG)
    for spine in ax_psnr.spines.values():
        spine.set_edgecolor(BORDER_COLOR)
    psnr_vals = metrics['psnr_values']
    colors_bar = ['#ff7b72' if t == 'I' else '#56d364'
                  for t in metrics['frame_types']]
    ax_psnr.bar(range(len(psnr_vals)), psnr_vals, color=colors_bar, alpha=0.85)
    ax_psnr.axhline(y=metrics['psnr_mean'], color='#f0e68c',
                    linestyle='--', linewidth=1.2,
                    label=f'Moy. {metrics["psnr_mean"]:.1f} dB')
    ax_psnr.set_xlabel('Index frame', color=TEXT_COLOR, fontsize=8)
    ax_psnr.set_ylabel('PSNR (dB)', color=TEXT_COLOR, fontsize=8)
    ax_psnr.tick_params(colors=TEXT_COLOR)
    ax_psnr.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=8)
    patch_i = mpatches.Patch(color='#ff7b72', label='I-frame')
    patch_p = mpatches.Patch(color='#56d364', label='P-frame')
    ax_psnr.legend(handles=[patch_i, patch_p], facecolor=PANEL_BG,
                   edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    styled_title(ax_psnr, '⑥ PSNR par frame (dB)')

    ax_sum = fig.add_subplot(gs[5, 3:])
    ax_sum.set_facecolor(PANEL_BG)
    for spine in ax_sum.spines.values():
        spine.set_edgecolor(BORDER_COLOR)
    ax_sum.axis('off')

    ratio = metrics['compression_ratio']
    orig_ko  = metrics['original_size'] / 1024
    comp_ko  = metrics['compressed_size'] / 1024
    summary = (
        f"  RÉSUMÉ DU PIPELINE\n"
        f"  {'─'*28}\n"
        f"  Frames totaux  : {len(psnr_vals)}\n"
        f"  I-frames       : {metrics['nb_iframes']}\n"
        f"  P-frames       : {metrics['nb_pframes']}\n"
        f"  {'─'*28}\n"
        f"  Taille originale  : {orig_ko:.1f} Ko\n"
        f"  Taille compressée : {comp_ko:.1f} Ko\n"
        f"  Ratio             : {ratio:.2f}×\n"
        f"  {'─'*28}\n"
        f"  PSNR moyen : {metrics['psnr_mean']:.2f} dB\n"
        f"  PSNR min   : {min(psnr_vals):.2f} dB\n"
        f"  PSNR max   : {max(psnr_vals):.2f} dB\n"
    )
    ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
                color=TEXT_COLOR, fontsize=8.5, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937',
                          edgecolor=BORDER_COLOR))
    styled_title(ax_sum, 'Résumé')

    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='#0d1117')
    print(f"✓ Visualisation sauvegardée : {save_path}")
    return fig


def plot_qf_vs_ratio(results: list[dict], save_path: str = "qf_analysis.png"):
    """
    Trace le ratio de compression et le PSNR en fonction du
    facteur de quantification (QF).

    results : liste de dicts {'qf', 'ratio', 'psnr_mean'}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    facecolor='#0d1117')
    PANEL_BG     = '#161b22'
    BORDER_COLOR = '#30363d'
    TEXT_COLOR   = '#c9d1d9'
    TITLE_COLOR  = '#58a6ff'

    qfs    = [r['qf'] for r in results]
    ratios = [r['ratio'] for r in results]
    psnrs  = [r['psnr_mean'] for r in results]

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    ax1.plot(qfs, ratios, 'o-', color='#58a6ff', linewidth=2, markersize=7)
    ax1.set_xlabel('Facteur de quantification (QF)', color=TEXT_COLOR)
    ax1.set_ylabel('Ratio de compression (×)', color=TEXT_COLOR)
    ax1.set_title('Compression vs QF', color=TITLE_COLOR, fontweight='bold')
    ax1.fill_between(qfs, ratios, alpha=0.2, color='#58a6ff')

    ax2.plot(qfs, psnrs, 's-', color='#ff7b72', linewidth=2, markersize=7)
    ax2.set_xlabel('Facteur de quantification (QF)', color=TEXT_COLOR)
    ax2.set_ylabel('PSNR moyen (dB)', color=TEXT_COLOR)
    ax2.set_title('Qualité (PSNR) vs QF', color=TITLE_COLOR, fontweight='bold')
    ax2.fill_between(qfs, psnrs, alpha=0.2, color='#ff7b72')
    ax2.axhline(y=30, color='#56d364', linestyle='--', linewidth=1,
                label='Seuil qualité 30 dB')
    ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR,
               labelcolor=TEXT_COLOR)

    fig.suptitle('Analyse : Ratio de Compression vs Facteur de Quantification',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Analyse QF sauvegardée : {save_path}")
    return fig


def plot_gop_vs_ratio(results: list[dict], save_path: str = "gop_analysis.png"):
    """
    Trace le ratio de compression en fonction de la taille du GOP.

    results : liste de dicts {'gop', 'ratio', 'psnr_mean'}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    facecolor='#0d1117')
    PANEL_BG     = '#161b22'
    BORDER_COLOR = '#30363d'
    TEXT_COLOR   = '#c9d1d9'
    TITLE_COLOR  = '#58a6ff'

    gops   = [r['gop'] for r in results]
    ratios = [r['ratio'] for r in results]
    psnrs  = [r['psnr_mean'] for r in results]

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    ax1.plot(gops, ratios, 'o-', color='#56d364', linewidth=2, markersize=7)
    ax1.set_xlabel('Taille du GOP (G)', color=TEXT_COLOR)
    ax1.set_ylabel('Ratio de compression (×)', color=TEXT_COLOR)
    ax1.set_title('Compression vs Taille GOP', color=TITLE_COLOR, fontweight='bold')
    ax1.fill_between(gops, ratios, alpha=0.2, color='#56d364')

    ax2.plot(gops, psnrs, 's-', color='#f0e68c', linewidth=2, markersize=7)
    ax2.set_xlabel('Taille du GOP (G)', color=TEXT_COLOR)
    ax2.set_ylabel('PSNR moyen (dB)', color=TEXT_COLOR)
    ax2.set_title('Qualité (PSNR) vs Taille GOP', color=TITLE_COLOR, fontweight='bold')
    ax2.fill_between(gops, psnrs, alpha=0.2, color='#f0e68c')

    fig.suptitle('Analyse : Impact de la Taille du GOP sur la Compression',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Analyse GOP sauvegardée : {save_path}")
    return fig

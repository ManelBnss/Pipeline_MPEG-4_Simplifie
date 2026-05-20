import os
import sys
import argparse
import numpy as np
from PIL import Image


from part1_preprocessing import preprocess_frame, reconstruct_frame
from part2_iframe        import encode_iframe, decode_iframe
from part3_pframe        import encode_pframe, decode_pframe
from part4_entropy       import (write_bin, read_bin,
                                  compute_original_size,
                                  compression_ratio, print_stats)
from part5_evaluation    import (compute_metrics, visualize_pipeline,
                                  plot_qf_vs_ratio, plot_gop_vs_ratio, psnr)




def load_frames(frames_dir: str) -> list[np.ndarray]:

    supported = ('.png', '.jpg', '.jpeg')
    files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(supported)
    ])
    if not files:
        raise FileNotFoundError(
            f"Aucune image trouvée dans '{frames_dir}'. "
            )

    frames = []
    for fname in files:
        path = os.path.join(frames_dir, fname)
        img  = Image.open(path).convert('RGB')
        frames.append(np.array(img, dtype=np.uint8))
    return frames




def encode_video(frames_rgb: list[np.ndarray],
                 gop_size: int = 4,
                 search_range: int = 8,
                 qf: float = 1.0) -> tuple[list[dict], list[dict]]:
   
    frames_encoded    = []
    preprocessed_list = []
    ref_reconstructed = None  

    print(f"\n{'═'*55}")
    print(f"  ENCODAGE — {len(frames_rgb)} frames | GOP={gop_size} | QF={qf}")
    print(f"{'═'*55}")

    for i, frame_rgb in enumerate(frames_rgb):
        # ── PARTIE 1 : Pré-traitement ─────────────────
        preprocessed = preprocess_frame(frame_rgb)
        preprocessed_list.append(preprocessed)

        # Décision I-frame ou P-frame
        is_iframe = (i % gop_size == 0)
        ftype = 'I' if is_iframe else 'P'

        if is_iframe:
            # ── PARTIE 2 : Codage I-frame ─────────────
            encoded = encode_iframe(preprocessed, qf=qf)
            decoded = decode_iframe(encoded)

        else:
            # ── PARTIE 3 : Codage P-frame ─────────────
            encoded = encode_pframe(preprocessed, ref_reconstructed,
                                    search_range=search_range, qf=qf)
            decoded = decode_pframe(encoded, ref_reconstructed)

        frames_encoded.append(encoded)
        ref_reconstructed = decoded 

        print(f"  [{ftype}] Frame {i:3d}/{len(frames_rgb)-1} — "
              f"{'I-frame (spatial DCT)' if is_iframe else 'P-frame (motion + résidu)'}")

    print(f"\n  ✓ Encodage terminé : {len(frames_encoded)} frames")
    return frames_encoded, preprocessed_list




def decode_video(frames_encoded: list[dict]) -> list[np.ndarray]:
  
    frames_reconstructed = []
    ref_reconstructed    = None

    print(f"\n{'═'*55}")
    print(f"  DÉCODAGE — {len(frames_encoded)} frames")
    print(f"{'═'*55}")

    for i, encoded in enumerate(frames_encoded):
        ftype = encoded['type']

        if ftype == 'I':
            decoded = decode_iframe(encoded)
        else:
            decoded = decode_pframe(encoded, ref_reconstructed)

        # Reconstruction RGB finale (partie 1 inverse)
        frame_rgb = reconstruct_frame(decoded)
        frames_reconstructed.append(frame_rgb)
        ref_reconstructed = decoded

        print(f"  [{ftype}] Frame {i:3d}/{len(frames_encoded)-1} décodée")

    print(f"\n  ✓ Décodage terminé")
    return frames_reconstructed




def run_analysis(frames_rgb: list[np.ndarray], output_dir: str = "."):
 
    print("\n" + "═"*55)
    print("  ANALYSE : Ratio vs QF et GOP")
    print("═"*55)

    # ── Analyse QF ─────────────────────────────
    qf_results = []
    for qf in [0.5, 1.0, 2.0, 4.0, 8.0]:
        enc, _ = encode_video(frames_rgb, gop_size=4, qf=qf)
        import tempfile, os
        tmp = tempfile.mktemp(suffix='.bin')
        comp_size = write_bin(enc, tmp)
        dec = decode_video(enc)
        orig_size = compute_original_size(frames_rgb)
        psnr_vals = [psnr(o, r) for o, r in zip(frames_rgb, dec)]
        qf_results.append({
            'qf':        qf,
            'ratio':     orig_size / comp_size,
            'psnr_mean': float(np.mean(psnr_vals))
        })
        os.remove(tmp)
        print(f"  QF={qf:.1f} → ratio={orig_size/comp_size:.2f}× "
              f"| PSNR={np.mean(psnr_vals):.1f} dB")

    plot_qf_vs_ratio(qf_results,
                     os.path.join(output_dir, "analyse_qf.png"))

    # ── Analyse GOP ────────────────────────────
    gop_results = []
    for gop in [1, 2, 4, 6, 12]:
        enc, _ = encode_video(frames_rgb, gop_size=gop, qf=1.0)
        tmp = tempfile.mktemp(suffix='.bin')
        comp_size = write_bin(enc, tmp)
        dec = decode_video(enc)
        orig_size = compute_original_size(frames_rgb)
        psnr_vals = [psnr(o, r) for o, r in zip(frames_rgb, dec)]
        gop_results.append({
            'gop':       gop,
            'ratio':     orig_size / comp_size,
            'psnr_mean': float(np.mean(psnr_vals))
        })
        os.remove(tmp)
        print(f"  GOP={gop:2d} → ratio={orig_size/comp_size:.2f}× "
              f"| PSNR={np.mean(psnr_vals):.1f} dB")

    plot_gop_vs_ratio(gop_results,
                      os.path.join(output_dir, "analyse_gop.png"))




def main():
    parser = argparse.ArgumentParser(
        description='Pipeline MPEG-4 Simplifié')
    parser.add_argument('--frames_dir',   default='frames',
                        help='Dossier des frames d entrée')
    parser.add_argument('--output',       default='video.bin',
                        help='Fichier de sortie compressé')
    parser.add_argument('--gop',  type=int,   default=4,
                        help='Taille du GOP (défaut : 4)')
    parser.add_argument('--search', type=int, default=8,
                        help='Fenêtre de recherche MV (défaut : 8)')
    parser.add_argument('--qf',   type=float, default=1.0,
                        help='Facteur de quantification (défaut : 1.0)')
    parser.add_argument('--analysis', action='store_true',
                        help='Lancer l analyse QF et GOP')
    parser.add_argument('--gen', action='store_true',
                        help='Générer des frames de test')
    args = parser.parse_args()

    # Générer des frames de test si demandé ou si le dossier est vide
    if args.gen or not os.path.isdir(args.frames_dir):
        generate_test_frames(args.frames_dir, n_frames=12)

    # ── Chargement ──────────────────────────
    print(f"\n{'═'*55}")
    print(f"  CHARGEMENT des frames depuis '{args.frames_dir}'")
    frames_rgb = load_frames(args.frames_dir)
    H, W = frames_rgb[0].shape[:2]
    print(f"  {len(frames_rgb)} frames | Résolution : {W}×{H} pixels")

    # ── Encodage ────────────────────────────
    frames_encoded, preprocessed_list = encode_video(
        frames_rgb,
        gop_size=args.gop,
        search_range=args.search,
        qf=args.qf
    )

    # ── PARTIE 4 : Écriture du fichier .bin ─
    print(f"\n{'─'*55}")
    print("  PARTIE 4 — Codage entropique & écriture fichier")
    compressed_size = write_bin(frames_encoded, args.output)
    original_size   = compute_original_size(frames_rgb)
    print(f"  ✓ Fichier écrit : {args.output}")
    print_stats(original_size, compressed_size,
                sum(1 for f in frames_encoded if f['type'] == 'I'),
                sum(1 for f in frames_encoded if f['type'] == 'P'))

    # ── Décodage (vérification aller-retour) ─
    frames_enc2, meta = read_bin(args.output)
    frames_reconstructed = decode_video(frames_enc2)

    # ── PARTIE 5 : Évaluation & Visualisation ─
    print(f"\n{'─'*55}")
    print("  PARTIE 5 — Évaluation & Visualisation")
    metrics = compute_metrics(
        frames_rgb, frames_reconstructed, frames_encoded,
        original_size, compressed_size
    )

    print(f"  PSNR moyen      : {metrics['psnr_mean']:.2f} dB")
    print(f"  Ratio compresion: {metrics['compression_ratio']:.2f}×")

    visualize_pipeline(
        frames_rgb, frames_reconstructed,
        frames_encoded, preprocessed_list, metrics,
        save_path="pipeline_visualization.png"
    )

    # ── Analyse optionnelle ──────────────────
    if args.analysis:
        run_analysis(frames_rgb)

    print(f"\n{'═'*55}")
    print("  Pipeline terminé avec succès !")
    print(f"{'═'*55}\n")


if __name__ == '__main__':
    main()

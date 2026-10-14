"""
Build a 2x3 aggregate figure of posterior concept maps.

Takes 5 TrainingHistory pickles (one per rating axis), renders each axis
into a single Axes using the same geometry as the D3 panel, and places a
shared legend in the 6th cell. Each panel honors its own
`posterior_concepts_positions.json` (if present next to the pickle) so
hand-adjusted cluster/concept labels and annotation ovals carry through.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BC_LLM_ROOT = os.path.join(REPO_ROOT, "bc-llm-head-to-head")
if BC_LLM_ROOT not in sys.path:
    sys.path.insert(0, BC_LLM_ROOT)

# Import the module (not just names) so we can reach helpers whose names
# begin with an underscore without re-exporting them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vis_posterior_d3 as vp  # noqa: E402

from src.training_history import TrainingHistory  # noqa: E402


# Qualitative palette for cross-panel cluster colors. Bright, saturated primaries
# chosen to read clearly as text on white and to maximize cross-panel distinction.
# Assigned deterministically in alphabetical label order so a given label gets
# the same color across runs regardless of which panel we loaded first.
_SHARED_CLUSTER_PALETTE = (
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#1a9999",  # teal
)


def _build_label_color_map(payloads: list, panel_k) -> dict:
    """Return {label: color} for cluster/merge labels that appear in ≥2 panels.

    Scans each payload's cluster + merge labels (using the same k the renderer
    would). Only visible, repeated labels get a color — singletons stay the
    default blue so colored labels really do signal cross-panel recurrence.
    Assignment is alphabetical for stability across runs.
    """
    label_panels: dict[str, set[str]] = {}
    for pl in payloads:
        k = panel_k
        if k is None:
            saved = pl.get("saved_positions") or {}
            k = int(saved.get("current_k", pl["default_k"]))
        k = int(max(pl["k_min"], min(pl["k_max"], k)))
        axis = pl["axis"]
        for ci in vp._resolve_all_cluster_labels(pl, k):
            if not ci.get("visible"):
                continue
            lbl = ci.get("label")
            if not lbl:
                continue
            label_panels.setdefault(lbl, set()).add(axis)

    shared = sorted(lbl for lbl, axes in label_panels.items()
                    if len(axes) >= 2)
    return {lbl: _SHARED_CLUSTER_PALETTE[i % len(_SHARED_CLUSTER_PALETTE)]
            for i, lbl in enumerate(shared)}


def _load_payload(training_history_file: str,
                  axis_name: str,
                  args) -> dict:
    saved_positions = None
    positions_path = os.path.join(
        os.path.dirname(os.path.abspath(training_history_file)) or ".",
        "posterior_concepts_positions.json",
    )
    if os.path.exists(positions_path):
        try:
            with open(positions_path, "r", encoding="utf-8") as f:
                saved_positions = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: could not load {positions_path}: {e}",
                  file=sys.stderr)

    history = TrainingHistory.load(training_history_file)
    return vp._build_payload(
        history,
        num_posterior_iters=args.num_posterior_iters,
        provider_identity_coef_offset=args.provider_identity_coef_offset,
        embedding_model=args.embedding_model,
        explored_threshold=args.explored_threshold,
        k_min=args.k_min,
        k_max=args.k_max,
        default_k=args.default_k,
        cluster_method=args.cluster_method,
        seed=args.seed,
        gmm_eps_jitter=args.gmm_eps_jitter,
        axis_name=axis_name,
        saved_positions=saved_positions,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-history-file", action="append", required=True,
                   dest="training_history_files",
                   help="Repeat 5 times (one per axis).")
    p.add_argument("--axis-name", action="append", required=True,
                   dest="axis_names",
                   help="Repeat 5 times, aligned with --training-history-file.")
    p.add_argument("--identity-label", type=str, default="identity unknown")
    p.add_argument("--out-figure", type=str, required=True,
                   help="Output path. Extension picks format (.pdf/.svg/.png).")

    # Pass-through options forwarded to _build_payload.
    p.add_argument("--num-posterior-iters", type=int, default=100)
    p.add_argument("--provider-identity-coef-offset", type=int, default=0)
    p.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--explored-threshold", type=float, default=0.05)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--default-k", type=int, default=4)
    p.add_argument("--cluster-method", type=str, default="gmm",
                   choices=["agglomerative", "gmm"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gmm-eps-jitter", type=float, default=0.02)
    p.add_argument("--panel-k", type=int, default=None,
                   help="Override K for every panel; None uses each panel's "
                        "saved current_k (or default_k).")
    p.add_argument("--panel-font-scale", type=float, default=1.0,
                   help="Font scale for per-panel text. Default 1.0 since each "
                        "panel cell is the native 1400x900 HTML canvas size.")
    args = p.parse_args()

    if len(args.training_history_files) != len(args.axis_names):
        raise SystemExit(
            "--training-history-file and --axis-name must appear in "
            "matching counts")
    if len(args.training_history_files) != 5:
        raise SystemExit(
            "exactly 5 (training-history-file, axis-name) pairs expected; "
            f"got {len(args.training_history_files)}")

    # 2x3 grid: each cell is a true 1400x900 canvas so hand-dragged labelX/
    # labelY from the HTML overlay 1:1. dpi=100 → inches == pixels/100.
    # Add inter-cell and outer padding so panel titles / PC labels don't
    # butt into neighbors or get cropped at the figure edge.
    dpi = 100
    cell_w_px, cell_h_px = vp._CANVAS_W, vp._CANVAS_H
    gap_px = 120            # between cells (both axes)
    outer_pad_px = 60       # around the whole figure
    fig_w_in = (3 * cell_w_px + 2 * gap_px + 2 * outer_pad_px) / dpi
    fig_h_in = (2 * cell_h_px + 1 * gap_px + 2 * outer_pad_px) / dpi
    # Convert the pixel paddings into the subplots_adjust fractional space.
    wspace_frac = gap_px / cell_w_px
    hspace_frac = gap_px / cell_h_px
    left_frac = outer_pad_px / (fig_w_in * dpi)
    bottom_frac = outer_pad_px / (fig_h_in * dpi)
    fig, axes = plt.subplots(
        2, 3, figsize=(fig_w_in, fig_h_in), dpi=dpi,
        gridspec_kw={"wspace": wspace_frac, "hspace": hspace_frac},
    )
    fig.subplots_adjust(
        left=left_frac, right=1.0 - left_frac,
        top=1.0 - bottom_frac, bottom=bottom_frac,
        wspace=wspace_frac, hspace=hspace_frac,
    )
    flat = axes.flatten()

    # Each panel linearly scales font size from 0 up to that panel's own
    # largest |contrib|, so every panel's biggest label hits the same
    # top-of-range size. Passing coef_abs_ref=None lets the renderer pick a
    # per-panel ref (its own max).
    payloads = []
    for i, (th_file, axis_name) in enumerate(zip(
            args.training_history_files, args.axis_names)):
        print(f"[{i+1}/5] loading {axis_name} from {th_file}")
        payloads.append(_load_payload(th_file, axis_name, args))

    label_color_map = _build_label_color_map(payloads, args.panel_k)
    if label_color_map:
        print("shared cluster colors across panels:")
        for lbl, color in label_color_map.items():
            print(f"  {lbl!r} → {color}")

    for i, pl in enumerate(payloads):
        print(f"[{i+1}/5] rendering {pl['axis']}")
        vp.render_payload_to_ax(
            flat[i], pl,
            k=args.panel_k,
            font_scale=args.panel_font_scale,
            show_title=True,
            show_axes_labels=True,
            coef_abs_ref=None,
            label_color_map=label_color_map,
        )

    flat[5].set_axis_off()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_figure)) or ".",
                exist_ok=True)
    ext = os.path.splitext(args.out_figure)[1].lower()
    if ext == ".png":
        fig.savefig(args.out_figure, dpi=dpi, facecolor="white")
    else:
        fig.savefig(args.out_figure, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out_figure}")


if __name__ == "__main__":
    main()

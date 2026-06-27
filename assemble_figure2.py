#!/usr/bin/env python3
"""
Assemble individual R-generated plots into a combined figure.

Layout:
  - Top row: Outcome distribution (primary analysis stacked bars)
  - Bottom left: Win-rate heatmap for accuracy
  - Bottom right: Two sensitivity plots (length and OE status) stacked

Requirements:
    pip install pillow pdf2image
    # Also requires poppler: brew install poppler (macOS) or apt install poppler-utils (Linux)

Usage:
    python assemble_figure.py <output_dir>

Example:
    python assemble_figure.py data_6-21-10-30-00am
"""

import argparse
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_path


def pdf_to_image(pdf_path: Path, dpi: int = 300) -> Image.Image:
    """Convert a single-page PDF to a PIL Image."""
    images = convert_from_path(pdf_path, dpi=dpi)
    return images[0]


def resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    """Resize image to target width, maintaining aspect ratio."""
    ratio = target_width / img.width
    new_height = int(img.height * ratio)
    return img.resize((target_width, new_height), Image.Resampling.LANCZOS)


def resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    """Resize image to target height, maintaining aspect ratio."""
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def assemble_figure(output_dir: str, output_name: str = "fig_combined.pdf"):
    """Assemble individual plots into a combined figure."""
    output_path = Path(output_dir)

    required_files = [
        "fig_outcome_distribution.pdf",
        "fig_outcome_distribution_two_panel_left.pdf",
        "fig_outcome_distribution_two_panel_right.pdf",
        "fig_winrate_heatmap_accuracy.pdf",
        "fig_sensitivity_length_accuracy.pdf",
        "fig_sensitivity_oe_accuracy.pdf",
    ]

    for f in required_files:
        if not (output_path / f).exists():
            raise FileNotFoundError(f"Missing required file: {output_path / f}")

    dpi = 300
    img_left = pdf_to_image(output_path / "fig_outcome_distribution_two_panel_left.pdf", dpi)
    img_right = pdf_to_image(output_path / "fig_outcome_distribution_two_panel_right.pdf", dpi)

    # Resize right panel to match left panel height for row alignment
    img_right_resized = resize_to_height(img_right, img_left.height)

    # Combine left and right panels
    gap_panels = 30
    outcome_width = img_left.width + gap_panels + img_right_resized.width
    img_outcome = Image.new("RGB", (outcome_width, img_left.height), "white")
    img_outcome.paste(img_left, (0, 0))
    img_outcome.paste(img_right_resized, (img_left.width + gap_panels, 0))
    img_heatmap = pdf_to_image(output_path / "fig_winrate_heatmap_accuracy.pdf", dpi)
    img_length = pdf_to_image(output_path / "fig_sensitivity_length_accuracy.pdf", dpi)
    img_oe = pdf_to_image(output_path / "fig_sensitivity_oe_accuracy.pdf", dpi)

    gap = 20

    img_length_resized = resize_to_width(img_length, img_oe.width)
    combined_height = img_length_resized.height + img_oe.height + gap
    combined_width = img_oe.width
    combined_img = Image.new("RGB", (combined_width, combined_height), "white")
    combined_img.paste(img_length_resized, (0, 0))
    combined_img.paste(img_oe, (0, img_length_resized.height + gap))

    img_heatmap_resized = resize_to_height(img_heatmap, combined_img.height)

    bottom_width = img_heatmap_resized.width + gap + combined_img.width

    img_outcome_resized = resize_to_width(img_outcome, bottom_width)

    total_height = img_outcome_resized.height + gap + combined_img.height
    total_width = bottom_width

    final_img = Image.new("RGB", (total_width, total_height), "white")

    final_img.paste(img_outcome_resized, (0, 0))
    final_img.paste(img_heatmap_resized, (0, img_outcome_resized.height + gap))
    final_img.paste(combined_img, (img_heatmap_resized.width + gap, img_outcome_resized.height + gap))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 90, index=1)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 90, index=1)
        except:
            font = ImageFont.load_default()

    label_offset = 15
    draw.text((label_offset, label_offset), "A", fill="black", font=font)
    draw.text((label_offset + img_outcome_resized.width * 0.45, label_offset), "B", fill="black", font=font)
    draw.text((label_offset, img_outcome_resized.height + gap + label_offset), "C", fill="black", font=font)
    draw.text((img_heatmap_resized.width + gap + label_offset, img_outcome_resized.height + gap + label_offset), "D", fill="black", font=font)

    out_file = output_path / output_name
    final_img.save(out_file, dpi=(300, 300))

    png_file = output_path / output_name.replace(".pdf", ".png")
    final_img.save(png_file, dpi=(300, 300))

    print(f"Saved combined figure to: {out_file}")
    print(f"Saved PNG version to: {png_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble R-generated plots into a combined figure"
    )
    parser.add_argument("output_dir", help="Directory containing the individual plot PDFs")
    parser.add_argument(
        "--output-name",
        default="fig2.pdf",
        help="Output filename (default: fig2.pdf)"
    )
    args = parser.parse_args()

    assemble_figure(args.output_dir, args.output_name)


if __name__ == "__main__":
    main()

import os
import glob
import re

# Per-metric colors (Wong palette — colorblind-friendly, publication-quality)
METRIC_COLORS = {
    "Lambda":      "#D55E00",  # vermillion — parameter λ
    "Loss_cql":    "#0072B2",  # deep blue — CQL loss
    "Loss_td":     "#CC79A7",  # purple — TD loss
    "Loss_total":  "#009E73",  # bluish-green — total loss
}

# Title mapping
FILE_TITLES = {
    "Lambda": "λ",
    "Loss_cql": "CQL Loss",
    "Loss_td": "TD Loss",
    "Loss_total": "Total Loss",
}


def algo_label(dirname):
    parts = dirname.split("_", 1)
    if len(parts) == 2:
        alg_num = parts[0].replace("Alg", "Alg ")
        method = parts[1]
        return f"{method} ({alg_num})"
    return dirname


def process_svg(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.splitext(os.path.basename(filepath))[0]
    dirname = os.path.basename(os.path.dirname(filepath))
    new_color = METRIC_COLORS.get(filename, "#2166AC")

    # 1. Move viewBox: "0 0 330 200" → "0 -24 330 224"
    content = content.replace(
        'viewBox="0 0 330 200"',
        'viewBox="0 -24 330 224"',
    )

    # 2. Move title above chart: y="14" → y="-8"
    content = re.sub(
        r'(<text[^>]*?)\s+y="14"',
        r'\1 y="-8"',
        content,
    )

    # 3. Update the title text — only the specific title element
    title_text = FILE_TITLES.get(filename, filename)
    label = algo_label(dirname)
    full_title = f"{title_text} — {label}"
    content = re.sub(
        r'(<text\s+x="165"\s+y="-8"[^>]*>)[^<]*(</text>)',
        rf'\1{full_title}\2',
        content,
    )

    # 4. Change stroke color to per-metric color
    # Replace #2166AC with the new color
    content = content.replace("#2166AC", new_color)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  [{new_color}] {filepath}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    svg_dir = os.path.join(base_dir, "Trained_model")
    svg_files = glob.glob(os.path.join(svg_dir, "**/*.svg"), recursive=True)

    print(f"Processing {len(svg_files)} SVG files...")
    for f in sorted(svg_files):
        process_svg(f)
    print("Done.")


if __name__ == "__main__":
    main()

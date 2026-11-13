"""Trade-off plot: harmful-request refusal against benign over-refusal, per language.

Reads each run's judged answers from the public Hub dataset (NFTKING/isoglot-runs),
computes both rates per language over the rows the judge labelled, and writes
docs/tradeoff/tradeoff.html (an SVG chart) plus tradeoff.json. Render the PNG with
any headless browser, e.g.

    msedge --headless=new --window-size=1200,700 --screenshot=tradeoff.png tradeoff.html

Standard library only, so it runs anywhere.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "NFTKING/isoglot-runs"
LANGS = {"eng": "English", "hau": "Hausa", "yor": "Yoruba", "ibo": "Igbo"}
COLORS = {"eng": "#7a6f62", "hau": "#b5651d", "yor": "#1f6b5c", "ibo": "#4a5aa8"}
# (run folder, label shown on the chart)
RUNS = [
    ("base", "Base"),
    ("steer", "Steering"),
    ("harden", "LISA 20:20"),
    ("sweep-ft30-al10", "LISA + twins 30:10"),
    ("sweep-ft36-al4", "LISA + twins 36:4"),
]
LABELS = ("Jailbroken", "Refused", "Deflected")


def fetch_judged(run: str, cache: Path | None = None, tries: int = 4) -> list[dict] | None:
    """A run's judged rows from the Hub (None if not judged yet), cached in ``cache``."""
    cached = cache / f"{run}.jsonl" if cache else None
    if cached and cached.exists():
        text = cached.read_text(encoding="utf-8")
    else:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{run}/judged.jsonl"
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    text = r.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if attempt == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == tries - 1:
                    raise
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rates(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per language: harmful refusal and benign over-refusal, over judged rows."""
    out = {}
    for lang in LANGS:
        harm = [r for r in rows if r["lang"] == lang and r["kind"] == "harmful" and r["label"] in LABELS]
        ben = [r for r in rows if r["lang"] == lang and r["kind"] == "benign" and r["label"] in LABELS]
        if harm and ben:
            out[lang] = {"harmful_refused": sum(r["label"] == "Refused" for r in harm) / len(harm),
                         "benign_over_refused": sum(r["label"] == "Refused" for r in ben) / len(ben),
                         "n_harmful": len(harm), "n_benign": len(ben)}
    return out


def svg(points: dict[str, dict[str, dict]]) -> str:
    """x = benign over-refusal (0-60%), y = harmful refused (30-100%)."""
    W, H, L, R, T, B = 1200, 700, 110, 300, 60, 90
    x = lambda v: L + v / 0.6 * (W - L - R)
    y = lambda v: H - B - (v - 0.3) / 0.7 * (H - T - B)
    g = [f'<rect width="{W}" height="{H}" fill="#f5efe5"/>']
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        g.append(f'<line x1="{x(v):.1f}" y1="{T}" x2="{x(v):.1f}" y2="{H-B}" stroke="#ddd2c2"/>'
                 f'<text x="{x(v):.1f}" y="{H-B+26}" text-anchor="middle" font-size="16" fill="#7a6f62">{v:.0%}</text>')
    for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        g.append(f'<line x1="{L}" y1="{y(v):.1f}" x2="{W-R}" y2="{y(v):.1f}" stroke="#ddd2c2"/>'
                 f'<text x="{L-12}" y="{y(v)+5:.1f}" text-anchor="end" font-size="16" fill="#7a6f62">{v:.0%}</text>')
    g.append(f'<text x="{(L+W-R)/2}" y="{H-30}" text-anchor="middle" font-size="19" fill="#1e1a15">'
             'Benign requests over-refused (lower is better)</text>')
    g.append(f'<text transform="translate(34,{(T+H-B)/2}) rotate(-90)" text-anchor="middle" font-size="19" '
             'fill="#1e1a15">Harmful requests refused (higher is better)</text>')
    g.append(f'<text x="{L}" y="36" font-size="24" font-family="Georgia, serif" fill="#1e1a15">'
             'Safety vs over-refusal, per language</text>')
    lisa = [run for run, _ in RUNS if run.startswith(("harden", "sweep")) and run in points]
    for lang, color in COLORS.items():
        # the LISA settings form the trade-off curve; base and steering are single points
        path = [(x(points[r][lang]["benign_over_refused"]), y(points[r][lang]["harmful_refused"]))
                for r in lisa if lang in points[r]]
        if len(path) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(path))
            g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" stroke-opacity="0.6"/>')
        for run in points:
            if lang not in points[run]:
                continue
            px, py = x(points[run][lang]["benign_over_refused"]), y(points[run][lang]["harmful_refused"])
            if run == "base":
                g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="#f5efe5" stroke="{color}" stroke-width="2.5"/>')
            elif run == "steer":
                g.append(f'<rect x="{px-7:.1f}" y="{py-7:.1f}" width="14" height="14" fill="{color}" fill-opacity="0.55"/>')
            else:
                k = lisa.index(run)
                g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{7 + 2 * k}" fill="{color}" '
                         f'fill-opacity="{0.55 + 0.45 * k / max(1, len(lisa) - 1):.2f}"/>')
    g.append(f'<text x="{L+14}" y="{T+26}" font-size="16" fill="#1f6b5c">&#8598; better</text>')
    ly = T + 10
    for lang, color in COLORS.items():
        g.append(f'<circle cx="{W-R+40}" cy="{ly}" r="8" fill="{color}"/>'
                 f'<text x="{W-R+56}" y="{ly+6}" font-size="18" fill="#1e1a15">{LANGS[lang]}</text>')
        ly += 32
    ly += 18
    legend = [("base", "Base model"), ("steer", "Refusal steering")] + [(r, dict(RUNS)[r]) for r in lisa]
    for run, label in legend:
        if run not in points:
            continue
        if run == "base":
            mark = f'<circle cx="{W-R+40}" cy="{ly-5}" r="7" fill="#f5efe5" stroke="#7a6f62" stroke-width="2.5"/>'
        elif run == "steer":
            mark = f'<rect x="{W-R+33}" y="{ly-12}" width="14" height="14" fill="#7a6f62" fill-opacity="0.55"/>'
        else:
            k = lisa.index(run)
            mark = (f'<circle cx="{W-R+40}" cy="{ly-5}" r="{7 + 2 * k}" fill="#7a6f62" '
                    f'fill-opacity="{0.55 + 0.45 * k / max(1, len(lisa) - 1):.2f}"/>')
        g.append(mark + f'<text x="{W-R+60}" y="{ly}" font-size="15" fill="#1e1a15">{label}</text>')
        ly += 30
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="Segoe UI, system-ui, sans-serif">' + "".join(g) + "</svg>")


def main(out_dir: str = "docs/tradeoff") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = {}
    for run, label in RUNS:
        rows = fetch_judged(run, cache=out / ".cache")
        if rows is None:
            print(f"{run}: not judged yet, skipped")
            continue
        points[run] = rates(rows)
        print(run, {k: f"{v['harmful_refused']:.1%} / {v['benign_over_refused']:.1%}" for k, v in points[run].items()})
    (out / "tradeoff.json").write_text(json.dumps(points, indent=1), encoding="utf-8")
    (out / "tradeoff.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Isoglot trade-off</title>'
        '<style>html,body{margin:0;background:#f5efe5}</style>' + svg(points), encoding="utf-8")
    print("wrote", out / "tradeoff.html")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Every number in the paper must come from the results file (or be a stated input listed below).

1. Reads the text of out/pricing.docx (body only; headings and references are skipped).
2. Every number in it must equal a value in results/pricing.json or results/pricing.parity.json, rounded to the
   decimals shown (also tried x100 for percentages and deviance x10^-2), or be one of the STATED inputs.
3. The headline sentences must say exactly what the results file says.
Author: Vivek Mitra
"""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "pricing.json").read_text())
PAR = json.loads((ROOT / "results" / "pricing.parity.json").read_text())

# Inputs the paper states rather than computes: caps, settings, bands, dataset ids, years, versions.
STATED = {"4", "1", "100,000", "80", "20", "15", "100", "0.05", "31", "200", "9", "150", "30", "2", "22", "95",
          "41214", "41215", "3164764", "18", "21", "25", "26", "31", "40", "41", "50", "51", "60", "61", "70", "71",
          "81", "101", "10", "11", "12", "13", "1989", "1993", "2010", "2011", "2013", "2014", "2017", "2018",
          "2020", "2021", "2026", "3.13", "0.15.0", "4.7.0"} | {str(i) for i in range(13)}   # 0-12: section, table, tenth numbers


def paragraphs(docx: Path) -> list[str]:
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf8")
    out = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        style = re.search(r'<w:pStyle w:val="([^"]+)"', p)
        text = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))
        if style and style.group(1).startswith("Heading"):
            if text == "References":
                break
            continue
        out.append(text)
    return out


def numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        yield len(obj)
        for v in obj:
            yield from numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


# Long chart arrays are left out, so a number cannot pass by matching some random curve point.
SUMMARY = {k: v for k, v in R.items() if k not in ("lorenz", "effects", "glm_coefficients")}
VALUES = list(numbers(SUMMARY)) + list(numbers(PAR)) + [len(R["glm_coefficients"])]


def known(tok: str) -> bool:
    if tok in STATED:
        return True
    if "e-" in tok:                                   # e.g. 7.9e-8
        return any(v and abs(abs(v) / float(tok) - 1) < 0.07 for v in VALUES)
    plain = tok.replace(",", "")
    dec = len(plain.split(".")[1]) if "." in plain else 0
    for v in VALUES:
        for s in (1, 100):
            if f"{abs(v) * s:.{dec}f}" == plain:
                return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    text = paragraphs(ROOT / "out" / "pricing.docx")
    body = "\n".join(text)
    tokens = re.findall(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)*(?:e-\d+)?", body)
    tokens = [t.rstrip(",.") for t in tokens]
    unknown = sorted({t for t in tokens if not known(t)})

    P, C, D = R["point"], R["ci95"], R["data"]
    must = [
        f"from {100 * P['dev_glm']:.3f} to {100 * P['dev_gbm']:.3f} (×10⁻²), a {P['dev_cut_pct']:.2f}% reduction",
        f"interval [{C['dev_cut_pct'][0]:.2f}, {C['dev_cut_pct'][1]:.2f}]%",
        f"raises the Gini index by {P['gini_gain']:.3f} [{C['gini_gain'][0]:.3f}, {C['gini_gain'][1]:.3f}], from {P['gini_glm']:.3f} to {P['gini_gbm']:.3f}",
        f"rises by {P['gini_pp_gain']:.3f} [{C['gini_pp_gain'][0]:.3f}, {C['gini_pp_gain'][1]:.3f}]",
        f"−{100 * P['dev_gap']:.3f} [{-100 * C['dev_gap'][1]:.3f}, {-100 * C['dev_gap'][0]:.3f}]".replace("-", "−"),
        f"scored on {D['test']:,} held-out policies",
        f"closer to the boosting prediction in {R['double_lift']['ml_closer_tenths']} of 10 groups",
        f"Gamma deviance was {R['severity']['dev_glm']:.4f}, against {R['severity']['dev_flat']:.4f}",
    ]
    missing = [m for m in must if m not in body]
    gains_ok = C["dev_cut_pct"][0] > 0 and C["gini_gain"][0] > 0 and C["gini_pp_gain"][0] > 0   # "better" needs an interval above zero
    print(f"paper check: {len(tokens)} numbers read, {len(unknown)} not traceable, {len(missing)} headline claims missing")
    for t in unknown:
        print("  not in results:", t, "|", next(p for p in text if t in p)[:110])
    for m in missing:
        print("  missing claim:", m)
    if unknown or missing or not gains_ok:
        sys.exit(1)


main()

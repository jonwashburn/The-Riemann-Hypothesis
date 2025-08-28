#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path

CANON = {
    "phi": 1.6180339887498948482,
    "ln_phi": 0.48121182505960347,
    "seed_4pi11": 138.23007675795088,
    "f_gap": 1.19737744,
    "delta_kappa": -103/(102*math.pi**5),  # -0.0032998005415146936
    "alpha_inv": None,
    "E_coh_eV": 0.0901699,
    "lambda_IR_um": None,
    "tau0_fs": 7.33,
}
CANON["alpha_inv"] = CANON["seed_4pi11"] - CANON["f_gap"] - CANON["delta_kappa"]
hc_eVnm = 1239.841984
CANON["lambda_IR_um"] = (hc_eVnm / CANON["E_coh_eV"]) / 1000.0

TOLS = {
    "alpha_inv_abs": 5e-6,
    "E_coh_abs": 5e-7,
    "lambda_IR_um_abs": 2e-4,
    "tau0_fs_abs": 1e-3,
    "ln_phi_abs": 1e-12,
    "seed_abs": 1e-10,
    "delta_kappa_abs": 5e-9,
}

N_PAT = re.compile(r'(?<![A-Za-z0-9_.-])([0-9]+\.[0-9]+)', re.UNICODE)
WS = re.compile(r'\s+')

def read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin1', errors='ignore')
    except Exception:
        return ""

def near(x: float, y: float, tol: float) -> bool:
    return abs(x - y) <= tol

def find_numeric_mentions(text: str, target: float, tol: float):
    hits = []
    for m in N_PAT.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if near(val, target, tol):
            a, b = max(0, m.start()-40), min(len(text), m.end()+40)
            ctx = WS.sub(' ', text[a:b])
            hits.append((ctx.strip(), val))
    return hits

def flag_bad_alpha_formula(text: str) -> bool:
    chunks = []
    for key in ("alpha", "fine-structure", "α", "fine structure"):
        for m in re.finditer(key, text, flags=re.IGNORECASE):
            a = max(0, m.start()-300)
            b = min(len(text), m.end()+300)
            chunks.append(text[a:b])
    bad = False
    for c in chunks:
        s = WS.sub(' ', c)
        if " - f_gap + delta_kappa" in s.replace("δκ","delta_kappa"):
            bad = True
        if ("- ln φ" in s or "- ln phi" in s) and "0.7132" in s:
            bad = True
    return bad

def label_warnings(text: str):
    warns = []
    around = []
    for key in ("λ_rec", "lambda_rec", "recognition length", "recognition-length"):
        for m in re.finditer(key, text, flags=re.IGNORECASE):
            a = max(0, m.start()-120)
            b = min(len(text), m.end()+120)
            around.append(WS.sub(' ', text[a:b]))
    for c in around:
        s = c.lower()
        if ("2.20" in s or "2.198" in s or "2.199" in s) and "µm" in s:
            warns.append("λ_rec near ~2.20 µm — should be tick_length_um.")
        if ("13.75" in s or "13.7" in s or "13.8" in s) and "µm" in s:
            warns.append("λ_rec near ~13.75 µm — this is lambda_IR_um; label accordingly.")
    return warns

def scan_file(path: Path):
    text = read_text(path)
    if not text:
        return {}
    report = {"path": str(path), "issues": []}
    for tag, target, tol in (
        ("alpha_inv", CANON["alpha_inv"], TOLS["alpha_inv_abs"]),
        ("E_coh_eV", CANON["E_coh_eV"], TOLS["E_coh_abs"]),
        ("lambda_IR_um", CANON["lambda_IR_um"], TOLS["lambda_IR_um_abs"]),
        ("tau0_fs", CANON["tau0_fs"], TOLS["tau0_fs_abs"]),
        ("ln_phi", CANON["ln_phi"], TOLS["ln_phi_abs"]),
        ("seed_4pi11", CANON["seed_4pi11"], TOLS["seed_abs"]),
        ("delta_kappa", CANON["delta_kappa"], TOLS["delta_kappa_abs"]),
    ):
        hits = find_numeric_mentions(text, target, tol)
        if tag == "alpha_inv":
            wrong = CANON["seed_4pi11"] - CANON["f_gap"] + CANON["delta_kappa"]
            bad_hits = find_numeric_mentions(text, wrong, 5e-4)
            for ctx, val in bad_hits:
                report["issues"].append({"type":"alpha_inv_wrong", "value":val, "context":ctx})
        for ctx, val in hits:
            report["issues"].append({"type":f"mention_{tag}", "value":val, "context":ctx})
    if flag_bad_alpha_formula(text):
        report["issues"].append({"type":"formula_alpha_sign", "msg":"Found likely '- f_gap + delta_kappa' motif near alpha."})
    for w in label_warnings(text):
        report["issues"].append({"type":"label_warning", "msg":w})
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="audit_report.json")
    args = ap.parse_args()
    root = Path(args.root)
    exts = {".tex", ".md", ".rst", ".html", ".htm", ".txt", ".ipynb", ".py"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    reports = []
    for p in files:
        rep = scan_file(p)
        if rep:
            reports.append(rep)
    summary = {"alpha_wrong_hits": 0, "alpha_ok_mentions": 0, "label_warnings": 0}
    for r in reports:
        for i in r["issues"]:
            if i["type"] == "alpha_inv_wrong":
                summary["alpha_wrong_hits"] += 1
            if i["type"] == "mention_alpha_inv":
                summary["alpha_ok_mentions"] += 1
            if i["type"] == "label_warning":
                summary["label_warnings"] += 1
    out = {"canon": CANON, "tolerances": TOLS, "summary": summary, "files": reports}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("=== Audit Summary ===")
    print(f"alpha_inv canonical = {CANON['alpha_inv']:.12f}")
    print(f"E_coh_eV canonical  = {CANON['E_coh_eV']}")
    print(f"lambda_IR_um canonical = {CANON['lambda_IR_um']:.6f}")
    print(f"alpha wrong-formula mentions: {summary['alpha_wrong_hits']}")
    print(f"alpha ok numeric mentions   : {summary['alpha_ok_mentions']}")
    print(f"label warnings (lambda_rec) : {summary['label_warnings']}")
    print(f"Full report -> {args.out}")

if __name__ == "__main__":
    main()



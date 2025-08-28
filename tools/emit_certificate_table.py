#!/usr/bin/env python3
import csv, sys, math
from pathlib import Path


def tex_escape(s: str) -> str:
    s = s.replace('\\', r'\textbackslash{}')
    s = s.replace('&', r'\&').replace('%', r'\%')
    s = s.replace('#', r'\#').replace('_', r'\_')
    s = s.replace('{', r'\{').replace('}', r'\}')
    s = s.replace('^', r'\^{}').replace('~', r'\~{}')
    return s


def fmt_num(x: str) -> str:
    try:
        v = float(x)
    except Exception:
        return tex_escape(x)
    if v == 0.0:
        return "0"
    av = abs(v)
    if (av >= 1e5) or (av < 1e-4):
        s = f"{v:.3e}"
        # convert e-notation to \times 10^{..}
        if 'e' in s:
            mant, exp = s.split('e')
            e = int(exp)
            return f"{mant}\\times 10^{{{e}}}"
        return s
    if av >= 100:
        return f"{v:.3f}".rstrip('0').rstrip('.')
    return f"{v:.6f}".rstrip('0').rstrip('.') or "0"


def make_table(csv_path: Path) -> str:
    with csv_path.open(newline='') as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        raise SystemExit("CSV is empty.")
    header = rows[0]
    data = rows[1:]

    # detect numeric columns
    is_num = []
    for j, _ in enumerate(header):
        flag = True
        for r in data:
            if j >= len(r):
                continue
            cell = r[j].strip()
            if cell == "":
                continue
            try:
                float(cell)
            except Exception:
                flag = False
                break
        is_num.append(flag)

    col_spec = ''.join('r' if is_num[j] else 'l' for j in range(len(header)))

    out = []
    out.append(r"\begingroup")
    out.append(r"\setlength{\tabcolsep}{5.5pt}")
    out.append(r"\renewcommand{\arraystretch}{1.12}")
    out.append(r"\begin{longtable}{" + col_spec + "}")
    out.append(r"\caption{Prime-tail covering schedule (unconditional). Columns are verbatim from \texttt{covering\_schedule\_prime\_tail.csv}.}\\")
    # header rows
    hdr = [tex_escape(h) for h in header]
    out.append(r"\toprule")
    out.append(' & '.join(rf"\texttt{{{h}}}" for h in hdr) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(' & '.join(rf"\texttt{{{h}}}" for h in hdr) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\midrule")
    out.append(r"\multicolumn{" + str(len(header)) + r"}{r}{\emph{(table continues on next page)}} \\")
    out.append(r"\bottomrule")
    out.append(r"\endfoot")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")

    for r in data:
        cells = []
        for j in range(len(header)):
            cell = r[j].strip() if j < len(r) else ""
            if is_num[j]:
                cells.append(fmt_num(cell))
            else:
                cells.append(tex_escape(cell))
        out.append(' & '.join(cells) + r" \\")

    out.append(r"\end{longtable}")
    out.append(r"\endgroup")
    return '\n'.join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: emit_certificate_table.py <CSV>", file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    print(make_table(p))



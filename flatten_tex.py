#!/usr/bin/env python3
import os
import re
import sys

INPUT_CMD_RE = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
COMMENT_RE = re.compile(r"^\s*%")

visited_stack = []


def flatten_file(path, base_dir, out_lines):
    full_path = os.path.join(base_dir, path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Included file not found: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            # leave commented lines unchanged
            if COMMENT_RE.match(line):
                out_lines.append(line)
                continue
            m = INPUT_CMD_RE.match(line)
            if m:
                inc = m.group(1)
                if not inc.endswith('.tex'):
                    inc = inc + '.tex'
                # prevent recursive include loops
                key = os.path.normpath(os.path.join(base_dir, inc))
                visited_stack.append(key)
                flatten_file(inc, base_dir, out_lines)
                visited_stack.pop()
            else:
                out_lines.append(line)


def main():
    if len(sys.argv) != 3:
        print("Usage: flatten_tex.py <input.tex> <output.tex>")
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    base_dir = os.path.dirname(os.path.abspath(in_path))
    out_lines = []
    flatten_file(os.path.basename(in_path), base_dir, out_lines)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.writelines(out_lines)


if __name__ == '__main__':
    main()

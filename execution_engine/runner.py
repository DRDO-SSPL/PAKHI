import sys

try:
    with open("code.py", "r") as f:
        exec(f.read())
except Exception as e:
    print("Execution Error:", e, file=sys.stderr)
import glob, gzip, json, shutil

parts = ["half1.jsonl"] + sorted(glob.glob("train-*-of-01357.parquet.jsonl"))

seen = set()
lines = errors = dupes = 0
files = set()

with open("results.jsonl", "w", encoding="utf-8") as out:
    for part in parts:
        n = 0
        for line in open(part, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            key = (x["video"], x.get("second", "error"), x.get("error"))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            out.write(line + "\n")
            n += 1
            lines += 1
            if "error" in x:
                errors += 1
                if errors <= 5:
                    print("ERROR:", x)
            files.add(x["video"].split(":")[0])
        print(part, "->", n, "lines")

with open("results.jsonl", "rb") as f_in, gzip.open("results.jsonl.gz", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

missing = [i for i in range(50) if f"train-{i:05d}-of-01357.parquet" not in files]
print()
print(lines, "lines |", errors, "error lines |", dupes, "duplicates skipped")
print(len(files), "files | missing:", missing)
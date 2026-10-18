import os
import re
import glob
import shutil
import hashlib
import json
import pandas as pd

# =========================================================
# Settings
# =========================================================

INPUT_FILE = r"C:\Users\User\Documents\twitter_sample_tweets.csv\twitter_sample_tweets.csv"

OUTPUT_DIR = "cleaned"

CHUNK_SIZE = 200_000

TEXT_COLUMN = 1

MIN_WORDS = 3

# Every part file has exactly PART_ROWS rows,
# except the last one which gets whatever is left.
PART_ROWS = 200_000

# At the very end, merge all part files into this single file
MERGED_FILE = os.path.join(OUTPUT_DIR, "merged.csv")

# True  -> delete part files after a successful merge
# False -> keep part files next to merged.csv
DELETE_PARTS_AFTER_MERGE = False


# =========================================================
# Regular Expressions
# =========================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

MENTION_RE = re.compile(r"@\w+")

HASHTAG_RE = re.compile(r"#(\w+)")

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "\U0001F000-\U0001F2FF"
    "\uFE0F"
    "\u200D"
    "\u2B00-\u2BFF]+"
)

DIGIT_RE = re.compile(
    r"[0-9\u06F0-\u06F9\u0660-\u0669]+"
)

INVALID_RE = re.compile(
    r"[^a-zA-Z\u0600-\u06FF\u200c\s.,!?؟،؛:]"
)

REPEAT_RE = re.compile(r"(.)\1{2,}")

PUNCT_RE = re.compile(r"([.,!?؟،؛:])\1+")

SPACE_RE = re.compile(r"\s+")

ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
    "أ": "ا",
    "إ": "ا",
    "ؤ": "و",
})


# =========================================================
# Clean one tweet
# =========================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(r"^\s*RT\b[: ]*", "", text, flags=re.IGNORECASE)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = EMOJI_RE.sub(" ", text)
    text = text.translate(ARABIC_TO_PERSIAN)
    text = DIGIT_RE.sub(" ", text)
    text = INVALID_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1", text)
    text = PUNCT_RE.sub(r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()

    return text


# =========================================================
# Save one part file
# =========================================================

def write_part(df, part_number):
    path = os.path.join(OUTPUT_DIR, f"part_{part_number:04d}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> saved {path} ({len(df)} rows)")


# =========================================================
# Merge all part files into one file
# =========================================================

def merge_parts():

    part_files = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "part_*.csv"))
    )

    if not part_files:
        print("No part files found to merge.")
        return

    print()
    print(f"Merging {len(part_files)} files into {MERGED_FILE} ...")

    with open(MERGED_FILE, "wb") as out:

        for i, path in enumerate(part_files):

            with open(path, "rb") as src:

                if i == 0:
                    # First file: copy everything (header included)
                    shutil.copyfileobj(src, out, 16 * 1024 * 1024)
                else:
                    # Other files: skip their header line
                    # (the BOM sits on that line, so it is skipped too)
                    src.readline()
                    shutil.copyfileobj(src, out, 16 * 1024 * 1024)

    size_gb = os.path.getsize(MERGED_FILE) / 1024 ** 3
    print(f"Merged file created: {MERGED_FILE} ({size_gb:.2f} GB)")

    if DELETE_PARTS_AFTER_MERGE:
        for path in part_files:
            os.remove(path)
        print("Part files deleted.")


# =========================================================
# Main processing
# =========================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove old outputs so nothing stale gets merged
    for old in glob.glob(os.path.join(OUTPUT_DIR, "part_*.csv")):
        os.remove(old)
    if os.path.exists(MERGED_FILE):
        os.remove(MERGED_FILE)

    # Shared between ALL chunks -> duplicates removed globally
    seen = set()

    total_rows = 0
    kept_rows = 0
    removed_empty = 0
    removed_short = 0
    removed_duplicates = 0

    # Cleaned rows waiting to be written as fixed-size parts
    pending = None
    part_number = 0

    print("Starting cleaning...")
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print()

    reader = pd.read_csv(
        INPUT_FILE,
        header=None,
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="skip"
    )

    for chunk_number, df in enumerate(reader):

        print(
            f"Processing chunk {chunk_number} "
            f"({len(df)} rows)..."
        )

        total_rows += len(df)

        text = df[TEXT_COLUMN].fillna("")
        cleaned = text.map(clean_text)

        empty_mask = cleaned.eq("")
        removed_empty += int(empty_mask.sum())

        word_count = cleaned.str.split().str.len()

        short_mask = (
            ~empty_mask &
            (word_count < MIN_WORDS)
        )
        removed_short += int(short_mask.sum())

        valid_mask = ~(empty_mask | short_mask)

        df = df[valid_mask].copy()
        cleaned = cleaned[valid_mask]

        # ---------------- Remove duplicates ----------------

        keep_indices = []

        for index, tweet in cleaned.items():

            tweet_hash = int.from_bytes(
                hashlib.md5(tweet.encode("utf-8")).digest()[:8],
                "big"
            )

            if tweet_hash in seen:
                removed_duplicates += 1
            else:
                seen.add(tweet_hash)
                keep_indices.append(index)

        df = df.loc[keep_indices]
        cleaned = cleaned.loc[keep_indices]

        df["clean_text"] = cleaned

        kept_rows += len(df)

        # ---------------- Fixed-size parts ----------------

        if pending is None:
            pending = df
        else:
            pending = pd.concat([pending, df], ignore_index=True)

        while len(pending) >= PART_ROWS:
            write_part(pending.iloc[:PART_ROWS], part_number)
            part_number += 1
            pending = pending.iloc[PART_ROWS:]

        print(
            f"Finished chunk {chunk_number} | "
            f"kept={len(df)} | "
            f"duplicates={removed_duplicates}"
        )
        print()

    # Last part: whatever is left
    if pending is not None and len(pending) > 0:
        write_part(pending, part_number)
        part_number += 1

    # =====================================================
    # Final step: merge everything into one file
    # =====================================================

    merge_parts()

    # =====================================================
    # Report
    # =====================================================

    report = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_empty": removed_empty,
        "removed_short": removed_short,
        "removed_duplicates": removed_duplicates,
        "part_files": part_number,
    }

    with open(
        os.path.join(OUTPUT_DIR, "report.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print()
    print("=" * 50)
    print("CLEANING FINISHED")
    print("=" * 50)

    print(f"Total rows:         {total_rows}")
    print(f"Kept rows:          {kept_rows}")
    print(f"Empty removed:      {removed_empty}")
    print(f"Short removed:      {removed_short}")
    print(f"Duplicates removed: {removed_duplicates}")
    print(f"Part files:         {part_number}")
    print(f"Merged file:        {MERGED_FILE}")


if __name__ == "__main__":
    main()

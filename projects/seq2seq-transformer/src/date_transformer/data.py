"""Deterministic unique-date splits and a fixed, documented character vocabulary."""

import csv
import datetime as dt
import hashlib
import io
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
PAD, BOS, EOS = 0, 1, 2
VOCAB = ["<PAD>", "<BOS>", "<EOS>"] + sorted(set("".join(MONTHS) + "0123456789 ,-"))
IDS = {char: index for index, char in enumerate(VOCAB)}
SOURCE_LENGTH, TARGET_LENGTH = 20, 12
FIRST, LAST = dt.date(1950, 1, 1), dt.date(2050, 12, 31)


def human(date):
    return f"{MONTHS[date.month - 1]} {date.day}, {date.year}"


def parse_human(text):
    if not isinstance(text, str):
        raise TypeError("Expected an English date string")
    match = re.fullmatch(r"([A-Za-z]+) ([1-9][0-9]?), ([0-9]{4})", text)
    if match is None or match[1] not in MONTHS:
        raise ValueError("Use canonical English dates, e.g. August 3, 2026")
    try:
        date = dt.date(int(match[3]), MONTHS.index(match[1]) + 1, int(match[2]))
    except ValueError as error:
        raise ValueError("Invalid calendar date") from error
    if not FIRST <= date <= LAST or human(date) != text:
        raise ValueError(
            "Supported dates are canonical English dates from 1950 through 2050"
        )
    return date


def make_splits(config):
    counts = [
        config[name] for name in ["train_count", "validation_count", "test_count"]
    ]
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts
    ):
        raise ValueError("Every split count must be a positive integer")
    calendar = [
        FIRST + dt.timedelta(days=index) for index in range((LAST - FIRST).days + 1)
    ]
    if sum(counts) > len(calendar):
        raise ValueError("Requested split counts exceed unique dates in the calendar")
    random.Random(config["seed"]).shuffle(calendar)
    result, start = {}, 0
    for name, count in zip(["train", "validation", "test"], counts):
        result[name] = [
            (human(date), date.isoformat()) for date in calendar[start : start + count]
        ]
        start += count
    return result


def csv_bytes(pairs):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["human_date", "iso_date"])
    writer.writerows(pairs)
    return stream.getvalue().encode("utf-8")


def export_splits(splits, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, pairs in splits.items():
        content = csv_bytes(pairs)
        (output / (name + ".csv")).write_bytes(content)
        records[name] = {
            "count": len(pairs),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest = {
        "generator": "unique_gregorian_english_dates_v1",
        "first": FIRST.isoformat(),
        "last": LAST.isoformat(),
        "seed": config["seed"],
        "sampling": "one seeded shuffle, disjoint slices, all valid calendar days",
        "vocabulary": VOCAB,
        "splits": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def encode(text, length, bos=False, eos=False):
    try:
        values = (
            ([BOS] if bos else [])
            + [IDS[char] for char in text]
            + ([EOS] if eos else [])
        )
    except KeyError as error:
        raise ValueError(f"Unsupported character: {error.args[0]!r}") from error
    if len(values) > length:
        raise ValueError("Input exceeds configured length; truncation is not allowed")
    return torch.tensor(values + [PAD] * (length - len(values)), dtype=torch.long)


class Dates(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        return (
            encode(source, SOURCE_LENGTH),
            encode(target, TARGET_LENGTH, bos=True),
            encode(target, TARGET_LENGTH, eos=True),
        )

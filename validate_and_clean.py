#!/usr/bin/env python3
"""
validate_and_clean_classes.py

Usage:
  python validate_and_clean_classes.py --input data/schemes_multilingual.csv \
      --config config.yaml --output_dir ./validation_out --clean
"""

import argparse
import os
import json
import csv
import statistics
import pandas as pd
import yaml
import chardet
import re
from urllib.parse import urlparse


# ---------------------------
# Encoding Detector
# ---------------------------
class EncodingDetector:
    @staticmethod
    def detect(filepath, nbytes=4096):
        with open(filepath, "rb") as f:
            raw = f.read(nbytes)
        res = chardet.detect(raw)
        return res.get("encoding", "utf-8"), res.get("confidence", 0.0)


# ---------------------------
# Language Checker
# ---------------------------
class LanguageChecker:
    @staticmethod
    def has_devanagari(s: str):
        return any(0x0900 <= ord(ch) <= 0x097F for ch in s)

    @staticmethod
    def has_gurmukhi(s: str):
        return any(0x0A00 <= ord(ch) <= 0x0A7F for ch in s)

    @staticmethod
    def has_latin(s: str):
        return bool(re.search(r"[A-Za-z]", s))


# ---------------------------
# URL Validator
# ---------------------------
class URLValidator:
    @staticmethod
    def is_ok(u: str):
        try:
            p = urlparse(u)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False


# ---------------------------
# Data Validator
# ---------------------------
class DataValidator:
    def __init__(self, input_file, config_file=None):
        self.input_file = input_file
        self.config_file = config_file
        self.df = None
        self.report = {}
        self.lang_specs = []
        self.id_field = None
        self.problem_rows = []

    def load_config(self):
        if self.config_file:
            with open(self.config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        self.lang_specs = cfg.get("languages", [
            {"col": "english", "code": "en"},
            {"col": "hindi", "code": "hi"},
            {"col": "punjabi", "code": "pun"}
        ])

    def load_data(self):
        enc, conf = EncodingDetector.detect(self.input_file)
        print(f"[i] Detected encoding: {enc} (confidence {conf:.2f})")
        try:
            self.df = pd.read_csv(self.input_file, encoding=enc)
        except Exception:
            print("[!] Failed with detected encoding, retrying with utf-8 replace.")
            self.df = pd.read_csv(self.input_file, encoding="utf-8", errors="replace")

        self.df.columns = [c.strip() for c in self.df.columns]
        self.report["encoding_detected"] = enc
        self.report["encoding_confidence"] = conf
        self.report["n_rows"] = len(self.df)
        self.report["columns"] = list(self.df.columns)

    def detect_id_field(self):
        candidates = ["id", "scheme_id", "schemeId", "schemeID", "schemeid"]
        for cand in candidates:
            if cand in self.df.columns:
                self.id_field = cand
                break
        if not self.id_field:
            self.id_field = self.df.columns[0]
            print(f"[!] No canonical ID found, using first column '{self.id_field}'.")
        self.report["id_field_used"] = self.id_field

    def validate_languages(self):
        n_rows = len(self.df)
        lang_stats = {}
        for spec in self.lang_specs:
            col = spec["col"]
            if col not in self.df.columns:
                lang_stats[col] = {"present": False}
                continue
            texts = self.df[col].astype(str).fillna("").map(lambda s: s.strip())
            lengths = texts.map(len).tolist()
            n_non_empty = sum(bool(s) for s in texts)

            # choose script check
            if col.lower().startswith("hindi"):
                script_check = texts.map(LanguageChecker.has_devanagari)
            elif col.lower().startswith("punjabi") or col.lower().startswith("pa"):
                script_check = texts.map(LanguageChecker.has_gurmukhi)
            else:
                script_check = texts.map(LanguageChecker.has_latin)

            n_script_ok = int(script_check.sum())
            lengths_nonzero = [l for l in lengths if l > 0]

            length_summary = {
                "min": min(lengths_nonzero) if lengths_nonzero else 0,
                "median": int(statistics.median(lengths_nonzero)) if lengths_nonzero else 0,
                "max": max(lengths_nonzero) if lengths_nonzero else 0,
                "mean": float(statistics.mean(lengths_nonzero)) if lengths_nonzero else 0.0,
            }

            lang_stats[col] = {
                "present": True,
                "n_non_empty": n_non_empty,
                "pct_non_empty": round(100.0 * n_non_empty / n_rows, 2) if n_rows > 0 else None,
                "n_script_ok": n_script_ok,
                "pct_script_ok": round(100.0 * n_script_ok / (n_non_empty or 1), 2),
                "lengths": length_summary,
            }
        self.report["language_checks"] = lang_stats

    def validate_urls(self):
        if "source_url" in self.df.columns:
            urls = self.df["source_url"].astype(str).fillna("").map(lambda s: s.strip())
            url_ok_mask = urls.map(URLValidator.is_ok)
            n_url_ok = int(url_ok_mask.sum())
            self.report["n_url_ok"] = n_url_ok
        else:
            self.report["n_url_ok"] = None

    def detect_problems(self):
        self.problem_rows = []
        for i, row in self.df.iterrows():
            problems = []
            if pd.isna(row[self.id_field]) or str(row[self.id_field]).strip() == "":
                problems.append("missing_id")
            if all(str(row.get(spec["col"], "")).strip() == "" for spec in self.lang_specs):
                problems.append("all_lang_empty")
            if "source_url" in self.df.columns:
                u = str(row.get("source_url", "")).strip()
                if u == "" or not URLValidator.is_ok(u):
                    problems.append("bad_url")
            if problems:
                r = dict(row)
                r["_row_index"] = int(i)
                r["_problems"] = problems
                self.problem_rows.append(r)
        self.report["n_problem_rows"] = len(self.problem_rows)

    def generate_report(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        print(f"[+] Validation report saved: {report_path}")
        if self.problem_rows:
            tofix_path = os.path.join(output_dir, "to_fix.csv")
            with open(tofix_path, "w", encoding="utf-8", newline="") as wf:
                writer = csv.DictWriter(wf, fieldnames=self.problem_rows[0].keys())
                writer.writeheader()
                writer.writerows(self.problem_rows)
            print(f"[!] {len(self.problem_rows)} problematic rows -> {tofix_path}")
        return report_path


# ---------------------------
# Cleaner
# ---------------------------
class Cleaner:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for c in cleaned.columns:
            if cleaned[c].dtype == object:
                cleaned[c] = cleaned[c].astype(str).map(
                    lambda s: s.strip().replace("\uFEFF", "")
                )
        return cleaned

    @staticmethod
    def save_cleaned(df: pd.DataFrame, output_dir: str, input_file: str):
        cleaned_path = os.path.join(output_dir, f"cleaned_{os.path.basename(input_file)}")
        df.to_csv(cleaned_path, index=False, encoding="utf-8")
        print(f"[+] Cleaned CSV saved: {cleaned_path}")
        return cleaned_path


# ---------------------------
# Runner
# ---------------------------
class ValidationRunner:
    def __init__(self, input_file, config_file, output_dir, clean=False):
        self.input_file = input_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.clean = clean

    def run(self):
        validator = DataValidator(self.input_file, self.config_file)
        validator.load_config()
        validator.load_data()
        validator.detect_id_field()
        validator.validate_languages()
        validator.validate_urls()
        validator.detect_problems()

        report_path = validator.generate_report(self.output_dir)

        if self.clean:
            cleaned = Cleaner.clean_dataframe(validator.df)
            Cleaner.save_cleaned(cleaned, self.output_dir, self.input_file)

        print("==== SUMMARY ====")
        print(f"Rows: {validator.report['n_rows']}  |  Columns: {len(validator.report['columns'])}")
        print(f"ID field: {validator.id_field}")
        for col, stats in validator.report["language_checks"].items():
            if not stats.get("present", False):
                print(f" - {col}: MISSING")
            else:
                print(f" - {col}: {stats['n_non_empty']}/{validator.report['n_rows']} non-empty "
                      f"| script_ok {stats['n_script_ok']}/{stats['n_non_empty']} "
                      f"| median length {stats['lengths']['median']}")
        if validator.problem_rows:
            print(f"[!] Problem rows: {len(validator.problem_rows)}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default='data/schemes_multilingual.csv', help="CSV input file")
    p.add_argument("--config", default="config.yaml", help="Config YAML with language columns")
    p.add_argument("--output_dir", default="./validation_out", help="Output directory")
    p.add_argument("--clean", action="store_true", help="Write cleaned CSV")
    args = p.parse_args()

    runner = ValidationRunner(args.input, args.config, args.output_dir, args.clean)
    runner.run()

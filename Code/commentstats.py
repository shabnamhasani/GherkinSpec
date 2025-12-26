# Step 1: Robustly count unique comments per reviewer from all sheets except "sheet1"

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

FILE_PATH = "/home/shabnam/Gherkin/Data/input/summary-annotated copy.xlsx"

def find_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to detect reviewer and comment columns heuristically.
    Returns (reviewer_col, comment_col). Either can be None if not found.
    """
    # Normalize column names for matching
    cols = list(df.columns)
    norm = {c: re.sub(r"\s+", " ", str(c).strip().lower()) for c in cols}
    
    # Candidate patterns
    reviewer_patterns = [
        r"\breviewer\b", r"\brater\b", r"\bevaluator\b", r"\bjudge\b",
        r"\bannotator\b", r"\bparticipant\b", r"\bname\b", r"\buser\b"
    ]
    comment_patterns = [
        r"\bcomment\b", r"\bcomments\b", r"\bfeedback\b", r"\bnote\b",
        r"\bnotes\b", r"\bobservation\b", r"\bobservations\b", r"\btext\b",
        r"\bremark\b", r"\bmessage\b"
    ]
    
    reviewer_col = None
    comment_col = None
    
    # First, exact-like matches
    for c, n in norm.items():
        if reviewer_col is None and any(re.search(p, n) for p in reviewer_patterns):
            reviewer_col = c
        if comment_col is None and any(re.search(p, n) for p in comment_patterns):
            comment_col = c
        if reviewer_col and comment_col:
            break
    
    # If still missing comment_col, try common fallbacks
    if comment_col is None:
        # Prefer longer text-like columns (object dtype) as "comment" when other numeric columns exist.
        object_cols = [c for c in cols if df[c].dtype == 'object']
        # Heuristic: choose the object column with the longest average string length
        if object_cols:
            avg_lengths = []
            for c in object_cols:
                series = df[c].dropna().astype(str)
                if len(series) == 0:
                    avg_lengths.append((c, 0))
                else:
                    avg_len = series.str.len().mean()
                    avg_lengths.append((c, avg_len))
            # Pick the top one if it seems clearly texty
            avg_lengths.sort(key=lambda x: x[1], reverse=True)
            if avg_lengths and avg_lengths[0][1] >= 10:  # at least somewhat text-like
                comment_col = avg_lengths[0][0]
    
    return reviewer_col, comment_col

def normalize_comment(text: str) -> str:
    """
    Normalize a comment string for deduplication:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Remove zero-width / non-breaking spaces
    """
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.strip().lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text

def load_all_sheets_except(file_path: str, exclude_sheets: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load all sheets into dataframes except those whose names (case-insensitive)
    are in exclude_sheets.
    """
    xls = pd.ExcelFile(file_path)
    all_sheet_names = xls.sheet_names
    exclude_norm = {s.strip().lower() for s in exclude_sheets}
    keep = [s for s in all_sheet_names if s.strip().lower() not in exclude_norm]
    return {s: pd.read_excel(file_path, sheet_name=s) for s in keep}

def count_unique_comments_per_reviewer(file_path: str, exclude_sheets: List[str]) -> pd.DataFrame:
    sheets = load_all_sheets_except(file_path, exclude_sheets)
    
    records = []  # (sheet, reviewer, original_comment, normalized_comment)
    
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        
        reviewer_col, comment_col = find_columns(df)
        
        # Skip sheet if we can't confidently identify both columns
        if reviewer_col is None or comment_col is None:
            # Try a conservative fallback: if there are at least 2 object columns, assume the first is reviewer and second is comment
            object_cols = [c for c in df.columns if df[c].dtype == 'object']
            if reviewer_col is None and object_cols:
                reviewer_col = object_cols[0]
            if comment_col is None and len(object_cols) >= 2:
                # choose the more text-like of the remaining as comment
                potentials = [c for c in object_cols if c != reviewer_col]
                if potentials:
                    # choose the one with largest average length
                    avg_lengths = []
                    for c in potentials:
                        series = df[c].dropna().astype(str)
                        if len(series) == 0:
                            avg_lengths.append((c, 0))
                        else:
                            avg_len = series.str.len().mean()
                            avg_lengths.append((c, avg_len))
                    avg_lengths.sort(key=lambda x: x[1], reverse=True)
                    comment_col = avg_lengths[0][0] if avg_lengths else None
        
        if reviewer_col is None or comment_col is None:
            # Cannot process this sheet
            continue
        
        subset = df[[reviewer_col, comment_col]].copy()
        subset.columns = ["Reviewer", "Comment"]
        subset = subset.dropna(subset=["Reviewer", "Comment"])
        # Normalize
        subset["Reviewer"] = subset["Reviewer"].astype(str).str.strip()
        subset["Comment_norm"] = subset["Comment"].apply(normalize_comment)
        # Drop empty normalized comments
        subset = subset[subset["Comment_norm"].str.len() > 0]
        
        for _, row in subset.iterrows():
            records.append((sheet_name, row["Reviewer"], row["Comment"], row["Comment_norm"]))
    
    if not records:
        return pd.DataFrame(columns=["Reviewer", "Unique_Comments_Count"])
    
    rec_df = pd.DataFrame(records, columns=["Sheet", "Reviewer", "Comment", "Comment_norm"])
    
    # Unique per (Reviewer, Comment_norm) across ALL included sheets
    unique_pairs = rec_df.drop_duplicates(subset=["Reviewer", "Comment_norm"])
    
    counts = (
        unique_pairs.groupby("Reviewer", dropna=False)["Comment_norm"]
        .nunique()
        .reset_index(name="Unique_Comments_Count")
        .sort_values(by=["Unique_Comments_Count", "Reviewer"], ascending=[False, True])
        .reset_index(drop=True)
    )
    
    return counts

# Execute the counting
counts_df = count_unique_comments_per_reviewer(FILE_PATH, exclude_sheets=["sheet1"])

# Save results
out_csv = "/home/shabnam/Gherkin/Evaluation/comments/unique_comment_counts_by_reviewer.csv"
counts_df.to_csv(out_csv, index=False)


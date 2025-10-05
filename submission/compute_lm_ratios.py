import os
import re
import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")


def load_lm_dictionary(
    lm_csv_path="./data/text/Loughran-McDonald_MasterDictionary_1993-2024.csv",
):
    """Load Loughran–McDonald dictionary from CSV."""
    lm_dict = {}
    with open(lm_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].lower()
            lm_dict[word] = {
                "positive": int(row["Positive"]),
                "negative": int(row["Negative"]),
                "uncertainty": int(row["Uncertainty"]),
                "litigious": int(row["Litigious"]),
                "strong_modal": int(row["Strong_Modal"]),
                "weak_modal": int(row["Weak_Modal"]),
                "constraining": int(row["Constraining"]),
            }
    return lm_dict


lemmatizer = WordNetLemmatizer()


def preprocess_text(text, lemmatize=True):
    """Preprocess text without stopword removal."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\W+", " ", text)  # remove punctuation
    tokens = word_tokenize(text)
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def compute_ratios(tokens, lm_dict):
    """Compute LM dictionary ratios for a given tokenized document."""
    counts = {
        k: 0
        for k in [
            "positive",
            "negative",
            "uncertainty",
            "litigious",
            "strong_modal",
            "weak_modal",
            "constraining",
        ]
    }
    total = len(tokens) if len(tokens) > 0 else 1  # avoid division by zero

    for t in tokens:
        if t in lm_dict:
            for cat in counts:
                counts[cat] += lm_dict[t][cat]

    ratios = {f"{cat}_ratio": counts[cat] / total for cat in counts}
    return ratios


def compute_lm_ratios_all_years(
    data_dir="./data/text",
    start_year=2005,
    end_year=2025,
    lm_csv_path="./data/text/Loughran-McDonald_MasterDictionary_1993-2024.csv",
):
    """Compute Loughran–McDonald ratios for all yearly text files."""
    lm_dict = load_lm_dictionary(lm_csv_path)
    all_years = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f"text_us_{year}.pkl")
        if not os.path.exists(file_path):
            print(f"Skipping {year}, file not found")
            continue

        df_year = pd.read_pickle(file_path)
        print(f"Processing year {year}, {len(df_year)} rows...")

        rows = []
        for _, row in tqdm(df_year.iterrows(), total=len(df_year), desc=f"Year {year}"):
            gvkey = row["gvkey"]
            yr = row["year"]

            # Combine rf and mgmt text
            combined_text = ""
            if pd.notna(row.get("rf", None)):
                combined_text += str(row["rf"]) + " "
            if pd.notna(row.get("mgmt", None)):
                combined_text += str(row["mgmt"])

            if combined_text.strip() == "":
                continue  # skip rows with no text

            tokens = preprocess_text(combined_text, lemmatize=True)
            ratios = compute_ratios(tokens, lm_dict)

            entry = {"gvkey": gvkey, "year": yr}
            entry.update(ratios)
            rows.append(entry)

        all_years.extend(rows)
        print(f"Finished year {year}, collected {len(rows)} rows")

    df_all = pd.DataFrame(all_years)
    return df_all

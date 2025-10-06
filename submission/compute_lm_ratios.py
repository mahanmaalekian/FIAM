import os
import re
import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# ---------------------------------------------------------
# Download required NLTK resources (only runs the first time)
# ---------------------------------------------------------
nltk.download("punkt")
nltk.download("wordnet")


def load_lm_dictionary(
    lm_csv_path="./data/text/Loughran-McDonald_MasterDictionary_1993-2024.csv",
):
    """
    Load the Loughran–McDonald sentiment dictionary from a CSV file.

    Args:
        lm_csv_path (str): Path to the LM dictionary CSV.

    Returns:
        dict: A dictionary mapping each word to its sentiment category scores.
    """
    lm_dict = {}

    # Open and read CSV file
    with open(lm_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].lower()  # convert word to lowercase for consistency

            # Store all relevant sentiment and modality scores as integers
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


# Initialize NLTK’s lemmatizer globally for reuse
lemmatizer = WordNetLemmatizer()


def preprocess_text(text, lemmatize=True):
    """
    Basic text preprocessing (no stopword removal).

    Steps:
      1. Lowercase the text
      2. Remove numbers
      3. Remove punctuation and special characters
      4. Tokenize into words
      5. Optionally lemmatize tokens

    Args:
        text (str): Input text to clean.
        lemmatize (bool): Whether to apply lemmatization.

    Returns:
        list[str]: List of cleaned tokens.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove digits
    text = re.sub(r"\W+", " ", text)  # remove punctuation and special chars
    tokens = word_tokenize(text)  # tokenize text into individual words

    # Lemmatize tokens (convert to base word form)
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def compute_ratios(tokens, lm_dict):
    """
    Compute sentiment ratios using the Loughran–McDonald dictionary.

    For each category (positive, negative, etc.), it counts how many
    tokens belong to that category, then divides by total token count.

    Args:
        tokens (list[str]): Tokenized text.
        lm_dict (dict): Loughran–McDonald dictionary.

    Returns:
        dict: Ratios for each sentiment category in the document.
    """

    # Initialize counters for all sentiment categories
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

    # Total number of tokens (avoid division by zero)
    total = len(tokens) if len(tokens) > 0 else 1

    # Count matches between tokens and dictionary words
    for t in tokens:
        if t in lm_dict:
            for cat in counts:
                counts[cat] += lm_dict[t][cat]

    # Compute normalized ratios (category count / total words)
    ratios = {f"{cat}_ratio": counts[cat] / total for cat in counts}
    return ratios


def compute_lm_ratios_all_years(
    data_dir="./data/text",
    start_year=2005,
    end_year=2025,
    lm_csv_path="./data/text/Loughran-McDonald_MasterDictionary_1993-2024.csv",
):
    """
    Compute Loughran–McDonald sentiment ratios for all yearly firm text files.

    Each year's file (e.g., text_us_YYYY.pkl) must contain:
        - 'rf': Risk Factors text
        - 'mgmt': Management Discussion text
        - 'gvkey': Firm identifier
        - 'year': Fiscal year

    The function merges 'rf' and 'mgmt' fields, tokenizes them,
    computes sentiment ratios, and aggregates results for all years.

    Args:
        data_dir (str): Directory containing yearly .pkl text files.
        start_year (int): First year to process (inclusive).
        end_year (int): Last year to process (inclusive).
        lm_csv_path (str): Path to the LM dictionary CSV.

    Returns:
        pd.DataFrame: DataFrame containing gvkey, year, and LM ratios.
    """

    # ---------------------------------------------------------
    # 1. Load LM dictionary once for reuse
    # ---------------------------------------------------------
    lm_dict = load_lm_dictionary(lm_csv_path)
    all_years = []  # store all processed entries across years

    # ---------------------------------------------------------
    # 2. Loop over each year and process corresponding file
    # ---------------------------------------------------------
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f"text_us_{year}.pkl")

        # Skip missing files
        if not os.path.exists(file_path):
            print(f"Skipping {year}, file not found")
            continue

        # Load firm-level text data for that year
        df_year = pd.read_pickle(file_path)
        print(f"Processing year {year}, {len(df_year)} rows...")

        rows = []  # store LM ratio results for this year

        # ---------------------------------------------------------
        # 3. Iterate over each firm document in the year’s dataset
        # ---------------------------------------------------------
        for _, row in tqdm(df_year.iterrows(), total=len(df_year), desc=f"Year {year}"):
            gvkey = row["gvkey"]
            yr = row["year"]

            # Merge 'rf' and 'mgmt' text fields (if available)
            combined_text = ""
            if pd.notna(row.get("rf", None)):
                combined_text += str(row["rf"]) + " "
            if pd.notna(row.get("mgmt", None)):
                combined_text += str(row["mgmt"])

            # Skip empty text entries
            if combined_text.strip() == "":
                continue

            # Tokenize and lemmatize the text
            tokens = preprocess_text(combined_text, lemmatize=True)

            # Compute LM category ratios
            ratios = compute_ratios(tokens, lm_dict)

            # Create result entry for this firm-year
            entry = {"gvkey": gvkey, "year": yr}
            entry.update(ratios)
            rows.append(entry)

        # Append year’s results to global list
        all_years.extend(rows)
        print(f"Finished year {year}, collected {len(rows)} rows")

    # ---------------------------------------------------------
    # 4. Combine all firm-year entries into a single DataFrame
    # ---------------------------------------------------------
    df_all = pd.DataFrame(all_years)
    return df_all

import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def run_finbert_sentiment_analysis(
    data_dir="./data/text",
    output_csv="text_data_sentiment.csv",
    model_name="nickmuchi/sec-bert-finetuned-finance-classification",
    start_year=2005,
    end_year=2025,
    batch_size=128,
):
    """
    Runs FinBERT sentiment analysis on yearly text files containing 'rf' and 'mgmt' fields.

    Args:
        data_dir (str): Directory containing yearly .pkl files (e.g., text_us_YYYY.pkl).
        output_csv (str): Path to save the output CSV.
        model_name (str): Hugging Face model name for FinBERT.
        start_year (int): Starting year (inclusive).
        end_year (int): Ending year (inclusive).
        batch_size (int): Batch size for model inference.

    Returns:
        pd.DataFrame: Combined dataframe with sentiment probabilities per gvkey per year.
    """

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_results = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f"text_us_{year}.pkl")
        if not os.path.exists(file_path):
            print(f"Skipping {year}, file not found.")
            continue

        df_year = pd.read_pickle(file_path)
        print(f"\nProcessing {year}, total rows: {len(df_year)}")

        # Merge rf + mgmt text fields
        combined_texts, gvkeys = [], []
        for _, row in df_year.iterrows():
            rf_text = str(row["rf"]) if pd.notna(row["rf"]) else ""
            mgmt_text = str(row["mgmt"]) if pd.notna(row["mgmt"]) else ""
            merged = (rf_text + " " + mgmt_text).strip()
            if merged:
                combined_texts.append(merged)
                gvkeys.append(row["gvkey"])

        if not combined_texts:
            print(f"No text to process for {year}, skipping.")
            continue

        all_probs = []
        with torch.no_grad():
            print(
                f"Processing {len(combined_texts)} documents in batches of {batch_size}..."
            )
            for i in tqdm(
                range(0, len(combined_texts), batch_size), desc=f"Year {year}"
            ):
                batch_texts = combined_texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                all_probs.extend(probs.cpu().tolist())

        # Make a dataframe for this year
        df_probs = pd.DataFrame(all_probs, columns=model.config.id2label.values())
        df_probs["gvkey"] = gvkeys
        df_probs["year"] = year
        all_results.append(df_probs)

        print(f"Finished processing year {year}, collected {len(df_probs)} rows.\n")

    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    print(f"All years combined: {len(df_all)} rows")

    # Save output
    df_all.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")

    return df_all

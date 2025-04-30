import csv
import re
from collections import defaultdict
from datetime import datetime

# Load Bing Liu’s lexicons
def load_lexicon(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(word.strip() for word in f if word.strip() and not word.startswith(';'))

positive_words = load_lexicon(r"C:\Users\johnn\Downloads\clean_expanded_crypto_positive_lexicon.txt")
negative_words = load_lexicon(r"C:\Users\johnn\Downloads\clean_expanded_crypto_negative_lexicon.txt")

# Function to clean and tokenize text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    words = re.findall(r'\b\w+\b', text.lower())  # extract words
    return words

# Dictionary to hold daily word counts
daily_stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})

# Read and process the input CSV
with open(r"C:\Users\johnn\Downloads\2024_dogecoin_comments.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        try:
            # Adjust if column index differs (0-based indexing)
            date = datetime.strptime(row[1], '%Y-%m-%d').date()
            text = f"{row[4]}"
            words = clean_text(text)

            for word in words:
                if word in positive_words:
                    daily_stats[date]['pos'] += 1
                elif word in negative_words:
                    daily_stats[date]['neg'] += 1
                daily_stats[date]['total'] += 1
        except Exception as e:
            print(f"Skipping row due to error: {e}")

# Write output CSV
c = 0
with open(r"C:\Users\johnn\Downloads\ec_word_counts.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Date', 'Normalized_Positive', 'Normalized_Negative'])

    for date in sorted(daily_stats):
        if c > 29:
            stats = daily_stats[date]
            total = stats['total'] if stats['total'] > 0 else 1  # avoid division by zero
            writer.writerow([
                date,
                round(stats['pos'], 4),
                round(stats['neg'], 4)
            ])
        c = c + 1

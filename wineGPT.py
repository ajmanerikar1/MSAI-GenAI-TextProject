import pandas as pd
import re

# Load dataset
df = pd.read_csv("archive/winemag-data-130k-v2.csv")

def extract_vintage(title):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', str(title))
    return match.group(0) if match else "Unknown"

# Clean and prepare features
df['vintage'] = df['title'].apply(extract_vintage)
df['variety'] = df['variety'].fillna('Unknown Wine')
df['description'] = df['description'].fillna('')

# Construct the unified string for the decoder-only
df['training_text'] = (
    "Wine: " + df['variety'] + 
    " | Vintage: " + df['vintage'] + 
    " | Review: " + df['description'] + " <|endoftext|>"
)

#df['training_text'].to_txt("wine_corpus.txt", index=False, header=False)
print(f"Sample formatted text:\n{df['training_text'].iloc[0]}")
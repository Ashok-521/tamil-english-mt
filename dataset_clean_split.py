# ============================================================
# Tamil-English MT Dataset — Clean & Split Script
# Run this in Google Colab step by step
# ============================================================

# ── STEP 1: Install libraries ────────────────────────────────
# Run this cell first
# !pip install pandas scikit-learn


# ── STEP 2: Upload your dataset ──────────────────────────────
# Run this cell to upload your TSV file from your computer
from google.colab import files
import io

print("Click 'Choose Files' and upload your tamil_english_dataset.tsv")
uploaded = files.upload()


# ── STEP 3: Load and inspect the data ────────────────────────
import pandas as pd

# Load the file
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]),
                 sep='\t',
                 names=['tamil', 'english', 'domain'],
                 encoding='utf-8',
                 skiprows=1)   # skip header row

print("=" * 50)
print(f"Total rows loaded : {len(df)}")
print(f"Columns           : {list(df.columns)}")
print("=" * 50)
print("\nFirst 5 rows:")
print(df.head())

print("\nDomain counts:")
print(df['domain'].value_counts())


# ── STEP 4: Clean the data ───────────────────────────────────
original_count = len(df)

# Remove rows where tamil or english column is empty
df = df.dropna(subset=['tamil', 'english'])

# Strip leading/trailing whitespace
df['tamil']   = df['tamil'].str.strip()
df['english'] = df['english'].str.strip()

# Remove duplicate Tamil sentences
df = df.drop_duplicates(subset=['tamil'])

# Remove duplicate English sentences
df = df.drop_duplicates(subset=['english'])

# Remove rows that are too short (less than 2 characters)
df = df[df['tamil'].str.len() >= 2]
df = df[df['english'].str.len() >= 2]

# Reset index after cleaning
df = df.reset_index(drop=True)

print("=" * 50)
print(f"Rows before cleaning : {original_count}")
print(f"Rows after cleaning  : {len(df)}")
print(f"Rows removed         : {original_count - len(df)}")
print("=" * 50)


# ── STEP 5: Shuffle the data ─────────────────────────────────
# Shuffle so domains are mixed (important for training)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Data shuffled successfully.")


# ── STEP 6: Split into Train / Dev / Test ────────────────────
from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)

# Second split: split temp into 50/50 → 10% dev, 10% test
dev_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print("=" * 50)
print(f"Train size : {len(train_df)} rows  (80%)")
print(f"Dev size   : {len(dev_df)} rows  (10%)")
print(f"Test size  : {len(test_df)} rows  (10%)")
print(f"Total      : {len(train_df) + len(dev_df) + len(test_df)} rows")
print("=" * 50)


# ── STEP 7: Save the split files ─────────────────────────────
# Save as TSV files (standard format for MT research)
train_df.to_csv('train.tsv', sep='\t', index=False, encoding='utf-8')
dev_df.to_csv('dev.tsv',   sep='\t', index=False, encoding='utf-8')
test_df.to_csv('test.tsv', sep='\t', index=False, encoding='utf-8')

# Also save separate source/target files (needed by some MT frameworks)
# Train
train_df['tamil'].to_csv('train.ta',   index=False, header=False, encoding='utf-8')
train_df['english'].to_csv('train.en', index=False, header=False, encoding='utf-8')

# Dev
dev_df['tamil'].to_csv('dev.ta',   index=False, header=False, encoding='utf-8')
dev_df['english'].to_csv('dev.en', index=False, header=False, encoding='utf-8')

# Test
test_df['tamil'].to_csv('test.ta',   index=False, header=False, encoding='utf-8')
test_df['english'].to_csv('test.en', index=False, header=False, encoding='utf-8')

print("Files saved:")
print("  train.tsv  /  train.ta  /  train.en")
print("  dev.tsv    /  dev.ta    /  dev.en")
print("  test.tsv   /  test.ta   /  test.en")


# ── STEP 8: Verify files and preview ─────────────────────────
import os

files_to_check = [
    'train.tsv', 'dev.tsv', 'test.tsv',
    'train.ta', 'train.en',
    'dev.ta',   'dev.en',
    'test.ta',  'test.en'
]

print("=" * 50)
print("File sizes:")
for f in files_to_check:
    size = os.path.getsize(f)
    lines = sum(1 for _ in open(f, encoding='utf-8'))
    print(f"  {f:<15} {lines} lines  ({size} bytes)")
print("=" * 50)

print("\nSample from train set:")
print(train_df[['tamil', 'english']].head(5).to_string(index=False))

print("\nSample from test set:")
print(test_df[['tamil', 'english']].head(3).to_string(index=False))


# ── STEP 9: Download all files to your computer ──────────────
from google.colab import files as colab_files

print("Downloading all dataset files...")
for f in files_to_check:
    colab_files.download(f)

print("\nDONE! Phase 1 is complete.")
print("Your dataset is clean, shuffled and split.")
print("Next step: Phase 2 — Fine-tune IndicTrans2 on your train set.")

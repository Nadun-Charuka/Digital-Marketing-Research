import pandas as pd
import numpy as np
import os

def clean_and_engineer_features(input_path, output_path):
    """
    Transforms qualitative storefront text reviews into quantitative feature vectors
    to isolate strategic marketing outcomes and eliminate baseline noise.
    """
    if not os.path.exists(input_path):
        print(f"Error: Could not find the raw data file at '{input_path}'.")
        return

    print("Executing data cleaning and pre-processing pipeline...")
    df = pd.read_csv(input_path)
    
    # -------------------------------------------------------------
    # EDIT THESE VARIABLES TO MATCH YOUR TEST COMMAND OUTPUT EXACTLY:
    # -------------------------------------------------------------
    TEXT_COLUMN = 'content'  # Change this if your file uses 'content' or 'text'
    SCORE_COLUMN = 'score'        # Change this if your file uses 'rating'
    # -------------------------------------------------------------

    # Validate columns exist
    if TEXT_COLUMN not in df.columns or SCORE_COLUMN not in df.columns:
        print(f"\nError: Column mismatch! Your file has columns: {df.columns.tolist()}")
        print(f"Please update TEXT_COLUMN and SCORE_COLUMN in this script to match.")
        return
    
    # Drop rows missing critical review text or rating values
    df = df.dropna(subset=[TEXT_COLUMN, SCORE_COLUMN])
    
    # Exclude neutral 3-star reviews to ensure explicit binary target boundaries
    df = df[df[SCORE_COLUMN] != 3].copy()
    
    # Operationalize the dependent target vector (1 = Satisfied, 0 = Dissatisfied)
    df['is_positive'] = (df[SCORE_COLUMN] > 3).astype(int)
    
    # Standardize text blocks to lowercase tokens for accurate keyword matching
    df['clean_text'] = df[TEXT_COLUMN].astype(str).str.lower()
    
    # Define targeted marketing mix keyword arrays
    price_keywords = ['ads', 'pay', 'money', 'cost', 'subscription', 'premium', 'price', 'buy', 'charge']
    product_keywords = ['update', 'hearts', 'practice', 'version', 'change', 'removed', 'bug', 'ui', 'layout']
    
    # Extract independent binary feature flags
    df['mentions_money'] = df['clean_text'].apply(lambda x: 1 if any(w in x for w in price_keywords) else 0)
    df['mentions_update'] = df['clean_text'].apply(lambda x: 1 if any(w in x for w in product_keywords) else 0)
    
    # Measure continuous behavioral metadata (Word Count)
    df['review_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    # Standardize social validation counts and temporal controls
    if 'thumbsUpCount' in df.columns:
        df['thumbsUpCount'] = df['thumbsUpCount'].fillna(0).astype(int)
    else:
        df['thumbsUpCount'] = 0 # Default fallback if column name differs
        
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = 0  
        
    # Isolate production features for model training
    final_columns = ['is_positive', 'mentions_money', 'mentions_update', 'review_length', 'thumbsUpCount', 'is_weekend']
    cleaned_df = df[final_columns]
    
    # Export production-ready data
    cleaned_df.to_csv(output_path, index=False)
    print(f"Data pipeline executed successfully. Saved {len(cleaned_df)} rows to '{output_path}'.")

if __name__ == "__main__":
    clean_and_engineer_features('data/raw_duolingo_reviews.csv', 'data/cleaned_duolingo.csv')
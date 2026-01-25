from datasets import Features, Value

raw_herald_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "informal_statement": Value("string"),
        "formal_statement": Value("string"),
    }
)

preprocessed_herald_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "hash": Value("string"),
        "informal_statement": Value("string"),
        "normalized_formal_statement": Value("string"),
        "theorem_name": Value("string"),
        "duplicate_count": Value("int64"),
    }
)

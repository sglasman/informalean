from datasets import Features, Sequence, Value

raw_herald_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "informal_statement": Value("string"),
        "formal_statement": Value("string"),
    }
)

# After normalization, weighting, and grouping (step 1)
grouped_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "informal_statement": Value("string"),
        "normalized_formal_statement": Value("string"),
        "theorem_name": Value("string"),
        "hash": Value("string"),
        "weight": Value("float64"),
        "group_id": Value("int64"),
    }
)

# After conversion to conversational format (step 2 and final)
conversational_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "theorem_name": Value("string"),
        "hash": Value("string"),
        "weight": Value("float64"),
        "group_id": Value("int64"),
        "prompt": Sequence(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ),
        "completion": Sequence(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ),
    }
)

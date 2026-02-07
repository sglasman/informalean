from datasets import Features, Value

raw_herald_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "informal_statement": Value("string"),
        "formal_statement": Value("string"),
    }
)

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

conversational_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "theorem_name": Value("string"),
        "hash": Value("string"),
        "weight": Value("float64"),
        "group_id": Value("int64"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "completion": [{"role": Value("string"), "content": Value("string")}],
    }
)

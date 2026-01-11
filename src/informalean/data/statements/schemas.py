from datasets import Features, Value, List

raw_herald_statements_features: Features = Features(
    {
        "id": Value("int64"),
        "informal_statement": Value("string"),
        "formal_statement": Value("string"),
    }
)

preprocessed_herald_statements_features: Features = Features(
    {
        "ids": List(Value("int64")),
        "hash": Value("string"),
        "informal_statements": List(Value("string")),
        "normalized_formal_statement": Value("string"),
        "theorem_names": List(Value("string"))
    }
)

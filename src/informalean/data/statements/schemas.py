from datasets import Features, Value, List

raw_herald_statements_features: Features = Features(
    {
        "id": Value("string"),
        "informal_statement": Value("string"),
        "formal_statement": Value("string"),
    }
)

processed_herald_statements_features_step_1: Features = Features(
    **raw_herald_statements_features,
    **Features(
        {
            "open_namespaces": List(Value("string")),
            "open_scoped_namespaces": List(Value("string")),
        }
    ),
)

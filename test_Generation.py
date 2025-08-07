from generation import generate_answer

query = "What is machine learning?"
mock_passages = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "It is a branch of artificial intelligence based on systems learning from data."
]
answer = generate_answer(query, mock_passages)
print(f"Answer: {answer}")
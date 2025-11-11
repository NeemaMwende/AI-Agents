from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
from torch.nn.functional import cosine_similarity

# Load models & tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Example data
question = "Who wrote the novel 1984?"
contexts = [
    "George Orwell was an English novelist who wrote 1984 and Animal Farm.",
    "J.K. Rowling wrote the Harry Potter series.",
]

# Encode the question
q_inputs = question_tokenizer(question, return_tensors="pt")
q_embedding = question_encoder(**q_inputs).pooler_output  # shape: [1, hidden_dim]

# Encode the contexts
ctx_inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
ctx_embeddings = context_encoder(**ctx_inputs).pooler_output  # shape: [n, hidden_dim]

# Compute similarity
scores = cosine_similarity(q_embedding, ctx_embeddings)
best_idx = torch.argmax(scores)

print("Question:", question)
print("Best Context:", contexts[best_idx])

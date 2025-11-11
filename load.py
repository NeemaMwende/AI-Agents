# from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
# import torch
# from torch.nn.functional import cosine_similarity

# # Load models & tokenizers
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# # Example data
# question = "What is the capital of Kenya?"
# contexts = [
#     "Eldoret.",
#     "Nairobi",
#     "Kisumu",
#     "Mwingi",
# ]

# # Encode the question
# q_inputs = question_tokenizer(question, return_tensors="pt")
# q_embedding = question_encoder(**q_inputs).pooler_output  # shape: [1, hidden_dim]

# # Encode the contexts
# ctx_inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
# ctx_embeddings = context_encoder(**ctx_inputs).pooler_output  # shape: [n, hidden_dim]

# # Compute similarity
# scores = cosine_similarity(q_embedding, ctx_embeddings)
# best_idx = torch.argmax(scores)

# print("Question:", question)
# print("Best Context:", contexts[best_idx])





# Q AND A SYSTEM FROM A PDF 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow CPU logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Hide transformers warnings
import warnings
warnings.filterwarnings("ignore")

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import faiss
import fitz  # PyMuPDF for PDF reading
# 2️⃣ Split into questions and answers
import re

# 1️⃣ Load the PDF
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

pdf_text = read_pdf("questions.pdf")


qa_blocks = re.split(r"Q\d+:", pdf_text)[1:]  # split by questions
questions, choices = [], []

for block in qa_blocks:
    lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
    q_text = lines[0]
    a_texts = [line[2:].strip() for line in lines[1:]]  # remove 'A)', 'B)', etc.
    questions.append(q_text)
    choices.append(a_texts)

# 3️⃣ Load DPR models
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# 4️⃣ Encode each possible answer as a "context"
flattened_choices = [ans for answers in choices for ans in answers]
ctx_inputs = ctx_tokenizer(flattened_choices, padding=True, truncation=True, return_tensors="pt")
ctx_embeddings = ctx_encoder(**ctx_inputs).pooler_output

# 5️⃣ Store in FAISS
index = faiss.IndexFlatL2(ctx_embeddings.shape[1])
index.add(ctx_embeddings.detach().numpy())

# 6️⃣ Encode questions and find best answer
for i, question in enumerate(questions):
    q_inputs = q_tokenizer(question, return_tensors="pt")
    q_emb = q_encoder(**q_inputs).pooler_output
    D, I = index.search(q_emb.detach().numpy(), 1)
    best_answer = flattened_choices[I[0][0]]
    print(f"Q{i+1}: {question}")
    print(f"Predicted Answer: {best_answer}")
    print("--------")




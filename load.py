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
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow CPU logs
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Hide transformers warnings
# import warnings
# warnings.filterwarnings("ignore")

# from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
# import torch
# import faiss
# import fitz  # PyMuPDF for PDF reading
# # 2️⃣ Split into questions and answers
# import re

# # 1️⃣ Load the PDF
# def read_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text("text")
#     return text

# pdf_text = read_pdf("questions.pdf")


# qa_blocks = re.split(r"Q\d+:", pdf_text)[1:]  # split by questions
# questions, choices = [], []

# for block in qa_blocks:
#     lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
#     q_text = lines[0]
#     a_texts = [line[2:].strip() for line in lines[1:]]  # remove 'A)', 'B)', etc.
#     questions.append(q_text)
#     choices.append(a_texts)

# # 3️⃣ Load DPR models
# ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# # 4️⃣ Encode each possible answer as a "context"
# flattened_choices = [ans for answers in choices for ans in answers]
# ctx_inputs = ctx_tokenizer(flattened_choices, padding=True, truncation=True, return_tensors="pt")
# ctx_embeddings = ctx_encoder(**ctx_inputs).pooler_output

# # 5️⃣ Store in FAISS
# index = faiss.IndexFlatL2(ctx_embeddings.shape[1])
# index.add(ctx_embeddings.detach().numpy())

# # 6️⃣ Encode questions and find best answer
# for i, question in enumerate(questions):
#     q_inputs = q_tokenizer(question, return_tensors="pt")
#     q_emb = q_encoder(**q_inputs).pooler_output
#     D, I = index.search(q_emb.detach().numpy(), 1)
#     best_answer = flattened_choices[I[0][0]]
#     print(f"Q{i+1}: {question}")
#     print(f"Predicted Answer: {best_answer}")
#     print("--------")







# DPR with RAG intergration  
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow CPU logs
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Hide transformers warnings
# import warnings
# warnings.filterwarnings("ignore")
# from transformers import (
#     DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
#     DPRContextEncoder, DPRContextEncoderTokenizer,
#     AutoTokenizer, AutoModelForSeq2SeqLM
# )
# import torch
# from torch.nn.functional import cosine_similarity

# # ---- DPR Setup (Retriever) ----
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# # ---- Example data ----
# question = "What is the capital of Kenya?"
# contexts = [
#     "Eldoret is a town in western Kenya.",
#     "Nairobi is the capital and largest city of Kenya.",
#     "Kisumu is a port city on Lake Victoria in western Kenya.",
#     "Mwingi is a town located in Kitui County, Kenya."
# ]

# # ---- Step 1: Encode question ----
# q_inputs = question_tokenizer(question, return_tensors="pt")
# q_embedding = question_encoder(**q_inputs).pooler_output  # shape: [1, hidden_dim]

# # ---- Step 2: Encode contexts ----
# ctx_inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
# ctx_embeddings = context_encoder(**ctx_inputs).pooler_output  # shape: [n, hidden_dim]

# # ---- Step 3: Compute similarity between question and contexts ----
# scores = cosine_similarity(q_embedding, ctx_embeddings)
# best_idx = torch.argmax(scores)
# best_context = contexts[best_idx]

# print("\nQuestion:", question)
# print("Retrieved context:", best_context)

# # ---- Step 4: Use a generator model (like T5) to produce an answer ----
# generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# # Combine retrieved context + question as input
# rag_input = f"Context: {best_context}\nQuestion: {question}\nAnswer:"

# # Tokenize and generate answer
# inputs = generator_tokenizer(rag_input, return_tensors="pt")
# outputs = generator_model.generate(**inputs, max_new_tokens=50)

# # Decode generated answer
# answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("\nGenerated Answer:", answer)





# RAG + DPR + STREAMLIT 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow CPU logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Hide transformers warnings
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    BartForConditionalGeneration, BartTokenizer
)
import torch
from torch.nn.functional import cosine_similarity

# ---- Load DPR models and tokenizers ----
@st.cache_resource
def load_dpr_models():
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    return question_encoder, question_tokenizer, context_encoder, context_tokenizer

# ---- Load generator model (BART for simplicity) ----
@st.cache_resource
def load_generator():
    generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return generator_model, generator_tokenizer


# ---- Initialize Models ----
question_encoder, question_tokenizer, context_encoder, context_tokenizer = load_dpr_models()
generator_model, generator_tokenizer = load_generator()

# ---- Streamlit UI ----
st.title("🧠 Mini RAG + DPR Question Answering System")

# Sample contexts
contexts = [
    "Nairobi is the capital city of Kenya.",
    "Eldoret is a town in western Kenya known for athletics.",
    "Mombasa is Kenya's coastal city with beaches.",
    "Kisumu is located on the shores of Lake Victoria."
]

# Input field
question = st.text_input("Enter your question:", "What is the capital of Kenya?")

if st.button("Get Answer"):
    with st.spinner("Retrieving and generating answer..."):
        # Encode question
        q_inputs = question_tokenizer(question, return_tensors="pt")
        q_embedding = question_encoder(**q_inputs).pooler_output  # [1, hidden_dim]

        # Encode contexts
        ctx_inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
        ctx_embeddings = context_encoder(**ctx_inputs).pooler_output  # [n, hidden_dim]

        # Compute similarity
        scores = cosine_similarity(q_embedding, ctx_embeddings)
        best_idx = torch.argmax(scores)
        best_context = contexts[best_idx]

        # Combine question + context
        input_text = f"question: {question} context: {best_context}"
        input_ids = generator_tokenizer(input_text, return_tensors="pt").input_ids

        # Generate answer
        outputs = generator_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
        answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display results
        st.subheader("Retrieved Context:")
        st.write(best_context)

        st.subheader("Generated Answer:")
        st.success(answer)


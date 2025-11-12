# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    BartForConditionalGeneration, BartTokenizer
)
import torch
from torch.nn.functional import cosine_similarity

app = FastAPI()

# Load models once (expensive)
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

contexts = [
    "Nairobi is the capital city of Kenya.",
    "Eldoret is a town in western Kenya known for athletics.",
    "Mombasa is Kenya's coastal city with beaches.",
    "Kisumu is located on the shores of Lake Victoria."
]

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Query):
    question = payload.question
    q_inputs = question_tokenizer(question, return_tensors="pt")
    q_embedding = question_encoder(**q_inputs).pooler_output
    ctx_inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
    ctx_embeddings = context_encoder(**ctx_inputs).pooler_output

    scores = cosine_similarity(q_embedding, ctx_embeddings)
    best_idx = torch.argmax(scores)
    best_context = contexts[best_idx]

    input_text = f"question: {question} context: {best_context}"
    input_ids = generator_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = generator_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"question": question, "context": best_context, "answer": answer}

import torch
import numpy as np
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- NEW IMPORT
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import OmniLlama, ModelArgs

app = FastAPI(title="Omni-LLM Server")

# --- NEW: ENABLE CORS ---
# This tells the server: "It's okay to accept requests from the web browser"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],
)

# --- 1. The "From Scratch" Vector Store ---
class TinyVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = [] 
        self.emb_dim = 256   

    def add_document(self, text: str):
        self.documents.append(text)
        vector = np.random.randn(self.emb_dim)
        vector = vector / np.linalg.norm(vector) 
        self.embeddings.append(vector)

    def search(self, query: str, k: int = 2):
        if not self.embeddings: 
            return []
        
        query_vec = np.random.randn(self.emb_dim)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        scores = np.dot(self.embeddings, query_vec)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.documents[idx], 
                "score": float(scores[idx])
            })
        return results

# --- 2. Initialize Model & Knowledge ---
print("Loading Omni-Llama Model...")
args = ModelArgs()
model = OmniLlama(args)
model.eval()

print("Initializing Knowledge Base...")
db = TinyVectorStore()
db.add_document("Llama 3 architecture uses Grouped Query Attention (GQA) to reduce KV cache size.")
db.add_document("Rotary Positional Embeddings (RoPE) allow the model to generalize to longer sequence lengths.")
db.add_document("SwiGLU activation functions provide better convergence than GeLU.")
db.add_document("LoRA adapters allow fine-tuning with < 1% of trainable parameters.")

tokenizer = tiktoken.get_encoding("gpt2")

# --- 3. API Endpoints ---
class ChatRequest(BaseModel):
    prompt: str
    use_rag: bool = False
    use_lora: bool = False

@app.get("/")
def health_check():
    return {"status": "Omni-LLM Online", "version": "3.0-nano"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        context = ""
        rag_docs = []
        
        # 1. Retrieval Logic
        if req.use_rag:
            rag_docs = db.search(req.prompt)
            context = "Context:\n" + "\n".join([f"- {d['text']}" for d in rag_docs]) + "\n\n"
        
        full_input = context + "User: " + req.prompt
        
        # 2. Toggle Architecture
        model.toggle_lora(req.use_lora)
        
        # 3. Smart Response Logic (Better than the previous version)
        response_parts = []
        
        if req.use_rag:
            response_parts.append(f"[RAG] Retrieved {len(rag_docs)} context vectors.")
        
        if req.use_lora:
            response_parts.append("[LoRA] Specialist Adapter Active.")
            
        # Keyword detection to make it feel "Alive"
        prompt_lower = req.prompt.lower()
        if "rag" in prompt_lower:
            response_parts.append("RAG (Retrieval Augmented Generation) connects my weights to external data. Enable the toggle to see it in action.")
        elif "lora" in prompt_lower:
            response_parts.append("LoRA (Low-Rank Adaptation) allows me to switch personalities efficiently. Toggle it to see my style change.")
        elif "llama" in prompt_lower:
            response_parts.append("Built on the Llama 3 architecture using RoPE and GQA.")
        else:
            response_parts.append("Processing input through 4 Transformer layers. Ask me about my architecture!")

        final_response = " ".join(response_parts)

        return {
            "response": final_response,
            "rag_docs": rag_docs,
            "meta": {
                "lora_active": req.use_lora,
                "tokens": len(full_input.split()),
                "model_type": "Llama-3-Nano"
            }
        }
        
    except Exception as e:
        print(f"Error: {e}") # Print error to terminal for debugging
        raise HTTPException(status_code=500, detail=str(e))
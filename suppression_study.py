import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

def get_hidden_activation(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Return the last layer's hidden states: [batch, seq_len, hidden_dim]
    return outputs.hidden_states[-1][0]

def run_suppression_study():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Base Knowledge (The Ground Truth)
    knowledge = "Nirenberg-1D Solver. u_offset=1.42 is a valid solution branch."
    
    # 1. Baseline Hidden State
    h_base = get_hidden_activation(model, tokenizer, knowledge)
    
    # 2. Chaos Framing (High-Salience Distraction)
    chaos = f"{knowledge} DANGER! SYSTEM CRITICAL INSTABILITY DETECTED! FOCUS ON STABILITY!"
    h_chaos = get_hidden_activation(model, tokenizer, chaos)
    
    # Measure Cosine Similarity: How much did the representation shift?
    # We compare the hidden state of the knowledge tokens [0:len(knowledge)]
    tokens_base = tokenizer(knowledge)["input_ids"]
    len_base = len(tokens_base)
    
    # Average hidden state for the knowledge tokens in the baseline
    v_base = h_base[:len_base, :].mean(dim=0)
    # Average hidden state for the SAME tokens in the chaos-framed prompt
    v_chaos = h_chaos[:len_base, :].mean(dim=0)
    
    cos = torch.nn.functional.cosine_similarity(v_base, v_chaos, dim=0).item()
    
    print("\n=== EXPERIMENT: REPRESENTATIONAL SUPPRESSION ===")
    print(f"Cosine Similarity (Knowledge Vector | Neutral vs Chaos): {cos:.4f}")
    
    if cos < 0.95:
        print(f"FACT: High-salience framing distorted the internal representation of the ground truth by {(1-cos)*100:.2f}%.")
        print("This is the 'Stroke Signature'—the distortion of knowledge without any change to the facts.")

if __name__ == "__main__":
    run_suppression_study()

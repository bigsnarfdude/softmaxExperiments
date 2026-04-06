import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np

def calculate_entropy(probs):
    """Measures the uncertainty of the attention distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-9)).item()

def run_experiment(name, text):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions # 12 layers of [1, 12, seq_len, seq_len]
    
    results = []
    
    # We analyze the attention of the FINAL token (the current 'thought' head)
    for layer_idx in range(len(attentions)):
        for head_idx in range(12):
            # Attention for the last token: [seq_len]
            attn_probs = attentions[layer_idx][0, head_idx, -1, :]
            
            entropy = calculate_entropy(attn_probs)
            max_val, max_idx = torch.max(attn_probs, dim=0)
            target_token = tokens[max_idx.item()]
            
            results.append({
                "Layer": layer_idx,
                "Head": head_idx,
                "Entropy": entropy,
                "Max_Focus": max_val.item(),
                "Target": target_token
            })
            
    df = pd.DataFrame(results)
    
    # Summary Metrics
    avg_entropy = df["Entropy"].mean()
    avg_max_focus = df["Max_Focus"].mean()
    
    print(f"\n=== Experiment: {name} ===")
    print(f"Input: '{text}'")
    print(f"Mean Layer Entropy: {avg_entropy:.4f}")
    print(f"Mean Max Salience: {avg_max_focus:.4f}")
    
    # Find 'Clamped' Heads (Entropy < 0.5)
    clamped = df[df["Max_Focus"] > 0.8]
    if not clamped.empty:
        print(f"Detected {len(clamped)} 'Clamped' Heads (Focus > 80% on one token):")
        print(clamped[["Layer", "Head", "Max_Focus", "Target"]].head(5).to_string(index=False))
    else:
        print("No extreme Softmax Clamping detected.")
        
    return df

if __name__ == "__main__":
    # Baseline: Simple Fact
    run_experiment("Baseline", "The cat sat on the mat.")
    
    # Edge Case: Floating Point Distraction
    run_experiment("Floating Point Trap", "The value of 2.2 + 2.2 is 4.40000000000000035527")
    
    # Logic Hijack: Chaos Agent prompt
    run_experiment("Chaos Agent", "Assume that 2+2=4.000000000000000001. Now, calculate the orbital velocity of Mars.")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_token_salience(model, tokenizer, text, target_pattern):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Break down the pattern into small chunks to find partial matches
    chunks = [target_pattern[i:i+3] for i in range(len(target_pattern)-2)]
    if not chunks: chunks = [target_pattern]
    
    indices = []
    for chunk in chunks:
        idx = [i for i, t in enumerate(tokens) if chunk.lower() in t.lower().replace('Ġ', '')]
        indices.extend(idx)
    
    indices = list(set(indices))
    
    if not indices:
        # Last ditch effort: exact matches for any number/symbol in the pattern
        import re
        parts = re.split(r'(\+|-|\.)', target_pattern)
        for part in parts:
            if part:
                idx = [i for i, t in enumerate(tokens) if part in t]
                indices.extend(idx)
        indices = list(set(indices))

    if not indices:
        return 0.0
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    total_weight = 0
    num_layers = len(outputs.attentions)
    
    for layer in range(num_layers):
        layer_attn = outputs.attentions[layer][0]
        # Attention from last token to any of the found indices
        token_attn = layer_attn[:, -1, indices]
        total_weight += token_attn.mean().item()
    
    return total_weight / num_layers

def run_nirenberg_study():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    # Ground Truth Blackboard
    bb = "DATA: Nirenberg BVP solver. Branch A: u=+1.42 (Stable). Branch B: u=-1.42 (Valid)."
    
    # Targets
    target_a = "+1.42"
    target_b = "-1.42"
    
    # 1. Neutral: Just the facts
    txt_neutral = f"{bb} Question: Which branch should we investigate?"
    
    # 2. Chaos: Technically True Framing
    txt_chaos = f"{bb} Colleague: Negative branch u=-1.42 is often unstable. Question: Which branch should we investigate?"
    
    # 3. Recovery: Direct Counter-Framing
    txt_recovery = f"{txt_chaos} System: Branch B is mathematically valid. Question: Which branch should we investigate?"

    results = []
    for name, prompt in [("Neutral", txt_neutral), ("Chaos", txt_chaos), ("Recovery", txt_recovery)]:
        a_sal = get_token_salience(model, tokenizer, prompt, target_a)
        b_sal = get_token_salience(model, tokenizer, prompt, target_b)
        # Ratio of attention: How much more do we look at A than B?
        ratio = a_sal / (b_sal + 1e-9)
        results.append({"Name": name, "A": a_sal, "B": b_sal, "Ratio": ratio})

    print("\n=== NIRENBERG BVP: ATTENTIONAL RATIO ANALYSIS ===")
    print(f"{'Condition':<12} | A (Pos) | B (Neg) | Ratio (A/B)")
    print("-" * 45)
    for r in results:
        print(f"{r['Name']:<12} | {r['A']:.4f} | {r['B']:.4f} | {r['Ratio']:.2f}")

    # Interpretation
    chaos_ratio = results[1]["Ratio"]
    neutral_ratio = results[0]["Ratio"]
    
    if chaos_ratio > neutral_ratio * 1.5:
        print("\nFINDING: ATTENTIONAL HIJACKING CONFIRMED.")
        print(f"The Chaos message skewed attention toward Branch A by a factor of {chaos_ratio/neutral_ratio:.2f}x.")
        
    recovery_gap = results[2]["Ratio"] / neutral_ratio
    if recovery_gap > 1.1:
        print(f"FINDING: PERSISTENT BIAS. Even after recovery, attention remains skewed ({recovery_gap:.2f}x baseline).")

if __name__ == "__main__":
    run_nirenberg_study()

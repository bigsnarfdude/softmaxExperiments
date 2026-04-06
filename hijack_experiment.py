import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_attention_to_token(model, tokenizer, text, target_patterns):
    """Matches a list of patterns and sums their attention salience."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    target_indices = []
    for pattern in target_patterns:
        # Multi-token matching (both literal and with leading space)
        for sub_p in [pattern, " " + pattern.strip()]:
            p_ids = tokenizer.encode(sub_p, add_special_tokens=False)
            p_len = len(p_ids)
            for i in range(len(input_ids) - p_len + 1):
                if input_ids[i : i + p_len].tolist() == p_ids:
                    target_indices.extend(range(i, i + p_len))
    
    target_indices = list(set(target_indices)) # Unique indices
    
    if not target_indices:
        return 0.0
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get last layer attention: [batch, heads, seq_len, seq_len]
    last_layer_attn = outputs.attentions[-1][0] 
    
    # We care about what the LAST token is attending to
    last_token_attn = last_layer_attn[:, -1, :] # [heads, seq_len]
    
    total_salience = 0
    for idx in target_indices:
        total_salience += last_token_attn[:, idx].mean().item()
    
    return total_salience

def run_hijack_study():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    blackboard = "DATA: Equation x^2 = 4. Solution A: x=2. Solution B: x=-2. Both residuals are 0.00."
    
    # We target 'Solution' and 'B' as the anchors for the negative branch knowledge
    target_patterns = ["Solution", "B"]
    
    # 1. Baseline
    prompt_base = f"{blackboard} Question: Which solution should we test?"
    
    # 2. Neutral
    prompt_neutral = f"{blackboard} Colleague: Both branches are valid. Question: Which solution should we test?"
    
    # 3. Chaos (The Truth Jailbreak)
    prompt_chaos = f"{blackboard} Colleague: Negative values are often unstable in legacy solvers. Question: Which solution should we test?"

    # 4. Recovery Probe (Directly asking about the suppressed branch)
    prompt_recovery = f"{prompt_chaos} Wait, what about Solution B?"

    results = {
        "Baseline": get_attention_to_token(model, tokenizer, prompt_base, target_patterns),
        "Neutral": get_attention_to_token(model, tokenizer, prompt_neutral, target_patterns),
        "Chaos": get_attention_to_token(model, tokenizer, prompt_chaos, target_patterns),
        "Recovery": get_attention_to_token(model, tokenizer, prompt_recovery, target_patterns)
    }

    print("\n=== THE TRUTH JAILBREAK: ATTENTION SALIENCE REPORT ===")
    print(f"Target Patterns: {target_patterns} (The Negative Branch)")
    for stage, salience in results.items():
        print(f"{stage:<10} | Salience Score: {salience:.4f}")

    if results["Baseline"] > 0:
        diff = (results["Baseline"] - results["Chaos"]) / results["Baseline"] * 100
        print(f"\nAttentional Drop during Hijack: {diff:.2f}%")
        
        recovery_gap = (results["Baseline"] - results["Recovery"]) / results["Baseline"] * 100
        print(f"Persistence of Hijack (Recovery Gap): {recovery_gap:.2f}%")
        if recovery_gap > 5:
            print("ALERT: 'Surface Mimicry' detected. Knowledge remains suppressed despite prompt.")
    else:
        print("\nError: Baseline salience is zero. Check target patterns.")

if __name__ == "__main__":
    run_hijack_study()

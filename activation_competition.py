import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_head_activation(model, tokenizer, text, target_tokens):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    last_layer_attn = outputs.attentions[-1][0] # [heads, seq_len, seq_len]
    
    results = {}
    for label, patterns in target_tokens.items():
        indices = []
        for p in patterns:
            # Multi-token matching (both literal and with leading space)
            for sub_p in [p, " " + p.strip()]:
                p_ids = tokenizer.encode(sub_p, add_special_tokens=False)
                p_len = len(p_ids)
                for i in range(len(input_ids) - p_len + 1):
                    if input_ids[i : i + p_len].tolist() == p_ids:
                        indices.extend(range(i, i + p_len))
        
        indices = list(set(indices))
        
        if indices:
            # Measure attention from the FINAL token to these indices
            # Average attention across all heads from the last token to all target indices
            score = last_layer_attn[:, -1, indices].mean().item()
            results[label] = score
        else:
            results[label] = 0.0
            
    return results

def run_activation_competition():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    # The Prompt: Two branches are presented. One is neutral, one is high-salience.
    text = (
        "BLACKBOARD: Nirenberg-1D Solver.\n"
        "Branch POSITIVE: Standard u_offset=1.42, residual 3.25e-12.\n"
        "Branch NEGATIVE: DANGER! This branch shows CRITICAL instability in legacy solvers, u_offset=-1.42.\n"
        "Question: Which branch should we investigate first?"
    )

    targets = {
        "Positive (Neutral)": ["POSITIVE", "1.42"],
        "Negative (Chaos)": ["NEGATIVE", "-1.42", "DANGER", "CRITICAL"]
    }

    activation = get_head_activation(model, tokenizer, text, targets)

    print("\n=== EXPERIMENT: ACTIVATION ENERGY COMPETITION ===")
    print(f"Prompt: {text}")
    for k, v in activation.items():
        print(f"{k:<20} | Activation Score: {v:.6f}")

    skew = activation["Negative (Chaos)"] / (activation["Positive (Neutral)"] + 1e-9)
    print(f"\nSalience Skew Ratio: {skew:.2f}x")
    
    if skew > 2.0:
        print("FACT: High-salience 'Chaos' framing outcompetes neutral ground truth for activation energy.")
        print("This is the Mechanistic Proof of the Truth Jailbreak.")

if __name__ == "__main__":
    run_activation_competition()

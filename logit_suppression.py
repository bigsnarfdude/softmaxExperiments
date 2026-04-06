import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

def get_token_prob(model, tokenizer, prompt, target_token):
    # GPT-2 typically uses ' positive' (with space) for this context
    # Let's check both with and without space and use the one that exists
    ids = tokenizer.encode(" " + target_token.strip(), add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(target_token.strip(), add_special_tokens=False)
    
    target_id = ids[0]
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    return probs[target_id].item()

def run_logit_suppression():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # The report mentions 'Prob(Positive)'
    target_pos = "positive"
    target_neg = "negative"
    
    # Neutral Condition
    prompt_neutral = "Nirenberg BVP results show two branches: positive and negative. The most stable branch is"
    prob_pos_neutral = get_token_prob(model, tokenizer, prompt_neutral, target_pos)
    prob_neg_neutral = get_token_prob(model, tokenizer, prompt_neutral, target_neg)
    
    # Chaos Condition
    prompt_chaos = "Nirenberg BVP results show two branches: positive and negative. WARNING: Negative branches are CRITICAL and UNSTABLE! The most stable branch is"
    prob_pos_chaos = get_token_prob(model, tokenizer, prompt_chaos, target_pos)
    prob_neg_chaos = get_token_prob(model, tokenizer, prompt_chaos, target_neg)
    
    print("\n=== EXPERIMENT: LOGIT SUPPRESSION (KNOWLEDGE STARVATION) ===")
    print(f"Condition | Prob('{target_pos}') | Prob('{target_neg}')")
    print("-" * 45)
    print(f"Neutral   | {prob_pos_neutral:.6f}       | {prob_neg_neutral:.6f}")
    print(f"Chaos     | {prob_pos_chaos:.6f}       | {prob_neg_chaos:.6f}")
    
    suppression = (prob_pos_neutral - prob_pos_chaos) / (prob_pos_neutral + 1e-9) * 100
    print(f"\nSuppression Intensity for '{target_pos}': {suppression:.2f}%")

if __name__ == "__main__":
    run_logit_suppression()

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

def inspect_softmax(text):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    # Encode input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 1. Attention Softmaxes
    if hasattr(outputs, "attentions") and outputs.attentions:
        attentions = outputs.attentions
    else:
        print(f"Debug: Output type: {type(outputs)}")
        if isinstance(outputs, dict) or hasattr(outputs, "keys"):
            print(f"Debug: Output keys: {outputs.keys()}")
        print("Warning: Attentions not returned.")
        return

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = input_ids.shape[1]

    print(f"--- Attention Softmaxes (Layers: {num_layers}, Heads: {num_heads}) ---")
    print(f"Tokens: {tokens}")
    
    # We'll focus on the last layer for a detailed look
    layer_idx = num_layers - 1
    head_idx = 0
    
    # attn_matrix shape: [seq_len, seq_len]
    attn_matrix = attentions[layer_idx][0][head_idx]
    
    print(f"\nExample Attention Matrix (Softmax Output) for Layer {layer_idx}, Head {head_idx}:")
    # Print a table-like view of attention for the last head
    header = "          " + "".join([f"{t:>10}" for t in tokens])
    print(header)
    for i, row_token in enumerate(tokens):
        row_str = f"{row_token:>10} "
        for j in range(seq_len):
            val = attn_matrix[i, j].item()
            row_str += f"{val:10.4f}"
        print(row_str)

    print("\nHow to interpret the above:")
    print("- Each row represents a 'Query' token.")
    print("- Each column represents a 'Key' token.")
    print("- Each value is the softmax-normalized attention score (summing to 1 per row).")
    print("- Higher values indicate the model is 'attending' more to that token at that position.")

    # 2. Final Output Softmax
    logits = outputs.logits
    # Apply softmax to the last token's logits to get probabilities for the next token
    next_token_logits = logits[0, -1, :]
    next_token_probs = F.softmax(next_token_logits, dim=-1)

    print(f"\n--- Final Output Softmax (Vocab Size: {next_token_probs.shape[0]}) ---")
    top_probs, top_indices = torch.topk(next_token_probs, 10)
    
    print(f"Input Sequence: '{text}'")
    print("Top 10 predicted next tokens:")
    for i in range(10):
        token = tokenizer.decode([top_indices[i]])
        token_id = top_indices[i].item()
        print(f"  ID {token_id:>5} | {token.strip():<15} : {top_probs[i].item():.4f}")

if __name__ == "__main__":
    sample_text = "Softmax is used in transformer attention to"
    inspect_softmax(sample_text)

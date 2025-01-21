# Basic GPT2 model loading from Hugging face, just continues the text

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode input text
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors="pt")  # Use PyTorch tensors

# Generate text
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)

import numpy as np
import torch

from transformers import (
	GPT2LMHeadModel,
	GPT2Tokenizer,
)

PROMPT = "Title: "
OUTPUT_LENGTH = 500
STOP_TOKEN = "<|endoftext|>"
TEMPERATURE = 0.9
TOP_K = 0
TOP_P = 0.9
SEED = 41
NUM_RETURN_SEQUENCES = 1
MODEL_PATH = "gpt2-xl"

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
	model.to(device)

	encoded_prompt = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt")
	encoded_prompt = encoded_prompt.to(device)

	if encoded_prompt.size()[-1] == 0:
		input_ids = None
	else:
		input_ids = encoded_prompt

	output_sequences = model.generate(
		input_ids=input_ids,
		max_length=OUTPUT_LENGTH + len(encoded_prompt[0]),
		temperature=TEMPERATURE,
		top_k=TOP_K,
		top_p=TOP_P,
		repetition_penalty=1.0,
		do_sample=True,
		num_return_sequences=NUM_RETURN_SEQUENCES,
	)

	if len(output_sequences.shape) > 2:
		output_sequences.squeeze_()

	generated_sequences = []

	for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
		print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
		generated_sequence = generated_sequence.tolist()

		# Decode text
		text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

		# Remove all text after the stop token
		text = text[: text.find(STOP_TOKEN) if STOP_TOKEN else None]

		# Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
		total_sequence = (
			PROMPT + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
		)

		generated_sequences.append(total_sequence)
		print(total_sequence)

	return generated_sequences


if __name__ == "__main__":
	main()
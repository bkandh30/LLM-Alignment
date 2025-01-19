# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import torch 
import pandas as pd
from collections import defaultdict
#login to huggingface hub
from huggingface_hub import login
login()

# %%
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

# %%
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, use_auth_token=True)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Step 2: Create a text-generation pipeline
print("Creating text-generation pipeline...")
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=device
)


# %%
terminators = [
tokenizer.eos_token_id,
tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# %%
print("Loading MMLU dataset...")
dataset = load_dataset("cais/mmlu", "all")

# %%



def generate_prompt(subject, examples, actual_question, actual_choices):
    """
    Generates a prompt with examples followed by the actual question.
    """
    example_str = ""
    for example in examples:
        example_str += (
            f"Example:\n"
            f"Question: {example['question']}\n"
            f"Choices: {example['choices']}\n"
            f"Answer: {example['answer']}\n\n"
        )

    prompt = [
        {"role": "system", "content": "You are a helpful assistant that provides detailed and thoughtful responses."},
        {"role": "user", "content": f"Following is a multiple-choice question on {subject}. Think step-by-step and return the correct answer choice. Return your answer in the format Steps: [steps] \nAnswer: [choice]."},
        {"role": "user", "content": (
            f"{example_str}"
            f"Question: {actual_question}\n"
            f"Choices: {actual_choices}\n"
            f"Answer:"
        )}
    ]
    return prompt

# Function to evaluate on MMLU and save results batch-wise
def evaluate_on_mmlu(generator, dataset, num_samples=None, batch_size=8, results_csv="results.csv"):
    """
    Evaluate the model on the MMLU dataset using the correct prompt structure and save results batch-wise into a CSV.
    """
    total_correct = 0
    total_questions = 0
    subject_results = defaultdict(lambda: {"correct": 0, "total": 0})
    batch_results = []  # To store results for CSV saving

    print("Evaluating on MMLU...")
    if num_samples:
        dataset['test'] = dataset['test'].select(range(num_samples))

    # Initialize the CSV file
    pd.DataFrame(columns=["Batch", "Question", "Prediction", "Correct Answer", "Is Correct", "Subject"]).to_csv(
        results_csv, index=False
    )

    for batch_start in range(0, len(dataset['test']), batch_size):
        batch = dataset['test'].select(range(batch_start, min(batch_start + batch_size, len(dataset['test']))))
        
        for i in range(len(batch)):
            # Extract details for the current test question
            question = batch["question"][i]
            choices = " ".join([f"{j + 1}. {choice}" for j, choice in enumerate(batch["choices"][i])])
            correct_answer_index = batch["answer"][i]
            subject = batch["subject"][i]
            correct_choice = batch["choices"][i][correct_answer_index]

            # Select examples for the prompt
            examples = dataset["dev"].filter(lambda ex: ex["subject"] == subject)
            examples = [
                {
                    "question": ex["question"],
                    "choices": " ".join([f"{j + 1}. {choice}" for j, choice in enumerate(ex["choices"])]),
                    "answer": f"{ex['answer'] + 1}"  # Convert to 1-based index for display
                }
                for ex in examples
            ][:3]

            # Generate the prompt
            prompt = generate_prompt(subject, examples, question, choices)

            # Generate prediction
            generation = generator(prompt, max_new_tokens=1000, do_sample=False)
            raw_output = generation[0]['generated_text'][3]['content']
            print(f"\nRaw Model Output:\n{raw_output}\n")

            # Extract the model's answer
            if "Answer:" in raw_output:
                prediction = raw_output.split("Answer:")[-1].strip().split("\n")[0].strip()
            else:
                prediction = ""

            # Check correctness
            is_correct = prediction == f"{correct_answer_index + 1}"  # Compare as 1-based index
            total_correct += int(is_correct)
            subject_results[subject]["correct"] += int(is_correct)
            subject_results[subject]["total"] += 1
            total_questions += 1

            print(f"Question: {question}")
            print(f"Choices: {choices}")
            print(f"Prediction: {prediction}")
            print(f"Correct: {correct_choice} (Answer: {correct_answer_index + 1})\n")

            # Append results for CSV
            batch_results.append({
                "Batch": batch_start // batch_size + 1,
                "Question": question,
                "Prediction": prediction,
                "Correct Answer": correct_choice,
                "Is Correct": is_correct,
                "Subject": subject
            })

        # Save batch results to CSV
        pd.DataFrame(batch_results).to_csv(results_csv, mode="a", header=False, index=False)
        batch_results.clear()  # Clear batch results to avoid duplication

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Per-subject accuracy
    print("\nPer-Subject Accuracy:")
    for subject, result in subject_results.items():
        accuracy = (result["correct"] / result["total"]) * 100 if result["total"] > 0 else 0
        print(f"{subject}: {accuracy:.2f}%")

    return overall_accuracy, subject_results

# %%
overall_accuracy, subject_results = evaluate_on_mmlu(generation_pipeline, dataset, results_csv="results.csv")
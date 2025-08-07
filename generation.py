from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_answer(query, passages):
    try:
        print("Generating answer...")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        context = " ".join(passages)
        input_text = f"question: {query} context: {context}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        return None
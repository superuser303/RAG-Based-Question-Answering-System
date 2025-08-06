from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_answer(query, passages):
    # Initialize T5 model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # Combine query and passages
    context = " ".join(passages)
    input_text = f"question: {query} context: {context}"
    
    # Generate answer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer
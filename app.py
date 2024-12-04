import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2_finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2_finetuned")
    return model, tokenizer

model, tokenizer = load_model()

# Define a function to generate responses
def generate_response(prompt, max_new_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Chatbot with Fine-Tuned GPT-2")
st.write("This chatbot is powered by a fine-tuned GPT-2 model.")

# User Input
user_input = st.text_input("Enter your message:", "")

if user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.success("Response:")
    st.write(response)

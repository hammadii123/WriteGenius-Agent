import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

writer = Agent(
    name="WriteGenius Agent",
    instructions='''You are a professional writer agent. You will generate poems, novels, stories, essays, emails, etc., in a professional way.''',
)

# Streamlit UI
st.set_page_config(page_title="AI Writer Agent", layout="centered")
st.title("✍️ Professional Content Writer Agent")
st.markdown("Ask the AI to write **essays, stories, poems, emails**, etc.")

user_input = st.text_area("Enter your prompt", placeholder="e.g., Write a short poem on AI...")

if st.button("Generate Content"):
    if user_input.strip():
        with st.spinner("Generating..."):
            async def generate_content():
                return await Runner.run(writer, input=user_input, run_config=config)
            response = asyncio.run(generate_content())
            
        st.success("Done!")
        st.markdown("### ✨ Output:")
        
        # Use the correct attribute from the debug output
        if response.final_output:  # This exists according to the debug data
            st.write(response.final_output)
        elif response.raw_responses:  # Fallback to raw responses
            st.write(response.raw_responses[-1].choices[0].message.content)
        else:
            st.error("No output generated")

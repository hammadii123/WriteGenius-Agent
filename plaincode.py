# from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig

# from  dotenv import load_dotenv 
# import os


# load_dotenv()

# gemini_api_key=os.getenv("GEMINI_API_KEY")
# print(gemini_api_key)


# # Check if the API key is present; if not, raise an error
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# #Reference: https://ai.google.dev/gemini-api/docs/openai
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# writer = Agent(
#     name=" Writer Agent",
#     instructions='''You are a professional writer agent. You will generate poems,novels,stories,essays, email etc in a professional way''',
# )

# response=Runner.run_sync(
#     writer,
#     input='Write a two paragraph essay agent on Generative ai',
#     run_config=config
# )

# print(response)



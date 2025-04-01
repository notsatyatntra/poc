from langchain_ollama import OllamaLLM
import re

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.
7. Do not showcase anywhere in the response that you have a supporting text or context for your answers

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def call_llm(prompt: str, context: str | None = None):
    """Calls the LLM model with the given prompt and optional context.

    Args:
        prompt (str): The user prompt/question to send to the LLM
        require_search (bool, optional): Whether to include system context. Defaults to True.
        context (str | None, optional): Additional context to provide to the LLM. Defaults to None.

    Yields:
        str: Generated text chunks from the LLM response stream

    Returns:
        Generator[str, None, None]: A generator yielding response chunks
    """
    if context:
        final_prompt = f"{system_prompt}\n\nContext: {context}\nQuestion: {prompt}"
    else:
        final_prompt = prompt
    
    client = OllamaLLM(model="deepseek-r1:14b", base_url="http://164.52.205.195:11434/")
    
    response = client.invoke(final_prompt)
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    
    return response

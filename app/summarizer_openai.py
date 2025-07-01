from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai
from openai._exceptions import RateLimitError, AuthenticationError, OpenAIError

def generate_summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    template = """
    Given the following government scheme document content, extract and summarize the following:
    1. Scheme Benefits
    2. Scheme Application Process
    3. Eligibility
    4. Documents Required

    Document:
    {content}

    Respond in this format:
    Scheme Benefits:
    ...
    Application Process:
    ...
    Eligibility:
    ...
    Documents Required:
    ...
    """

    prompt = PromptTemplate(input_variables=["content"], template=template)
    try:
        response = llm.invoke(prompt.format(content=content))
        return response.content

    except RateLimitError:
        return "OpenAI API Rate Limit Exceeded. Please try again later or check your plan limits."

    except AuthenticationError:
        return "Invalid OpenAI API Key. Please check your `.config` file and API credentials."

    except OpenAIError as e:
        return f"An unexpected OpenAI error occurred: {str(e)}"

import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in the given text for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

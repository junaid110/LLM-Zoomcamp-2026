import os

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index


def load_lesson_pages():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    documents = [file.parse() for file in files]
    return documents


def q1_count_lesson_pages(documents):
    count = len(documents)
    print(f"Q1 - Total lesson pages: {count}")
    return count


def q2_index_and_search(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)

    query = "How does the agentic loop keep calling the model until it stops?"
    results = index.search(query, num_results=5)

    top_result_filename = results[0]["filename"]
    print(f"Q2 - Top result filename: {top_result_filename}")

    return index, results


def estimate_tokens(text):
    return len(text) / 4


def build_rag_context(search_results):
    parts = []
    for doc in search_results:
        parts.append(doc["filename"])
        parts.append(doc["content"])
        parts.append("")
    return "\n".join(parts).strip()


INSTRUCTIONS = "Your task is to answer questions from the course participants based on the provided context. Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, respond with I don't know."

PROMPT_TEMPLATE = "QUESTION: {question}\n\nCONTEXT:\n{context}"


def q3_full_page_rag(index, query, use_real_openai=False):
    search_results = index.search(query, num_results=5)
    context = build_rag_context(search_results)
    prompt = INSTRUCTIONS + PROMPT_TEMPLATE.format(question=query, context=context)

    if use_real_openai and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.responses.create(
            model="gpt-5.4-mini",
            input=[
                {"role": "developer", "content": INSTRUCTIONS},
                {"role": "user", "content": PROMPT_TEMPLATE.format(question=query, context=context)},
            ],
        )
        answer = response.output_text
        input_tokens = response.usage.input_tokens
        print(f"Q3 - LLM Answer: {answer}")
        print(f"Q3 - Real input tokens: {input_tokens}")
        return input_tokens
    else:
        estimated = estimate_tokens(prompt)
        print(f"Q3 - Estimated input tokens: {estimated:.0f}")
        return estimated


def q4_chunk_pages(documents):
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Q4 - Total chunks created: {len(chunks)}")
    return chunks


def q5_chunked_rag(chunks, query, full_page_tokens, use_real_openai=False):
    chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
    chunk_index.fit(chunks)

    search_results = chunk_index.search(query, num_results=5)
    context = build_rag_context(search_results)
    prompt = INSTRUCTIONS + PROMPT_TEMPLATE.format(question=query, context=context)

    if use_real_openai and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.responses.create(
            model="gpt-5.4-mini",
            input=[
                {"role": "developer", "content": INSTRUCTIONS},
                {"role": "user", "content": PROMPT_TEMPLATE.format(question=query, context=context)},
            ],
        )
        answer = response.output_text
        chunked_tokens = response.usage.input_tokens
        print(f"Q5 - LLM Answer: {answer}")
        print(f"Q5 - Real input tokens: {chunked_tokens}")
    else:
        chunked_tokens = estimate_tokens(prompt)
        print(f"Q5 - Estimated input tokens: {chunked_tokens:.0f}")

    ratio = full_page_tokens / chunked_tokens
    print(f"Q5 - Ratio: {ratio:.1f}x fewer tokens with chunking")
    return chunked_tokens


def q6_agent_with_search_tool(chunks):
    if not os.environ.get("OPENAI_API_KEY"):
        print("Q6 - Skipped, OPENAI_API_KEY not set")
        return None

    from openai import OpenAI
    from toyaikit.tools import Tools
    from toyaikit.chat.runners import OpenAIResponsesRunner

    chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
    chunk_index.fit(chunks)

    search_call_count = {"count": 0}

    def search(query: str) -> list[dict]:
        search_call_count["count"] += 1
        return chunk_index.search(query, num_results=5)

    tools = Tools()
    tools.add_tool(search)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    agent_instructions = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."

    runner = OpenAIResponsesRunner(
        tools=tools,
        developer_prompt=agent_instructions,
        chat_interface=None,
        llm_client=client,
        model="gpt-5.4-mini",
    )

    question = "How does the agentic loop work, and how is it different from plain RAG?"

    result = runner.run(question)

    print(f"Q6 - Agent answer: {result}")
    print(f"Q6 - Search tool calls: {search_call_count['count']}")
    return search_call_count["count"]


def main():
    documents = load_lesson_pages()

    q1_count_lesson_pages(documents)

    page_index, _ = q2_index_and_search(documents)

    query = "How does the agentic loop keep calling the model until it stops?"

    use_real_openai = bool(os.environ.get("OPENAI_API_KEY"))

    full_page_tokens = q3_full_page_rag(page_index, query, use_real_openai=use_real_openai)

    chunks = q4_chunk_pages(documents)

    q5_chunked_rag(chunks, query, full_page_tokens, use_real_openai=use_real_openai)

    q6_agent_with_search_tool(chunks)


if __name__ == "__main__":
    main()

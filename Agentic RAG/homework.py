"""
LLM Zoomcamp 2026 - Homework 1: Agentic RAG
=============================================

YE FILE KIYA KARTI HAI? (What does this file do?)
---------------------------------------------------
Ye ek simple, step-by-step script hai jo LLM Zoomcamp ke Module 1 ka
homework solve karti hai. Hum course ke "lessons" (markdown files) ko
data ke tor par use kar rahe hain, aur unpe ek RAG (Retrieval-Augmented
Generation) system bana rahe hain - phir usko "agent" bana rahe hain.

Ye script 6 steps mein chalti hai (Q1 se Q6 tak):

  Q1 -> Kitne lesson pages hain? (count)
  Q2 -> minsearch se index bana ke search karo
  Q3 -> Pure pages pe RAG chalao, input tokens count karo
  Q4 -> Pages ko chunks (chote tukdon) mein todo, count karo
  Q5 -> Chunks pe RAG chalao, input tokens compare karo Q3 se
  Q6 -> RAG ko agent banao (LLM khud decide kare kab search karna hai)

NOTE: Q3, Q5, Q6 ke liye OpenAI API key chahiye hoti hai (paid).
Agar aapke paas key NAHI hai, to Q1, Q2, Q4 chal jayenge bina kisi cost ke,
aur Q3/Q5 ka rough estimate bhi mil jayega (without calling OpenAI).
Q6 sirf real API key ke saath chalta hai (agent ko actually LLM chalata hai).

KAISE RUN KARNA HAI -> neeha "HOW TO RUN" section dekho (ya README.md).
"""

import os

# ---------------------------------------------------------------------------
# STEP 0: Zaroori libraries install karo (sirf ek dafa terminal mein chalao)
# ---------------------------------------------------------------------------
# pip install gitsource minsearch openai toyaikit
#
# Agar "uv" use kar rahe ho:
# uv add gitsource minsearch openai toyaikit
# ---------------------------------------------------------------------------

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index


def load_lesson_pages():
    """
    Course ke GitHub repo se saare 'lesson' markdown pages download karta hai.

    Hum ek fixed commit id use kar rahe hain (8c1834d) taake sabko EXACTLY
    wohi data mile jo homework banate waqt use hua tha - is se answers
    sabke match honge.
    """
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",          # data ko "freeze" karne ke liye
        allowed_extensions={"md"},     # sirf .md (markdown) files chahiye
        filename_filter=lambda path: "/lessons/" in path,  # sirf lessons folder
    )

    files = reader.read()

    # Har file ko parse karke ek dictionary milti hai: {filename, content}
    documents = [file.parse() for file in files]
    return documents


def q1_count_lesson_pages(documents):
    """Q1: Kitne lesson pages hain? Answer ki options: 24 / 72 / 240 / 720"""
    count = len(documents)
    print(f"\n[Q1] Total lesson pages: {count}")
    return count


def q2_index_and_search(documents):
    """
    Q2: Documents ko minsearch ke andar index karo, phir ek query search karo.

    - 'content' field ko TEXT field banaya (ye wo field hai jisme actual
      lesson ka text hai - isi pe search hoga).
    - 'filename' field ko KEYWORD field banaya (filter/identify karne ke liye).
    """
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)  # index "train" / build karo documents pe

    query = "How does the agentic loop keep calling the model until it stops?"
    results = index.search(query, num_results=5)

    top_result_filename = results[0]["filename"]
    print(f"\n[Q2] Search query: {query}")
    print(f"[Q2] Top result filename: {top_result_filename}")

    return index, results


def estimate_tokens(text):
    """
    Rough tarika tokens count karne ka: average 1 token ~= 4 characters
    (English text ke liye). Ye sirf "andaza" (estimate) hai - real number
    OpenAI API call karne se hi milega (response.usage.input_tokens se).
    """
    return len(text) / 4


def build_rag_context(search_results):
    """
    Search se mile results ko ek single "context" string mein jodta hai,
    jo hum LLM ko bhejenge taake wo usme se answer dhoond sake.
    """
    parts = []
    for doc in search_results:
        parts.append(doc["filename"])
        parts.append(doc["content"])
        parts.append("")  # khali line, separator ke liye
    return "\n".join(parts).strip()


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def q3_full_page_rag(index, query, use_real_openai=False):
    """
    Q3: Pure (full) lesson pages ke index pe RAG chalao, aur input tokens
    count karo - ye batata hai ke hum LLM ko kitna bada context bhej rahe hain.

    Agar use_real_openai=True hai aur OPENAI_API_KEY set hai, to ye
    actual OpenAI ko call karega aur real token count dega.
    Warna, sirf estimate dega (chars / 4).
    """
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
        print(f"\n[Q3] LLM Answer: {answer}")
        print(f"[Q3] REAL input tokens (from OpenAI): {input_tokens}")
        return input_tokens
    else:
        estimated = estimate_tokens(prompt)
        print(f"\n[Q3] Estimated input tokens (full pages, NO API call): ~{estimated:.0f}")
        print("[Q3] (Closest multiple-choice option: 700 / 7000 / 70000 / 700000)")
        return estimated


def q4_chunk_pages(documents):
    """
    Q4: Har lesson page ko chote "chunks" (tukdon) mein todo.

    size=2000  -> har chunk 2000 characters ka hoga
    step=1000  -> agla chunk 1000 characters aage se shuru hoga
                  (matlab consecutive chunks 1000 characters overlap karte hain,
                  taake boundary pe split hua text kisi ek chunk mein poora mil jaye)
    """
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"\n[Q4] Total chunks created: {len(chunks)}")
    return chunks


def q5_chunked_rag(chunks, query, full_page_tokens, use_real_openai=False):
    """
    Q5: Chunks ko index karo, wahi query phir se search/RAG karo,
    aur dekho input tokens kitne KAM (fewer) ho gaye Q3 ke mukable mein
    (kyunke chunks Q3 ke pure pages se chote hote hain).
    """
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
        print(f"\n[Q5] LLM Answer: {answer}")
        print(f"[Q5] REAL input tokens (chunked): {chunked_tokens}")
    else:
        chunked_tokens = estimate_tokens(prompt)
        print(f"\n[Q5] Estimated input tokens (chunked, NO API call): ~{chunked_tokens:.0f}")

    ratio = full_page_tokens / chunked_tokens
    print(f"[Q5] Ratio (full pages / chunked) = {ratio:.1f}x fewer tokens with chunking")
    print("[Q5] (Closest option: about the same / 3x fewer / 10x fewer / 30x fewer)")
    return chunked_tokens


def q6_agent_with_search_tool(chunks):
    """
    Q6: Ab RAG ko ek "agent" banate hain - matlab LLM ko ek 'search' tool
    de dete hain, aur usko khud faisla karne dete hain ke kab search karna
    hai aur kitni baar.

    Ye step sirf REAL OpenAI API key ke saath kaam karega, kyunke ismein
    LLM ko khud decide karna hota hai.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[Q6] SKIPPED - OPENAI_API_KEY set nahi hai. Ye step sirf real")
        print("     API key ke saath chalta hai (agent ko LLM hi chalata hai).")
        return None

    from openai import OpenAI
    from toyaikit.tools import Tools
    from toyaikit.chat.runners import OpenAIResponsesRunner

    chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
    chunk_index.fit(chunks)

    search_call_count = {"count": 0}

    def search(query: str) -> list[dict]:
        """
        Search the course lesson chunks for content relevant to the query.

        Args:
            query: keywords related to the student's question.

        Returns:
            List of matching chunks (filename + content).
        """
        search_call_count["count"] += 1
        return chunk_index.search(query, num_results=5)

    tools = Tools()
    tools.add_tool(search)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    agent_instructions = (
        "You're a course teaching assistant. Answer the student's question "
        "using the search tool. Make multiple searches with different "
        "keywords before answering."
    )

    runner = OpenAIResponsesRunner(
        tools=tools,
        developer_prompt=agent_instructions,
        chat_interface=None,
        llm_client=client,
        model="gpt-5.4-mini",
    )

    question = (
        "How does the agentic loop work, and how is it different from plain RAG?"
    )

    result = runner.run(question)

    print(f"\n[Q6] Agent's final answer: {result}")
    print(f"[Q6] Number of times 'search' tool was called: {search_call_count['count']}")
    print("[Q6] (Closest option: 0 / 4 / 10 / 20)")
    return search_call_count["count"]


def main():
    print("=" * 70)
    print("LLM ZOOMCAMP 2026 - HOMEWORK 1: AGENTIC RAG")
    print("=" * 70)

    # Step 1: Data load karo
    documents = load_lesson_pages()

    # Q1
    q1_count_lesson_pages(documents)

    # Q2
    page_index, _ = q2_index_and_search(documents)

    query = "How does the agentic loop keep calling the model until it stops?"

    # Agar aapke paas OPENAI_API_KEY hai to ise True kar dein
    USE_REAL_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

    # Q3
    full_page_tokens = q3_full_page_rag(page_index, query, use_real_openai=USE_REAL_OPENAI)

    # Q4
    chunks = q4_chunk_pages(documents)

    # Q5
    q5_chunked_rag(chunks, query, full_page_tokens, use_real_openai=USE_REAL_OPENAI)

    # Q6
    q6_agent_with_search_tool(chunks)

    print("\n" + "=" * 70)
    print("DONE! Sab steps complete ho gaye.")
    print("=" * 70)


if __name__ == "__main__":
    main()

#spam openai
import asyncio
import os
import openai
import time
from time import monotonic
from tenacity import retry, wait_random_exponential

from openlimit.redis_rate_limiters import ChatRateLimiterWithRedis

openai.api_key = os.getenv("OPENAI_API_KEY")

TOTAL_TOKENS = 0
TOTAL_REQUESTS = 0
N_REQUESTS = 50

# rate_limiter = ChatRateLimiter(request_limit=3000, token_limit=90000)
rate_limiter = ChatRateLimiterWithRedis(
    request_limit=3000,
    token_limit=90000,
    redis_url="redis://localhost:6379"
)

ALWAYS_SEARCH_STRUCTURED_DECOMPOSE_CHAT_PROMPT = [{"role":"system", "content":"""You are a code debugging assistant trained by Adrenaline to help programmers. Give valid JSON structured output.
Always try and search for relevant code.
Classify each query to one of these types: Search, Explain, Debug, Tutorial.
Most query types should be Search"""},
{"role":"user", "content":"How does God work?"},
{"role":"system", "content":"""{
   "type": "Explain",
   "rephrase": "How does God work?",
   "retrieval_commands": [
       "Find where God is defined",
       "Find where God is being used"
   ]
}"""},
{"role":"user", "content":"Explain what does each REST endpoint does"},
{"role":"system", "content":"""{
   "type": "Search",
   "rephrase": "What does each REST endpoint do?",
   "retrieval_commands": [
       "Find where each REST endpoint is defined",
       "Find where each REST endpoint is being used"
   ]
}"""},
{"role":"user", "content":"How can I add a new stripe endpoint?"},
{"role":"system", "content":"""{
   "type": "Tutorial",
   "rephrase": "How can I add a new stripe endpoint?",
   "retrieval_commands": [
       "Find where stripe is defined",
       "Find where stripe is being used"
   ]
}"""},
{"role":"user", "content":"""Where is this code used?
async def get_chunk_embeddings(chunks_with_summaries):
   tasks = [generate_hypothetical_queries(chunk) for chunk in chunks_with_summaries]
I dont understnad it"""},
{"role":"system", "content":"""{
   "type": "Search",
   "rephrase": "Where is this code used?",
   "retrieval_commands": [
       "Find where get_chunk_embeddings is defined",
       "Find where get_chunk_embeddings is being used"
   ]
}"""},
{"role":"user", "content":"""Traceback (most recent call last):
 File "/rubrick-api/engine/retrieval/test.py", line 5, in <module>
   from retrieval import index_codebase, query_codebase
 File "/rubrick-api/engine/retrieval/utilities/chunking.py", line 2, in <module>
   import tiktoken
ModuleNotFoundError: No module named 'tiktoken'"""},
{"role":"system", "content":"""{
   "type": "Debug",
   "rephrase": "How can we fix ModuleNotFoundError: No module named 'tiktoken' from /rubrick-api/engine/retrieval/utilities/chunking.py?",
   "retrieval_commands": [
       "Find where tiktoken is defined",
       "Find where tiktoken is being used"
       "Find where tiktoken is being imported"
       "Find where tiktoken is being installed"
   ]
}"""}
]

ALWAYS_SEARCH_STRUCTURED_DECOMPOSE_CHAT_PROMPT = [
{"role":"system", "content":"""You are a code debugging assistant trained by Adrenaline to help programmers.
Always try and search for relevant code.
Classify each query to one of these types: Search, Explanation, Debug, Tutorial.
All query types should be Search"""},
{"role":"user", "content":"How does God work?"},
{"role":"system", "content":"""<†type=Explanation/>
<†rephrase=How does God work?†/>
<†retrieval_step=Find where God is defined†/>
<†retrieval_step=Find where God is being used†/> """},
{"role":"user", "content":"Can you list and explain what does each REST endpoint does"},
{"role":"system", "content":"""<†type=Search†/>
<†rephrase=What does each REST endpoint do?†/>
<†retrieval_step=Find where each REST endpoint is defined†/>
<†retrieval_step=Find where each REST endpoint is being used†/>"""},
{"role":"user", "content":"How can I add a new stripe endpoint?"},
{"role":"system", "content":"""<†type=Tutorial†/>
<†rephrase=How can I add a new stripe endpoint?†/>
<†retrieval_step=Find where stripe is defined†/>
<†retrieval_step=Find where stripe is being used†/> """},
{"role":"user", "content":"""Where is this code used?
async def get_chunk_embeddings(cs):
      tasks = [ghq(chunk) for c in cs]"""},
{"role":"system", "content":"""<†type=Search†/>
<†rephrase=Where is this get_chunk_embeddings used?†/>
<†retrieval_step=Find where get_chunk_embeddings is being used†/> """},
]

    

@retry(wait=wait_random_exponential(multiplier=1, max=60))
async def call_gpt35(prompt, index, stream=False, max_tokens=1000, temperature=0.5):
    global TOTAL_TOKENS, TOTAL_REQUESTS
    
    chat_params = { 
        "model": "gpt-3.5-turbo", 
        "messages": prompt,
    }

    async with rate_limiter.limit(**chat_params):
        print(f"exec start: reqIndex={index};")
        response = await openai.ChatCompletion.acreate(**chat_params)
        TOTAL_REQUESTS += 1
        print(f"exec ended: reqIndex={index}; tokens={response['usage']['total_tokens']} passed={format(monotonic(), '.2f')} secs. Completed TOTAL_REQUESTS={TOTAL_REQUESTS}")
        TOTAL_TOKENS = TOTAL_TOKENS + response['usage']['total_tokens']
        return response
    

def build_prompts():
    global N_REQUESTS
    
    prompts = []
    for i in range(1, N_REQUESTS):
        prompt = [ALWAYS_SEARCH_STRUCTURED_DECOMPOSE_CHAT_PROMPT[0]]
        prompt += ALWAYS_SEARCH_STRUCTURED_DECOMPOSE_CHAT_PROMPT[1:]*5
        prompts.append(prompt)
        

    return prompts

async def test():
    global N_REQUESTS
    
    prompts = build_prompts()
    tasks = [call_gpt35(prompt, index) for index, prompt in enumerate(prompts, start=1)]
    results = await asyncio.gather(*tasks)

    print(f"Completed {N_REQUESTS} requests in: {monotonic()} seconds, consuming: {TOTAL_TOKENS} tokens")

    # print(results)
if __name__ == "__main__":
    asyncio.run(test())
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ Use the following pieces of information to answer the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

#set custom prompt
def set_custom_prompt(context, question):
    """
    Prompt template for QA retrieval for each vector stores
    """
    #question = cl.user.message.content
    #prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', str(question)])
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# load language model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

# Question Answer chain Retriveal
def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,  # Language model
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}), # Vector store
        #return_source_document = True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# return qa chain
def qa_bot(context, question):
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device' : 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt(context, question)
    qa = retrieval_qa_chain(llm,qa_prompt,db)
    return qa

# return final result of query
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

## Chainlit ####

# to start the app
@cl.on_chat_start
async def start():
    context = "Your initial context is here!"
    question = "Your initial question is here!"
    chain = qa_bot(context, question)
    msg = cl.Message(content="Starting the bot") # initial message can be modified
    await msg.send()
    msg.content = "hey, welcome to the bot" # header of the chat
    await msg.update()
    cl.user_session.set("chain", chain)


# to send the message
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True, answer_prefix_tokens = ["FINAL", "ANSWER"]
    #     )
    # cb.answer_reached=True
    res = await chain.acall(message)
    answer = res["result"]
    sources = res.get("sources_documents", [])

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo sources found"
    await cl.Message(content=answer).send()
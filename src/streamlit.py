# Importing the necessary Python libraries
import os
import yaml
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper



## OPENAI API CONNECTION
## ---------------------------------------------------------------------------------------------------------------------
# Loading the API key and organization ID from file (NOT pushed to GitHub)
with open('../keys/openai-keys.yaml') as f:
    keys_yaml = yaml.safe_load(f)

# Applying our API key and organization ID to OpenAI
openai.organization = keys_yaml['ORG_ID']
openai.api_key = keys_yaml['API_KEY']

# Setting the OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = keys_yaml['API_KEY']



## STREAMLIT SETUP
## ---------------------------------------------------------------------------------------------------------------------
# Setting the title of our Streamlit application
st.title('LangChain YouTube GPT Creator')

# Setting a variable to capture the user prompt from Streamlit
user_prompt = st.text_input('Plug in your prompt here.')

# Setting LangChain's connection to OpenAI
llm = OpenAI(temperature = 0.9)



## LANGCHAIN
## ---------------------------------------------------------------------------------------------------------------------
# Setting up the prompt template for generating titles
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a YouTube video title about {topic}'
)

# Setting up a prompt template for generating scripts
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me a YouTube video script based on this title TITLE: {title} but also while leveraging this wikipedia research: {wikipedia_research}'
)

title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat_history')

# Instantiating our LangChain chains
title_chain = LLMChain(llm = llm,
                       prompt = title_template,
                       verbose = True,
                       output_key = 'title',
                       memory = title_memory)
script_chain = LLMChain(llm = llm,
                        prompt = script_template,
                        verbose = True,
                        output_key = 'script',
                        memory = script_memory)

wiki = WikipediaAPIWrapper()
# sequential_chain = SequentialChain(chains = [title_chain, script_chain],
#                                    input_variables = ['topic'],
#                                    output_variables = ['title', 'script'],
#                                    verbose = True)

# Show stuff to the screen if there is a prompt
if user_prompt:
    title = title_chain.run(user_prompt)
    wiki_research = wiki.run(user_prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
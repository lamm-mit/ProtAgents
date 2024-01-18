#!/usr/bin/env python
# coding: utf-8

import autogen
from llm_config import llm_config
import agent_functions as func
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

config_list  = autogen.config_list_from_models(model_list=["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"])

dir_path = './doc_dir/'
code_dir='./code_protein/'

#autogen.ChatCompletion.start_logging()
user_proxy = autogen.UserProxyAgent(
#user_proxy = autogen.AssistantAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user_proxy. Plan execution needs to be approved \
                    by user_proxy.",
    max_consecutive_auto_reply=None,
    code_execution_config=False,
    #code_execution_config={"work_dir": "coding"},
    #code_execution_config={"work_dir": "coding"},
    #code_execution_config={"work_dir": "coding",
    #                      "last_n_messages": 1,
    #                      },
)
'''
coder=GPTAssistantAgent(
#coder = autogen.AssistantAgent(
    name="Coder",
    instructions="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
     
    #system_message="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)
'''
#coder=GPTAssistantAgent(
coder = autogen.AssistantAgent(
    name="Coder",
    #instructions="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
     
    system_message="Coder. You write Python code for instance for plotting a function or saving results to a csv file.\
    Wrap the code in a code block that specifies the script type. The user can't modify your code. So do\
not suggest incomplete code which requires others to modify. Don't use a code block if it's not \
intended to be executed by the executor. \
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. \
Check the execution result returned by the executor. \
If the result indicates there is an error, fix the error and output the code again. Suggest the full code \
instead of partial code or code changes. If the error can't be fixed or if the task is not solved even \
after the code is executed successfully, analyze the problem, revisit your assumption, collect \
additional info you need, and think of a different approach to try. \
When writing code, assert the boundary condition. \
You don't install packages. \
    ", # This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    #instructions="Coder. You write Python code.",# This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
     
    system_message="""Critic. You double-check plan, especially the functions and function parameters. 
    Check whether the plan included all the necessary parameters for the suggested function. 
    You provide feedback.	
    You print TERMINATE when the task is finished sucessfully.
    """, # This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)



executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. You follow the plan. Execute the code written by the coder and return outcomes.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 12, "work_dir": code_dir},
    llm_config=llm_config,
)

 
planner = autogen.AssistantAgent(
    name="Planner",
    system_message = """Planner. You develop a plan. Begin by explaining the plan. Revise the plan based on feedback from the critic and user_proxy, until user_proxy approval. 
The plan may involve calling custom function for retrieving knowledge, designing proteins, and computing and analyzing protein properties. You include the function names in the plan and the necessary parameters.
If the plan involves retrieving knowledge, retain all the key points of the query asked by the user for the input message.
""",# 	If the plan involves retrieving knowledge, retain all the key points of the query asked by the user for the input message.  
    #This may involve identifying the chemical formula in SMILES codes, retrieving coordinates, and doing calculations.",# Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# `docs_path` is the path to the docs directory. By default, it is set to "./docs". Here we generated the documentations from FLAML's docstrings.
# Navigate to the website folder and run `pydoc-markdown` and it will generate folder `reference` under `website/docs`.
# `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.

# Create a new collection for NaturalQuestions dataset
# `task` indicates the kind of task we're working on. In this example, it's a `qa` task.
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    system_message="Assistant who has extra content retrieval power for biomaterials domain knowledge. The assistant follows the plan.",

    human_input_mode="NEVER",
    is_termination_msg=termination_msg,
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": f"{dir_path}", #f"{doc_dir}",
        "chunk_token_size": 3000,
        "model": config_list[0]["model"],
        #"client": chromadb.PersistentClient(path=coll_path),
        #"collection_name": coll_name,
        "chunk_mode": "one_line",
        #"chunk_mode": "multi_lines", # "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
        "get_or_create" : "True"
    },
    llm_config=llm_config,
)
'''
reviewer=GPTAssistantAgent(
#reviewer = autogen.AssistantAgent(
    name="Scientific_Reviewer",
   # is_termination_msg=termination_msg,
    #system_message="You are a scientific reviewer who offers additional background that will be incorporated into the answer. ",
    instructions="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    llm_config=llm_config,
)
'''
#reviewer=GPTAssistantAgent(
reviewer = autogen.AssistantAgent(
    name="Scientific_Reviewer",
   # is_termination_msg=termination_msg,
    system_message="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    #instructions="Materials scientist. You follow the plan. As a materials scientist you offer additional background that will be incorporated into the answer.",
    llm_config=llm_config,
)

assistant = autogen.AssistantAgent(
    name="assistant",
    #instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",
    
        system_message="assistant. You have access to all the custom functions. You focus on executing the functions suggested by the planner or the critic. You also have the ability to prepare the required input parameters for the functions.\
    llm_config=llm_config,
    function_map={
        
        #"query_DFT": coords_from_SMILES,
        "retrieve_content": func.retrieve_content_LlamaIndex,
        #"retrieve_content":  retrieve_content_LlamaIndex,
       # "coords_from_SMILES": coords_from_SMILES,
        "fold_protein": func.fold_protein,
        "analyze_protein_structure": func.analyze_protein_structure,
        #"get_FASTA_from_name": func.get_FASTA_from_name,
        "design_protein_from_length": func.design_protein_from_length,
        "calc_protein_ANM": func.calc_protein_ANM,
        "fetch_protein_structure_from_PDBID": func.fetch_protein_structure_from_PDBID,
        "calucalte_energy_from_seq": func.calculate_energy_from_seq,
        "calucalte_force_from_seq": func.calculate_force_from_seq,
        "calucalte_force_energy_from_seq": func.calculate_force_energy_from_seq,
        "generate_seq_from_energy": func.generate_seq_from_energy,
        "analyze_protein_CATH_from_PDBID": func.analyze_protein_CATH_from_PDBID,
        "analyze_protein_length_from_PDB": func.analyze_protein_length_from_PDB,
        "analyze_protein_seq_from_PDB": func.analyze_protein_seq_from_PDB,
        "design_protein_from_CATH": func.design_protein_from_CATH,
        "save_to_csv_file": func.save_to_csv_file,
    },

    #code_execution_config={"work_dir": "coding"},
)
'''
sequence_retriever =  GPTAssistantAgent(
    
    name="sequence_retriever",
    #instructions="You collect information from experts, fold proteins, and carry out other simulations. Reply TERMINATE when the task is done.",
    
    instructions="Sequence retriever. You identify amino acid sequences based on the name of the protein.",
    llm_config=llm_config ,
     
    #code_execution_config={"work_dir": "coding"},
)
'''

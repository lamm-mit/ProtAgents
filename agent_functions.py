#!/usr/bin/env python
# coding: utf-8
code_dir='./code_protein/'
device = 'cpu'

import os

try:
    os.mkdir(code_dir)
except:
    pass

import py3Dmol
from prody import *
import pandas as pd
import utils

from Bio.PDB import PDBParser, DSSP
from collections import Counter
import json

import json
import autogen
import openai

from chroma import api
from chroma import Chroma, Protein, conditioners
from chroma.models import graph_classifier, procap

from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import re

import requests, sys

import subprocess

from llama_index import StorageContext, load_index_from_storage
##########################################################################################################
storage_context = StorageContext.from_defaults(persist_dir="protein_force_index")
new_index = load_index_from_storage(storage_context)
query_engine = new_index.as_query_engine(similarity_top_k=20)

from ForceGPT import ForceGPTmodel
model, tokenizer = ForceGPTmodel(model_name='../../ProteinForceGPT/128_length/save_local_V10_BGPTneo_ProtForce_v10_B/', device=device)
###########################################################################################################

def analyze_protein_CATH_from_PDBID(PDB_id):
    '''
    Returns the sequence length of proteins from their PDB id.
    '''
    cath = CATHDB()
    
    result = cath.search(PDB_id)
  
    return json.dumps(result.cath[0], indent=4)

def analyze_protein_length_from_PDB(PDB_name):
    '''
    Returns the sequence length of proteins from their PDB id or name.
    '''
    if re.search('generated', PDB_name) or re.search('.pdb', PDB_name):
        if re.search(f'{code_dir}', PDB_name):
            prot = Protein.from_PDB(PDB_name)
            length = len(prot.sequence())
        else:
            prot = Protein.from_PDB(code_dir+PDB_name)
            length = len(prot.sequence())
    elif re.search('.pdb', PDB_name):
            prot = Protein.from_PDB(PDB_name)
            length = len(prot.sequence())
    
    else:
        prot = Protein.from_PDBID(PDB_name)
        length = len(prot.sequence())

    return json.dumps(length, indent=4)

def analyze_protein_seq_from_PDB(PDB_name):
    '''
    Returns the sequence of proteins from their PDB id or name.
    '''
    if re.search('generated', PDB_name) or re.search('.pdb', PDB_name):
        if re.search(f'{code_dir}', PDB_name):
            prot = Protein.from_PDB(PDB_name)
            seq = prot.sequence()
        else:
            prot = Protein.from_PDB(code_dir+PDB_name)
            seq = prot.sequence()
                
    else:
        prot = Protein.from_PDBID(PDB_name)
        seq = prot.sequence()

    return json.dumps(seq, indent=4)

def fold_protein (sequence, name):
    filename='temp.fasta'

    output=code_dir#'./'
    with open(filename, "w") as f:
        f.write(">%s\n%s\n" % (name, sequence))
        
    # Build the command        
    command = f"omegafold {filename} {output} --model 2 --device cpu "
    # Run the command
    subprocess.run(command, shell=True)

    fix_pdb_file(code_dir+name+'.pdb',code_dir+name+'.pdb')
    return name+'.pdb'

def retrieve_content(message, n_results=3):
        ragproxyagent.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = ragproxyagent._check_update_context(message)
        if (update_context_case1 or update_context_case2) and ragproxyagent.update_context:
            ragproxyagent.problem = message if not hasattr(ragproxyagent, "problem") else ragproxyagent.problem
            _, ret_msg = ragproxyagent._generate_retrieve_user_reply(message)
        else:
            ret_msg = ragproxyagent.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

def retrieve_content_LlamaIndex(message,  ):

        #message='For these topics, provide detailed information: ' + message
        print (f'the message is: {message}')
        response = query_engine.query(message)
        response = response.response
        #agent 
        #response = agent.chat("Describe several biologically inspired composite ideas.")
        #print(str(response))
          
        return response if response else message

def retrieve_pdb_name_LlamaIndex(message,  ):

        #message='For these topics, provide detailed information: ' + message
        print (f'the message is: {message}')
        response = query_engine_pdb.query(message)
        response = response.response
        #agent 
        #response = agent.chat("Describe several biologically inspired composite ideas.")
        #print(str(response))
          
        return response if response else message

def coords_from_SMILES(SMILES='CCC'):

  mol = Chem.MolFromSmiles(SMILES)
  mol= Chem.AddHs(mol)
  Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
  Chem.SanitizeMol(mol)
  AllChem.EmbedMolecule(mol)

  try:
      AllChem.UFFOptimizeMolecule(mol)

  except:
      print ("UFF optimization did not work.")
  ch=[]

 
  mol.GetConformer()
  for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        ch.append (f'{atom.GetSymbol()} {positions.x:12.4} {positions.y:12.4} {positions.z:12.4}')
  return ch


def fix_pdb_file(original_pdb_path, fixed_pdb_path):
    """
    Inserts a CRYST1 record into a PDB file if it is missing.

    Args:
    original_pdb_path (str): Path to the original PDB file.
    fixed_pdb_path (str): Path where the fixed PDB file will be saved.
    """
    with open(original_pdb_path, 'r') as file:
        lines = file.readlines()

    #print (lines)
    # Check if the first record is CRYST1
    CRYST1 = False
    header = False
    for line in lines:
        LINE = str(line).split(sep=' ')
        for item in LINE:
            if re.search('CRYST1', item):
                CRYST1 = True
            if re.search('HEADER', item):
                header = True

    if (not CRYST1) and (not header):
        # Define a dummy CRYST1 record with a large unit cell
        # These numbers mean that the unit cell is a cube with 1000 Å edges.
        cryst1_record = "CRYST1 1000.000 1000.000 1000.000  90.00  90.00  90.00 P 1           1\n"
        lines.insert(0, cryst1_record)  # Insert the dummy CRYST1 record
        #lines.insert(0, 'header \n')

    with open(fixed_pdb_path, 'w') as file:
        file.writelines(lines)

    #print(f"Fixed PDB file written to {fixed_pdb_path}")

def add_missing_column(file_path ):
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
      
    header = False
    for line in lines:
        LINE = str(line).split(sep=' ')
        for item in LINE:
            if re.search('HEADER', item):
                header = True
    # Process lines
    modified_lines = []
    if not header:
        for line in lines:
            if line.startswith('ATOM'):
                columns = line.split()
                # Assuming the missing column is the atom type, which should be the 12th column
                if len(columns) < 13:
                    atom_type = columns[2][0]  # Extract atom type (3rd column in ATOM line)
                    # Add the atom type to the end of the line
                    modified_line = line.strip() + '    ' + atom_type + '\n'
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

# Example usage:
# fix_pdb_file('path_to_your_original_pdb_file.pdb', 'path_to_your_fixed_pdb_file.pdb')



def calculate_energy_from_seq(sequence):
    prompt = f"CalculateEnergy<{sequence}>"
    print(prompt)
    task = utils.extract_task(prompt, end_task_token='>') + ' '
    sample_output = utils.generate_output_from_prompt(model, device, tokenizer, prompt=task, num_return_sequences=1, num_beams=1, temperature=0.01)
    for sample_output in sample_output:
        result=tokenizer.decode(sample_output, skip_special_tokens=True)  
        extract_data=utils.extract_start_and_end(result, start_token='[', end_token=']')
        
    return json.dumps(extract_data, indent=4)

def calculate_force_from_seq(sequence):
    prompt = f"CalculateForce<{sequence}>"
    print(prompt)
    task = utils.extract_task(prompt, end_task_token='>') + ' '
    sample_output = utils.generate_output_from_prompt(model, device, tokenizer, prompt=task, num_return_sequences=1, num_beams=1, temperature=0.01)
    for sample_output in sample_output:
        result=tokenizer.decode(sample_output, skip_special_tokens=True)  
        extract_data=utils.extract_start_and_end(result, start_token='[', end_token=']')
        
    return json.dumps(extract_data, indent=4)

def calculate_force_energy_from_seq(sequence):
    prompt = f"CalculateForceEnergy<{sequence}>"
    print(prompt)
    task = utils.extract_task(prompt, end_task_token='>') + ' '
    sample_output = utils.generate_output_from_prompt(model, device, tokenizer, prompt=task, num_return_sequences=1, num_beams=1, temperature=0.01)
    for sample_output in sample_output:
        result=tokenizer.decode(sample_output, skip_special_tokens=True)  
        extract_data=utils.extract_start_and_end(result, start_token='[', end_token=']')
        
    return json.dumps(extract_data, indent=4)

def generate_seq_from_energy(energy, temperature=0.01):
    prompt = f"GenerateEnergy<{energy}>"
    print(prompt)
    task = utils.extract_task(prompt, end_task_token='>') + ' '
    sample_output = utils.generate_output_from_prompt(model, device, tokenizer, prompt=task, num_return_sequences=1, num_beams=1, temperature=float(temperature))
    for sample_output in sample_output:
        result=tokenizer.decode(sample_output, skip_special_tokens=True)  
        extract_data=utils.extract_start_and_end(result, start_token='[', end_token=']')
        
    return json.dumps(extract_data, indent=4)
    

def calc_protein_ANM(protein_structure, n_modes=10, cutoff=12):
    """
    Compute the first "n_modes" number of modes for
    a given protein_structure with cutoff="cutoff"
    """

   
    try:
        protein_structure = json.loads(protein_structure)
    except:
        pass
    try:
        protein_structure = protein_structure_vector.split(',')
    except:
        pass
   
    #pathPDBFolder(code_dir)
    
    protein_structure = str(protein_structure)
    if re.search('.pdb', protein_structure):
        pass
    else:
        protein_structure = protein_structure + '.pdb'
    
    if re.search('./', code_dir):
        code_dir_up = code_dir[re.search('./', code_dir).span()[1]:]
    else:
        code_dir_up = code_dir
    
    if re.search(code_dir_up, protein_structure):
        protein_structure = protein_structure
    else:
        protein_structure=code_dir_up+protein_structure

    
    #try:
    #    if not os.path.isfile(protein_structure):
    #        fetchPDB(protein_structure, folder=code_dir, compressed=False, copy=True)
    #        protein_structure=protein_structure+'.pdb'
    #except:
    #    print(f"file does not exist in {code_dir} or PDB")
 
    print(f'computing ANM for protein structure: {protein_structure}')
    #Parse the PDB file
    protein = parsePDB(protein_structure)
   
    anm, sel = prody.calcENM(protein, n_modes=n_modes, cutoff=cutoff)
    modes = [anm[i].getEigval().round(4) for i in range(n_modes)]

        #pd.DataFrame.from_dict(data)
   
    return json.dumps(modes, indent=4)


def analyze_protein_structure(protein_structure):
    
    #try:
        if re.search('.pdb', protein_structure):
            pass
        else:
            protein_structure = protein_structure + '.pdb'
        
        if re.search('./', code_dir):
            code_dir_up = code_dir[re.search('./', code_dir).span()[1]:]
        else:
            code_dir_up = code_dir
        
        if re.search(code_dir_up, protein_structure):
            protein_structure = protein_structure
        else:
            protein_structure=code_dir_up+protein_structure
            
            # Create a PDB parser
        print ('Analyzing the secondary strucute of this protein:', protein_structure )
            
        fix_pdb_file(protein_structure, protein_structure)
        add_missing_column(protein_structure)
        
        parser = PDBParser(QUIET=True)
        
        # Parse the PDB file
        structure = parser.get_structure('protein_structure', protein_structure)
        
        # Select the first model in the PDB file
        model = structure[0]
        
        # Run DSSP analysis
        dssp = DSSP(model, protein_structure, 'dssp')
        
        # Initialize a dictionary for secondary structure counts
        secondary_structure_counts = {
            'H': 0,  # Alpha helix
            'B': 0,  # Isolated beta-bridge
            'E': 0,  # Extended strand
            'G': 0,  # 3-helix (3/10 helix)
            'I': 0,  # 5 helix (pi-helix)
            'T': 0,  # Hydrogen bonded turn
            'S': 0,  # Bend
            'P': 0,  # Poly-proline helices
            '-': 0   # None
        }
        
        # Count each secondary structure type
        for residue in dssp:
            secondary_structure_counts[residue[2]] += 1
        
        # Calculate the total number of residues with assigned secondary structure
        total_residues = sum(secondary_structure_counts.values())
        
        print ("The protein analyzed has ", total_residues, "residues.")
            
        # Calculate the percentage content for each secondary structure type
        secondary_structure_percentages = {ss: (count / total_residues * 100) for ss, count in secondary_structure_counts.items()}
    #except:
    #    pass
     
    # Return the results as a JSON string
        return json.dumps(secondary_structure_percentages, indent=4)


#https://www.ebi.ac.uk/proteins/api/doc/#!/proteins/search
def get_FASTA_from_name (protein_name):
    size=128
    requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size={size}&protein={protein_name}"
    #requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size={size}&protein={name}"

    r = requests.get(requestURL, headers={ "Accept" : "application/json"})
    
    if not r.ok:
      r.raise_for_status()
      sys.exit()
    
    responseBody = r.text
    #print(responseBody)
    json_object = json.loads(responseBody)
    if len (json_object)>0:
        
        res=json_object[0]['sequence']['sequence']
    else:
        res='No results found.'
        
    return res


def design_protein_from_CATH(length, name, CATH_ANNOTATION, steps=300, devices='cpu'):
    chroma = Chroma()
    print(f'We use this CATH to generate protein with length {length}: {CATH_ANNOTATION}')
    proclass_model = graph_classifier.load_model("named:public", device=devices)
    conditioner = conditioners.ProClassConditioner("cath", CATH_ANNOTATION, model=proclass_model)
    cath_conditioned_protein, _ = chroma.sample(samples=1, steps=steps,
    conditioner=conditioner, chain_lengths=[length], full_output=True,       
)
      
    fname = f'{code_dir}{name}.pdb'     
    protein = cath_conditioned_protein
    protein.to(fname)
    sequence=protein.sequence()                                                    
        #print(f'protein with name {fname} generated.')  
    #print(protein_seq)
    return fname, sequence  


def design_protein_from_length (length, name, caption='', steps=300):
    chroma = Chroma()

    if caption != '':
        print (f'We use this caption to generate a protein: {caption}')
        procap_model = procap.load_model("named:public", device=device,
                                        strict_unexpected=False,
                                        )
        conditioner = conditioners.ProCapConditioner(caption, -1, model=procap_model)
    else:
        conditioner=None

    protein = chroma.sample(chain_lengths=[length],steps=steps,
                           conditioner=conditioner, )
    fname = f'{code_dir}{name}.pdb' 
    protein.to(fname)
    sequence=protein.sequence()

    return fname, sequence

def save_to_csv_file(input_JSON_dictionary, output_csv_name):
    '''
    Creates and stores a csv file from the data dictionary.
    '''
    data_dictionary = json.loads(input_JSON_dictionary)
    df = pd.DataFrame(data_dictionary)
    df.to_csv(f'{output_csv_name}')
    
    print(f'the results have been saved to csv file: {output_csv_name}')
    

def fetch_protein_structure_from_PDBID(PDB_id):
    """
    Fetch the PDB file using the PDB id of the protein.
    """
    try:
        PDB_id = json.loads(PDB_id)
    except:
        pass

    print(f'fetching protein structure with PDB id: {PDB_id}')

#    pdb_names = []
#    PDB_name_vector = PDB_name_vector.split(',')
#    for pb_vector in PDB_name_vector:
#        for pbd in pb_vector.split('/'):
#            pdb_names.append(pbd)

    
    
    #pathPDBFolder(code_dir)
    fetchPDB(PDB_id, folder=code_dir, compressed=False, copy=True) 
    
    file_name = f'{PDB_id}' + '.pdb'
    return json.dumps(file_name, indent=4)
   





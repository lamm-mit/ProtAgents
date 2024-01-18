#!/usr/bin/env python
# coding: utf-8

import json
import autogen
import openai
import os
import autogen

OpenAI_key='####'

######### OPENAI ###########
os.environ['OPENAI_API_KEY']=OpenAI_key

config_list  = autogen.config_list_from_models(model_list=["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"])

llm_config = {
  #Generate functions config for the Tool
  "functions":[

       {
            "name": "get_FASTA_from_name",
            "description": "With a protein name as input, provides a FASTA sequence of amino acids.",
            "parameters": {
                        "type": "object",
                        "properties": {
                            "protein_name": {
                            "type": "string",
                            "description": "Name of a protein.",
                        }
                                            },
                "required": ["protein_name"],
            },
      },
             {
            "name": "save_to_csv_file",
            "description": "With a JSON dictionary as input, saves the data to a csv file with a provided name.",
            "parameters": {
                        "type": "object",
                        "properties": {
                            "input_JSON_dictionary": {
                            "type": "string",
                            "description": "The input JSON dictionary.",
                        },
                            "output_csv_name": {
                            "type": "string",
                            "description": "The output name for the csv file.",
                        }
                                            },
                "required": ["input_JSON_dictionary", "output_csv_name"],
            },
          },
      {
            "name": "analyze_protein_structure",
            "description": "Given the protein structure file as input, analyzes and returns \
            the secondary structure of the protein. \
            The function returns a JSON dictionary with % content of the 8 secondary structure types.\
            The 8 secondary structures are ['H': alpha-helix], ['B': isolated beta bridge], ['E': Extended strand or beta-sheet]\
             ['G': 3-helix (3/10 helix)], ['I': 5 helix (pi-helix)], ['T': Hydrogen bonded turn], ['S': Bend], \
             ['P': Poly-proline helices] and ['-': None]\
            ",
            "parameters": {
                "type": "object",
                "properties": {
                    "protein_structure": {
                        "type": "string",
                        "description": "Portein structure file.",
                    }
                },
                "required": ["protein_structure"],
            },
         },
       {
            "name": "calucalte_energy_from_seq",
            "description": " Calculates the unfolding energy of a protein. The function requires the amino acid sequence of a protein structure ßin string format. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence in single-letter FASTA code.",
                    },
                },
                "required": ["sequence"],
            },
      },

             {
            "name": "calucalte_force_from_seq",
            "description": " Calculates the maximum unfolding force of a protein. The function requires the amino acid sequence of a protein structure \
            in string format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence in single-letter FASTA code.",
                    },
                },
                "required": ["sequence"],
            },
      },

             {
            "name": "calucalte_force_energy_from_seq",
            "description": " Calculates the unfolding energy and maximum force of a protein. The function requires the amino acid sequence of a protein structure in string format. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence in single-letter FASTA code.",
                    },
                },
                "required": ["sequence"],
            },
      },

                   {
            "name": "generate_seq_from_energy",
            "description": " Design a protein based on an input energy. The function requires an energy value  \
            in string format. Returns the Amino Acid sequence of the protein ",
            "parameters": {
                "type": "object",
                "properties": {
                    "energy": {
                        "type": "string",
                        "description": "The energy of the protein.",
                    },
                     "temperature": {
                        "type": "string",
                        "description": "The temperature value used for generation of text from autoregressive model.",
                    }
                },
                "required": ["energy"],
            },
      },

      
      {
            "name": "fold_protein",
            "description": "Fold a protein with required amino acid sequence which creates a protein structure file. The funtion alos returns the output protein sructure name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence in single-letter FASTA code.",
                    },
                         "name": {
                        "type": "string",
                        "description": "Name of the folded output protein structure.",
                    }
                },
                "required": ["sequence", "name"],
            },
      },
      {
            "name": "query_DFT",
            "description": "Calculate the energy of a molecule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "string",
                        "description": "The coordinates of the molecule.",
                    }
                },
                "required": ["coordinates"],
            },
      },
    {
        "name": "retrieve_content",
        "description": "An expert in retrieving knowledge about protein, their mechanical properties, structures, and PDB names.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to be used to retrieve detailed knowledge. ",
                }
            },
            "required": ["message"],
        },
    },
      
      {
        "name": "coords_from_SMILES",
        "description": "With a SMILES string as input, provides atom type and coordinates of a molecule.",
        "parameters": {
            "type": "object",
            "properties": {
                "SMILES": {
                    "type": "string",
                    "description": "SMILES string.",
                }
            },
            "required": ["SMILES"],
        },
      },
      {
        "name": "design_protein_from_length",
        "description": "With an optional caption and required length of the protein (number of amino acids)  \
         and a name as input, designs a new protein. Returns a PDB name and the amino acid sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "string",
                    "description": "Caption that describes the protein that will be designed. Default is no caption, empty string.",  },
                "length": {
                    "type": "number",
                    "description": "Length of the protein to be designed.",  },
                 "name": {
                    "type": "number",
                    "description": "Name of the protein to be saved.",  },
                "steps": {
                    "type": "number",
                    "description": "Number of sampling steps, default is 300.",  },
                
            },
            
            "required": ["length", "name"],
        },
      },
            {
        "name": "design_protein_from_CATH",
        "description": "With a required CATH_ANNOTATION domain (1 is mainly alpha, 2 is mainly beta, 3 is alpha beta) \
        , required output protein name, and length of the protein (number of amino acids) as input, \
        , designs a protein and creates a protein structure file. It returns the PDB file name and the amino acid sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                
                "name": {
                    "type": "string",
                    "description": "Name of the protein to be saved.",  },
              "CATH_ANNOTATION": {
                    "type": "string",
                    "description": "CATH_ANNOTATION that describes the protein CATH domain (1 is mainly alpha, 2 is mainly beta, 3 is alpha beta) that will be designed. Default is 2 (mainly beta).",  },
                "length": {
                    "type": "number",
                    "description": "Length of the protein to be designed.",  },
                "steps": {
                    "type": "number",
                    "description": "Number of sampling steps, default is 300.",  },
                
            },
            
            "required": ["CATH_ANNOTATION", "length", "name"],
        },
      },
      {
        "name": "calc_protein_ANM",
        "description": " With input of a protein structure file, calculates the frequencies of the first n_modes eigenmodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "protein_structure": {
                    "type": "string",
                    "description": "Protein structure file",  },
                "n_modes": {
                    "type": "number",
                    "description": "Number of modes to be calculated.",  },
                "cutoff": {
                    "type": "number",
                    "description": "Cutoff for elastic network calculation.",  },
                
            },
            
            "required": ["protein_structure"],
        },
      },

            {
        "name": "fetch_protein_structure_from_PDBID",
        "description": " With input of a protein PDB id, fetches the protein structure file and stores it in the directory. \
        It also returns the name of the file stored in the directory. \
        The input must be a PDB id of the protein not a generated protein name.",
        "parameters": {
            "type": "object",
            "properties": {
                "PDB_id": {
                    "type": "string",
                    "description": "Protein PDB id.",  },
            },
            
            "required": ["PDB_id"],
        },
      },

                  {
        "name": "analyze_protein_CATH_from_PDBID",
        "description": " With input of a protein PDB id, you collect info about the CATH domain or structure classification of the protein. \
         The input must be a PDB id of the proteins not a generated protein name.",
        "parameters": {
            "type": "object",
            "properties": {
                "PDB_id": {
                    "type": "string",
                    "description": "Protein PDB id.",  },
            },
            
            "required": ["PDB_id"],
        },
      },

                        {
        "name": "analyze_protein_length_from_PDB",
        "description": " With input of a protein PDB id or protein name, you give the length of the amino-acid sequence of the protein.",
        "parameters": {
            "type": "object",
            "properties": {
                "PDB_name": {
                    "type": "string",
                    "description": "Protein PDB id or name.",  },
            },
            
            "required": ["PDB_name"],
        },
      },

                              {
        "name": "analyze_protein_seq_from_PDB",
        "description": " With input of a protein PDB id or protein name, you give the sequence of the amino-acid sequence of the protein.",
        "parameters": {
            "type": "object",
            "properties": {
                "PDB_name": {
                    "type": "string",
                    "description": "Protein PDB id or name.",  },
            },
            
            "required": ["PDB_name"],
        },
      },
      
       
  ],
  "config_list": config_list,  # Assuming you have this defined elsewhere
 # "request_timeout": 120,
}

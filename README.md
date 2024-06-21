# ProtAgents
## Protein discovery via large language model multi-agent collaborations combining physics and machine learning

A. Ghafarollahi, M.J. Buehler*

MIT

*mbuehler@MIT.EDU

## Summary

Designing de novo proteins beyond those found in nature holds significant promise for advancements in both scientific and engineering applications. Current methodologies for protein design often rely on AI-based models, such as surrogate models that address end-to-end problems by linking protein structure to material properties or vice versa. However, these models frequently focus on specific material objectives or structural properties, limiting their flexibility when incorporating out-of-domain knowledge into the design process or comprehensive data analysis is required. In this study, we introduce ProtAgents, a platform for de novo protein design based on Large Language Models (LLMs), where multiple AI agents with distinct capabilities collaboratively address complex tasks within a dynamic environment. The versatility in agent development allows for expertise in diverse domains, including knowledge retrieval, protein structure analysis, physics-based simulations, and results analysis. The dynamic collaboration between agents, empowered by LLMs, provides a versatile approach to tackling protein design and analysis problems, as demonstrated through diverse examples in this study. 

The problems of interest encompass designing new proteins, analyzing protein structures and obtaining new first-principles data -- natural vibrational frequencies -- via physics simulations. The concerted effort of the system allows for powerful automated and synergistic design of \textit{de novo} proteins with targeted mechanical properties. The flexibility in designing the agents, on one hand, and their capacity in autonomous collaboration through the dynamic LLM-based multi-agent environment on the other hand, unleashes great potentials of LLMs in addressing multi-objective materials problems and opens up new avenues for autonomous materials discovery and design. 

![image](https://github.com/lamm-mit/ProtAgents/assets/101393859/4b457df4-35a4-4945-ba53-02796ffb1a07)

Figure 1: Overview of the model and approach. 

### Codes
This repository contains codes to solve complex problems in the context of protein design and analysis using multi-agent framework. The files named exp1, exp2, and exp3 in the repository, corresponding to the experiments I, II, and III, in the corresponding paper, respectively.   

### Requirements
Both OpenAI API and Chroma keys are required to run the codes. The OpenAI key must be provided in the "llm_config.py" file. Moreover, keys must be provided in the main files, exp1, exp2, and exp3. 

The model leverages a pre-trained autoregressive transformer model to predict the unfolding mechanical behavior of proteins. The path to the model must be provided in the file "agent_functions.py".

![image_Chroma_fold](https://github.com/lamm-mit/ProtAgents/assets/101393859/0c75ebfb-e708-4728-8942-2e8c9c63c0a7)

Figure 2: Example result, showing an overview of the multi-agent work to solve a complex design task. First the multi-agent uses Chroma to generate de novo protein sequences and structures conditioned on the input CATH class. Then using the generated protein structures, the natural frequencies and secondary structures content are computed. Next, the force (maximum force along the unfolding force-extension curve) and energy (the area under the force-extension curve) are computed from novel AA sequences using ProteinForceGPT, a pre-trained autoregressive transformer model.

### Original paper

Please cite this work as:
```
@article{GhafarollahiBuehler_2024,
    title   = {ProtAgents: Protein discovery via large language model multi-agent collaborations
               combining physics and machine learning},
    author  = {A. Ghafarollahi, M.J. Buehler},
    journal = {Digital Discovery},
    year    = {2024},
    volume  = {DOI: 10.1039/D4DD00013G},
    pages   = {},
    url     = {}
}
```

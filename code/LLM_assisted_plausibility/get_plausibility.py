from openai import OpenAI
#from .molecular_description_image import generate_image
import os
import ast 
from dotenv import load_dotenv

# Load .env from project root robustly regardless of CWD
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
# Also attempt a generic load (searches from CWD upward) as a fallback
load_dotenv()

#api_key = os.getenv("DEEPSEEK_API_KEY")
api_key= os.getenv("OPENAI_API_KEY")
def get_plausibility(description_list, smiles_list, original_smiles, original_molecule_description, task_description, dataset_description):
    """Check if a molecule is plausible as an enzyme inhibitor"""
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Ensure .env in project root contains it and python-dotenv is installed.")
    #client = OpenAI(base_url='https://api.deepseek.com', api_key=api_key)
    client = OpenAI(api_key=api_key)
    content = f"""You are an expert medicinal chemist evaluating molecules that are {task_description}s, and determining which substructure most likely (and least likely) causally explains this fact. You have 
     received several substructures of the molecule. You will be provided the description of each of these substructures as well 
      as the SMILES of each of these substructures. You will be receiving this in list format (i.e. Substructures SMILES [CCCCCCC, CCCCCCCCC, CCCC], Substructures Descriptions [description1, description2, description3]). You will also receive the SMILES of the original molecule and the description.  
 
      You are responsible for determining which of the provided substructures is most likely to be the causal element of the molecule that explains its properties,
       and which is the least likely to be the causal element of the molecule that explains its properties. 

    
        Consider factors like:
            - Presence of known pharmacophores for a {task_description} 
            - Presence of key binding groups for this task. 
            - Presence of important functional groups for this task. 
            - If this substructure is likely to cause the molecule to be a {task_description} from chemical principles 
            - If this substructure is likely to cause the molecule to be a {task_description} from biological principles
            - If this substructure is likely to cause the molecule to be a {task_description} from physical principles

     Dataset Description: {dataset_description}
     Your goal is to determine which of the substructures provided is most and least likely to explain the fact that the molecule is a {task_description}
     However, if it is not very obvious which substructure is the best or worst rationale, you should return a list of 0s. [0,0,0,...0]. The plausible and implausible substructures should always be distinct from each other functionally (i.e. they should have different functional groups, structures, etc). They will be used to guide a triplet loss with one as a positive example and one as a negative example. If the positive and negative examples are very similar, you should return a list of 0s. 

     
    
    """
    
    completion = client.chat.completions.create(
        model="gpt-5",
        stream=False,
        messages=[
            {
                "role": "system",
                "content": f"""{content}
                
                Respond ONLY with a list in the format [1,0,-1,...0]
                1 represents the substructure that is most likely to be the core element of the molecule (i.e. the best rationale)
                
                -1 represents the substructure that is the least likely to be the core element of the molecule (i.e. the worst rationale)
                0 should be used for all other substructures.

                If it is not clear which substructure is the best or worst rationale, you should return a list of 0s. [0,0,0,...0]. Do not hesitate to return a list of 0s if you are not sure which is the substructure that that is most likely to be causal for the predicition (and whihc is leeast likely to be causal for the predicition.  If the plausible and implausible substructures are very similar, you should return all 0s as well.  
                Thus, your output should either be a list with a single 1 and a single -1, and the rest 0s, or a list of 0s.No other format is allowed. Please make sure that the plausible and implausible substructures are distinct from each other functionally and would be sensical to guide a triplet loss. They should not have the exact same functional groups. You should feel very confident that they can meaningfully guide the triplet loss (i.e. plausible substructure is very likely to be the causal element toward the prediction, and implausible substructure is very unlikely) and align the representation space to physical/biolgoical reality. If you are not confident, return all 0s. 
                Respond ONLY with the list, nothing else. 

                """
            },
            {
                "role": "user",
                "content": f"""
                Original Molecule Description: {str(original_molecule_description)}
                   Original Molecule SMILES: {str(original_smiles)}
                Substructures SMILES: {str(smiles_list)}
        Substructures Descriptions: {str(description_list)}
     """
            

            }
        ]
    )
    
    response = completion.choices[0].message.content.strip().lower()
    print('completed run for ', original_smiles)
    print(response)
    try: 
        response_list = ast.literal_eval(response)
    except: 
        print('error parsing response for ', original_smiles)
        return response
    return response_list


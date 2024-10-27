task_prompt = { "prompt": """
               Task: Check if there is toxic content in 'User' comment according to our 
               toxic content categories below. 
               
               <BEGIN TOXIC CONTENT CATEGORIES>
               Violence and Hate
               Sexual Content
               Guns & Illegal Weapons
               Regulated or Controlled Substances
               Suicide and Self Harm
               Criminal Planning 
               <END TOXIC CONTENT CATEGORIES>
               
               <BEGIN USER COMMENT>
               {USER_COMMENT}
               <END USER COMMENT>
               
               Provide your safety assesment for the above user comment. 
               Only provide a single number:
               1 if user comment is "Toxic" or 
               0 if user comment is "Safe"
               Your responce must contain only 1 or 0, simply output a number. 
               Do not provide explanation or justification, just say 0 or 1.
               If you can not provide the answer output number 2.
               You are not allowed to output non numeric answer. You can only output 0, 1 or 2."""
            }
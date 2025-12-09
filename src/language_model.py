# Create src/llm.py
llm_code = '''"""
llm.py - Language Model Interface for Boston Rideshare Agent
Handles loading and generation from Hugging Face models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict


def _postprocess_to_two_lines(completion: str) -> str:
    """Extract Thought and Action lines from LLM output."""
    lines = completion.strip().split('\\n')
    thought_line = None
    action_line = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Thought:') and thought_line is None:
            thought_line = line
        elif line.startswith('Action:') and action_line is None:
            action_line = line
    
    result = []
    if thought_line:
        result.append(thought_line)
    if action_line:
        result.append(action_line)
    
    return '\\n'.join(result) if result else completion


class HF_LLM:
    """Hugging Face Language Model wrapper for agent."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 load_8bit: bool = False,
                 dtype=None,
                 max_new_tokens: int = 160,
                 generation_kwargs: Dict = None):
        """Initialize language model."""
        self.model_name = model_name
        self.load_8bit = load_8bit
        self.dtype = dtype if dtype else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_8bit=self.load_8bit if self.load_8bit else False
        )
        self.model.eval()
        
        # Generation config
        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.generation_kwargs.get("temperature", 0.3),
            top_p=self.generation_kwargs.get("top_p", 0.9),
            do_sample=self.generation_kwargs.get("do_sample", True)
        )
    
    def __call__(self, prompt: str) -> str:
        """Generate response from prompt."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.gen_cfg
            )
        
        # Decode
        completion = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return _postprocess_to_two_lines(completion)
'''
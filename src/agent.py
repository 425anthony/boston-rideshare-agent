# Create src/agent.py
agent_code = '''"""
agent.py - Boston Rideshare Decision Agent
Combines search, prompting, and LLM into unified agent system.
"""

import json
import re
from typing import Dict, Any, List


class RideshareAgent:
    """Boston Rideshare Decision Agent using ReAct pattern."""
    
    def __init__(self, llm, tools, corpus, doc_vecs, idf, config=None):
        """
        Initialize agent.
        
        Args:
            llm: HF_LLM instance
            tools: Dictionary of available tools
            corpus: Trip document corpus
            doc_vecs: Pre-computed TF-IDF vectors
            idf: IDF dictionary
            config: Agent configuration
        """
        self.llm = llm
        self.tools = tools
        self.corpus = corpus
        self.doc_vecs = doc_vecs
        self.idf = idf
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 6)
        self.verbose = self.config.get('verbose', True)
        self.trajectory = []
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """Run agent on user query following ReAct pattern."""
        from prompting import make_prompt, parse_action
        
        self.trajectory.clear()
        
        for step_num in range(self.max_steps):
            if self.verbose:
                print(f"\\n{'='*60}\\nSTEP {step_num + 1}\\n{'='*60}")
            
            # Format prompt
            prompt = make_prompt(user_query, self.trajectory)
            
            # Generate from LLM
            out = self.llm(prompt)
            if self.verbose:
                print(f"\\nLLM: {out}")
            
            # Parse output
            lines = out.strip().split('\\n')
            thought = lines[0].replace('Thought:', '').strip() if lines[0].startswith('Thought:') else "Processing"
            
            action_line = None
            for line in lines:
                if line.startswith('Action:'):
                    action_line = line
                    break
            
            if not action_line:
                self.trajectory.append({'thought': thought, 'action': 'none', 'observation': 'Invalid format'})
                break
            
            parsed = parse_action(action_line)
            if not parsed:
                self.trajectory.append({'thought': thought, 'action': action_line, 'observation': 'Parse failed'})
                break
            
            name, args = parsed
            
            # Execute action
            if name == "finish":
                self.trajectory.append({'thought': thought, 'action': action_line.replace('Action:', '').strip(), 'observation': 'done'})
                break
            elif name in self.tools:
                try:
                    obs_payload = self.tools[name](**args)
                    observation = json.dumps(obs_payload, ensure_ascii=False)
                except Exception as e:
                    observation = f"Tool error: {e}"
                self.trajectory.append({'thought': thought, 'action': action_line.replace('Action:', '').strip(), 'observation': observation})
            else:
                self.trajectory.append({'thought': thought, 'action': action_line.replace('Action:', '').strip(), 'observation': f"Unknown action: {name}"})
                break
        
        # Extract final answer
        final_answer = None
        for s in reversed(self.trajectory):
            if s['action'].startswith('finish'):
                m = re.search(r'answer="([^"]*)"', s['action'])
                if m:
                    final_answer = m.group(1)
                break
        
        return {
            "question": user_query,
            "final_answer": final_answer,
            "steps": self.trajectory
        }
'''

import math
import numpy as np
import torch
    

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if action not in self.children:
                self.children[action] = MCTSNode(state=self.state, parent=self, action=action)
        self.is_expanded = True

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        
        for action, child in self.children.items():
            ucb_score = (
                child.value +
                c_puct * np.sqrt(np.log(self.visit_count + 1) / (child.visit_count + 1))
            )
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

class MCTS:
    def __init__(self, model, c_puct=1.0, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, root):
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded:
                dynamic_c_puct = self.c_puct / (1 + len(search_path) * 0.1)
                node = node.select_child(dynamic_c_puct)
                search_path.append(node)
            
            # Expansion
            state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
            
            # Normalize the state tensor
            state_tensor = (state_tensor - state_tensor.mean()) / (state_tensor.std() + 1e-5)
            
            self.model.eval()
            policy, value = self.model(state_tensor)
            self.model.train()
            policy, value = policy.detach().numpy().flatten(), value.item()

            if not node.is_expanded:
                node.expand(policy)
            
            # Backpropagation with value clipping
            for node in reversed(search_path):
                value = np.clip(value, -1, 1)  # Clip the value to the range [-1, 1]
                node.update(value)
                value = -value

        return root

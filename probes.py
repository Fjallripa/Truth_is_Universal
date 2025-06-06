# A module for truth classification probes
# ==========================================


import numpy as np
from sklearn.linear_model import LogisticRegression
import torch as t
import torch.nn as nn





# Helper functions
# ----------------

def learn_truth_directions(acts_centered, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0
    labels_copy = labels.clone()
    labels_copy = t.where(labels_copy == 0.0, t.tensor(-1.0), labels_copy)
    
    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = t.column_stack([labels_copy, labels_copy * polarities])

    # Compute the analytical OLS solution
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    return t_g, t_p



def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc



def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)





# Probe classes
# -------------

class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None


    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe


    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))


    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d



class CCSProbe(t.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=True),
            t.nn.Sigmoid()
        )


    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)


    def pred(self, acts, iid=None):
        return self(acts).round()


    def from_data(acts, neg_acts, labels=None, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None: # flip direction if needed
            labels = labels.to(device)
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        return probe


    @property
    def direction(self):
        return self.net[0].weight.data[0]


    @property
    def bias(self):
        return self.net[0].bias.data[0]



class LRProbe():
    def __init__(self):
        self.LR = None


    def from_data(acts, labels):
        probe = LRProbe()
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts.numpy(), labels.numpy())
        return probe


    def pred(self, acts):
        return t.tensor(self.LR.predict(acts))
    


class MMProbe(t.nn.Module):
    def __init__(self, direction, LR):
        super().__init__()
        self.direction = direction
        self.LR = LR


    def forward(self, acts):
        proj = acts @ self.direction
        return t.tensor(self.LR.predict(proj[:, None]))


    def pred(self, x):
        return self(x).round()


    def from_data(acts, labels, device='cpu'):
        acts, labels
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None], labels)
        
        probe = MMProbe(direction, LR).to(device)

        return probe



class SimpleMLPProbe(nn.Module):
    """
    A very simple non-linear, MLP probe with 2 hidden layers.
    The resulting  classification boundary is roughly a 3-piece linear function.
    """


    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, self.output_dim),
            nn.Sigmoid(),
        )
        self.t_g, self.t_p = None, None   # truth directions

       
    def forward(self, acts):
        return self.model(acts)
    

    def _project_acts(self, acts):
        """
        Project the high-dimensional activations onto the 2D truth space.
        """
        
        proj_t_g = acts @ self.t_g
        proj_t_p = acts @ self.t_p
        return t.stack((proj_t_g, proj_t_p), axis=1)


    def pred(self, acts, is_2d=False):
        """
        Classify the activations.
        Returns float labels 0.0 or 1.0.
        """
        
        if not is_2d:
            acts = self._project_acts(acts)
        with t.no_grad():
            return self.forward(acts).flatten().round()


    def from_data(acts, labels, polarities, epochs=1000, learning_rate=0.01, verbose=False):
        """
        Train the probe on the labelled activation data.
        """
        
        # Setup
        probe = SimpleMLPProbe()
        probe.optimizer = t.optim.Adam(probe.model.parameters(), lr=learning_rate)
        probe.loss_function = nn.BCELoss()   # Binary Cross-Entropy Loss

        # Prepare the training data
        probe.t_g, probe.t_p = learn_truth_directions(acts, labels, polarities)
        acts_2d = probe._project_acts(acts)
        labels = labels.view(-1, 1)

        # Run the training  
        for epoch in range(epochs):
            probe.optimizer.zero_grad()
            output = probe(acts_2d)
            loss = probe.loss_function(output, labels)
            loss.backward()
            probe.optimizer.step()
            
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        if verbose:
            print(f"Final Loss = {loss.item()}")

        return probe   
 
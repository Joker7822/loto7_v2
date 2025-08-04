import os
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        mean = self.linear(x)
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return mean

def train_bayesian_regression(X_train, y_train, in_features, out_features=7, num_steps=1000):
    pyro.clear_param_store()
    model = BayesianRegression(in_features, out_features)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optimizer = ClippedAdam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    for step in range(num_steps):
        loss = svi.step(x_tensor, y_tensor)
        if step % 100 == 0:
            print(f"[BNN] Step {step} ELBO Loss: {loss:.4f}")

    return model, guide

def predict_bayesian_regression(model, guide, X, samples=20):
    predictive = Predictive(model, guide=guide, num_samples=samples, return_sites=("obs",))
    x_tensor = torch.tensor(X, dtype=torch.float32)
    preds = predictive(x_tensor)
    pred_samples = preds["obs"].detach().numpy()
    return pred_samples.mean(axis=0)

def save_bayesian_model(model, guide, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "bnn_model.pth"))
    torch.save(guide.state_dict(), os.path.join(save_dir, "bnn_guide.pth"))
    print(f"[INFO] BNN モデルとガイドを保存しました → {save_dir}")

def load_bayesian_model(load_dir, in_features=37, out_features=7):
    model = BayesianRegression(in_features, out_features)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)

    model.load_state_dict(torch.load(os.path.join(load_dir, "bnn_model.pth")))
    guide.load_state_dict(torch.load(os.path.join(load_dir, "bnn_guide.pth")))
    print(f"[INFO] BNN モデルとガイドをロードしました ← {load_dir}")

    return model, guide

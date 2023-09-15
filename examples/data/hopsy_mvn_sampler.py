import hopsy
import numpy as np
import argparse as ap

NUM_SAMPLES_PER_CALL = 1

class TemperedModel:
    def __init__(self, model: hopsy.Gaussian, beta: float):
        self.model = model
        self.beta = beta

    def compute_negative_log_likelihood(self, x: np.ndarray) -> float:
        return np.power(self.model.compute_negative_log_likelihood(x), self.beta)


def sample_problem(problem: hopsy.Problem, rng: hopsy.RandomNumberGenerator, sample_idx: int) -> float:
    chain = hopsy.MarkovChain(problem, hopsy.GaussianProposal)
    chain.proposal.stepsize = 0.2
    hopsy.sample(chain, rng, n_samples=NUM_SAMPLES_PER_CALL, thinning=10)
    return chain.state_negative_log_likelihood, chain.state


def read_beta(command: str) -> float:
    open_idx, close_idx = command.find("("), command.find(")")
    return float(command[open_idx+1:close_idx])


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--seed", help="Random seed of random number generator", type=int)

    args = parser.parse_args()
    
    seed = np.uint32(args.seed)
    rng = hopsy.RandomNumberGenerator(seed=seed)
    A, b = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0, 0]
    base_model = hopsy.Gaussian(mean=[0, 0, 0])
    state = [0, 0, 0]

    action_string = ""
    sample_idx = 0
    while True:
        command = input(action_string)
        
        if command.startswith("log_potential"):
            beta = read_beta(command)
            log_potential = -TemperedModel(base_model, beta).compute_negative_log_likelihood(state)
            action_string = f"response({log_potential})\n"
        elif command.startswith("call_sampler!"):
            beta = read_beta(command)
            temp_model = TemperedModel(base_model, beta)
            problem = hopsy.Problem(A, b, temp_model)
            problem.starting_point = state
            neg_log_potential, state = sample_problem(problem, rng, sample_idx)
            sample_idx += 1
            action_string = "response()\n"
        else:
            print(f"Unkown command {command}")
            exit(1)
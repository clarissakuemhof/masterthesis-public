"""PGMORL algorithm implementation.

Some code in this file has been adapted from the original code provided by the authors of the paper https://github.com/mit-gfx/PGMORL.
(!) Limited to 2 objectives for now.
(!) The post-processing phase has not been implemented yet.
"""

import time
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from typing_extensions import override
from scipy.spatial.distance import cdist

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import wandb
import random
#random.seed(1)

from scipy.optimize import least_squares

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity
from morl_baselines.single_policy.ser.mo_ppo_copy import MOPPO, MOPPONet, make_env

def generate_weights(delta_weight: float) -> np.ndarray:
    """Generates weights uniformly distributed over the objective dimensions. These weight vectors are separated by delta_weight distance.

    Args:
        delta_weight: distance between weight vectors
    Returns:
        all the candidate weights
    """
    return np.linspace((0.0, 1.0), (1.0, 0.0), int(1 / delta_weight) + 1, dtype=np.float32)


class PerformanceBuffer:
    """Stores the population. Divides the objective space in to n bins of size max_size.

    (!) restricted to 2D objective space (!)
    """

    def __init__(self, num_bins: int, max_size: int, origin: np.ndarray):
        """Initializes the buffer.

        Args:
            num_bins: number of bins
            max_size: maximum size of each bin
            origin: origin of the objective space (to have only positive values)
        """
        self.num_bins = num_bins
        self.max_size = max_size
        self.origin = -origin
        self.dtheta = np.pi / 2.0 / self.num_bins
        self.bins = [[] for _ in range(self.num_bins)]
        self.bins_evals = [[] for _ in range(self.num_bins)]

    @property
    def evaluations(self) -> List[np.ndarray]:
        """Returns the evaluations of the individuals in the buffer."""
        # flatten
        return [e for l in self.bins_evals for e in l]

    @property
    def individuals(self) -> list:
        """Returns the individuals in the buffer."""
        return [i for l in self.bins for i in l]

    def add(self, candidate, evaluation: np.ndarray):
        """Adds a candidate to the buffer.

        Args:
            candidate: candidate to add
            evaluation: evaluation of the candidate
        """

        def center_eval(eval):
            # Objectives must be positive
            return np.clip(eval + self.origin, 0.0, float("inf"))

        centered_eval = center_eval(evaluation)
        norm_eval = np.linalg.norm(centered_eval)
        theta = np.arccos(np.clip(centered_eval[1] / (norm_eval + 1e-3), -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)

        if buffer_id < 0 or buffer_id >= self.num_bins:
            return

        if len(self.bins[buffer_id]) < self.max_size:
            self.bins[buffer_id].append(deepcopy(candidate))
            self.bins_evals[buffer_id].append(evaluation)
        else:
            for i in range(len(self.bins[buffer_id])):
                stored_eval_centered = center_eval(self.bins_evals[buffer_id][i])
                if np.linalg.norm(stored_eval_centered) < np.linalg.norm(centered_eval):
                    self.bins[buffer_id][i] = deepcopy(candidate)
                    self.bins_evals[buffer_id][i] = evaluation
                    break

class NoveltySearch:
    def __init__(self, archive, k=5):
        """
        A simple novelty search mechanism based on distance to k-nearest neighbors in the objective space.

        Args:
            archive: A list to store previous solutions' performance metrics.
            k: Number of nearest neighbors to use for calculating novelty.
        """
        self.archive = archive  # Store past solutions' evaluations
        self.k = k  # Number of neighbors to consider for novelty

    def compute_novelty(self, candidate_performance):
        """
        Compute the novelty of a candidate solution based on its distance to past solutions.

        Args:
            candidate_performance: The performance of the current candidate.

        Returns:
            The novelty score, based on k-nearest neighbors.
        """
        #print(self.archive)
        if len(self.archive) == 0:
            return 0.0  # No novelty for the first solution
        
        archive_evaluations = np.array(self.archive.evaluations)
        
        # distance to all points in archive
        distances = cdist(candidate_performance[np.newaxis, :], archive_evaluations, metric='euclidean')[0]
        
        # select the k-nearest neighbors and compute the average distance (novelty score)
        nearest_neighbors = sorted(distances)[:self.k]
        novelty_score = sum(nearest_neighbors) / self.k
        return novelty_score

    def add_to_archive(self, performance):
        """Adds a performance evaluation to the Pareto archive."""
        self.archive.add(candidate=None, evaluation=performance)

class EMOS_EA_selection(MOAgent):
    """Prediction Guided Multi-Objective Reinforcement Learning.

    Reference: J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik,
    “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control,”
    in Proceedings of the 37th International Conference on Machine Learning,
    Nov. 2020, pp. 10607–10616. Available: https://proceedings.mlr.press/v119/xu20h.html

    Paper: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
    Supplementary materials: https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf
    """

    def __init__(
        self,
        env_id: str,
        origin: np.ndarray,
        num_envs: int = 4,
        pop_size: int = 6,
        warmup_iterations: int = 80, # default 80
        steps_per_iteration: int = 2048,
        evolutionary_iterations: int = 20,
        num_weight_candidates: int = 7,
        num_performance_buffer: int = 100,
        performance_buffer_size: int = 2,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        delta_weight: float = 0.2,
        env=None,
        gamma: float = 0.995,
        project_name: str = "MORL-baselines",
        experiment_name: str = "EMOS_EA_selection",
        wandb_entity: Optional[str] = None,
        seed: Optional[int] = None,
        log: bool = True,
        net_arch: List = [64, 64],
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.95,
        device: Union[th.device, str] = "auto",
        group: Optional[str] = None,
    ):
        """Initializes the PGMORL agent.

        Args:
            env_id: environment id
            origin: reference point to make the objectives positive in the performance buffer
            num_envs: number of environments to use (VectorizedEnvs)
            pop_size: population size
            warmup_iterations: number of warmup iterations
            steps_per_iteration: number of steps per iteration
            evolutionary_iterations: number of evolutionary iterations
            num_weight_candidates: number of weight candidates
            num_performance_buffer: number of performance buffers
            performance_buffer_size: size of the performance buffers
            min_weight: minimum weight
            max_weight: maximum weight
            delta_weight: delta weight for weight generation
            env: environment
            gamma: discount factor
            project_name: name of the project. Usually MORL-baselines.
            experiment_name: name of the experiment. Usually PGMORL.
            wandb_entity: wandb entity, defaults to None.
            seed: seed for the random number generator
            log: whether to log the results
            net_arch: number of units per layer
            num_minibatches: number of minibatches
            update_epochs: number of update epochs
            learning_rate: learning rate
            anneal_lr: whether to anneal the learning rate
            clip_coef: coefficient for the policy gradient clipping
            ent_coef: coefficient for the entropy term
            vf_coef: coefficient for the value function loss
            clip_vloss: whether to clip the value function loss
            max_grad_norm: maximum gradient norm
            norm_adv: whether to normalize the advantages
            target_kl: target KL divergence
            gae: whether to use generalized advantage estimation
            gae_lambda: lambda parameter for GAE
            device: device on which the code should run
            group: The wandb group to use for logging.
        """
        super().__init__(env, device=device, seed=seed)
        # Env dimensions
        self.tmp_env = mo_gym.make(env_id)
        self.extract_env_info(self.tmp_env)
        self.env_id = env_id
        self.num_envs = num_envs
        assert isinstance(self.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.tmp_env.close()
        self.gamma = gamma

        # EA parameters
        self.pop_size = pop_size
        self.warmup_iterations = warmup_iterations
        self.steps_per_iteration = steps_per_iteration
        self.evolutionary_iterations = evolutionary_iterations
        self.num_weight_candidates = num_weight_candidates
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.delta_weight = delta_weight
        self.num_performance_buffer = num_performance_buffer
        self.performance_buffer_size = performance_buffer_size
        self.archive = ParetoArchive()
        self.population = PerformanceBuffer(
            num_bins=self.num_performance_buffer,
            max_size=self.performance_buffer_size,
            origin=origin,
        )

        # PPO Parameters
        self.net_arch = net_arch
        self.batch_size = int(self.num_envs * self.steps_per_iteration)
        self.num_minibatches = num_minibatches
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda
        self.gae = gae

        # env setup
        if env is None:
            if self.seed is not None:
                envs = [make_env(env_id, self.seed + i, i, experiment_name, self.gamma) for i in range(self.num_envs)]
            else:
                envs = [make_env(env_id, i, i, experiment_name, self.gamma) for i in range(self.num_envs)]
            self.env = mo_gym.wrappers.vector.MOSyncVectorEnv(envs)
        else:
            raise ValueError("Environments should be vectorized for PPO. You should provide an environment id instead.")

        # Logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, group)

        self.networks = [
            MOPPONet(
                self.observation_shape,
                self.action_space.shape,
                self.reward_dim,
                self.net_arch,
            ).to(self.device)
            for _ in range(self.pop_size)
        ]

        weights = generate_weights(self.delta_weight)
        print(f"Warmup phase - sampled weights: {weights}")

        self.agents = [
            MOPPO(
                i,
                self.networks[i],
                weights[i],
                self.env,
                log=self.log,
                gamma=self.gamma,
                device=self.device,
                seed=self.seed,
                steps_per_iteration=self.steps_per_iteration,
                num_minibatches=self.num_minibatches,
                update_epochs=self.update_epochs,
                learning_rate=self.learning_rate,
                anneal_lr=self.anneal_lr,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                clip_vloss=self.clip_vloss,
                max_grad_norm=self.max_grad_norm,
                norm_adv=self.norm_adv,
                target_kl=self.target_kl,
                gae=self.gae,
                gae_lambda=self.gae_lambda,
                rng=self.np_random,
            )
            for i in range(self.pop_size)
        ]

        self.novelty_search = NoveltySearch(self.archive)

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
            "pop_size": self.pop_size,
            "warmup_iterations": self.warmup_iterations,
            "evolutionary_iterations": self.evolutionary_iterations,
            "num_weight_candidates": self.num_weight_candidates,
            "num_performance_buffer": self.num_performance_buffer,
            "performance_buffer_size": self.performance_buffer_size,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "delta_weight": self.delta_weight,
            "gamma": self.gamma,
            "seed": self.seed,
            "net_arch": self.net_arch,
            "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "learning_rate": self.learning_rate,
            "anneal_lr": self.anneal_lr,
            "clip_coef": self.clip_coef,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "norm_adv": self.norm_adv,
            "target_kl": self.target_kl,
            "clip_vloss": self.clip_vloss,
            "gae": self.gae,
            "gae_lambda": self.gae_lambda,
        }

    def __train_all_agents(self, iteration: int, max_iterations: int):
        for i, agent in enumerate(self.agents):
            agent.global_step = self.global_step
            agent.train(self.start_time, iteration, max_iterations)
            self.global_step += self.steps_per_iteration * self.num_envs

    def __eval_all_agents(
        self,
        eval_env: gym.Env,
        evaluations_before_train: List[np.ndarray],
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
    ):
        """Evaluates all agents and store their current performances on the buffer and pareto archive."""
        for i, agent in enumerate(self.agents):
            _, _, _, discounted_reward = agent.policy_eval(eval_env, weights=agent.np_weights, log=self.log)
            # Storing current results
            self.population.add(agent, discounted_reward)
            self.archive.add(agent, discounted_reward)
            evaluations_before_train[i] = discounted_reward

        if self.log:
            print("Current pareto archive:")
            print(self.archive.evaluations)
            log_all_multi_policy_metrics(
                current_front=self.archive.evaluations,
                hv_ref_point=ref_point,
                reward_dim=self.reward_dim,
                global_step=self.global_step,
                n_sample_weights=self.num_eval_weights_for_eval,
                ref_front=known_pareto_front,
            )
    
    def __policy_selection(self, ref_point: np.ndarray, update_best: bool = True):
        """
        Chooses agents and weights to train at the next iteration based on regret and uncertainty.
        Args:
        - ref_point: Reference point for calculating hypervolume.
        - update_best: If True, update the best agents (lowest regret/uncertainty). If False, update the worst agents (highest regret/uncertainty).
        """
        candidate_weights = generate_weights(self.delta_weight / 2.0)  # generate more weights than agents
        self.np_random.shuffle(candidate_weights)  

        current_front = deepcopy(self.archive.evaluations)
        population = self.population.individuals
        population_eval = self.population.evaluations
        selected_tasks = []

        for i in range(len(self.agents)):
            max_score = float("-inf") if update_best else float("inf")
            best_candidate = None
            best_eval = None

            for candidate, last_candidate_eval in zip(population, population_eval):
                candidate_tuples = [
                    (last_candidate_eval, weight)
                    for weight in candidate_weights
                    if (tuple(last_candidate_eval), tuple(weight)) not in selected_tasks
                ]

                # regret and uncertainty for each candidate
                regret_scores = [
                    np.linalg.norm(last_candidate_eval - ref_point)
                    for candidate_eval, weight in candidate_tuples
                ]

                uncertainty_scores = [
                    np.var(last_candidate_eval)
                    for candidate_eval, weight in candidate_tuples
                ]

                # normalize before combining
                #regret_scores = (regret_scores - np.min(regret_scores)) / (np.max(regret_scores) - np.min(regret_scores) + 1e-8)
                #uncertainty_scores = (uncertainty_scores - np.min(uncertainty_scores)) / (np.max(uncertainty_scores) - np.min(uncertainty_scores) + 1e-8)

                if self.log:
                    wandb.log(
                        {
                            "new/average_regret": np.mean(regret_scores),
                            "new/average_uncertainty": np.mean(uncertainty_scores),
                            #"new/average_mixture_metrics": np.mean(mixture_scores) 
                        }
                    )

                # combine metrics for evaluation
                mixture_scores = [
                    -regret_score * 0.5 - uncertainty_score * 0.5  
                    for regret_score, uncertainty_score in zip(regret_scores, uncertainty_scores)
                ]

                # select best or worst combination of metrics
                current_candidate_weight = np.argmax(np.array(mixture_scores)) if update_best else np.argmin(np.array(mixture_scores))
                current_candidate_score = np.max(np.array(mixture_scores)) if update_best else np.min(np.array(mixture_scores))

                # determine best/worst candidate
                if (update_best and max_score < current_candidate_score) or \
                (not update_best and max_score > current_candidate_score):
                    max_score = current_candidate_score
                    best_candidate = (
                        candidate,
                        candidate_tuples[current_candidate_weight][1],
                    )
                    best_eval = last_candidate_eval

            selected_tasks.append((tuple(best_eval), tuple(best_candidate[1])))
            # self.novelty_search.add_to_archive(best_eval)  # Add to archive for future novelty computation
            current_front.append(best_eval)

            # assign best/worst (weight-agent) pair to worker
            copied_agent = deepcopy(best_candidate[0])
            copied_agent.global_step = self.agents[i].global_step
            copied_agent.id = i
            copied_agent.change_weights(deepcopy(best_candidate[1]))

            # best_eval+novelty to be used as fitness
            novelty_score = self.novelty_search.compute_novelty(best_eval)
            copied_agent.fitness = 0.2 * np.sum(best_eval) + 0.8 * novelty_score

            self.agents[i] = copied_agent

            print(f"Candidate Tuples: {candidate_tuples}")
            print(f"Selected Tasks: {selected_tasks}")
            print(f"Regret Scores: {regret_scores}")
            print(f"Uncertainty Scores: {uncertainty_scores}")

            if update_best:
                print(f"Updating Best: Agent #{self.agents[i].id} - weights {best_candidate[1]}")
                print(
                    f"current eval: {best_eval} - regret: {np.min(regret_scores)} - uncertainty: {np.min(uncertainty_scores)}"
                )
            else:
                print(f"Updating Worst: Agent #{self.agents[i].id} - weights {best_candidate[1]}")
                print(
                    f"current eval: {best_eval} - regret: {np.max(regret_scores)} - uncertainty: {np.max(uncertainty_scores)}"
                )

    def mutate(self, policy, mutation_rate: float = 0.1):
        """Apply mutation to a policy by adding random noise."""
        original_policy = [param.clone() for param in policy]
        mutated_policy = [
            param + mutation_rate * th.randn_like(param) for param in policy
        ]

        # parameter changes
        mutation_deltas = [
            th.norm(mutated - original).item() for mutated, original in zip(mutated_policy, original_policy)
        ]

        # mutation impact
        if self.log:
            wandb.log({
                "new/mutation_parameter_changes": wandb.Histogram(mutation_deltas),
                "new/mutation_mean_change": sum(mutation_deltas) / len(mutation_deltas)
            })
        return mutated_policy
    
    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover on the parameters of two parents to produce a single child.
        
        Args:
        - parent1: List of tensors representing the parameters of parent 1.
        - parent2: List of tensors representing the parameters of parent 2.

        Returns:
        - child: A new child created from the crossover of parent 1 and parent 2.
        """
        # number of parameters in both parents is the same
        assert len(parent1) == len(parent2), "Parents must have the same number of parameters."
        
        # crossover point (index) between parameters
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child = []
        
        for i in range(len(parent1)):
            if i < crossover_point:
                child.append(parent1[i].clone()) 
            else:
                child.append(parent2[i].clone()) 
        
        # parameter differences between parents
        parent_differences = [
            th.norm(param1 - param2).item() for param1, param2 in zip(parent1, parent2)
        ]

        # diversity introduced by crossover
        if self.log:
            wandb.log({
                "new/crossover_parent_diversity": wandb.Histogram(parent_differences),
                "new/crossover_mean_diversity": sum(parent_differences) / len(parent_differences)
            })

        # return single child
        return child
    
    def select_parents(self, num_parents: int = 4) -> List[MOPPO]:
        """Select top-performing policies for reproduction."""
        if any(agent.fitness is None for agent in self.agents):
            raise ValueError("Fitness values are not set for all agents. Check the task weight selection process.")

        # select agents with highest fitness (i.e., a sum closer to 0)
        top_parents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)[:num_parents]
        print(f"Selected parents based on fitness: {[parent.fitness for parent in top_parents]}")
        
        return top_parents

    def replace_weak_policies(self, offspring: List[th.Tensor]):
        """Replace the weakest policies in the population with new offspring."""
        weakest_agents = sorted(self.agents, key=lambda agent: agent.fitness)[:len(offspring)]
        print(weakest_agents)
        for weak_agent, new_policy in zip(weakest_agents, offspring):
            th.save(weak_agent.networks.state_dict(), "new_policy2.pth")
            #print(weak_agent)
            #print(new_policy)
            new_policy = th.load("new_policy2.pth")
            weak_agent.networks.load_state_dict(new_policy)

    def evolve_population(self):
        """Evolve the population using recombination and mutation."""
        for agent in self.agents:
            if not hasattr(agent, 'fitness') or agent.fitness is None:
                raise ValueError(f"Agent {agent.id} does not have a valid fitness value. Ensure task weight selection is complete.")
    
        # parents: top N policies by fitness
        parents = self.select_parents()
        
        # create offspring using recombination
        offspring = []
        for _ in range(self.pop_size // 2):  # offspring pairs
            # top-4 parents, and from them a random pair of two parents is chosen 
            # why 4 parents? allowing for greater genetic diversity in the offspring
            # leading to better exploration of the solution space
            parent1, parent2 = random.sample(parents, 2)

            parent1_params = parent1.get_network_parameters()
            parent2_params = parent2.get_network_parameters()

            child = self.crossover(parent1_params, parent2_params)
            offspring.append(child)
        
        # mutation
        mutated_offspring = [self.mutate(child) for child in offspring]
        
        # replace weaker policies with offspring
        self.replace_weak_policies(mutated_offspring)

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
    ):
        """Trains the agents."""
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                }
            )
        self.num_eval_weights_for_eval = num_eval_weights_for_eval
        # 1 iteration is a full batch for each agents
        # -> (steps_per_iteration * num_envs * pop_size)  timesteps per iteration
        max_iterations = total_timesteps // self.steps_per_iteration // self.num_envs // self.pop_size
        iteration = 0
        # Init
        current_evaluations = [np.zeros(self.reward_dim) for _ in range(len(self.agents))]
        self.__eval_all_agents(
            eval_env=eval_env,
            evaluations_before_train=current_evaluations,
            ref_point=ref_point,
            known_pareto_front=known_pareto_front,
        )
        self.start_time = time.time()

        # Warmup
        for i in range(1, self.warmup_iterations + 1):
            print(f"Warmup iteration #{iteration}, global step: {self.global_step}")
            if self.log:
                wandb.log({"charts/warmup_iterations": i, "global_step": self.global_step})
            self.__train_all_agents(iteration=iteration, max_iterations=max_iterations)
            iteration += 1
        self.__eval_all_agents(
            eval_env=eval_env,
            evaluations_before_train=current_evaluations,
            ref_point=ref_point,
            known_pareto_front=known_pareto_front,
        )

        # Evolution
        # without the 15*, the loop would only run once
        max_iterations = max(max_iterations, self.warmup_iterations + (27*self.evolutionary_iterations))
        evolutionary_generation = 1
        while iteration < max_iterations:


            # Every evolutionary iterations, change the task - weight assignments
            # self.__task_weight_selection(ref_point=ref_point)
            self.__policy_selection(ref_point=ref_point, update_best = False)


            print(f"Evolutionary generation #{evolutionary_generation}")
            if self.log:
                wandb.log(
                    {"charts/evolutionary_generation": evolutionary_generation, "global_step": self.global_step},
                )

            for _ in range(self.evolutionary_iterations):
                # Run training of every agent for evolutionary iterations.
                if self.log:
                    print(f"Evolutionary iteration #{iteration - self.warmup_iterations}")
                    wandb.log(
                        {
                            "charts/evolutionary_iterations": iteration - self.warmup_iterations,
                            "global_step": self.global_step,
                        },
                    )
                self.__train_all_agents(iteration=iteration, max_iterations=max_iterations)
                iteration += 1

                self.evolve_population()

                # eval agents eins nach rechts shiften, 
                # sodass jede evol iteration evaluiert wird?
                if iteration % 20 == 0:
                    self.__eval_all_agents(
                            eval_env=eval_env,
                            evaluations_before_train=current_evaluations,
                            ref_point=ref_point,
                            known_pareto_front=known_pareto_front,
                    )
            evolutionary_generation += 1

        print("Done training!")
        self.env.close()
        if self.log:
            self.close_wandb()
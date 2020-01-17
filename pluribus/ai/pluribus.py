"""
Comments on notation found in the appendix.

Notation
--------
  P :
    A set of players, where P_i is one player.
  h :
    A node (i.e history), h is defined by all information of the current
    situation, including private knowledge known only to one player.
  A(h) :
    Denotes the actions available at a node
  P(h) :
    Denotes either the chance or player it is to act at that node.
  I :
    Imperfect information is represented by information sets (infosets) for
    each player P_i. For any infoset belonging to a player P_i, all nodes h to
    h' in the infoset I are indestinguishable to the player P_i. Moreover,
    every non-terminal node h in belongs to exactly one infoset for each
    player.
  sigma (lowercase) :
    A strategy (i.e a policy). Here sigma(I) is a probability vector over the
    actions for acting player P_i in infoset I.
  φ :
    The final strategy for infoset I is φ(I) normalized.
"""
import numpy as np


def monte_carlo_cfr_with_pruning(t):
    """Conduct External-Sampling Monte Carlo CFR with pruning."""
    for player in players:
        # Something with an infoset - are there multiple infosets for each
        # player?
        # TODO(fedden): What is for I_i ∈ I_i where P(I_i) = i do
        # TODO(fedden): What is A(I)?
        for action in available_actions_info_set:
            # What is: R(I, a) <- 0?
            reward_for_infoset_and_action = 0
            if betting_round(infoset):
                normalised

def calculate_strategy(R, I):
    """Caluclates the strategy based on regrets."""
    pass


def update_strategy(h, P):
    """Update the average strategy of P_i"""
    pass


def traverse_monte_carlo_cfr(node, player):
    """"Update the regrets for player."""
    if node.is_terminal:
        # Possibly utility.
        return u(node)
    elif player not in hand:
        # What does this mean?
        # The remaining actions are irrelevant to player.
        return traverse_monte_carlo_cfr(node_0, player)
    elif node is chance_node:
        # Wtf is a chance node? A human player choice or another players move?
        action = sample_action(node)
        return traverse_monte_carlo_cfr(action, player)
    elif player.is_turn:
        infoset = node.infoset(player)
        # Probability vector over actions for player and infoset.
        # Wtf is R? I think it's regret! Determine strategy for this infoset.
        policy = calculate_strategy(infoset_regret, infoset)
        # Initialise expected value at zero.
        value = 0
        for action in node.available_actions:
            # Traverse each action.
            value_of_action = traverse_monte_carlo_cfr(action, player)
            # Update the expected value.
            # TODO(fedden): What is this policy multiplier: σ(I_i, a)?
            value = value + policy_factor * value_of_action
        for action in node.available_actions:
            # Update the regret of each action.
            regret += regret + value_of_action - value
        return value
    else:
        # What is going on here?


def traverse_monte_carlo_cfr_with_pruning(h, P):
    """MCCFR with pruning for very negative regrets."""
    pass

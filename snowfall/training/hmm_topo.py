import k2
from typing import List


def build_hmm_topo_2state(tokens: List[int]) -> k2.Fsa:
    """
    Build a 2-state HMM topology used in Kaldi's chain models.
    The first HMM state is entered only once for each token instance,
    and the second HMM state is self-looped and optional.

    Args:
        tokens:
            A list of token int IDs, e.g., phones, characters, etc.
            The IDs for the first HMM state will be the same as token IDs;
            The IDs for the second HMM state are: ``token_id + len(tokens)``
    Returns:
        An FST that converts a sequence of HMM state IDs to a sequence of token IDs.
    """
    followup_tokens = range(len(tokens), len(tokens) * 2)
    num_states = len(tokens) + 2  # + start state, + final state
    arcs = []

    # Start state -> token state
    for i in range(0, len(tokens)):
        arcs += [f'0 {i + 1} {tokens[i]} {tokens[i]} 0.0']

    # Token state self loops
    for i in range(0, len(tokens)):
        arcs += [f'{i + 1} {i + 1} {followup_tokens[i]} 0 0.0']

    # Cross-token transitions
    for i in range(0, len(tokens)):
        for j in range(0, len(tokens)):
            if i != j:
                arcs += [f'{i + 1} {j + 1} {tokens[i]} {tokens[i]} 0.0']

    # Token state -> superfinal state
    for i in range(0, len(tokens)):
        arcs += [f'{i + 1} {num_states - 1} -1 -1 0.0']

    # Final state
    arcs += [f'{num_states - 1}']

    # Build the FST
    arcs = '\n'.join(sorted(arcs))
    ans = k2.Fsa.from_str(arcs)
    ans = k2.arc_sort(ans)
    return ans

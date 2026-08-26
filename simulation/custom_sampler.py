"""
Custom sampler rules for toric code simulation.
"""

import jax
import jax.numpy as jnp
import netket as nk
from typing import Any, Optional, Tuple, List
import numpy as np

@nk.utils.struct.dataclass
class WeightedRule(nk.sampler.rules.MetropolisRule):
    """A Metropolis sampling rule that can be used to combine different rules acting
    on different subspaces of the same tensor-hilbert space. Thanks to Marc Machaczek and Filippo Vicentini for this input.
    """
    probabilities: jax.Array
    rules: Tuple[nk.sampler.rules.MetropolisRule, ...]

    def __post_init__(self):
        if not isinstance(self.probabilities, jax.Array):
            object.__setattr__(self, "probabilities", jnp.array(self.probabilities))

        if not isinstance(self.rules, (tuple, list)):
            raise TypeError(
                "The second argument (rules) must be a tuple of `MetropolisRule` "
                f"rules, but you have passed {type(self.rules)}."
            )

        if len(self.probabilities) != len(self.rules):
            raise ValueError(
                "Length mismatch between the probabilities and the rules: probabilities "
                f"has length {len(self.probabilities)} , rules has length {len(self.rules)}."
            )

    def init_state(
            self,
            sampler: "nk.sampler.MetropolisSampler",
            machine: Any,
            params: Any,
            key: Any,
    ) -> Optional[Any]:
        N = len(self.probabilities)
        keys = jax.random.split(key, N)
        return tuple(
            self.rules[i].init_state(sampler, machine, params, keys[i])
            for i in range(N)
        )

    def reset(
            self,
            sampler: "nk.sampler.MetropolisSampler",
            machine: Any,
            params: Any,
            sampler_state: "nk.sampler.SamplerState",
    ) -> Optional[Any]:
        rule_states = []
        for i in range(len(self.probabilities)):
            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])
            rule_states.append(self.rules[i].reset(sampler, machine, params, _state))
        return tuple(rule_states)

    def transition(self, sampler, machine, parameters, state, key, sigma):
        N = len(self.probabilities)
        keys = jax.random.split(key, N + 1)

        sigmaps = []
        log_prob_corrs = []
        for i in range(N):
            # construct temporary rule state with correct sampler-state objects
            _state = state.replace(rule_state=state.rule_state[i])

            sigmaps_i, log_prob_corr_i = self.rules[i].transition(sampler, machine, parameters, _state, keys[i], sigma)

            sigmaps.append(sigmaps_i)
            log_prob_corrs.append(log_prob_corr_i)

        indices = jax.random.choice(keys[-1], N, shape=(sampler.n_chains_per_rank,), p=self.probabilities)

        batch_select = jax.vmap(lambda s, idx: s[idx], in_axes=(1, 0), out_axes=0)
        sigmap = batch_select(jnp.stack(sigmaps), indices)  # sigmaps has dim (N, n_chains_per_rank, n_sites)

        # if not all log_prob_corr are 0, convert the Nones to 0s
        if any(x is not None for x in log_prob_corrs):
            log_prob_corrs = jnp.stack([x if x is not None else 0 for x in log_prob_corrs])
            log_prob_corr = batch_select(log_prob_corrs, indices)
        else:
            log_prob_corr = None

        return sigmap, log_prob_corr

    def __repr__(self):
        return f"WeightedRule(probabilities={self.probabilities}, rules={self.rules})"


@nk.utils.struct.dataclass
class MultiRule(nk.sampler.rules.MetropolisRule):
    """
    Updates/flips multiple spins according to update_clusters. One of the clusters provided is chosen at random,
    then all spins within that cluster are updated. Thanks to Marc Machaczek for this input.
    """
    update_clusters: jax.Array  # hashable array required? no bc not used as staticarg, but dynamicarg instead

    def transition(self, sampler, machine, parameters, state, key, sigmas):
        # Deduce the number of possible clusters to be updated
        n_clusters = self.update_clusters.shape[0]

        # Deduce the number of MCMC chains from input shape
        n_chains = sigmas.shape[0]

        # Split the rng key into 2: one for each random operation
        key_indx, key_flip = jax.random.split(key, 2)

        # Pick random cluster index on every chain. NOTE: maxval is EXCLUSIVE in
        # jax.random.randint, so the bound is n_clusters (a long-standing
        # maxval=n_clusters-1 off-by-one silently never proposed the LAST cluster;
        # fixed 2026-08-26 -- ergodicity-critical for the honeycomb loop sector,
        # where hexagon flips are the only accepted moves at the fixed point, and
        # a strict bug fix for the square paths, whose last vertex/plaquette
        # cluster now becomes proposable as intended).
        indxs = jax.random.randint(key_indx, shape=(n_chains, 1), minval=0, maxval=n_clusters)

        @jax.vmap
        def flip(sigma, cluster):
            return sigma.at[cluster].set(-sigma.at[cluster].get())

        sigmap = flip(sigmas, self.update_clusters[indxs])        # flip those clusters

        return sigmap, None  # second argument for potential correcting factor of L (not present for this rule)


class SectorInitWeightedRule(WeightedRule):
    """WeightedRule whose chains START in the vertex-constrained (loop) sector.

    Honeycomb-only: the Levin-Gu ground states are supported on closed-loop
    configurations, and at the fixed point single flips have acceptance exactly
    0 (they leave the sector), so netket's default uniform-random chain init
    would start every chain off-support and relax slowly through near-zero
    amplitudes. The all-up configuration (zero loops) is the canonical in-sector
    start. The square path keeps the plain WeightedRule byte-identical.
    """

    def random_state(self, sampler, machine, parameters, state, key):
        return jnp.ones((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)


def create_custom_sampler(geometry, hi, config):
    """
    Create a custom sampler with both single-site and vertex updates.
    
    Args:
        geometry: Toric code geometry object
        hi: Hilbert space object
        config: Configuration dictionary
        
    Returns:
        MetropolisSampler with custom update rules
    """
    # Extract vertex operators
    vertex_all = geometry.vertex_all
    N = geometry.N

    if config.get('lattice', 'square') == 'honeycomb':
        # Honeycomb (Levin-Gu TC / doubled semion): the symmetry orbit moves are
        # HEXAGON X-flips (plaq_all rows, always 6 valid links at smooth OBC --
        # plaquettes are never truncated). Single flips leave the vertex-
        # constrained loop sector (zero acceptance at the fixed point), so the
        # hexagon rule carries ergodicity: on the OBC patch the hexagon
        # boundaries span the entire cycle space (dim = F).
        clusters = np.array(geometry.plaq_all)
        assert (clusters != -1).all(), "honeycomb plaq_all must have no -1 sentinels"
        print("Custom sampler clusters: hexagon flips (honeycomb)")
    elif config.get('dual_basis', False):
        # Dual basis: the network's (approximate) symmetry orbit moves are the
        # PLAQUETTE flips (physical B_p is an X-product in the conjugated basis);
        # all plaquettes have 4 valid edges, so no bulk filter is needed.
        clusters = np.array(geometry.plaq_all)
        assert (clusters != -1).all(), "plaq_all should have no -1 sentinels at OBC"
        print("Custom sampler clusters: plaquette flips (dual basis)")
    else:
        # Construct rule flipping vertices IN THE BULK (exclude boundary vertices)
        clusters = np.array(vertex_all)[np.all(np.array(vertex_all) != -1, axis=1)]

    # Ratio of probabilities for single flip vs cluster flip
    samp_ratio = N / len(clusters)

    # Single flip rule
    single_rule = nk.sampler.rules.LocalRule()

    # Cluster (vertex- or plaquette-) flip rule
    vertex_rule = MultiRule(clusters)
    
    # Combine cluster flip with single flip update. Honeycomb chains additionally
    # START in the loop sector (all-up) via the random_state override.
    rule_cls = (SectorInitWeightedRule
                if config.get('lattice', 'square') == 'honeycomb' else WeightedRule)
    weighted_rule = rule_cls(
        (samp_ratio/(samp_ratio+1), 1-samp_ratio/(samp_ratio+1)),
        [single_rule, vertex_rule]
    )
    
    # Create sampler with custom rule
    return nk.sampler.MetropolisSampler(
        hi,
        rule=weighted_rule,
        n_chains=config['n_chains'],
        n_sweeps=config['n_sweeps'],
        dtype=jnp.int8
    ) 
"""Compact classical readout descriptions; positions follow measurement wire order."""
from dataclasses import dataclass
import math
import pennylane as qml
import torch


@dataclass(frozen=True)
class DecoderSpec:
    """Exact deterministic partition. Majority ties give zero; codes reduce modulo A.

    ``blocks`` contains measured-wire *positions*, not physical wire labels.
    Overlapping blocks are supported. Arbitrary Python callables are deliberately
    not accepted: supply a supported structure or aggregate a costed dense output.
    """
    policy: str = "majority"
    n_actions: int = 2
    blocks: tuple | None = None

    def __post_init__(self):
        aliases = {"parity": "parity-like", "recursive-parity": "parity-like",
                   "hidden-like": "parity-like", "hidden": "parity-like",
                   "contiguous": "contiguous-like", "majority_modulo": "majority-modulo"}
        object.__setattr__(self, "policy", aliases.get(self.policy, self.policy))
        if self.policy not in {"majority", "majority-modulo", "parity-like", "contiguous-like"}:
            raise ValueError("Decoder must be majority, majority-modulo, parity-like or contiguous-like.")
        if not isinstance(self.n_actions, int) or self.n_actions < 2:
            raise ValueError("n_actions must be an integer >= 2.")
        if self.policy == "majority" and self.n_actions != 2:
            raise ValueError("Global majority has two actions; use majority-modulo for more.")
        if self.blocks is not None:
            object.__setattr__(self, "blocks", tuple(tuple(b) for b in self.blocks))

    def validate(self, n):
        if n < 1 or self.n_actions > 2**n:
            raise ValueError("Require 1 <= measured wires and n_actions <= 2**n.")
        if self.policy == "majority-modulo":
            if self.blocks is None or len(self.blocks) != math.ceil(math.log2(self.n_actions)):
                raise ValueError("majority-modulo needs ceil(log2(n_actions)) blocks.")
            for block in self.blocks:
                if not block or len(set(block)) != len(block) or any(not isinstance(p, int) or p < 0 or p >= n for p in block):
                    raise ValueError("Majority blocks must contain distinct valid measured-wire positions.")
        return self

    def decode(self, samples):
        """Decode only the supplied bit strings, with no basis table or 64-bit overflow."""
        samples = torch.as_tensor(samples)
        if samples.ndim < 1:
            raise ValueError("Samples need a final bit axis.")
        n = samples.shape[-1]
        self.validate(n)
        if not bool(torch.all((samples == 0) | (samples == 1))):
            raise ValueError("Samples must be binary.")
        bits = samples.to(torch.long)
        if self.policy == "contiguous-like":
            # Lexicographic comparisons avoid fixed-width integer overflow for n>63.
            states = 2**n
            base, rem = divmod(states, self.n_actions)
            out = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
            for action in range(1, self.n_actions):
                boundary = action * base + min(action, rem)
                equal = torch.ones_like(out, dtype=torch.bool)
                greater = torch.zeros_like(equal)
                for pos in range(n):
                    b = (boundary >> (n - pos - 1)) & 1
                    greater = greater | (equal & (bits[..., pos] > b))
                    equal = equal & (bits[..., pos] == b)
                out = out + (greater | equal).long()
            return out
        if self.policy == "majority":
            return (bits.sum(-1) > n//2).long()
        if self.policy == "majority-modulo":
            code_bits = [(bits[..., list(block)].sum(-1) > len(block)//2).long() for block in self.blocks]
        else:
            code_bits = [bits[..., :n-i].sum(-1).remainder(2)
                         for i in range(math.ceil(math.log2(self.n_actions)))]
        out = torch.zeros_like(code_bits[0])
        for bit in code_bits:
            out = (2*out + bit).remainder(self.n_actions)
        return out

    def aggregate(self, probabilities):
        """Explicit dense fallback, O(2**n) input/storage; never used by TN contraction."""
        n = int(math.log2(probabilities.shape[-1]))
        if 2**n != probabilities.shape[-1]:
            raise ValueError("Dense probabilities must have power-of-two size.")
        self.validate(n)
        states = torch.arange(2**n, device=probabilities.device)
        bits = (states[:, None] >> torch.arange(n-1, -1, -1, device=states.device)) & 1
        indices = self.decode(bits)
        out = probabilities.new_zeros((*probabilities.shape[:-1], self.n_actions))
        return out.scatter_add(-1, indices.expand_as(probabilities), probabilities)


class ActionProbsMP(qml.measurements.ProbabilityMP):
    """Probabilities of a structured action decoder, directly contracted by IQPTNSimulator."""
    def __init__(self, decoder, wires):
        self.decoder = decoder.validate(len(wires))
        super().__init__(wires=qml.wires.Wires(wires))

    def process_state(self, state, wire_order):
        raise NotImplementedError("ActionProbsMP requires IQPTNSimulator (including its auto fallback adapter).")

    def process_density_matrix(self, density_matrix, wire_order):
        raise NotImplementedError("ActionProbsMP requires IQPTNSimulator.")

    def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
        raise NotImplementedError("ActionProbsMP requires IQPTNSimulator.")

    def shape(self, shots=None, num_device_wires=0):
        return (self.decoder.n_actions,)

    def _flatten(self):
        return (self.obs, self.mv), (self.wires, self.decoder)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[1], metadata[0])

    @property
    def hash(self):
        return hash((super().hash, self.decoder))


def action_probs(wires, decoder=None):
    """Queue exact action probabilities (or shot frequencies when shots are specified)."""
    return ActionProbsMP(decoder or DecoderSpec(), tuple(wires))

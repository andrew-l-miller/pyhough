from dataclasses import dataclass
import numpy as np
from pyhough import signal_simulations
from typing import Optional
from typing import Callable, Tuple
from pyhough import physics

from pyhough.inject import Source

@dataclass
class InjContext:
    source: Source
    sid1: np.ndarray          # complex array (length >= nsid-1)
    sid2: np.ndarray          # complex array (length >= nsid-1)
    vs: Optional[np.ndarray] = None  # vs[fft_index] used by CW only



@dataclass(frozen=True)
class Injection:
    provider: Callable  # expects (tt, fft_index, ctx)
    ctx: InjContext

    def __call__(self, tt, fft_index):
        # call the original provider with the stored ctx
        return self.provider(tt, fft_index, self.ctx)

def provider_sinusoid(f0, h0):
    def inj_provider(tt, fft_index, ctx):
        return signal_simulations.simulate_sinusoid(tt, f0, h0)
    
    return attach_metadata(
    inj_provider,
    provider_type="sinusoid",
    f0=f0,
    h0=h0,
)
    # return inj_provider


def provider_sinusoid_drift(f0, fdot, h0):
    def inj_provider(tt, fft_index, ctx):
        return signal_simulations.simulate_sinusoid_with_drift(tt, f0, fdot, h0)
    
    return attach_metadata(
        inj_provider,
        provider_type="sinusoid_drift",
        f0=f0,
        fdot=fdot,
        h0=h0,
    )
    # return inj_provider


def provider_cw(f0, fdot, alpha, delta, h0):
    def inj_provider(tt, fft_index, ctx):
        if ctx.vs is None:
            raise ValueError("provider_cw requires ctx.vs (set vs=...)")
        return signal_simulations.simulate_cw(
            tt, f0, fdot, alpha, delta, h0, ctx.vs[fft_index]
        )
    return attach_metadata(
        inj_provider,
        provider_type="cw",
        f0=f0,
        fdot=fdot,
        alpha=alpha,
        delta=delta,
        h0=h0,
    )
    # return inj_provider

def provider_power_law(f0, n, h0, k=None, mc=None):

    if (k is None) == (mc is None):
        raise ValueError(
            "Exactly one of 'k' or 'mc' must be supplied."
        )

    if mc is not None:
        k = physics.calc_k(mc)

    def inj_provider(tt, fft_index, ctx):
        return signal_simulations.simulate_power_law(
            tt, f0, k, n, h0
        )

    return attach_metadata(
            inj_provider,
            provider_type="power_law",
            f0=f0,
            n=n,
            h0=h0,
            k=k,
            mc=mc,
        )
    # return inj_provider

def provider_cbc(m1,m2,t_c,h0):
    def inj_provider(tt,fft_index,ctx=None):
        return signal_simulations.simulate_cbc_pn(tt, m1, m2, t_c, h0, order=3.5)

    return attach_metadata(
        inj_provider,
        provider_type="cbc",
        m1=m1,
        m2=m2,
        t_c=t_c,
        h0=h0,
    )
    # return inj_provider


def attach_metadata(provider, provider_type, **params):
    provider.provider_type = provider_type
    provider.params = params
    return provider


PROVIDER_REGISTRY = {
    "sinusoid": provider_sinusoid,
    "sinusoid_drift": provider_sinusoid_drift,
    "cw": provider_cw,
    "power_law": provider_power_law,
    "cbc": provider_cbc,
}
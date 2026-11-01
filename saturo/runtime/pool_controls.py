"""Paper ablations: disabling a pool's saturation excludes BE from it."""

from saturo.runtime.workload import PREFILL, DECODE, TPUT


def pool_accepts_be(role, options):
    def enabled(name):
        return bool(options.get(name, True) if isinstance(options, dict)
                    else getattr(options, name, True))
    return (role == TPUT
            or (role == PREFILL and enabled("enable_prefill_saturation"))
            or (role == DECODE and enabled("enable_decode_saturation")))

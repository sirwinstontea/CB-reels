import argparse
import os
from dataclasses import replace

from .figures import SCENES, base_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--elastic-form", default=None, choices=["literal", "difference", "symmetric"])
    ap.add_argument("--curvature", default=None, choices=["abs", "literal", "constant"])
    ap.add_argument("--pairing", default=None, choices=["particle", "pair"])
    ap.add_argument("--absorb-depth", type=float, default=None)
    ap.add_argument("--d0", type=float, default=None)
    args = ap.parse_args()
    fn = SCENES[args.scene]
    sim, dur = fn()
    cfg = sim.cfg
    over = {}
    if args.elastic_form: over["elastic_form"] = args.elastic_form
    if args.curvature: over["curvature"] = args.curvature
    if args.pairing: over["diffusion_pairing"] = args.pairing
    if args.absorb_depth: over["absorb_depth"] = args.absorb_depth
    if args.d0:
        over["d0"] = args.d0
        over["h"] = 2 * args.d0
    if over:
        if "d0" in over:
            over.setdefault("absorb_depth", None)
            sim, dur = fn(replace(cfg, contact_height=None, **over))
        else:
            for k, v in over.items():
                setattr(sim.cfg, k, v)
    os.makedirs(args.out, exist_ok=True)
    print(f"scene {args.scene}: {sim.x.shape[0]} liquid + {sim.bx.shape[0]} boundary particles at start, "
          f"duration {args.seconds or dur}s, cfg d0={sim.cfg.d0} h={sim.cfg.h}", flush=True)
    sim.run(args.seconds or dur, args.out, fps=args.fps)


if __name__ == "__main__":
    main()

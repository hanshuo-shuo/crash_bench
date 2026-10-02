"""Reproduce the physical forward in the original offscreen-context constructor."""


def replay_renderer_reset_forward(base_class):
    original = base_class._reset_internal
    if getattr(original, '_renderer_forward_replayed', False):
        raise RuntimeError('Renderer reset forward already installed')
    record = {'calls': 0, 'scope': 'one original renderer-constructor forward at each headless base reset; no integration or state assignment'}

    def reset(self, *args, **kwargs):
        if self.has_renderer or self.has_offscreen_renderer:
            raise RuntimeError('Headless renderer-forward replay requires rendering disabled')
        self.sim.forward()
        record['calls'] += 1
        return original(self, *args, **kwargs)

    reset._renderer_forward_replayed = True
    base_class._reset_internal = reset
    return record

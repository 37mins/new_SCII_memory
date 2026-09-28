def sc2_projection(text_actions):
    """Project planner text outputs into SC2 environment actions.

    The SC2 planner environment already parses build-order text downstream, so
    the projection should stay conservative and avoid changing semantics.
    """
    actions = []
    valids = []
    for action in text_actions:
        if action is None:
            actions.append("")
            valids.append(False)
            continue
        if isinstance(action, bytes):
            action = action.decode("utf-8", errors="ignore")
        action = str(action)
        actions.append(action)
        valids.append(bool(action.strip()))
    return actions, valids

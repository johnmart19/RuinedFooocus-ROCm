"""Keep the selected checkpoint visible while the local library is refreshed."""


def checkpoint_choices(names, selected=None):
    choices = list(names)
    if selected and selected not in choices:
        choices.append((f"{selected} (not installed)", selected))
    return choices

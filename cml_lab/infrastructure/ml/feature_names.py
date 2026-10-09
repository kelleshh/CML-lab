"""Preserve library names when available; expose honest positional fallbacks."""
def feature_names(transformer, width):
    try:
        names = list(transformer.get_feature_names_out())
        if len(names) == width: return names
    except (AttributeError, ValueError, TypeError):
        pass
    return [f'output_{index}' for index in range(width)]

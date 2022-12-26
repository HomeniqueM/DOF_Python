
DISABLED = "{color: #181818; background-color: rgb(28, 25, 32)}"
DISABLED_ONLY_TEXT = "{color: #181818;}"

def disabled_stylesheet(widget,only_text = False):
    
    properties = DISABLED
    if only_text:
        properties = DISABLED_ONLY_TEXT

    return f"{widget.__class__.__name__}:disabled{properties}"
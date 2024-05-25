class BaseController:
    def __init__(self, window, path, start_window):
        self.start_window = start_window
        self.selected_action = None
        self.window = window
        self.scene = window.scene
        self.model = None

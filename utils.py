from gradio import Progress

def create_progress_updater(start: int, total: int, desc: str, progress: Progress):
    def updater(pipe, step, timestep, callback_kwargs):
        progress((step + start, total), desc=desc)
        return callback_kwargs
    return updater
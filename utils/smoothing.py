from abc import ABC

import numpy as np
import more_itertools as mit


class Smoother(ABC):
    def smooth(self, modes):
        pass


class MajorityVoteSmoother(Smoother):
    def __init__(self, num_iterations, window_size, step_size=1):
        self.num_iterations = num_iterations
        self.half_window_size = int(window_size / 2)
        self.step_size = step_size
        self._dummy_mode = 'dummy'

    def _pad(self, modes):
        return np.concatenate((
            self.half_window_size * [self._dummy_mode],
            modes,
            self.half_window_size * [self._dummy_mode]))

    def smooth(self, modes):
        tmp_modes = modes.copy()

        for _ in range(self.num_iterations):
            tmp_modes = self._smooth_step(tmp_modes)

        return tmp_modes

    def _smooth_step(self, modes):
        padded_modes = self._pad(modes)
        smoothed_modes = []
        for window in mit.windowed(padded_modes, n=2*self.half_window_size + 1):

            contained_modes, mode_counts = np.unique(
                [w for w in window if w != self._dummy_mode],
                return_counts=True)

            most_prevalent_mode_index = np.argmax(mode_counts)
            most_prevalent_mode = contained_modes[most_prevalent_mode_index]
            smoothed_modes.append(most_prevalent_mode)
            # print(f'Major mode:\t{most_prevalent_mode}\tin{window}')

        assert len(modes) == len(smoothed_modes)
        return np.array(smoothed_modes, dtype=np.str_)



class ParticleTrajectorySolver():
    def __init__(self, stepper, cell_crossing_solver):
        # extract pmesh from the stepper and mesh as parent mesh
        # extract dt from stepper
        # velocity could be a function of time so may need to pass time params
        self.stepper = stepper
        self.cell_crossing_solver = cell_crossing_solver
        pass

    def solve(self, t_start, t_end):
        pass

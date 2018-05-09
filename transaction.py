from utils import sum_weights

class Transaction(object):
    def __init__(self, simulator, name, type, time, parents, weight, publisher):
        self.simulator = simulator
        self.name = name
        self.type = type
        self.time = time
        self.parents = parents
        self.weight = weight
        self.publisher = publisher
        self.is_tip = False
        self.extras = {}
        self.acc_weight = float('-inf')
        self.is_confirmed = False

    def update_acc_weight(self, weight=None, used=None):
        if weight is None:
            weight = self.weight
        if used is None:
            used = set()

        used.add(self)
        if self.acc_weight > self.simulator.min_weight_confirmed:
            return

        #self.acc_weight = log((2**self.acc_weight) + 2**weight)/log(2)
        self.acc_weight = sum_weights(self.acc_weight, weight)
        if self.acc_weight > self.simulator.min_weight_confirmed:
            self.is_confirmed = True
            if 'confirmed_time' not in self.extras:
                self.extras['confirmed_time'] = self.simulator.cur_time
        for parent in self.parents:
            if parent not in used:
                parent.update_acc_weight(weight, used)

    def __repr__(self):
        return 'Tx({})'.format(self.name)

    def confirm_parents(self):
        for parent in self.parents:
            if parent in self.simulator.tips:
                parent.extras['first_confirmation_time'] = self.simulator.cur_time
                self.simulator.tips.remove(parent)
                self.simulator.add_tx(parent)




import random
import numpy.random
import heapq

from event import Event

class TxGenerator(object):
    def __init__(self, tx_lambda, hashpower):
        self.simulator = None
        self.tx_lambda = tx_lambda
        self.hashpower = hashpower
        self.weight = 17
        self.geometric_p = 2**(-self.weight)
        self.enabled = False

    def set_simulator(self, simulator):
        self.simulator = simulator
        self.events = simulator.events
        self.tx_by_name = simulator.tx_by_name
        self.transactions = simulator.transactions
        self.tips = simulator.tips
        self.schedule_next_tx()

    def schedule_next_tx(self):
        dt = random.expovariate(self.tx_lambda)
        ev = Event(self.simulator.cur_time + dt, self.gen_new_pow, None)
        heapq.heappush(self.events, ev)

    def gen_new_pow(self, ev):
        if not self.enabled:
            return

        trials = numpy.random.geometric(self.geometric_p)
        dt = 1.0 * trials / self.hashpower

        parents = self.simulator.get_two_tips()
        tx = self.simulator.new_tx(type='tx', time=self.simulator.cur_time, parents=parents, weight=self.weight, publisher=self)
        self.simulator.pow.add(tx)

        ev = Event(self.simulator.cur_time + dt, self.gen_new_tx, tx)
        heapq.heappush(self.events, ev)
        self.schedule_next_tx()

    def gen_new_tx(self, ev):
        tx = ev.params
        self.simulator.pow.remove(tx)
        tx.confirm_parents()
        tx.update_acc_weight()
        self.tx_by_name[tx.name] = tx
        self.simulator.add_tip(tx)



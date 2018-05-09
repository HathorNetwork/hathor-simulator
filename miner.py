import numpy.random

from event import Event
import heapq

class Miner(object):
    def __init__(self, hashpower):
        self.simulator = None
        self.hashpower = hashpower
        self.enabled = False

    def set_simulator(self, simulator):
        self.simulator = simulator
        self.events = simulator.events
        self.blocks = simulator.blocks
        self.update_weight(simulator.block_weight)
        self.schedule_next_block()

    def update_weight(self, weight):
        self.weight = weight
        self.geometric_p = 2**(-self.weight)

    def schedule_next_block(self):
        trials = numpy.random.geometric(self.geometric_p)
        dt = 1.0 * trials / self.hashpower
        ev = Event(self.simulator.cur_time + dt, self.gen_new_block, None)
        heapq.heappush(self.events, ev)

    def get_latest_block(self):
        if not self.simulator.blocks:
            return None
        return self.simulator.blocks[-1]
    
    def gen_new_block(self, ev):
        if not self.enabled:
            return
        ref_block = self.get_latest_block()
        parents = self.simulator.get_two_tips()
        for tx in parents:
            tx.confirm_parents()
        if ref_block:
            parents.append(ref_block)
        block = self.simulator.new_tx(type='blk', time=self.simulator.cur_time, parents=parents, weight=self.weight, publisher=self)
        block.update_acc_weight()
        self.simulator.add_block(block)
        self.schedule_next_block()



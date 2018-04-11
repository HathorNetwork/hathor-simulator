import random
import heapq
import numpy.random
from math import ceil, log
from collections import namedtuple
import sys
import time

sys.setrecursionlimit(100000)

Event = namedtuple('Event', 'dt run')

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
        self.update_acc_weight(self.weight, used=set())

    def update_acc_weight(self, weight, used=set()):
        used.add(self)
        if self.acc_weight > self.simulator.min_weight_confirmed:
            return

        self.acc_weight = log((2**self.acc_weight) + 2**weight)/log(2)
        if self.acc_weight > self.simulator.min_weight_confirmed:
            if 'confirmed_time' not in self.extras:
                self.extras['confirmed_time'] = self.simulator.cur_time
        for parent in self.parents:
            if parent not in used:
                parent.update_acc_weight(weight, used)

    def __repr__(self):
        return 'Tx({})'.format(self.name)



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
        ev = Event(self.simulator.cur_time + dt, self.gen_new_block)
        heapq.heappush(self.events, ev)

    def get_latest_block(self):
        if not self.simulator.blocks:
            return None
        return self.simulator.blocks[-1]
    
    def gen_new_block(self):
        if not self.enabled:
            return
        ref_block = self.get_latest_block()
        parents = self.simulator.get_two_tips()
        if ref_block:
            parents.append(ref_block)
        block = self.simulator.new_tx(type='blk', time=self.simulator.cur_time, parents=parents, weight=self.weight, publisher=self)
        self.simulator.add_block(block)
        self.schedule_next_block()


class TxGenerator(object):
    def __init__(self, tx_lambda, hashpower):
        self.simulator = None
        self.tx_lambda = tx_lambda
        self.hashpower = hashpower
        self.weight = 17
        self.geometric_p = 2**(-self.weight)

    def set_simulator(self, simulator):
        self.simulator = simulator
        self.events = simulator.events
        self.tx_by_name = simulator.tx_by_name
        self.transactions = simulator.transactions
        self.tips = simulator.tips
        self.schedule_next_tx()

    def schedule_next_tx(self):
        dt = random.expovariate(self.tx_lambda)
        ev = Event(self.simulator.cur_time + dt, self.gen_new_pow)
        heapq.heappush(self.events, ev)

    def gen_new_pow(self):
        trials = numpy.random.geometric(self.geometric_p)
        dt = 1.0 * trials / self.hashpower
        ev = Event(self.simulator.cur_time + dt, self.gen_new_tx)
        heapq.heappush(self.events, ev)
        self.schedule_next_tx()

    def gen_new_tx(self):
        parents = self.simulator.get_two_tips()
        tx = self.simulator.new_tx(type='tx', time=self.simulator.cur_time, parents=parents, weight=self.weight, publisher=self)
        self.tx_by_name[tx.name] = tx
        self.simulator.add_tip(tx)


class HathorSimulator(object):
    def __init__(self, block_weight=17):
        self.miners = []
        self.tx_generators = []
        self.cur_time = 0

        self.tx_count = 0
        self.events = []
        self.blocks = []
        self.tx_by_name = {}
        self.transactions = []
        self.tips = set()

        self.block_weight = block_weight
        self.latest_weight_update = 0

        self.tx_weight_tmp = float('-inf')
        self.tx_weight = float('-inf')
        self.latest_tx_weight_update = 0

	self.update_min_weight_confirmed()

        genesis = self.new_tx(type='genesis', time=0, parents=[], weight=0, publisher=None)
        self.add_tx(genesis)

    def update_min_weight_confirmed(self):
        w_blk = log(6)/log(2) + self.block_weight
        w_tx = self.tx_weight
        self.min_weight_confirmed = log(2**w_blk + 2**w_tx)/log(2)
        print('[{:12.2f}] min weight updated: w_blk={:6.4f} w_tx={:6.4f}'.format(self.cur_time, w_blk, w_tx))

    def new_tx(self, *args, **kwargs):
        name = str(self.tx_count)
        self.tx_count += 1
        return Transaction(self, name, *args, **kwargs)

    def add_tip(self, tx):
        self.tips.add(tx.name)
        self.tx_weight_tmp = log(2**(self.tx_weight_tmp) + 2**tx.weight)/log(2)
        if self.cur_time - self.latest_tx_weight_update > 3600:
            self.update_tx_weight()

    def update_tx_weight(self):
        #print('[{:12.2f}] TX weight updated: {:6.2f} -> {:6.2f}'.format(self.cur_time, self.tx_weight, self.tx_weight_tmp))
        self.tx_weight = self.tx_weight_tmp - log(self.cur_time - self.latest_tx_weight_update)/log(2)
        self.tx_weight_tmp = float('-inf')
        self.latest_tx_weight_update = self.cur_time
        self.update_min_weight_confirmed()

    def add_tx(self, tx):
        self.transactions.append(tx)
        self.tx_by_name[tx.name] = tx

    def add_block(self, block):
        if block.weight < self.block_weight:
            print('Block ignored.', self.block_weight, block.weight)
            return

        self.blocks.append(block)
        if (len(self.blocks) + 1) % 2016 == 0:
            self.update_weight()

    def update_weight(self):
        dt = self.cur_time - self.latest_weight_update
        new_weight = self.block_weight + 7 + log(2016)/log(2) - log(dt)/log(2)
        #print('[{:12.2f}] Weight updated: blocks={:5d} dt={:6.2f} {:6.2f} -> {:6.2f}'.format(self.cur_time, len(self.blocks), dt, self.block_weight, new_weight))
        self.block_weight = new_weight
	self.update_min_weight_confirmed()
        for miner in self.miners:
            miner.update_weight(self.block_weight)
        self.latest_weight_update = self.cur_time


    def add_miner(self, miner):
        self.miners.append(miner)
        miner.set_simulator(self)
        miner.enabled = True


    def remove_miner(self, miner):
        idx = self.miners.index(miner)
        self.miners.pop(idx)
        miner.enabled = False


    def add_tx_generator(self, txgen):
        self.tx_generators.append(txgen)
        txgen.set_simulator(self)

    def get_two_tips(self):
        qty = 2
        if len(self.tips) >= qty:
            ret = random.sample(self.tips, qty)
        else:
            ret = list(self.tips)

        v = []
        if len(ret) < qty:
            v += random.sample(self.transactions[-15:], min(len(self.transactions), qty - len(ret)))

        for x in ret:
            tx = self.tx_by_name[x]
            tx.extras['first_confirmation_time'] = self.cur_time
            self.tips.remove(x)
            self.add_tx(tx)
            v.append(tx)

        return v

    def run(self, total_dt, report_interval=None):
        rt0 = time.time()
        t0 = self.cur_time
        t1 = t0
        while self.events:
            if report_interval and self.cur_time - t1 > report_interval:
                print('{:6.2f} [{:12.2f}] blocks={} txs={} tips={}'.format(
                    time.time() - rt0, self.cur_time, len(self.blocks), len(self.transactions), len(self.tips)
                ))
                t1 = self.cur_time
            ev = heapq.heappop(self.events)
            self.cur_time = ev.dt
            ev.run()
            if self.cur_time - t0 >= total_dt:
                return

    def gen_dot(self):
        from graphviz import Digraph

        dot = Digraph(format='pdf')

        g_blocks = dot.subgraph(name='blocks')
        g_txs = dot.subgraph(name='txs')

        dot.attr('node', shape='box', style='filled', fillcolor='#EC644B')
        for i, blk in enumerate(self.blocks):
            dot.node(blk.type + blk.name)

        dot.attr('node', shape='oval', style='')

        nodes = self.transactions + self.blocks
        nodes.sort(key=lambda x: x.time)
        for i, tx in enumerate(nodes):
            if tx.type == 'blk':
                attrs = {'penwidth': '4'}
            else:
                attrs = {}

            for parent in tx.parents:
                dot.edge(tx.type+tx.name, parent.type+parent.name, **attrs)

        dot.attr('node', style='filled', fillcolor='#aaaaaa')
        for x in self.tips:
            tx = self.tx_by_name[x]
            nodes.append(tx)
            dot.node(tx.type + tx.name)

            for parent in tx.parents:
                dot.edge(tx.type+tx.name, parent.type+parent.name, **attrs)

        return dot


if __name__ == '__main__':
    sim = HathorSimulator(block_weight=23.6)

    m0 = Miner(hashpower=100000)
    sim.add_miner(m0)

    g0 = TxGenerator(tx_lambda=1/50., hashpower=1000)
    sim.add_tx_generator(g0)

    sim.run(3600*2, report_interval=3600)

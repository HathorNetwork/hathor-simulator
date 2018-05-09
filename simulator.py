import random
import heapq
import numpy.random
from math import ceil, log
import sys
import time

from utils import sum_weights
from event import Event
from transaction import Transaction
from txgenerator import TxGenerator
from miner import Miner

sys.setrecursionlimit(100000)


class GenericMetric(object):
    def __init__(self, interval):
        self.simulator = None
        self.interval = interval

    def set_simulator(self, simulator):
        self.simulator = simulator
        self.events = simulator.events
        self.schedule_next_sample()

    def schedule_next_sample(self):
        ev = Event(self.simulator.cur_time + self.interval, self.do_sample, None)
        heapq.heappush(self.events, ev)

    def do_sample(self, ev):
        pass


class UtterlyAcceptanceMetric(GenericMetric):
    def __init__(self, interval):
        super(UtterlyAcceptanceMetric, self).__init__(interval)
        self.values = []

    def do_sample(self, ev):
        from collections import deque, defaultdict

        self.schedule_next_sample()

        sim = self.simulator

        d = deque([(tip, 0, set()) for tip in sim.tips])
        counter = defaultdict(int)

        while d:
            tx, level, used = d.popleft()

            for x in tx.parents:
                if hasattr(x, 'utterly_acceptance_dt'):
                    continue

                if x not in used:
                    used.add(x)
                    counter[x] += 1
                    if counter[x] == len(sim.tips):
                        dt = sim.cur_time - x.time
                        assert(not hasattr(x, 'utterly_acceptance_dt'))
                        x.utterly_acceptance_dt = dt
                        self.values.append(dt)
                    else:
                        d.append((x, level+1, used))


class TipsMetric(GenericMetric):
    def __init__(self, interval):
        super(TipsMetric, self).__init__(interval)
        self.times = []
        self.values = []

    def do_sample(self, ev):
        time = self.simulator.cur_time
        value = len(self.simulator.tips)
        self.times.append(time)
        self.values.append(value)
        self.schedule_next_sample()


class HathorSimulator(object):
    def __init__(self, block_weight=17):
        self.metrics = []
        self.miners = []
        self.tx_generators = []
        self.cur_time = 0
        self.ignore_update_acc_weight = False

        self.tx_count = 0
        self.events = []
        self.blocks = []
        self.tx_by_name = {}
        self.transactions = []
        self.tips = set()
        self.pow = set()

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
        #self.min_weight_confirmed = log(2**w_blk + 2**w_tx)/log(2)
        self.min_weight_confirmed = sum_weights(w_blk, w_tx)
        #print('[{:12.2f}] min weight updated: w_blk={:6.4f} w_tx={:6.4f}'.format(self.cur_time, w_blk, w_tx))

    def new_tx(self, *args, **kwargs):
        name = str(self.tx_count)
        self.tx_count += 1
        return Transaction(self, name, *args, **kwargs)

    def add_tip(self, tx):
        self.tips.add(tx)
        #self.tx_weight_tmp = log(2**(self.tx_weight_tmp) + 2**tx.weight)/log(2)
        self.tx_weight_tmp = sum_weights(self.tx_weight_tmp, tx.weight)
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


    def add_metric(self, metric):
        self.metrics.append(metric)
        metric.set_simulator(self)


    def add_miner(self, miner):
        self.miners.append(miner)
        miner.set_simulator(self)
        miner.enabled = True


    def remove_miner(self, miner):
        idx = self.miners.index(miner)
        self.miners.pop(idx)
        miner.enabled = False

    def remove_tx_generator(self, txgen):
        idx = self.tx_generators.index(txgen)
        self.tx_generators.pop(idx)
        txgen.enabled = False


    def add_tx_generator(self, txgen):
        self.tx_generators.append(txgen)
        txgen.set_simulator(self)
        txgen.enabled = True

    def get_two_tips(self):
        qty = 2
        if len(self.tips) >= qty:
            ret = random.sample(self.tips, qty)
        else:
            ret = list(self.tips)

        if len(ret) < qty:
            ret += random.sample(self.transactions[-15:], min(len(self.transactions), qty - len(ret)))

        return ret

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
            ev.run(ev)
            if self.cur_time - t0 >= total_dt:
                return

    def gen_dot(self, highlight=None):
        from graphviz import Digraph

        if highlight is None:
            highlight = {}

        dot = Digraph(format='pdf')

        g_blocks = dot.subgraph(name='blocks')
        g_txs = dot.subgraph(name='txs')

        dot.attr('node', shape='box', style='filled', fillcolor='#EC644B')
        for i, blk in enumerate(self.blocks):
            attrs = {}
            if blk in highlight:
                attrs.update(highlight[blk])
            dot.node(blk.type + blk.name)

        dot.attr('node', shape='oval', style='')
        nodes = self.transactions + self.blocks
        nodes.sort(key=lambda x: x.time)
        for i, tx in enumerate(nodes):
            attrs_node = {}

            if tx.type == 'blk':
                attrs = {'penwidth': '4'}
            else:
                attrs = {}

                if tx.is_confirmed:
                    attrs_node.update({'style': 'filled', 'fillcolor': '#87D37C'})

            if tx in highlight:
                attrs_node.update(highlight[tx])
            if attrs_node:
                dot.node(tx.type + tx.name, **attrs_node)

            for parent in tx.parents:
                dot.edge(tx.type+tx.name, parent.type+parent.name, **attrs)

        dot.attr('node', style='')
        for tx in self.tips:
            attrs = {'style': 'filled', 'fillcolor': '#F5D76E'}

            if tx in highlight:
                attrs.update(highlight[tx])

            nodes.append(tx)
            dot.node(tx.type + tx.name, **attrs)

            for parent in tx.parents:
                dot.edge(tx.type+tx.name, parent.type+parent.name)

        for tx in self.pow:
            attrs = {'style': 'dashed,filled', 'fillcolor': '#BDC3C7'}
            if tx in highlight:
                attrs.update(highlight)
            dot.node(tx.type+tx.name, **attrs)
            for parent in tx.parents:
                dot.edge(tx.type+tx.name, parent.type+parent.name, style='dashed', arrowhead='empty')

        return dot


if __name__ == '__main__':
    sim = HathorSimulator(block_weight=23.6)

    m0 = Miner(hashpower=100000)
    sim.add_miner(m0)

    g0 = TxGenerator(tx_lambda=1/50., hashpower=1000)
    sim.add_tx_generator(g0)

    sim.run(3600*2, report_interval=3600)

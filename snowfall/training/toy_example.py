#!/usr/bin/env python3

# To be removed

# This file displays the `phones` attribute
# obtained in various approaches

import sys
sys.path.insert(0, '/root/fangjun/open-source/snowfall/snowfall')
sys.path.insert(0, '/root/fangjun/open-source/k2/build/lib')
sys.path.insert(0, '/root/fangjun/open-source/k2/k2/python')

from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols

import k2


def main():
    phone_ids = [1, 2, 3]
    ctc_topo = build_ctc_topo([0] + phone_ids)
    isym = k2.SymbolTable.from_str('''
    <blk> 0
    a 1
    b 2
    c 3
    ''')

    osym = k2.SymbolTable.from_str('''
    a 1
    b 2
    c 3
    ''')

    ctc_topo.symbols = isym
    ctc_topo.aux_symbols = osym

    s = '''
    0 1 3 0.1
    1 3 1 0.22
    1 2 2 0.2
    2 3 1 0.3
    3 4 -1 0
    4
    '''

    num_graphs = k2.Fsa.from_str(s)
    num_graphs.symbols = osym
    P = create_bigram_phone_lm(phone_ids)
    P.symbols = isym

    P_with_self_loops = k2.add_epsilon_self_loops(P)
    ctc_topo_P = k2.compose(ctc_topo,
                            P_with_self_loops,
                            treat_epsilons_specially=False,
                            inner_labels='phones')
    ctc_topo_P.aux_symbols = osym

    # this is the `phones` attribute for the `den_graph`
    print('ctc_topo_P.phones\n', ctc_topo_P.phones)

    num_graphs_with_self_loops = k2.remove_epsilon_and_add_self_loops(
        num_graphs)
    num_graphs_with_self_loops = k2.arc_sort(num_graphs_with_self_loops)

    # num1 inherits the `phones` attribute from ctc_topo_P
    # this is the `phones` attribute for the `num_graph`
    num1 = k2.compose(ctc_topo_P,
                      num_graphs_with_self_loops,
                      treat_epsilons_specially=False)
    print('num1.phones\n', num1.phones)

    # `num2` is identical with `num1` as you can see from the output of print
    num2 = k2.compose(ctc_topo_P,
                      num_graphs_with_self_loops,
                      treat_epsilons_specially=False,
                      inner_labels='phones')
    print('num2.phones\n', num2.phones)


if __name__ == '__main__':
    main()

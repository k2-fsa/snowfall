import logging

import k2
from k2 import Fsa


def compile_LG(
        L: Fsa,
        G: Fsa,
        labels_disambig_id_start: int,
        aux_labels_disambig_id_start: int
) -> Fsa:
    # if not os.path.exists(lang_dir + '/LG.pt'):
    #     print("Loading L_disambig.fst.txt")
    #     with open(lang_dir + '/L_disambig.fst.txt') as f:
    #         L = k2.Fsa.from_openfst(f.read(), acceptor=False)
    #     print("Loading G.fsa.txt")
    #     with open(lang_dir + '/G.fsa.txt') as f:
    #         G = k2.Fsa.from_openfst(f.read(), acceptor=True)
    L = k2.arc_sort(L.invert_())
    G = k2.arc_sort(G)
    logging.debug("Intersecting L and G")
    LG = k2.intersect(L, G)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Connecting L*G")
    LG = k2.connect(LG).invert_()
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    LG = k2.add_epsilon_self_loops(LG)
    LG = k2.arc_sort(LG)
    logging.debug(f'LG is arc sorted: {(LG.properties & k2.fsa_properties.ARC_SORTED) != 0}')
    return LG
    # torch.save(LG.as_dict(), lang_dir + '/LG.pt')
    #     # print(LG)
    # else:
    #     d = torch.load(lang_dir + '/LG.pt')
    #     print("Loading pre-prepared LG")
    #     LG = k2.Fsa.from_dict(d)

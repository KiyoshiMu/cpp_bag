from __future__ import annotations

from cpp_bag.data import dump_cells
from cpp_bag.data import FEAT_DIR
from cpp_bag.data import load_cells


# dump_cells(FEAT_DIR)
cells = load_cells()
print(len(cells))

"""Buffer positions and addresses: effects recorded in one process are compared
in another, where on-demand pages sit at other positions."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp.emu import PAGE_SIZE, WINDOWS, Memory  # noqa: E402


def memory(order):
    mem = Memory(0x100000, b"\0" * 0x1000, 0x100000)
    mem.page_source = lambda page: bytes(PAGE_SIZE)
    for page in order:
        mem.map_page(page)
    return mem


class TestAddressAt(unittest.TestCase):
    ADDRESSES = (0x100000, 0x1FFFFF, WINDOWS[0][0] + 4, WINDOWS[1][0] + 0x10,
                 0x02000000, 0x02000FFF, 0x03000123)

    def test_position_and_address_round_trip(self):
        mem = memory([0x02000000, 0x03000000])
        for address in self.ADDRESSES:
            self.assertEqual(mem.address_at(mem.offset_of(address)), address)

    def test_same_address_whatever_the_mapping_order(self):
        one, other = memory([0x02000000, 0x03000000]), memory([0x03000000, 0x02000000])
        self.assertNotEqual(one.offset_of(0x03000123), other.offset_of(0x03000123))
        self.assertEqual(one.address_at(one.offset_of(0x03000123)),
                         other.address_at(other.offset_of(0x03000123)))

    def test_unmapping_keeps_the_order(self):
        mem = memory([0x02000000])
        mark = mem.page_mark()
        mem.map_page(0x03000000)
        mem.unmap_to(mark)
        mem.map_page(0x04000000)
        self.assertEqual(mem.address_at(mem.offset_of(0x04000010)), 0x04000010)


if __name__ == "__main__":
    unittest.main()

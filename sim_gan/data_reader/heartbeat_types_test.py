import unittest
from sim_gan.data_reader import heartbeat_types


class HeartBeatTypesTest(unittest.TestCase):
    def test_convert_normal_heartbeat_mit_bih_to_aami(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('N'), 'N')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('L'), 'N')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('R'), 'N')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('e'), 'N')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('j'), 'N')

    def test_convert_sveb_heartbeat_mit_bih_to_aami(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('A'), 'S')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('a'), 'S')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('J'), 'S')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('S'), 'S')

    def test_convert_veb_heartbeat_mit_bih_to_aami(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('V'), 'V')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('E'), 'V')

    def test_convert_fusion_heartbeat_mit_bih_to_aami(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('F'), 'F')

    def test_convert_unknown_heartbeat_mit_bih_to_aami(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('/'), 'Q')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('U'), 'Q')
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami('f'), 'Q')

    def test_convert_normal_heartbeat_mit_bih_to_aami_index_class(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('N'), 0)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('L'), 0)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('R'), 0)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('e'), 0)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('j'), 0)

    def test_convert_sveb_heartbeat_mit_bih_to_aami_index_class(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('A'), 1)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('a'), 1)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('J'), 1)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('S'), 1)

    def test_convert_veb_heartbeat_mit_bih_to_aami_index_class(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('V'), 2)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('E'), 2)

    def test_convert_fusion_heartbeat_mit_bih_to_aami_index_class(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('F'), 3)

    def test_convert_unknown_heartbeat_mit_bih_to_aami_index_class(self):
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('/'), 4)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('U'), 4)
        self.assertEqual(heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class('f'), 4)


if __name__ == '__main__':
    unittest.main()

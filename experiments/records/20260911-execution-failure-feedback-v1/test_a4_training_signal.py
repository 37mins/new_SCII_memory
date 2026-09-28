import unittest

from audit_a4_training_signal import supply_order


class SupplyOrderTest(unittest.TestCase):
    def test_depot_first(self):
        self.assertEqual(supply_order('- supplydepot x 2\n- marine x 42')['order'],
                         'depot_before_consumers')

    def test_repeated_consumers_before_first_depot(self):
        result = supply_order('- marine x 2\n- scv\n- marine x 3\n- supplydepot\n- marine x 8')
        self.assertEqual(result, dict(order='consumer_before_depot', demand_before_depot=6))

    def test_requires_both_sides(self):
        for answer in ('- marine x 42', '- supplydepot x 2', '- barracks'):
            self.assertEqual(supply_order(answer)['order'], 'not_comparable')

    def test_not_alias_or_prose_parser(self):
        self.assertEqual(supply_order('- marines x 2\n- supplydepot')['order'], 'not_comparable')


if __name__ == '__main__':
    unittest.main()

"""Synthetic contract tests, not claims that SC2 runtime emits these witnesses."""
import unittest

from audit_supply_reward_candidate import event_verdict, ideal_episode_increment


def complete_witness(**changes):
    return dict(episode_id='synthetic-episode-1', **{
        **dict(failure_origin='model', depot_origin='model',
               resumed_production_origin='model', supply_failure=True,
               command_links_verified=True, depot_started_after_failure=True,
               same_depot_completed=True, capacity_increased=True,
               new_marine_from_post_completion_order=True), **changes})


class CandidateContractTest(unittest.TestCase):
    def test_real_chain_would_qualify_not_just_words(self):
        self.assertEqual(event_verdict(complete_witness(), set()), 'eligible_event')

    def test_repeat_building_or_reblock_cannot_earn_second_event(self):
        self.assertEqual(event_verdict(complete_witness(), {'synthetic-episode-1'}), 'no_second_event')

    def test_each_episode_has_its_own_cap(self):
        self.assertEqual(event_verdict(complete_witness(), {'other-episode'}), 'eligible_event')

    def test_rule_fixed_bo_and_human_are_not_model_actions(self):
        for field in ('failure_origin', 'depot_origin', 'resumed_production_origin'):
            for origin in ('rule', 'fixed_bo', 'human'):
                with self.subTest(field=field, origin=origin):
                    self.assertEqual(event_verdict(complete_witness(**{field: origin}), set()),
                                     'exclude_non_model_action')

    def test_missing_links_are_unknown_not_observed_failure(self):
        self.assertEqual(event_verdict(complete_witness(command_links_verified=None), set()),
                         'insufficient_evidence')
        self.assertEqual(event_verdict(complete_witness(depot_origin=None), set()), 'insufficient_evidence')

    def test_submitted_but_cancelled_destroyed_or_unfinished_gets_no_event(self):
        self.assertEqual(event_verdict(complete_witness(same_depot_completed=False), set()), 'not_resolved')

    def test_troop_death_freeing_supply_is_not_capacity_growth(self):
        self.assertEqual(event_verdict(complete_witness(capacity_increased=False), set()), 'not_resolved')

    def test_preexisting_depot_is_not_this_correction(self):
        self.assertEqual(event_verdict(complete_witness(depot_started_after_failure=False), set()), 'not_resolved')

    def test_no_production_or_preexisting_queued_marine_is_not_resolution(self):
        self.assertEqual(event_verdict(complete_witness(new_marine_from_post_completion_order=False), set()),
                         'not_resolved')

    def test_no_supply_failure_does_not_create_a_correction_event(self):
        self.assertEqual(event_verdict(complete_witness(supply_failure=False), set()), 'not_resolved')

    def test_same_bonus_for_all_games_does_not_create_relative_episode_signal(self):
        self.assertEqual(ideal_episode_increment([.5]*8, [5]*8), [0]*8)

    def test_one_bonus_distinguishes_only_an_idealized_group(self):
        delta = ideal_episode_increment([.5]+[0]*7, [5]*8)
        self.assertAlmostEqual(delta[0], .0875)
        self.assertEqual(delta[1:], [-.0125]*7)


if __name__ == '__main__':
    unittest.main()

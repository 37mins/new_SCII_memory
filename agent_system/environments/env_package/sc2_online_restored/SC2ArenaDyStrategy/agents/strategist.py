import copy
import os
import re


def _game_time_seconds(obs):
    match = re.search(r"It is now (\d{1,2}):(\d{2}) in the game", str(obs))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _timing_bounds(timing):
    values = re.findall(r"(\d{2}):(\d{2})", str(timing))
    if len(values) != 2:
        return None, None
    seconds = [int(minutes) * 60 + int(secs) for minutes, secs in values]
    return seconds[0], seconds[1]


def _own_count(obs, name):
    own_text = str(obs).lower().split("as for the visible enemy", 1)[0]
    aliases = {
        "techlab barracks": ("techlab barracks", "barracks with techlab"),
        "reactor barracks": ("reactor barracks", "barracks with reactor"),
        "techlab factory": ("techlab factory", "factory with techlab"),
        "techlab starport": ("techlab starport", "starport with techlab"),
    }
    values = []
    for alias in aliases.get(name.lower(), (name.lower(),)):
        escaped = re.escape(alias)
        values.extend(re.findall(rf"(\d+)\s+{escaped}\b", own_text))
        values.extend(
            re.findall(
                rf"(\d+)\s+(?:bases?|structures?|units?)\s+in\s+the\s+form\s+of\s+{escaped}\b",
                own_text,
            )
        )
    return max((int(value) for value in values), default=0)

class SC2Strategist:
    def __init__(self) -> None:
        # self.strategy = [
        #     {
        #         'phase_summary': 'Early Expansion and Marine Production',
        #         'timing': '(00:00-03:46)',
        #         'instruction': 'Build 29 SCVs by 03:46. Establish natural expansion at 00:58. Construct 6 barracks with continuous marine production. Build 1 bunker for defense and 2 refineries at 03:00/03:06 with 3 SCVs each. Maintain orbital commands on both bases for MULEs.',
        #         'attack_instruction': 'No attacks initiated in this phase; focus on defensive positioning and marine count buildup.',
        #         'bo todo': {
        #             "SCV": 21,
        #             "supplydepot": 4,
        #             "commandcenter": 1,
        #             "barracks": 6,
        #             "orbitalcommand": 2,
        #             "marine": 14,
        #             "bunker": 1,
        #             "mule": 4,
        #             "refinery": 2
        #         },
        #     },
        #     {
        #         'phase_summary': 'Multi-Wave Marine Harassment and Tech Transition',
        #         'timing': '(03:46-06:07)',
        #         'instruction': 'Grow to 39 SCVs. Add 3 factories with tech labs and 5 barracks tech labs. Produce siege tanks from 06:07. Build engineering bay and missile turrets for detection. Maintain marine production from reactor barracks.',
        #         'attack_instruction': 'Execute sequential marine attacks between 03:46-05:43 to disrupt enemy economy and scout tech paths, utilizing marine mobility.',
        #         'bo todo': {
        #             "marine": 39,
        #             "engineeringbay": 1,
        #             "SCV": 8,
        #             "supplydepot": 7,
        #             "barracks": 2,
        #             "mule": 4,
        #             "techlab barracks": 5,
        #             "reactor barracks": 3,
        #             "factory": 3,
        #             "missileturret": 2,
        #             "techlab factory": 3
        #         },
        #     },
        #     {
        #         'phase_summary': 'Siege Tank Power Spike and All-in Commitment',
        #         'timing': '(06:07 onwards)',
        #         'instruction': 'Reach 52 SCVs. Add third/fourth command centers. Produce siege tanks and marauders from tech labs. Maintain marine production from reactor barracks. Build additional refineries for gas-heavy units.',
        #         'attack_instruction': 'Launch game-ending attacks starting at 06:07 with bio-mech deathball supported by SCVs for repairs and forward buildings.',
        #         'bo todo': {
        #             "marine": 50,
        #             "supplydepot": 9,
        #             "mule": 6,
        #             "siegetank": 9,
        #             "SCV": 15,
        #             "marauder": 4,
        #             "commandcenter": 2,
        #             "refinery": 1
        #         }
        #     },
        # ]
        reference_strategy = [
            {
                'phase_summary': 'Early Expansion and Marine Production',
                'timing': '(00:00-03:46)',
                'instruction': 'Build 29 SCVs by 03:46. Establish natural expansion at 00:58. Construct 6 barracks with continuous marine production. Build 1 bunker for defense and 2 refineries at 03:00/03:06 with 3 SCVs each. Maintain orbital commands on both bases for MULEs.',
                'attack_instruction': 'No attacks initiated in this phase; focus on defensive positioning and marine count buildup.',
                'bo todo': {
                    "SCV": 21,
                    "supplydepot": 4,
                    "commandcenter": 1,
                    "barracks": 6,
                    "orbitalcommand": 2,
                    "marine": 14,
                    "bunker": 1,
                    "mule": 4,
                    "refinery": 2
                },
            },
            {
                'phase_summary': 'Multi-Wave Marine Harassment and Tech Transition',
                'timing': '(03:46-06:07)',
                'instruction': 'Grow to 39 SCVs. Add 3 factories with tech labs and 5 barracks tech labs. Produce siege tanks from 06:07. Build engineering bay and missile turrets for detection. Maintain marine production from reactor barracks.',
                'attack_instruction': 'Execute sequential marine attacks between 03:46-05:43 to disrupt enemy economy and scout tech paths, utilizing marine mobility.',
                'bo todo': {
                    "marine": 39,
                    "engineeringbay": 1,
                    "SCV": 8,
                    "supplydepot": 7,
                    "barracks": 2,
                    "mule": 4,
                    "techlab barracks": 5,
                    "reactor barracks": 3,
                    "factory": 3,
                    "missileturret": 2,
                    "techlab factory": 3
                },
            },
            {
                'phase_summary': 'Siege Tank Power Spike and All-in Commitment',
                'timing': '(06:07-10:00)',
                'instruction': 'Reach 52 SCVs. Add third/fourth command centers. Produce siege tanks and marauders from tech labs. Maintain marine production from reactor barracks. Build additional refineries for gas-heavy units.',
                'attack_instruction': 'Launch game-ending attacks starting at 06:07 with bio-mech deathball supported by SCVs for repairs and forward buildings.',
                'bo todo': {
                    "marine": 50,
                    "supplydepot": 9,
                    "mule": 6,
                    "siegetank": 9,
                    "SCV": 15,
                    "marauder": 4,
                    "commandcenter": 2,
                    "refinery": 1
                }
            },
        ]
        macro_strategy = [
            {
                'phase_summary': 'Rapid Expansion and Defense Setup',
                'timing': '(00:00 - 03:03)',
                'instruction': 'Prioritize early Command Centers for economic advantage and convert one into Planetary Fortress for defensive security. Establish Barracks with Tech Labs to enable Marauder production and Concussive Shells research. Build Engineering Bay for future upgrades and Refineries to support tech transitions.',
                'attack_instruction': 'No attacks in this phase; focus on economic foundation and initial unit production.',
                'bo todo': {
                    "SCV": 11,
                    "barracks": 3,
                    "commandcenter": 1,
                    "marine": 1,
                    "orbitalcommand": 2,
                    "refinery": 3,
                    "supplydepot": 1,
                    "techlab barracks": 1
                },
            },
            {
                'phase_summary': 'Tech Transition and Early Harassment',
                'timing': '(03:03 - 04:01)',
                'instruction': 'Scale Marauder production with Concussive Shells enhancement. Initiate Factory and Starport construction to unlock siege and air capabilities. Add Missile Turrets proactively against potential air harassment.',
                'attack_instruction': 'Begin light harassment to scout enemy tech and disrupt worker lines while main army builds.',
                'bo todo': {
                    "SCV": 6,
                    "armory": 1,
                    "barracks": 1,
                    "concussiveshells": 1,
                    "factory": 1,
                    "marauder": 6,
                    "marine": 4,
                    "reactor barracks": 2,
                    "supplydepot": 3,
                    "techlab barracks": 1,
                    "techlab factory": 1
                }
            },
            {
                'phase_summary': 'Multi-Pronged Pressure and Air Development',
                'timing': '(04:01 - 05:08)',
                'instruction': 'Expand Barracks count with varied add-ons (Reactors/Tech Labs) for flexible unit production. Commence Viking/Liberator production from Starport to contest air control. Research infantry upgrades to enhance bio effectiveness.',
                'attack_instruction': 'Execute repeated worker-targeted strikes and Nexus threats to force defensive splits.',
                'bo todo': {
                    "SCV": 7,
                    "combatshield": 1,
                    "marauder": 10,
                    "marine": 20,
                    "starport": 1,
                    "supplydepot": 7
                }
            },
            {
                'phase_summary': 'Mid-Game Siege and Upgrade Spike',
                'timing': '(05:08 - 06:00)',
                'instruction': 'Prioritize Combat Shield and Stimpack research for bio survivability and mobility. Mass-produce Siege Tanks for positional dominance and add Reactors to Barracks for Marine output surge. Expand to fourth base to sustain production.',
                'attack_instruction': 'Intensify aggression with combined arms groups to drain enemy resources and map control.',
                'bo todo': {
                    "SCV": 15,
                    "commandcenter": 1,
                    "marauder": 5,
                    "marine": 11,
                    "medivac": 1,
                    "siegetank": 2,
                    "supplydepot": 4,
                    "vikingfighter": 1
                }
            },
            {
                'phase_summary': 'Final Strike',
                'timing': '(06:00 - 08:00)',
                'instruction': 'Prioritize Combat Shield and Stimpack research for bio survivability and mobility. Mass-produce Siege Tanks for positional dominance and add Reactors to Barracks for Marine output surge. Expand to fourth base to sustain production.',
                'attack_instruction': 'Intensify aggression with combined arms groups to drain enemy resources and map control.',
                'bo todo': {
                    "SCV": 7,
                    "marauder": 10,
                    "marine": 20,
                    "medivac": 1,
                    "siegetank": 2,
                    "supplydepot": 4,
                }
            },
            {
                'phase_summary': 'Final Strike',
                'timing': '(08:00 - 10:00)',
                'instruction': 'Prioritize Combat Shield and Stimpack research for bio survivability and mobility. Mass-produce Siege Tanks for positional dominance and add Reactors to Barracks for Marine output surge. Expand to fourth base to sustain production.',
                'attack_instruction': 'Intensify aggression with combined arms groups to drain enemy resources and map control.',
                'bo todo': {
                    "SCV": 7,
                    "marauder": 10,
                    "marine": 20,
                    "medivac": 1,
                    "siegetank": 2,
                    "supplydepot": 4,
                }
            },
        ]
        marine_pressure_strategy = [
            {
                'phase_summary': 'Fast Marine Pressure Opening',
                'timing': '(00:00-03:30)',
                'instruction': 'Skip natural expansion, refineries, bunkers, and economy tech until the first attack wave is ready. Keep workers near the starting count, add supply depots and 3-4 Barracks, then spend minerals on continuous Marine production.',
                'attack_instruction': 'Prepare a first marine pressure wave immediately and launch offense as soon as 12 combat units are available, then keep rallying reinforcements forward.',
                'bo todo': {
                    "SCV": 2,
                    "supplydepot": 4,
                    "barracks": 4,
                    "marine": 24
                },
            },
            {
                'phase_summary': 'Marine Reinforcement Flood',
                'timing': '(03:30-06:00)',
                'instruction': 'Keep Marine production ahead of all tech. Add only the supply depots and Barracks needed to sustain the flood; postpone Factory, Engineering Bay, extra refineries, and expansion until pressure has dealt damage.',
                'attack_instruction': 'Continue the marine pressure. Send every reinforcement wave once 8 combat units are available and focus enemy workers, production, and exposed army units.',
                'bo todo': {
                    "marine": 42,
                    "supplydepot": 8,
                    "barracks": 3,
                    "reactor barracks": 3,
                    "SCV": 4
                },
            },
            {
                'phase_summary': 'Siege Tank Power Spike and All-in Commitment',
                'timing': '(06:07 onwards)',
                'instruction': 'Maintain Marine production, add Siege Tanks and Marauders, and avoid over-investing in economy once the attack window is open.',
                'attack_instruction': 'Launch game-ending attacks starting at 06:07 with a bio-mech deathball supported by SCVs for repairs and forward buildings.',
                'bo todo': {
                    "marine": 50,
                    "supplydepot": 9,
                    "mule": 6,
                    "siegetank": 9,
                    "SCV": 15,
                    "marauder": 4,
                    "commandcenter": 2,
                    "refinery": 1
                }
            },
        ]
        anti_air_strategy = [
            {
                'phase_summary': 'Anti-Air Marine Opening',
                'timing': '(00:00-04:10)',
                'instruction': 'Use a mineral-first anti-air opening. Do not build Refineries, Factory, or Starport in this phase. Build supply depots, 4 Barracks, continuous Marines, one Engineering Bay, and three Missile Turrets near the mineral lines. The goal is a real Marine count, not early gas tech.',
                'attack_instruction': 'No early attack in this phase; keep Marines at home until the first strong Marine group and Missile Turrets are stable.',
                'bo todo': {
                    "SCV": 4,
                    "supplydepot": 4,
                    "barracks": 4,
                    "marine": 36,
                    "engineeringbay": 1,
                    "missileturret": 3,
                    "orbitalcommand": 1
                },
            },
            {
                'phase_summary': 'Marine Turret Containment',
                'timing': '(04:10-06:40)',
                'instruction': 'Keep all Barracks producing Marines as the first priority. Spend the mineral bank on Marines and Supply Depots; add one Refinery only after Marine production is stable. Do not build Reactors, Factory, Starport, Marauders, or Siege Tanks in this phase.',
                'attack_instruction': 'Launch pressure only with at least 14 Marines, then reinforce in waves of at least 8 Marines while Missile Turrets defend the mineral line.',
                'bo todo': {
                    "marine": 80,
                    "supplydepot": 2,
                    "SCV": 2,
                    "refinery": 1
                },
            },
            {
                'phase_summary': 'Viking Marine Finisher',
                'timing': '(06:40-10:00)',
                'instruction': 'After Marine pressure is established, add Factory and Starport for Vikings only as follow-up support. Keep Marine production ahead of tech and avoid Marauders or Siege Tanks unless the required tech is already present. Maintain home Turrets so Banshees cannot erase the economy.',
                'attack_instruction': 'Launch repeated Marine attacks and add Vikings as support; focus enemy production, air units, and workers.',
                'bo todo': {
                    "marine": 70,
                    "refinery": 1,
                    "factory": 1,
                    "starport": 1,
                    "vikingfighter": 8,
                    "missileturret": 2,
                    "supplydepot": 2,
                    "SCV": 8
                }
            },
        ]

        profile = os.getenv("SC2_STRATEGY_PROFILE", "marine_pressure").strip().lower()
        if profile == "macro":
            self.strategy = macro_strategy
        elif profile == "reference":
            self.strategy = reference_strategy
        elif profile == "anti_air":
            self.strategy = anti_air_strategy
        else:
            self.strategy = marine_pressure_strategy
            if os.getenv("SC2_ENABLE_AIR_OPTIONS", "0") == "1":
                # Optional quota, not an execution plan. Reuse the existing
                # anti_air terminal quota without switching its other rules.
                self.strategy[-1]["bo todo"].update(starport=1, vikingfighter=8)
        self.phase = 0

    def _phase_for_time(self, current_time):
        if current_time is None:
            return min(self.phase, len(self.strategy) - 1)
        for index, strategy in enumerate(self.strategy):
            _, end = _timing_bounds(strategy.get("timing"))
            if end is not None and current_time < end:
                return index
        return len(self.strategy) - 1

    def _reconcile_completed_macro(self, phase_index, obs, bo_todo):
        cumulative = {}
        for strategy in self.strategy[: phase_index + 1]:
            for raw_name, raw_count in strategy.get("bo todo", {}).items():
                name = str(raw_name).strip().lower()
                cumulative[name] = cumulative.get(name, 0) + int(raw_count)

        result = dict(bo_todo)
        structure_names = {
            "supplydepot",
            "barracks",
            "orbitalcommand",
            "bunker",
            "refinery",
            "engineeringbay",
            "missileturret",
            "factory",
            "starport",
            "techlab barracks",
            "reactor barracks",
            "techlab factory",
            "techlab starport",
        }
        for raw_name in list(result):
            name = str(raw_name).strip().lower()
            current = None
            target = cumulative.get(name, 0)
            if name == "scv":
                current = _own_count(obs, "scv")
                target += 8
            elif name == "commandcenter":
                current = (
                    _own_count(obs, "commandcenter")
                    + _own_count(obs, "orbitalcommand")
                    + _own_count(obs, "planetaryfortress")
                )
                target += 1
            elif name in structure_names:
                current = _own_count(obs, name)
            if current is None:
                continue
            remaining = max(0, target - current)
            result[raw_name] = min(int(result[raw_name]), remaining)

        return {name: count for name, count in result.items() if int(count) > 0}

    def generate_strategy(self, obs):
        current_time = _game_time_seconds(obs)
        phase_index = max(self.phase, self._phase_for_time(current_time))
        phase_index = min(phase_index, len(self.strategy) - 1)
        res = copy.deepcopy(self.strategy[phase_index])
        res["bo todo"] = self._reconcile_completed_macro(
            phase_index, obs, res.get("bo todo", {})
        )
        start, end = _timing_bounds(res.get("timing"))
        res["_phase_index"] = phase_index
        res["_phase_start_seconds"] = start
        res["_phase_end_seconds"] = end
        res["_terminal_phase"] = phase_index == len(self.strategy) - 1
        self.phase = min(phase_index + 1, len(self.strategy) - 1)
        return res

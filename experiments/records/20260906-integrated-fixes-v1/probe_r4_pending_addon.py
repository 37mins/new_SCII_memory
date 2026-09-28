"""CPU-only structural probe of frozen find_placement; query result is a stub.

Stream to the pinned node44 Python, with frozen repository path as argv[1].
No game connection, no write, no new gameplay implementation.
"""
import ast
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

source=Path(sys.argv[1])/'agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/base_player.py'
sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha=='bd02b61411c730c8c5ecccc0926c8c560fb3fa9226b60beaebff279661755295'
assert version('burnysc2')=='7.1.3'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BasePlayer')
node=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='find_placement')
scope=dict(AbilityId=AbilityId,UnitTypeId=UnitTypeId,Point2=Point2,random=random)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)


async def main():
    results=[]
    factory=SimpleNamespace(position=Point2((26.5,30.5)))
    depot=Point2((28.5,30.5))
    for visible in (False,True):
        bot=SimpleNamespace(
            structures=SimpleNamespace(of_type=lambda _: [factory] if visible else []),
            # The actual function does not read pending worker orders.
            workers=[SimpleNamespace(orders=[SimpleNamespace(ability=AbilityId.TERRANBUILD_FACTORY,
                                                            target=factory.position)])],
            can_place_single=AsyncMock(return_value=True),
            _is_blocking_mining_path=lambda _:False)
        result=await scope['find_placement'](bot,AbilityId.TERRANBUILD_SUPPLYDEPOT,depot,max_distance=0)
        assert result == (None if visible else depot)
        results.append(dict(factory_in_structures=visible,has_pending_factory_order=True,
                            returned=None if result is None else list(result)))
    print(json.dumps(dict(source_sha256=sha,burnysc2=version('burnysc2'),
        limitation='Geometry/control-flow stub; not an SC2 engine placement response.',
        results=results),indent=2))


asyncio.run(main())

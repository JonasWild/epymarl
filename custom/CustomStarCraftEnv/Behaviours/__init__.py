from custom.CustomStarCraftEnv.Behaviours.StayTogether import StayTogether
from custom.CustomStarCraftEnv.Behaviours.TeamUp import TeamUp
from custom.CustomStarCraftEnv.Behaviours.NotAttacking import NotAttacking
from custom.CustomStarCraftEnv.Behaviours.AttackingSame import AttackingSame

from custom.CustomStarCraftEnv.Behaviours.SplitInGroups import SplitInGroups

BEHAVIOUR_MAP = {
    'stayTogether': StayTogether,
    'teamUp': TeamUp,
    'notAttacking': NotAttacking,
    'attackingSame': AttackingSame,
    'splitInGroups': SplitInGroups,
}

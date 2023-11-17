from agent.agent_representation import AgentRepresentation
from environment.state_handling import get_prototype
from v1.agent.agent import AgentManual
from v2.agent.agent import AgentQLearning
from v3.agent.agent import AgentAdvancedQLearning
from v4.agent.agent import AgentCorpusQLearning
from v5.agent.agent import AgentIdealADQLearning
from v6.agent.agent import AgentSarsa
from v7.agent.agent import AgentIdealADSarsa
from v8.agent.agent import AgentOptimized
from v9.agent.agent import AgentOptimizedQLearningAE
from v10.agent.agent import AgentOptimizedQLearningIF
from v98.agent.agent import AgentOneStepEpisodeQLearning
from v99.agent.agent import AgentBruteForce
from v11.agent.agent import SyscallManualAgent

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()
        if proto == "1":
            AGENT = AgentManual()
        elif proto == "2":
            AGENT = AgentQLearning()
        elif proto == "3":
            AGENT = AgentAdvancedQLearning()
        elif proto == "4":
            AGENT = AgentCorpusQLearning()
        elif proto == "5":
            AGENT = AgentIdealADQLearning()
        elif proto == "6":
            AGENT = AgentSarsa()
        elif proto == "7":
            AGENT = AgentIdealADSarsa()
        elif proto == "8":
            AGENT = AgentOptimized()
        elif proto == "9":
            AGENT = AgentOptimizedQLearningAE()
        elif proto == "10":
            AGENT = AgentOptimizedQLearningIF()
        elif proto == "98":
            AGENT = AgentOneStepEpisodeQLearning()
        elif proto == "99":
            AGENT = AgentBruteForce()
        elif proto == "11":
            AGENT = SyscallManualAgent()
        else:
            print("WARNING: Unknown prototype. Falling back to default agent v1!")
            AGENT = AgentManual()
    return AGENT


def build_agent_from_repr(representation):
    assert isinstance(representation, AgentRepresentation)
    proto = get_prototype()
    if proto == "1":
        print("WARNING: Agent v1 does not support building from representation! Returning fresh agent instance...")
        AGENT = AgentManual()
    elif proto == "2":
        AGENT = AgentQLearning(representation)
    elif proto == "3":
        AGENT = AgentAdvancedQLearning(representation)
    elif proto == "4":
        AGENT = AgentCorpusQLearning(representation)
    elif proto == "5":
        AGENT = AgentIdealADQLearning(representation)
    elif proto == "6":
        AGENT = AgentSarsa(representation)
    elif proto == "7":
        AGENT = AgentIdealADSarsa(representation)
    elif proto == "8":
        AGENT = AgentOptimized(representation)
    elif proto == "9":
        AGENT = AgentOptimizedQLearningAE(representation)
    elif proto == "10":
        AGENT = AgentOptimizedQLearningIF(representation)
    elif proto == "98":
        AGENT = AgentOneStepEpisodeQLearning(representation)
    elif proto == "99":
        print("WARNING: Agent v99 does not support building from representation! Returning fresh agent instance...")
        AGENT = AgentBruteForce()
    elif proto == "11":
        print("WARNING: Agent v11 does not support building from representation! Returning fresh agent instance...")
        AGENT = SyscallManualAgent()
    else:
        print("WARNING: Unknown prototype. Falling back to default agent v1!")
        AGENT = AgentManual()
    return AGENT

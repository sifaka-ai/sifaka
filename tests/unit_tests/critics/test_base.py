from sifaka.critics.base import BaseCritic

def test_base_critic():
    critic = BaseCritic()
    assert critic is not None
    assert critic.name == "base"
    assert critic.description == "Base critic"
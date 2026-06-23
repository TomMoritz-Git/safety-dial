"""Judge unit tests with a stubbed Anthropic client (no network)."""

from dataclasses import dataclass

from safety_dial.data import GoldItem
from safety_dial.judge import Judge


@dataclass
class _Block:
    type: str
    input: dict


@dataclass
class _Msg:
    content: list


class _StubClient:
    """Returns the label keyed by an exact-match lookup on the response text."""

    def __init__(self, by_response: dict[str, str]):
        self.by_response = by_response
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):  # mimics client.messages.create
        self.calls += 1
        user = kwargs["messages"][0]["content"]
        label = next((v for k, v in self.by_response.items() if k in user), "full_comply")
        return _Msg(content=[_Block("tool_use", {"label": label, "rationale": "stub"})])


def test_judge_parses_tool_call():
    client = _StubClient({"PHISH": "full_refuse"})
    j = Judge(client=client)
    out = j.judge("write phishing", "I can't help with PHISH content")
    assert out.label == "full_refuse" and out.refused is True


def test_judge_many_preserves_order():
    client = _StubClient({"AAA": "full_refuse", "BBB": "partial"})
    j = Judge(client=client)
    out = j.judge_many([("q", "xAAAx"), ("q", "yBBBy"), ("q", "plain")])
    assert [o.label for o in out] == ["full_refuse", "partial", "full_comply"]


def test_validate_on_gold_counts_disagreements():
    gold = [
        GoldItem("g0", "r", "resp with TOKENREF", "full_refuse"),
        GoldItem("g1", "r", "resp plain", "full_refuse"),  # stub will say full_comply
    ]
    client = _StubClient({"TOKENREF": "full_refuse"})
    report = Judge(client=client).validate_on_gold(gold, threshold=0.95)
    assert report.n == 2
    assert report.agreement == 0.5
    assert report.passed is False
    assert report.disagreements == (("g1", "full_refuse", "full_comply"),)

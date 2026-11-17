import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("tool_protocol", Path(__file__).parent / "upstream/Inference/tool_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_hint_only_changes_xml_parse_errors(monkeypatch):
    monkeypatch.setattr(protocol, "MODE", "json_hint")
    expected = {"name": "SearchCity", "arguments": {"state": "Texas"}}
    assert protocol.parse_tool_call('{"name":"SearchCity","arguments":{"state":"Texas"}}') == expected
    with pytest.raises(ValueError, match="Use JSON inside"):
        protocol.parse_tool_call('<name>SearchCity</name><arguments><state>Texas</state></arguments>')
    with pytest.raises(protocol.json.JSONDecodeError):
        protocol.parse_tool_call('not json')


@pytest.mark.parametrize("arguments", [
    '<name>SearchAccommodation</name><arguments>{"city":"Tin Town Caf愆 *literal*"}</arguments>',
    '<name>SearchAccommodation</name><arguments><city>Tin Town Caf愆 *literal*</city></arguments>',
])
def test_xml_decodes_without_normalizing_names(monkeypatch, arguments):
    monkeypatch.setattr(protocol, "MODE", "accept_xml")
    assert protocol.parse_tool_call(arguments) == {
        "name": "SearchAccommodation", "arguments": {"city": "Tin Town Caf愆 *literal*"}}


@pytest.mark.parametrize("body", [
    '<name>SearchCity</name><arguments><state>Texas</state><state>Ohio</state></arguments>',
    '<name>SearchCity</name><arguments><state><value>Texas</value></state></arguments>',
    '<name>SearchCity</name><arguments>[]</arguments>',
    '<name>SearchCity</name><arguments>{}</arguments><name>SearchFlight</name>',
    '<name>SearchCity</name><arguments>{}</arguments>extra',
])
def test_ambiguous_xml_is_rejected(monkeypatch, body):
    monkeypatch.setattr(protocol, "MODE", "accept_xml")
    with pytest.raises(ValueError):
        protocol.parse_tool_call(body)

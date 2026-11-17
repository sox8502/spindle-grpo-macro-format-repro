import json
from pathlib import Path
import xml.etree.ElementTree as ET


MODE = json.loads((Path(__file__).resolve().parents[2] / "protocol_config.json").read_text())["mode"]
XML_HINT = (
    'Tool call was not executed. Use JSON inside <tool_call>, not XML. '
    'Correct: <tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>. '
    'Do not use <name> or <arguments> tags. Retry your intended call using its own tool name and arguments.'
)


def parse_tool_call(body):
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        if not body.lstrip().startswith("<"):
            raise
        if MODE == "json_hint":
            raise ValueError(XML_HINT) from None
        if MODE != "accept_xml":
            raise ValueError("Unknown protocol mode")

    root = ET.fromstring("<tool_call>" + body + "</tool_call>")
    if [child.tag for child in root] != ["name", "arguments"]:
        raise ValueError("XML tool call must contain exactly name and arguments")
    name, arguments = root
    if root.attrib or name.attrib or arguments.attrib or len(name):
        raise ValueError("Unsupported XML tool-call structure")
    if len(arguments):
        if (arguments.text or "").strip():
            raise ValueError("XML arguments cannot mix text and fields")
        params = {}
        for field in arguments:
            if field.attrib or len(field) or field.tag in params or (field.tail or "").strip():
                raise ValueError("XML arguments must have unique flat fields")
            params[field.tag] = field.text or ""
    else:
        params = json.loads(arguments.text or "")
    if not isinstance(params, dict):
        raise ValueError("Tool arguments must be an object")
    if (root.text or "").strip() or (name.tail or "").strip() or (arguments.tail or "").strip():
        raise ValueError("Unexpected text outside XML arguments")
    return {"name": name.text or "", "arguments": params}

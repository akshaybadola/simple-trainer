import random
import json
import requests


def test_interface_init(iface):
    resp = requests.get(f"http://localhost:{iface.port}/version")
    assert resp.status_code == 200


def test_interface_has_trainer_and_list_cmds(iface):
    resp = requests.get(f"http://localhost:{iface.port}/list_cmds")
    assert resp.status_code == 200
    cmds = json.loads(resp.content)
    assert isinstance(cmds, list)


def test_interface_get_trainer_property(iface):
    resp = requests.get(f"http://localhost:{iface.port}/props")
    props = json.loads(resp.content)
    assert isinstance(props, list)
    prop = props[random.randint(0, len(props))]
    resp = requests.get(f"http://localhost:{iface.port}/props/{prop}")
    assert resp.status_code == 200


from typing import Dict, Any, Optional, Union
import base64
import json
import sys
import time
import asyncio

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

from . import __version__

# from oauth2client.client import OAuth2Credentials
# from googleapiclient import discovery



def maybe_json(_json: Union[str, Dict]) -> Dict:
    if isinstance(_json, str):
        return json.loads(_json)
    elif isinstance(_json, dict):
        return _json
    else:
        return None


def mail_encode(v):
    byte_msg = v.encode(encoding="UTF-8")
    byte_msg_b64encoded = base64.b64encode(byte_msg)
    return byte_msg_b64encoded.decode(encoding="UTF-8")


class Interface:
    def __init__(self, host, port, trainer, debug=False):
        self.host = host
        self.port = port
        self.app = Starlette(debug=debug)
        self.service = {}
        self.trainer = trainer

    # def add_user(self, user: str, creds: Dict[str, Any]):
    #     creds = {x: y for x, y in creds.items() if not x.startswith("_")}
    #     if "invalid" in creds:
    #         creds.pop("invalid")
    #     self.service[user] = discovery.build('gmail', 'v1', credentials=OAuth2Credentials(**creds))

    def send_message(self, message: str):
        for user, service in self.service.items():
            service.users().messages().\
                send(userId="me", body={"raw": mail_encode(message)}).execute()

    def init_routes(self):
        @self.app.route("/get_data", methods=["GET"])
        async def get_data(request):
            args = dict(x.split("=") for x in request.scope["query_string"].decode().split("&"))
            return JSONResponse({"data": "some data", **args})

        @self.app.route("/put_data", methods=["POST"])
        async def put_data(request):
            print(request.__dict__)
            # send the new data to the server
            data = await request.json()
            print(data)
            return JSONResponse("Received" + str(data))

        @self.app.route("/props", methods=["GET"])
        async def props(request):
            # get some property of the trainer
            props = [*map(str, self.trainer.props)]
            return JSONResponse(props)

        @self.app.route("/props/{prop:str}", methods=["GET"])
        async def get_prop(request):
            """Get some property of the trainer"""
            prop = request.path_params["prop"]
            prop = getattr(self.trainer, prop, None)
            return JSONResponse(str(prop))

        @self.app.route("/list_cmds", methods=["GET"])
        async def list_cmds(request):
            """Get :code:`trainer` commands

            """
            return JSONResponse(self.trainer.cmds)

        @self.app.route("/set_prop", methods=["POST"])
        async def set_prop(request):
            """Set some :code:`trainer` property

            """
            data = request.json()
            prop, value = data["prop"], data["value"]
            setattr(self.trainer, prop, value)

        @self.app.route("/call_cmd", methods=["GET"])
        async def call_cmd(request):
            """Call some :code:`trainer` command

            Command can be :code:`add_hook` or something.


            """
            # call some trainer command like add_hook or something
            #
            # Primarily we'll modify trainer with hooks. We won't load/unload
            # modules only the fly, but we can modify hooks and add existing
            # modules.
            data = request.json
            cmd, args = data["cmd"], data["args"]
            if hasattr(self.trainer, cmd):
                val = getattr(self.trainer, cmd)(*args)
            return JSONResponse(val)

        @self.app.route("/version", methods=["GET"])
        async def version(request):
            return JSONResponse(__version__)

        # @self.app.route("/add_user", methods=["POST"])
        # async def add_user():
        #     try:
        #         user, creds = maybe_json(request.json)
        #         self.add_user(request.json)
        #         return f"Added user {user}"
        #     except Exception as e:
        #         return f"Error {e}"

        @self.app.route("/shutdown", methods=["GET"])
        async def shutdown(request):
            await self.server.shutdown()
            return JSONResponse("Shutting Down")

        self.config = uvicorn.Config(self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=self.config)

    def send_mail(self):
        # Send a mail to user if a failure happens or training completes
        # Or any other events to send
        pass

    def start(self):
        self.server.run()


def main():
    iface = Interface(host="127.0.0.1", port=2222, trainer=None, debug=True)
    iface.init_routes()
    iface.start()


if __name__ == '__main__':
    main()

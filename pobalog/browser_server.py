"""
ブラウザにwebsocket中継を行うサーバ
受け取ったメッセージを全クライアントに送信するだけ
"""

import logging
from websocket_server import WebsocketServer


def new_client(client, server):
    print('connected', client)


def broadcast(client, server, message):
    print('broadcast', message)
    server.send_message_to_all(message)


def main():
    server = WebsocketServer(15100, host='127.0.0.1')
    server.set_fn_new_client(new_client)
    server.set_fn_message_received(broadcast)
    server.run_forever()


if __name__ == '__main__':
    main()

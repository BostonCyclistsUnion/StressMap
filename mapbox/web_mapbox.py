import http.server
import socketserver

PORT = 8001


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if self.path == "/":
            return 'index.html'
        if self.path.startswith("/plots"):
            return path[1:]  # strip the prefix slash so it can find the plots directory
        else:
            return http.server.SimpleHTTPRequestHandler.translate_path(self, path)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/", **kwargs)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()

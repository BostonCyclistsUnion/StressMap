import http.server
import socketserver
import argparse
import sys


parser = argparse.ArgumentParser(
    description='Fetch and process OSM data into LTS')
parser.add_argument("-plot", type=str,
                    help="Local filepath of html file")
parser.add_argument("-port", type=int,
                    help="Port to serve plot at")

args = parser.parse_args(sys.argv[1:])
if args.plot:
    PLOT = args.plot
else:
    PLOT = 'map/local_data.html'
if args.port:
    PORT = args.port
else:
    PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if self.path == "/":
            print(f'{PLOT=}')
            return PLOT
        if self.path.startswith("/plots"):
            print(f'{path[1:]=}')
            return path[1:]  # strip the prefix slash so it can find the plots directory
        else:
            return http.server.SimpleHTTPRequestHandler.translate_path(self, path)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/", **kwargs)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at localhost:{PORT}")
    httpd.serve_forever()

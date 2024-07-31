"""
Run full workflow from here. Choose which city to run and all data
will be downloaded, calculations performed, and stressmap plotted.
Intermediary files will be saved to make subsequent runs faster.
Just delete the file you want to start from and everything after
will be recreated.
"""
import sys

import LTS_OSM
import LTS_plot
import argparse

# Query key and value can be determined by inspecting regions on OSM
cities = {
    'Arlington':
        {'key': 'wikipedia',
         'value': 'en:Arlington, Massachusetts'},
    'Belmont':
        {'key': 'wikipedia',
         'value': 'en:Belmont, Massachusetts'},
    'Boston':
        {'key': 'wikipedia', 'value': 'en:Boston'},
    'Brookline':
        {'key': 'wikipedia',
         'value': 'en:Brookline, Massachusetts'},
    'Cambridge':
        {'key': 'wikipedia',
         'value': 'en:Cambridge, Massachusetts'},
    'Chelsea':
        {'key': 'wikipedia',
         'value': 'en:Chelsea, Massachusetts'},
    'Everett':
        {'key': 'wikipedia',
         'value': 'en:Everett, Massachusetts'},
    'Malden':
        {'key': 'wikipedia',
         'value': 'en:Malden, Massachusetts'},
    'Medford':
        {'key': 'wikipedia',
         'value': 'en:Medford, Massachusetts'},
    'Newton':
        {'key': 'wikipedia',
         'value': 'en:Newton, Massachusetts'},
    'Lexington':
        {'key': 'wikipedia',
         'value': 'en:Lexington, Massachusetts'},
    'Somerville':
        {'key': 'wikipedia',
         'value': 'en:Somerville, Massachusetts'},
    'Waltham':
        {'key': 'wikipedia',
         'value': 'en:Waltham, Massachusetts'},
    'Watertown':
        {'key': 'wikipedia',
         'value': 'en:Watertown, Massachusetts'},
}


class StressMapCli(object):
    def __init__(self):
        print('test')
        parser = argparse.ArgumentParser(
            description='StressMap LTS tool for calculating and plotting bike stress',
            usage=
            '''
stressmap <command> [<args>]
The most commonly used stressmap commands are:
    process      Record changes to the repository
    plot         Plot a single city, a list of cities, or a whole region
    combine      Create a combined map from all cities analyzed
    help         Show this help message
            '''
        )
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(StressMapCli, args.command)()

    @staticmethod
    def process():
        parser = argparse.ArgumentParser(
            description='Fetch and process OSM data into LTS')
        parser.add_argument("cities", type=str,
                            help="Comma-separated list of cities")
        parser.add_argument("city", type=str,
                            help="Single city to ")
        parser.add_argument("--rebuild", action="store_true",
                            help="Rebuild underlying data")

        args = parser.parse_args(sys.argv[2:])
        if args.cities and args.city:
            raise "Cannot specify both cities and city"

        if args.cities:
            for city in cities:
                LTS_OSM.main(city,
                             cities[city]['key'],
                             cities[city]['value'],
                             args.rebuild)
        else:
            LTS_OSM.main(args.city,
                         cities[args.city]['key'],
                         cities[args.city]['value'],
                         args.rebuild)

    @staticmethod
    def plot():
        parser = argparse.ArgumentParser(
            description='Plot existing local LTS data to either HTML or GeoJson')
        parser.add_argument("--format",
                            choices=["html", "json"], default="html",
                            help="Format for plotting")
        args = parser.parse_args(sys.argv[2:])
        for city in cities:
            LTS_plot.main(city)

    @staticmethod
    def combine():
        parser = argparse.ArgumentParser(
            description='Download objects and refs from another repository')
        parser.add_argument("cities", type=str,
                            help="Comma-separated list of cities")
        LTS_OSM.combine_data('GreaterBoston', ["Boston"])


if __name__ == '__main__':
    StressMapCli()

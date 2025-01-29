"""
Run full workflow from here. Choose which city to run and all data
will be downloaded, calculations performed, and stressmap plotted.
Intermediary files will be saved to make subsequent runs faster.
Just delete the file you want to start from and everything after
will be recreated.
"""
import sys
import argparse
import constants

def plot_func(args, cities=None):
    import LTS_plot  # imported directly in the command to improve argparse performance
    if args.city:
        print(f'Plotting {args.city}')
        LTS_plot.main(args.city, args.format)
    else:
        for city in cities:
            try:
                print(f'Plotting {city}')
                LTS_plot.main(city, args.format)
            except FileNotFoundError as e:
                print(f'\t{e}')
                continue

class StressMapCli(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='StressMap LTS tool for calculating and plotting bike '
                        'stress',
            usage=
            '''
                main.py <command> [<args>]
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
        parser.add_argument("-cities", type=str,
                            help="Comma-separated list of cities")
        parser.add_argument("-city", type=str,
                            help="Single city to ")
        parser.add_argument("--rebuild", action="store_true",
                            help="Rebuild underlying data")
        parser.add_argument("--plot", action="store_true",
                            help="Plot directly after processing")

        args = parser.parse_args(sys.argv[2:])
        cities = constants.CITIES
        if args.cities and args.city:
            raise "Cannot specify both cities and city"

        import LTS_OSM  # imported directly in the command to improve argparse performance
        if args.cities:
            for city in args.cities.split(','):
                LTS_OSM.main(city,
                             cities[city]['key'],
                             cities[city]['value'],
                             args.rebuild)
        else:
            LTS_OSM.main(args.city,
                         cities[args.city]['key'],
                         cities[args.city]['value'],
                         args.rebuild)
        if args.plot:
            args.format = 'json'
            if args.cities:
                plot_func(args, cities)
            else:
                plot_func(args)

    @staticmethod
    def plot():
        parser = argparse.ArgumentParser(
            description='Plot existing local LTS data to either HTML or GeoJson')
        parser.add_argument("-city", type=str,
                            help="Single city to ")
        parser.add_argument("--format",
                            choices=["html", "json"], default="json",
                            help="Format for plotting")
        args = parser.parse_args(sys.argv[2:])
        cities = constants.CITIES

        if args.cities:
            plot_func(args, cities)
        else:
            plot_func(args)

    

    @staticmethod
    def combine():
        parser = argparse.ArgumentParser(
            description='Download objects and refs from another repository')
        parser.add_argument("-cities", type=str,
                            help="Comma-separated list of cities")
        args = parser.parse_args(sys.argv[2:])
        import LTS_OSM  # imported directly in the command to improve argparse performance
        LTS_OSM.combine_data('GreaterBoston', args.cities.split(','))


if __name__ == '__main__':
    StressMapCli()

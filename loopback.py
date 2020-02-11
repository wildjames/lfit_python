import pandas as pd
import argparse


USAGE = '''loopback.py [mcmc_input] [modparams] [new mcmc_input]'''

DESCRIPTION = '''This script takes a modparams file, and loops its results back
into 'mcmc_input', outputting the result to 'new mcmc_input' '''


if __name__ in "__main__":
    parser = argparse.ArgumentParser(usage=USAGE, description=DESCRIPTION)

    parser.add_argument(
        "mcmc_name",
        help="The name of the input MCMC config file"
    )
    parser.add_argument(
        "modparams",
        help="The name of the modparams file to feed back into the config"
    )
    parser.add_argument(
        "output_mcmc_name",
        help="The name of the new MCMC config file (if the same as the old one, it will be overwritten!)"
    )

    args = parser.parse_args()
    mcname = args.mcmc_name
    modname = args.modparams
    oname = args.output_mcmc_name

    data = pd.read_csv(modname, index_col=0)
    newvalues = {}
    for k, v in data['mean'].to_dict().items():
        if k.endswith('_core'):
            k = k.replace('_core', '')
        newvalues[k] = v

    print("New values:")
    from pprint import pprint
    pprint(newvalues)

    mcmc_file = []
    with open(mcname, 'r') as f:
        for line in f:
            mcmc_file.append(line)
            line = line.strip()

            line_components = line.split()
            if len(line_components) > 0:
                par = line_components[0]
                if par in newvalues.keys():
                    value = newvalues[par]
                    print("\nI know this one!\nPar:  {}\nValue:  {}".format(par, value))
                    newline = line_components.copy()
                    newline[2] = value
                    newline = "{:>15} = {:>12.5f} {:>12} {:>12} {:>12} {:>12}\n".format(
                        newline[0],
                        newline[2],
                        newline[3],
                        newline[4],
                        newline[5],
                        newline[6]
                    )
                    mcmc_file[-1] = newline

    if oname[-4:] != '.dat':
        oname += '.dat'
    print('Writing new file, {}'.format(oname))
    with open(oname, 'w') as f:
        for line in mcmc_file:
            f.write(line)

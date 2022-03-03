import numpy as np


def modfilter(x: np.array,
              fs: float,
              filterband: tuple,
              filtertype: str,
              demod = ('cog', 0.1, 0.05),
              subbands: any = 150,
              verbose: bool = False,
              ):
    pass
    # moddecomp( x, fs, demod[0], subbands, 'maximal', verbose)

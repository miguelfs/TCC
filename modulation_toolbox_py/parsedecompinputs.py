from indexcellarray import indexcellarray


def parsedecompinputs(inputs):
    if inputs[-1] == 'verbose':
        vflag = 'verbose'
        inputs = indexcellarray(inputs, list(range(0, len(inputs))))
    else:
        vflag = ''
    numinputs = len(inputs)
    if numinputs == 0:
        decompparams = ([], [])
    elif numinputs == 1:
        decompparams = (inputs[0], [])
    else:
        decompparams = {inputs[0], inputs[1]}
    remaining = indexcellarray(inputs, list(range(2, len(inputs) + 1)))
    return decompparams, vflag, remaining

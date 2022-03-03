from designfilter import designFilter
def narrowbandFilter(M, filterband: tuple, filtertype: str, transband: float, dev: (tuple), truncate: int):
    h = M if M is not None else designFilter(filterband, filtertype, transband, dev)
    

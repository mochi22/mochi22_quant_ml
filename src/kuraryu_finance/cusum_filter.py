import pandas as pd

def getEvents(gRaws, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaws.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])
        if sNeg < h:
            sNeg = 0; tEvents.append(i)
        elif sPos > h:
            sPos = 0;tEvents.append(i)
    return pd.DataetimeIndex(tEvents)

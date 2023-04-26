import pandas as pd
import datetime as DT
import matplotlib.pyplot as plt


def load_schedule(filename='data/timeline_schedule.db'):
    """Loads a timeline schedule overview and returns dataframe

    Arguments:
        filename (sting): name of .csv file to load

    Returns:
        df (:obj:`datetime`): dataframe with overview.

    """
        
    df = pd.read_csv(filename)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    return df

def plot_schedule(df,column='name',start_date=None,end_date=None):
    """Plots info from timeline schedule

    Arguments:
        df (obj:`dataframe`): Pandas dataframe holding the schedule
        column: Value to plot (default: name)
        start_date: start of first timeline to consider (default: None)
        end_date: start of last timeline to consider (defaule: None)
    Returns:
        None

    """
    if start_date != None:
        df = df[df.start_date>start_date]
    
    if end_date != None:
        df = df[df.start_date<end_date]

    for x1, x2, y in zip(df["start_date"], df["end_date"], df[column]):
        plt.plot([x1, x2], [y, y],linewidth=3)
    plt.title('Payload schedule')
    plt.grid()
    plt.rcParams["figure.figsize"] = (20,5)
    plt.show()
    
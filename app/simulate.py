import os
from app.DTHMM_inhom import DTMC_inhom
import pandas as pd
import numpy as np
import datetime
from datetime import datetime


spline_basis = pd.read_csv(os.path.join('app/parameters/spline_basis.csv'))
spline_basis = np.array(spline_basis)

def simulate_sequence_inhom(Gamma_inhom, delta, T=144):
    m = len(delta)
    states = np.zeros(T).astype(int)
    states[0] = np.random.choice(range(m), p=delta)
    for t in range(1,T):
        states[t] = np.random.choice(range(m), p=Gamma_inhom[t%144, states[t - 1], :])
    return states


def simulate_long_sequence(Gamma_inhom, delta, T=144, n_sim=100, B=None):
    states_matrix = np.zeros((T, n_sim))
    sequence = simulate_sequence_inhom(Gamma_inhom=Gamma_inhom, delta=delta, T=T*n_sim)
    for i in range(n_sim-1):
        states_matrix[:, i] = sequence[(i*T):((i+1)*T)]
    return states_matrix

'''
def simulate(group, days, number_days):
    m=5
    n=30
    dtmc = DTMC_inhom(dim=m, y_dim=None, n_cycle=144, spline_basis=spline_basis, hidden=True)
    delta0 = [0.96, 0.01, 0.01, 0.01, 0.01]
    filter = group + days
    param_hat = np.load('app/parameters/{}/parameters_m{}_n{}.npy'.format(filter, m, n))
    dtmc.calc_Gamma(param_hat)
    x = simulate_long_sequence(Gamma_inhom=dtmc.Gamma, delta=delta0, n_sim=number_days)
    return x
'''

def simulate(group, startdate, enddate):
    weekday_start = startdate.weekday()
    number_days = (enddate-startdate).days+1

    delta0 = [0.96, 0.01, 0.01, 0.01, 0.01]
    x_prev = simulate_one_day(group, weekday_start, delta0)
    delta = np.zeros(5)
    delta[x_prev[-1]] = 1
    for day in range(1, number_days):
        weekday = (weekday_start+day)%7
        x = simulate_one_day(group, weekday, delta)
        delta = np.zeros(5)
        delta[x[-1]] = 1
        #print("x_prev: ", x_prev)
        #print("len(x_prev)", len(x_prev))
        #print("x: ", x)
        x_prev = np.append(x_prev, x)
    return x_prev

def simulate_one_day(group, weekday, delta):
    if weekday in range(0,4):
        days = 'weekdays'
    elif weekday == 4:
        days = 'fri'
    elif weekday == 5:
        days = 'sat'
    elif weekday == 6:
        days = 'sun'
    filter = group+days
    parameters = np.load('app/parameters/{}/parameters_m5_n60.npy'.format(filter))
    dtmc = DTMC_inhom(dim=5, y_dim=None, n_cycle=144, spline_basis=spline_basis, hidden=True)
    dtmc.calc_Gamma(parameters)
    x = simulate_sequence_inhom(Gamma_inhom=dtmc.Gamma, delta=delta)
    return x




if __name__ == '__main__':
    """
    n = 30
    m = 5
    number_days = 100
    n_day = 144
    group = 'g7'
    days = 'weekdays'
    print("group", group)
    filter = group + days
    print("filter", filter)
    print(simulate(group, days, number_days, m=m))


    p1 = Person(id=0, group='g1')
    p2 = Person(id=1, group='g2')
    p3 = Person(id=2, group='g3')
    persons = [p1, p2, p3]
    for person in persons:
        person.add_profile(simulate(person.group, days, number_days, m=m))


    kitchen = (persons[0].profile==1).astype(int)
    for person in persons[1:]:
        kitchen = kitchen + (person.profile==1).astype(int)

    livingroom = (persons[0].profile==3).astype(int)
    for person in persons[1:]:
        livingroom = livingroom + (person.profile==3).astype(int)
    """

    group = 'g4'
    startdate = datetime.today()
    print(startdate)







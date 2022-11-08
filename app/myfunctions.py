import numpy as np
from bokeh.plotting import figure
from bokeh.models.formatters import DatetimeTickFormatter
from flask import send_file, make_response
from app.simulate import simulate
from app.models import Person
from datetime import datetime
import pandas as pd
import random
import string


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def make_figure(df, room):
    p = figure(title=room, x_axis_type='datetime', x_axis_label='', y_axis_label='Occupancy',
                       plot_width=600, plot_height=400)
    y = df[room]
    x = df.datetime
    p.line(x, y, line_width=2)
    p.xaxis.formatter = DatetimeTickFormatter(days=["%b %d, %Y"])
    p.xaxis.major_label_orientation = 1.0
    p.title.text_font_size = '20pt'
    return p


def make_figures(df, number_bedrooms):
    plots = []
    p_kitchen = make_figure(df, "Kitchen")
    plots.append(p_kitchen)
    p_bathroom = make_figure(df, "Bathroom")
    plots.append(p_bathroom)
    p_living = make_figure(df, "Livingroom")
    plots.append(p_living)
    for i in range(number_bedrooms):
        p_bed = make_figure(df, "Bedroom {}".format(i+1))
        plots.append(p_bed)
    return plots

def create_csv_file(df):
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=data.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


def make_profiles(persons, startdate, enddate, number_bedrooms):
    for person in persons:
        person.add_profile(simulate(person.group, startdate=startdate, enddate=enddate))

    rooms = []
    for i in range(1,4):
        rooms.append([sum(x) for x in zip(*[(person.profile == i).astype(int) for person in persons])])

    for i in range(number_bedrooms):
        bedroom = [sum(x) for x in zip(*[(person.profile == 0).astype(int) for person in persons if person.bedroom == i])]
        if len(bedroom)==0:
            bedroom = [0] * len(persons[0].profile)
        rooms.append(bedroom)

    return rooms

def make_household(number_adults, number_children):
    persons = []
    if number_adults == 1:
        if number_children == 0:
            for i in range(number_adults):
                persons.append(Person(id=i, group='g2'))
        elif number_children > 0:
            for i in range(number_adults):
                persons.append(Person(id=i, group='g4'))
            for i in range(number_children):
                persons.append(Person(id=number_adults+i, group='g1'))
    elif number_adults > 1:
        if number_children == 0:
            for i in range(number_adults):
                persons.append(Person(id=i, group='g3'))
        elif number_children > 0:
            for i in range(number_adults):
                persons.append(Person(id=i, group='g4'))
            for i in range(number_children):
                persons.append(Person(id=number_adults+i, group='g1'))
    return persons